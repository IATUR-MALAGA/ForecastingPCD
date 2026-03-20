import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from back.api.routes import SARIMAX_model as sarimax_module
from back.database.repository import get_names_in_table_catalog


def format_mape(mape):
    return "N/A" if mape is None else f"{mape:.2f}%"


def _unique(seq):
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _choose_targets(entries, max_targets):
    names = sorted({e.get("nombre") for e in entries if e and e.get("nombre")})
    if max_targets is None or int(max_targets) <= 0:
        return names
    return names[: int(max_targets)]


def _candidate_exogs(target, entries, max_exogs):
    row = next((e for e in entries if e.get("nombre") == target), {})
    table = row.get("nombre_tabla")
    cat = row.get("categoria")

    same_table = [
        e.get("nombre")
        for e in entries
        if e.get("nombre") and e.get("nombre") != target and e.get("nombre_tabla") == table
    ]
    same_cat = [
        e.get("nombre")
        for e in entries
        if e.get("nombre") and e.get("nombre") != target and e.get("categoria") == cat
    ]
    all_other = [e.get("nombre") for e in entries if e.get("nombre") and e.get("nombre") != target]

    ordered = _unique([*same_table, *same_cat, *all_other])
    if max_exogs is None or int(max_exogs) <= 0:
        return ordered
    return ordered[: int(max_exogs)]


def _slice_for_cap(items, cap):
    if cap is None or int(cap) <= 0:
        return list(items)
    return list(items)[: int(cap)]


def build_cases(entries, max_targets=0, max_exogs=0, suite="ultra", max_pairs=0, max_trios=0):
    order_profiles = {
        "default": ((1, 1, 1), (0, 1, 1, 12)),
        "simple": ((1, 0, 0), (0, 0, 0, 0)),
        "trend": ((2, 1, 0), (0, 1, 1, 12)),
        "seasonal_light": ((1, 1, 0), (0, 1, 0, 12)),
    }

    suite = (suite or "ultra").strip().lower()

    cases = []
    for target in _choose_targets(entries, max_targets):
        exogs = _candidate_exogs(target, entries, max_exogs)

        cases.append({"indicator": target, "exog": [], "profile": "default"})
        cases.append({"indicator": target, "exog": [], "profile": "simple"})

        for ex in exogs:
            cases.append({"indicator": target, "exog": [ex], "profile": "default"})

        pairs = _slice_for_cap(combinations(exogs, 2), max_pairs)
        for pair in pairs:
            cases.append({"indicator": target, "exog": list(pair), "profile": "default"})

        trios = _slice_for_cap(combinations(exogs, 3), max_trios)
        for trio in trios:
            cases.append({"indicator": target, "exog": list(trio), "profile": "default"})

        if exogs:
            cases.append({"indicator": target, "exog": exogs, "profile": "simple"})

        if suite == "ultra":
            anchor_sets = [[], exogs[:3], exogs[:5], exogs]
            for aset in anchor_sets:
                aset = [x for x in aset if x]
                for profile in ["trend", "seasonal_light"]:
                    cases.append({"indicator": target, "exog": aset, "profile": profile})

    for c in cases:
        c["order"], c["seasonal_order"] = order_profiles[c["profile"]]

    dedup = []
    seen = set()
    for c in cases:
        key = (c["indicator"], tuple(c["exog"]), c["profile"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(c)

    return dedup


def evaluate_case(case, periods: int, train_ratio: float, optimize: bool):
    t0 = time.time()
    try:
        req = sarimax_module.SarimaxRunRequest(
            target_var=case["indicator"],
            predictors=list(case["exog"]),
            filters_by_var=None,
            train_ratio=float(train_ratio),
            auto_params=bool(optimize),
            s=12,
            order=None if optimize else tuple(case["order"]),
            seasonal_order=None if optimize else tuple(case["seasonal_order"]),
            horizon=int(periods),
            return_df=False,
        )
        resp = sarimax_module.sarimax_run(req)
        data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)

        return {
            "success": True,
            "case": case,
            "mape": float(data.get("mape")) if data.get("mape") is not None else None,
            "rmse": float(data.get("rmse")) if data.get("rmse") is not None else None,
            "mae": float(data.get("mae")) if data.get("mae") is not None else None,
            "train_size": data.get("n_train"),
            "test_size": data.get("n_test"),
            "elapsed": round(time.time() - t0, 3),
        }
    except HTTPException as e:
        return {
            "success": False,
            "case": case,
            "error": f"HTTP {e.status_code}: {e.detail}",
            "elapsed": round(time.time() - t0, 3),
        }
    except Exception as e:
        return {
            "success": False,
            "case": case,
            "error": f"{type(e).__name__}: {e}",
            "elapsed": round(time.time() - t0, 3),
        }


def save_reports(results, errors, threshold: float, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    ok = [r for r in results if r.get("success")]
    high = [r for r in ok if r.get("mape") is not None and r["mape"] > threshold]

    mape_values = [r["mape"] for r in ok if r.get("mape") is not None]
    summary = {
        "total": len(results),
        "successful": len(ok),
        "failed": len(errors),
        "high_mape": len(high),
        "threshold": threshold,
        "avg_mape": (sum(mape_values) / len(mape_values)) if mape_values else None,
    }

    payload = {"summary": summary, "results": ok, "errors": errors}

    json_path = outdir / "sarimax_stress_results.json"
    md_path = outdir / "sarimax_stress_report.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Stress Test SARIMAX (DB real)\n\n")
        f.write(f"- Total: {summary['total']}\n")
        f.write(f"- Exitosos: {summary['successful']}\n")
        f.write(f"- Fallidos: {summary['failed']}\n")
        f.write(f"- MAPE > {threshold}%: {summary['high_mape']}\n")
        if summary["avg_mape"] is not None:
            f.write(f"- MAPE promedio: {summary['avg_mape']:.2f}%\n")

        f.write("\n## Casos con MAPE alto\n\n")
        f.write("| Indicador | Exogenas | Perfil | MAPE | RMSE | MAE |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in sorted(high, key=lambda x: x["mape"], reverse=True):
            c = r["case"]
            ex = ", ".join(c["exog"]) if c["exog"] else "(ninguna)"
            f.write(
                f"| {c['indicator']} | {ex} | {c['profile']} | {format_mape(r['mape'])} | {r['rmse']:.2f} | {r['mae']:.2f} |\n"
            )

    return json_path, md_path, summary


def _print_case_breakdown(cases):
    by_target = {}
    by_len = {"0": 0, "1": 0, "2": 0, "3+": 0}
    for c in cases:
        t = c["indicator"]
        by_target[t] = by_target.get(t, 0) + 1
        n_ex = len(c.get("exog") or [])
        if n_ex == 0:
            by_len["0"] += 1
        elif n_ex == 1:
            by_len["1"] += 1
        elif n_ex == 2:
            by_len["2"] += 1
        else:
            by_len["3+"] += 1

    print("Distribucion de casos por indicador:")
    for t, n in sorted(by_target.items(), key=lambda kv: kv[0]):
        print(f"  - {t}: {n}")
    print(
        "Distribucion por numero de exogenas: "
        f"0={by_len['0']}, 1={by_len['1']}, 2={by_len['2']}, 3+={by_len['3+']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Stress test SARIMAX con datos reales de BD")
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--periods", type=int, default=3)
    parser.add_argument("--rows", type=int, default=0, help="No usado (compatibilidad)")
    parser.add_argument("--seed", type=int, default=42, help="No usado (compatibilidad)")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--suite", choices=["realista", "ultra"], default="ultra")
    parser.add_argument("--max-targets", type=int, default=0, help="0 = todos")
    parser.add_argument("--max-exogs", type=int, default=0, help="0 = todas")
    parser.add_argument("--max-pairs", type=int, default=0, help="0 = todos")
    parser.add_argument("--max-trios", type=int, default=0, help="0 = todos")
    parser.add_argument("--max-cases", type=int, default=0, help="0 = todos")
    parser.add_argument("--outdir", type=str, default="reports/stress")
    args = parser.parse_args()

    entries = get_names_in_table_catalog() or []
    cases = build_cases(
        entries,
        max_targets=args.max_targets,
        max_exogs=args.max_exogs,
        suite=args.suite,
        max_pairs=args.max_pairs,
        max_trios=args.max_trios,
    )
    if args.max_cases and int(args.max_cases) > 0:
        cases = cases[: int(args.max_cases)]

    print("=" * 60)
    print("STRESS TEST SARIMAX (DB real)")
    print("=" * 60)
    print(
        f"Suite: {args.suite} | Casos: {len(cases)} | Workers: {args.workers} | Optimize: {args.optimize}"
    )
    _print_case_breakdown(cases)

    results = []
    errors = []
    interrupted = False

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        fut_map = {
            pool.submit(evaluate_case, c, args.periods, args.train_ratio, args.optimize): c
            for c in cases
        }
        done = 0
        try:
            for fut in as_completed(fut_map):
                res = fut.result()
                done += 1
                if done % 5 == 0 or done == len(cases):
                    print(f"Progreso: {done}/{len(cases)}")
                if res.get("success"):
                    results.append(res)
                else:
                    errors.append(res)
        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupcion detectada. Guardando resultados parciales...")
            for fut in fut_map:
                fut.cancel()

    json_path, md_path, summary = save_reports(
        results=results + errors,
        errors=errors,
        threshold=args.threshold,
        outdir=Path(args.outdir),
    )

    print("\nResumen:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")
    if interrupted:
        print("Ejecucion interrumpida por usuario. Reportes parciales guardados.")


if __name__ == "__main__":
    main()
