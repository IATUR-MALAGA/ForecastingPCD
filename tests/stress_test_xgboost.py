import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from back.api.routes import XGBoost_model as xgb_module
from back.database.repository import (
    get_names_in_table_catalog,
    get_tableName_for_variable,
    get_variable_definition,
)
from back.services.dataframe_selection import create_dataframe_based_on_selection
from back.utils.column_utils import _safe_alias, build_time_index


def format_mape(mape):
    return "N/A" if mape is None else f"{mape:.2f}%"


def _mean(values):
    clean = [v for v in values if v is not None]
    return (sum(clean) / len(clean)) if clean else None


def _pct_change(current, baseline):
    if current is None or baseline in (None, 0):
        return None
    return ((current - baseline) / baseline) * 100.0


def _exog_usage_label(exog):
    return "Con exogenas" if exog else "Sin exogenas"


def _structural_hypothesis(indicator):
    if indicator == "Velocidad media":
        return (
            "Ahora mismo parece estructural: harian falta mas exogenas meteorologicas "
            "(rachas, direccion, presion, humedad) o un enfoque/modelo distinto."
        )
    if indicator == "Temperatura media por estación":
        return (
            "Ahora mismo parece estructural: haria falta tuning especifico de estacionalidad "
            "o mas senales meteorologicas complementarias."
        )
    return (
        "Ahora mismo parece estructural: haria falta una senal exogena mas informativa "
        "o tuning especifico por variable."
    )


def _structural_summary_text(row, threshold):
    return (
        f"Esta variable no tiene ninguna combinacion con MAPE < {threshold:.1f}% en este modelo. "
        f"Mejor MAPE observado: {format_mape(row['best_mape'])}. {row['improvement_hypothesis']}"
    )


def _summarize_by_indicator_and_exog(results, threshold):
    grouped = {}
    for row in results:
        if not row.get("success") or row.get("mape") is None:
            continue
        case = row["case"]
        key = (case["indicator"], _exog_usage_label(case.get("exog") or []))
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (indicator, exog_usage), rows in sorted(grouped.items()):
        best = min(rows, key=lambda item: item["mape"])
        worst = max(rows, key=lambda item: item["mape"])
        summary_rows.append(
            {
                "indicator": indicator,
                "exog_usage": exog_usage,
                "cases": len(rows),
                "avg_mape": _mean([r.get("mape") for r in rows]),
                "high_mape": sum(1 for r in rows if r["mape"] > threshold),
                "best_profile": best["case"]["profile"],
                "best_mape": best["mape"],
                "worst_profile": worst["case"]["profile"],
                "worst_mape": worst["mape"],
                "structural_limit": best["mape"] > threshold,
                "improvement_hypothesis": (
                    _structural_hypothesis(indicator)
                    if best["mape"] > threshold
                    else None
                ),
            }
        )
    return summary_rows


def _summarize_profiles(results, threshold):
    rows = [r for r in results if r.get("success") and r.get("mape") is not None]
    grouped = {}
    group_baselines = {}
    indicator_baselines = {}

    for row in rows:
        case = row["case"]
        indicator = case["indicator"]
        exog_usage = _exog_usage_label(case.get("exog") or [])
        grouped.setdefault((indicator, exog_usage, case["profile"]), []).append(row)

        if exog_usage == "Sin exogenas" and case["profile"] == "default":
            indicator_baselines.setdefault(indicator, []).append(row["mape"])
        if case["profile"] == "default":
            group_baselines.setdefault((indicator, exog_usage), []).append(row["mape"])

    summary_rows = []
    for (indicator, exog_usage, profile), profile_rows in sorted(grouped.items()):
        avg_mape = _mean([r["mape"] for r in profile_rows])
        baseline_group = _mean(group_baselines.get((indicator, exog_usage), []))
        baseline_indicator = _mean(indicator_baselines.get(indicator, []))
        delta_group = None if baseline_group is None else avg_mape - baseline_group
        delta_indicator = (
            None if baseline_indicator is None else avg_mape - baseline_indicator
        )
        summary_rows.append(
            {
                "indicator": indicator,
                "exog_usage": exog_usage,
                "profile": profile,
                "cases": len(profile_rows),
                "avg_mape": avg_mape,
                "high_mape": sum(1 for r in profile_rows if r["mape"] > threshold),
                "baseline_same_group": baseline_group,
                "delta_vs_group": delta_group,
                "delta_vs_group_pct": _pct_change(avg_mape, baseline_group),
                "baseline_no_exog_default": baseline_indicator,
                "delta_vs_no_exog_default": delta_indicator,
                "delta_vs_no_exog_default_pct": _pct_change(
                    avg_mape, baseline_indicator
                ),
            }
        )
    return summary_rows


MALAGA_FILTER_VALUES = {
    "Número de pasajeros por aeropuerto": ("aeropuerto", "MALAGA COSTA DEL SOL"),
    "Número de turistas por municipio": ("mun_dest", "Málaga"),
    "Temperatura media por estación": ("nombre", "MÁLAGA AEROPUERTO"),
    "Velocidad media": ("nombre", "MÁLAGA AEROPUERTO"),
}


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


def _normalize_temporality(value):
    text = (value or "").strip().lower()
    if not text:
        return ""
    if "dia" in text or "daily" in text:
        return "daily"
    if "mes" in text or "monthly" in text:
        return "monthly"
    if text in {"d", "day"}:
        return "daily"
    if text in {"m", "month"}:
        return "monthly"
    return text


def _infer_temporality(entry):
    temporality = _normalize_temporality(entry.get("temporalidad"))
    if temporality:
        return temporality
    granularity = _normalize_temporality(entry.get("granularidad"))
    if granularity:
        return granularity
    table = (entry.get("nombre_tabla") or "").strip().lower()
    if "diaria" in table or "daily" in table:
        return "daily"
    if "mensual" in table or "monthly" in table:
        return "monthly"
    return ""


def _enrich_entries(entries):
    enriched = []
    for entry in entries:
        row = dict(entry)
        meta = get_variable_definition(row.get("nombre")) or {}
        if meta:
            row.setdefault("temporalidad", meta.get("temporalidad"))
            row.setdefault("granularidad", meta.get("granularidad"))
        row["_temporality"] = _infer_temporality(row)
        enriched.append(row)
    return enriched


def _candidate_exogs(target, entries, max_exogs):
    row = next((e for e in entries if e.get("nombre") == target), {})
    table = row.get("nombre_tabla")
    cat = row.get("categoria")
    temporality = row.get("_temporality")

    compatible_entries = [
        e
        for e in entries
        if e.get("nombre")
        and e.get("nombre") != target
        and (not temporality or e.get("_temporality") == temporality)
    ]

    same_table = [
        e.get("nombre") for e in compatible_entries if e.get("nombre_tabla") == table
    ]
    same_cat = [
        e.get("nombre") for e in compatible_entries if e.get("categoria") == cat
    ]
    all_other = [e.get("nombre") for e in compatible_entries]

    ordered = _unique([*same_table, *same_cat, *all_other])
    if max_exogs is None or int(max_exogs) <= 0:
        return ordered
    return ordered[: int(max_exogs)]


def _slice_for_cap(items, cap):
    if cap is None or int(cap) <= 0:
        return list(items)
    return list(items)[: int(cap)]


def _resolve_predictor_columns(df, predictors):
    resolved = []
    seen = set()
    for predictor in predictors or []:
        base_alias = _safe_alias(predictor)
        matches = [
            col
            for col in df.columns
            if col == base_alias or col.startswith(f"{base_alias}__")
        ]
        candidate_cols = matches or [base_alias]
        for col in candidate_cols:
            if col in seen:
                continue
            seen.add(col)
            resolved.append(col)
    return resolved


def _contiguous_true_runs(mask):
    runs = []
    start = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _build_past_window(case, train_ratio):
    filters_by_var = _build_filters_by_var(case)
    df = create_dataframe_based_on_selection(
        target_var=case["indicator"],
        predictors=list(case["exog"]),
        filters_by_var=filters_by_var,
    )
    if df is None or df.empty:
        raise HTTPException(
            status_code=422, detail="El dataframe resultante está vacío"
        )

    df, _, _, _ = build_time_index(df)
    y_col = _safe_alias(case["indicator"])
    predictors = _resolve_predictor_columns(df, case["exog"])
    required_cols = [y_col, *predictors]
    valid_mask = df[required_cols].notna().all(axis=1).tolist()
    runs = _contiguous_true_runs(valid_mask)
    if not runs:
        raise HTTPException(
            status_code=422, detail="No hay tramo histórico válido para evaluar el caso"
        )

    best_start, best_end = max(runs, key=lambda item: item[1] - item[0] + 1)
    run_len = best_end - best_start + 1
    if run_len < 2:
        raise HTTPException(
            status_code=422,
            detail="Tramo histórico válido insuficiente para evaluar el caso",
        )

    split_offset = max(1, min(run_len - 1, int(run_len * float(train_ratio))))
    start_dt = df.iloc[best_start + split_offset]["__dt"]
    end_dt = df.iloc[best_end]["__dt"]
    return {
        "start": start_dt.strftime("%Y-%m-%d"),
        "end": end_dt.strftime("%Y-%m-%d"),
    }


def _build_filters_by_var(case):
    filters = {}
    variables = [case["indicator"], *(case.get("exog") or [])]
    for var in variables:
        filter_def = MALAGA_FILTER_VALUES.get(var)
        rows = get_tableName_for_variable(var) or []
        table = rows[0].get("nombre_tabla") if rows else None
        if not filter_def or not table:
            continue
        column, value = filter_def
        filters[var] = [{"table": table, "col": column, "values": [value]}]
    return filters


def build_cases(
    entries, max_targets=0, max_exogs=0, suite="ultra", max_pairs=0, max_trios=0
):
    xgb_profiles = {
        "default": {
            "n_estimators": 220,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": 42,
        },
        "conservador": {
            "n_estimators": 120,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 4.0,
            "reg_alpha": 0.6,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": 42,
        },
        "agresivo": {
            "n_estimators": 500,
            "max_depth": 7,
            "learning_rate": 0.1,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
            "reg_lambda": 0.2,
            "reg_alpha": 0.0,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": 42,
        },
        "regularizado": {
            "n_estimators": 250,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 8.0,
            "reg_alpha": 1.5,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": 42,
        },
        "ligero": {
            "n_estimators": 150,
            "max_depth": 3,
            "learning_rate": 0.08,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": 42,
        },
    }

    lag_profiles = {
        "sin_lags": 0,
        "lag2": 2,
        "lag3": 3,
        "lag6": 6,
        "lag12": 12,
    }

    suite = (suite or "ultra").strip().lower()

    cases = []
    for target in _choose_targets(entries, max_targets):
        exogs = _candidate_exogs(target, entries, max_exogs)

        # Baselines
        cases.append(
            {
                "indicator": target,
                "exog": [],
                "profile": "default",
                "max_lag": lag_profiles["lag6"],
            }
        )
        cases.append(
            {
                "indicator": target,
                "exog": [],
                "profile": "conservador",
                "max_lag": lag_profiles["lag3"],
            }
        )

        # Individuales
        for ex in exogs:
            cases.append(
                {
                    "indicator": target,
                    "exog": [ex],
                    "profile": "default",
                    "max_lag": lag_profiles["lag6"],
                }
            )

        # Pares y trios
        pairs = _slice_for_cap(combinations(exogs, 2), max_pairs)
        for pair in pairs:
            cases.append(
                {
                    "indicator": target,
                    "exog": list(pair),
                    "profile": "default",
                    "max_lag": lag_profiles["lag6"],
                }
            )

        trios = _slice_for_cap(combinations(exogs, 3), max_trios)
        for trio in trios:
            cases.append(
                {
                    "indicator": target,
                    "exog": list(trio),
                    "profile": "default",
                    "max_lag": lag_profiles["lag6"],
                }
            )

        if exogs:
            cases.append(
                {
                    "indicator": target,
                    "exog": exogs,
                    "profile": "default",
                    "max_lag": lag_profiles["lag3"],
                }
            )

        if suite == "ultra":
            anchor_sets = [[], exogs[:3], exogs[:5], exogs]
            for aset in anchor_sets:
                aset = [x for x in aset if x]
                for profile in ["conservador", "regularizado", "ligero", "agresivo"]:
                    for lag_name in ["sin_lags", "lag2", "lag3", "lag6", "lag12"]:
                        if not aset and lag_name == "sin_lags":
                            continue
                        cases.append(
                            {
                                "indicator": target,
                                "exog": aset,
                                "profile": profile,
                                "max_lag": lag_profiles[lag_name],
                            }
                        )

    for c in cases:
        c["xgb_params"] = xgb_profiles[c["profile"]]

    # Deduplicar
    dedup = []
    seen = set()
    for c in cases:
        key = (c["indicator"], tuple(c["exog"]), c["profile"], c["max_lag"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(c)

    return dedup


def evaluate_case(case, periods: int, train_ratio: float, optimize: bool):
    t0 = time.time()
    try:
        filters_by_var = _build_filters_by_var(case)
        window = _build_past_window(case, train_ratio)
        req = xgb_module.XGBoostRunRequest(
            target_var=case["indicator"],
            predictors=list(case["exog"]),
            filters_by_var=filters_by_var,
            train_ratio=float(train_ratio),
            auto_params=bool(optimize),
            xgb_params=None if optimize else dict(case["xgb_params"]),
            use_target_lags=True,
            max_lag=int(case["max_lag"]),
            recursive_forecast=True,
            scenario_mode="past",
            scenario_window=window,
            horizon=int(periods),
            return_df=False,
        )
        resp = xgb_module.xgboost_run(req)
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
    grouped_summary = _summarize_by_indicator_and_exog(ok, threshold)
    profile_summary = _summarize_profiles(ok, threshold)
    summary = {
        "total": len(results),
        "successful": len(ok),
        "failed": len(errors),
        "high_mape": len(high),
        "threshold": threshold,
        "avg_mape": (sum(mape_values) / len(mape_values)) if mape_values else None,
    }

    payload = {
        "summary": summary,
        "by_indicator_exog": grouped_summary,
        "by_indicator_exog_profile": profile_summary,
        "results": ok,
        "errors": errors,
    }

    json_path = outdir / "xgboost_stress_results.json"
    md_path = outdir / "xgboost_stress_report.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Stress Test XGBoost (DB real)\n\n")
        f.write(f"- Total: {summary['total']}\n")
        f.write(f"- Exitosos: {summary['successful']}\n")
        f.write(f"- Fallidos: {summary['failed']}\n")
        f.write(f"- MAPE > {threshold}%: {summary['high_mape']}\n")
        if summary["avg_mape"] is not None:
            f.write(f"- MAPE promedio: {summary['avg_mape']:.2f}%\n")

        f.write("\n## Como leer este reporte\n\n")
        f.write(
            "- Exogenas: variables externas usadas para ayudar a predecir la variable objetivo.\n"
        )
        f.write(
            "- MAPE: error porcentual medio. Mas bajo es mejor; por debajo del umbral se considera aceptable en este stress test.\n"
        )
        f.write("- RMSE: error cuadratico medio. Penaliza mas los fallos grandes.\n")
        f.write(
            "- MAE: error absoluto medio. Mide el error medio en unidades originales.\n"
        )
        f.write("- Perfil: configuracion concreta del modelo.\n")
        f.write(
            "- Baseline: referencia contra la que se compara una configuracion. Aqui usamos el perfil `default`.\n"
        )
        f.write(
            "- Delta vs default grupo: cuanto mejora o empeora un perfil frente al `default` del mismo indicador y mismo uso de exogenas. Negativo = mejora.\n"
        )
        f.write(
            "- Delta vs default sin exogenas: cuanto mejora o empeora un perfil frente al `default` sin exogenas del mismo indicador. Negativo = mejora.\n"
        )
        f.write(
            "- Limitacion estructural actual: incluso la mejor combinacion probada sigue por encima del umbral; probablemente no se arregla solo cambiando de perfil.\n"
        )
        f.write("\n## Resumen por indicador y exogenas\n\n")
        f.write(
            "| Indicador | Uso exogenas | Casos | MAPE prom. | MAPE > umbral | Mejor perfil | Peor perfil |\n"
        )
        f.write("|---|---|---|---|---|---|---|\n")
        for row in grouped_summary:
            f.write(
                f"| {row['indicator']} | {row['exog_usage']} | {row['cases']} | {format_mape(row['avg_mape'])} | {row['high_mape']}/{row['cases']} | {row['best_profile']} ({format_mape(row['best_mape'])}) | {row['worst_profile']} ({format_mape(row['worst_mape'])}) |\n"
            )

        structural_rows = [row for row in grouped_summary if row["structural_limit"]]
        if structural_rows:
            f.write("\n## Limitaciones estructurales actuales\n\n")
            for row in structural_rows:
                f.write(
                    f"- {row['indicator']} [{row['exog_usage']}]: {_structural_summary_text(row, threshold)}\n"
                )

        f.write("\n## Mejora por perfil vs baseline\n\n")
        f.write(
            "| Indicador | Uso exogenas | Perfil | MAPE prom. | Delta vs default grupo | Delta vs default sin exogenas | MAPE > umbral |\n"
        )
        f.write("|---|---|---|---|---|---|---|\n")
        for row in profile_summary:
            delta_group = format_mape(row["delta_vs_group"])
            delta_indicator = format_mape(row["delta_vs_no_exog_default"])
            f.write(
                f"| {row['indicator']} | {row['exog_usage']} | {row['profile']} | {format_mape(row['avg_mape'])} | {delta_group} | {delta_indicator} | {row['high_mape']}/{row['cases']} |\n"
            )

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
    by_temporality = {}
    for c in cases:
        t = c["indicator"]
        by_target[t] = by_target.get(t, 0) + 1
        temporality = c.get("temporality") or "unknown"
        by_temporality[temporality] = by_temporality.get(temporality, 0) + 1
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
    print("Distribucion por temporalidad:")
    for t, n in sorted(by_temporality.items(), key=lambda kv: kv[0]):
        print(f"  - {t}: {n}")
    print(
        "Distribucion por numero de exogenas: "
        f"0={by_len['0']}, 1={by_len['1']}, 2={by_len['2']}, 3+={by_len['3+']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Stress test XGBoost con datos reales de BD"
    )
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--periods", type=int, default=3)
    parser.add_argument("--rows", type=int, default=0, help="No usado (compatibilidad)")
    parser.add_argument(
        "--seed", type=int, default=42, help="No usado (compatibilidad)"
    )
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

    entries = _enrich_entries(get_names_in_table_catalog() or [])
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

    for case in cases:
        target_entry = next(
            (e for e in entries if e.get("nombre") == case["indicator"]), {}
        )
        case["temporality"] = target_entry.get("_temporality") or "unknown"

    print("=" * 60)
    print("STRESS TEST XGBOOST (DB real)")
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
            pool.submit(
                evaluate_case, c, args.periods, args.train_ratio, args.optimize
            ): c
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
