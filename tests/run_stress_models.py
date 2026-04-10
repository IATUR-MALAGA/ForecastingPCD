import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd):
    print("\n$ " + " ".join(cmd))
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nInterrumpido por usuario.")
        return 130


def main():
    parser = argparse.ArgumentParser(description="Runner para stress tests de XGBoost y SARIMAX")
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--periods", type=int, default=3)
    parser.add_argument("--rows", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
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

    repo_root = Path(__file__).resolve().parents[1]
    venv_py = repo_root / ".venv" / "Scripts" / "python.exe"
    py = str(venv_py) if venv_py.exists() else sys.executable

    base_args = [
        "--threshold",
        str(args.threshold),
        "--workers",
        str(args.workers),
        "--periods",
        str(args.periods),
        "--rows",
        str(args.rows),
        "--seed",
        str(args.seed),
        "--train-ratio",
        str(args.train_ratio),
        "--suite",
        args.suite,
        "--max-targets",
        str(args.max_targets),
        "--max-exogs",
        str(args.max_exogs),
        "--max-pairs",
        str(args.max_pairs),
        "--max-trios",
        str(args.max_trios),
        "--max-cases",
        str(args.max_cases),
        "--outdir",
        args.outdir,
    ]
    if args.optimize:
        base_args.append("--optimize")

    xgb_cmd = [py, str(repo_root / "tests" / "stress_test_xgboost.py"), *base_args]
    sarimax_cmd = [py, str(repo_root / "tests" / "stress_test_sarimax.py"), *base_args]

    rc1 = run_cmd(xgb_cmd)
    rc2 = run_cmd(sarimax_cmd)

    print("\nReportes esperados en:")
    print(f"- {Path(args.outdir) / 'xgboost_stress_results.json'}")
    print(f"- {Path(args.outdir) / 'xgboost_stress_report.md'}")
    print(f"- {Path(args.outdir) / 'sarimax_stress_results.json'}")
    print(f"- {Path(args.outdir) / 'sarimax_stress_report.md'}")

    if rc1 != 0 or rc2 != 0:
        raise SystemExit(rc1 if rc1 != 0 else rc2)


if __name__ == "__main__":
    main()
