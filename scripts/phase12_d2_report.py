from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


DATA_ROOT = (Path(__file__).resolve().parents[2] / "data").resolve()
OUT_DIR = DATA_ROOT / "processed" / "database_relative_model_checks"
DOCS_DIR = Path("docs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MatAlign v3 Phase D2 report.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def md_table(frame: pd.DataFrame, n: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.head(n).to_markdown(index=False)


def metric_from_summary(summary: pd.DataFrame, model: str, target: str) -> float:
    row = summary[(summary["model"] == model) & (summary["target_database"] == target)]
    if row.empty:
        return math.nan
    return float(row.iloc[0]["median_abs_error_eV_atom"])


def database_interpretation(summary: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if summary.empty:
        return ["- Database-relative summary is unavailable."]
    for model in sorted(summary["model"].dropna().unique()):
        med = {
            target: metric_from_summary(summary, str(model), target)
            for target in ["MP", "OQMD", "AFLOW", "JARVIS", "CONSENSUS_MEDIAN"]
        }
        finite = {key: value for key, value in med.items() if not math.isnan(value)}
        if not finite:
            continue
        closest = min(finite, key=finite.get)
        lines.append(
            "- "
            f"`{model}` closest target by median absolute error is `{closest}` "
            f"({finite[closest]:.4f} eV/atom); MP median is {med.get('MP', math.nan):.4f} eV/atom."
        )
        if closest != "MP":
            lines.append(
                "- "
                f"`{model}` does not support the simple 'nearest to MP training database' claim in this clean subset."
            )
    return lines


def main() -> None:
    args = parse_args()
    args.docs_dir.mkdir(parents=True, exist_ok=True)

    db_summary = read_csv(args.out_dir / "database_relative_summary.csv")
    db_tests = read_csv(args.out_dir / "database_relative_mp_closeness_tests.csv")
    db_nearest = read_csv(args.out_dir / "database_relative_nearest_database_counts.csv")
    leakage = read_json(args.out_dir / "training_leakage_summary.json")
    heldout = read_csv(args.out_dir / "heldout_saturation_summary.csv")
    frontier = read_csv(args.out_dir / "frontier_model_noise_floor_summary.csv")
    failures = read_csv(args.out_dir / "frontier_model_failures.csv")
    element = read_csv(args.out_dir / "element_headroom_map.csv")
    class_summary = read_csv(args.out_dir / "element_class_headroom_summary.csv")
    mn = read_csv(args.out_dir / "mn_heusler_sensitivity.csv")

    mace_mp = metric_from_summary(db_summary, "mace", "MP")
    mace_other = {
        db: metric_from_summary(db_summary, "mace", db)
        for db in ["OQMD", "AFLOW", "JARVIS"]
    }
    mptrj_blocked = not bool(leakage.get("exact_mptrj_index_available"))
    phase_summary = {
        "database_relative_rows": int(read_csv(args.out_dir / "database_relative_model_errors.csv").shape[0]),
        "training_leakage": leakage,
        "frontier_models": sorted(frontier["model"].dropna().unique().tolist()) if not frontier.empty else [],
        "frontier_failure_rows": int(len(failures)),
        "element_rows": int(len(element)),
        "mace_median_abs_error_vs_mp": mace_mp,
        "mace_median_abs_error_vs_other_databases": mace_other,
        "mptrj_exact_leakage_blocked": bool(mptrj_blocked),
    }
    (args.out_dir / "database_relative_model_checks_summary.json").write_text(
        json.dumps(phase_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = [
        "# MatAlign v3 Phase D2 Results",
        "",
        "## Summary",
        "",
        "- Phase D2 tests whether model accuracy is relative to the MP training database, whether the clean subset leaks into MPtrj training data, and whether frontier models broadly sit at the noise floor.",
        "- `model error < Ef_std` is not interpreted as a model being more accurate than DFT; it means the model is closer to the MP reference point than the observed spread among databases.",
        "- No VASP calculations are added in D2.",
        "",
        "## D2.1 Database-Relative Errors",
        "",
        md_table(db_summary.sort_values(["model", "target_database"]) if not db_summary.empty else db_summary, 40),
        "",
        "### MP-Closeness Tests",
        "",
        md_table(db_tests, 40),
        "",
        "### Nearest Database Counts",
        "",
        md_table(db_nearest, 40),
        "",
        "### D2.1 Interpretation",
        "",
        *database_interpretation(db_summary),
        "",
        "## D2.2 Training Leakage",
        "",
        f"- Exact MPtrj index available: `{leakage.get('exact_mptrj_index_available')}`.",
        f"- Exact in-sample count/rate: `{leakage.get('in_sample_exact_count')}` / `{leakage.get('in_sample_exact_rate')}`.",
        f"- Conservative MP2022 in-sample count/rate: `{leakage.get('in_sample_conservative_count')}` / `{leakage.get('in_sample_conservative_rate')}`.",
        f"- Held-out underpowered: `{leakage.get('heldout_underpowered')}`.",
        f"- Blocked reason: `{leakage.get('blocked_reason')}`.",
        "",
        "### Held-Out Saturation",
        "",
        md_table(heldout, 30),
        "",
        "## D2.3 Frontier Model Summary",
        "",
        md_table(frontier, 60),
        "",
        "### Frontier Failures",
        "",
        md_table(failures, 30),
        "",
        "## D2.4 Element Headroom Map",
        "",
        md_table(element.sort_values(["subset", "model", "saturation_call", "element"]) if not element.empty else element, 80),
        "",
        "### Element-Class Summary",
        "",
        md_table(class_summary, 60),
        "",
        "## D2.5 Mn-Heusler Sensitivity",
        "",
        md_table(mn, 30),
        "",
        "## Current Interpretation",
        "",
        "- D2.1 is the key test for the training-database-relative accuracy claim.",
        "- D2.2 cannot support held-out claims unless an exact MPtrj material-id index is available; conservative MP2022 membership is reported only as an upper-bound leakage screen.",
        "- D2.3 now includes the successful frontier-model expansion, but the interpretation remains noise-floor limited rather than 'more accurate than DFT'.",
    ]
    out_path = args.docs_dir / "database_relative_model_checks_results_20260606.md"
    out_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(phase_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
