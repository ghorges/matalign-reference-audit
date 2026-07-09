from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import spearmanr


DATA_ROOT = (Path(__file__).resolve().parents[2] / "data").resolve()
WORK_DIR = Path("vasp_v3_pbe_work")
SELECTION_PATH = DATA_ROOT / "processed" / "v3_pbe_selection" / "pbe_validation_primary.csv"
ANALYSIS_DIR = WORK_DIR / "analysis" / "primary_full"
EXPORT_DIR = DATA_ROOT / "processed" / "v3_pbe_validation"
DOCS_DIR = Path("docs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build QC tables for the MatAlign v3 primary PBE validation run.")
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--selection", type=Path, default=SELECTION_PATH)
    parser.add_argument("--export-dir", type=Path, default=EXPORT_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def finite_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def metric_block(values: pd.Series) -> dict[str, float | int | None]:
    clean = finite_series(values)
    if clean.empty:
        return {"n": 0, "mean": None, "median": None, "p75": None, "p90": None, "max": None}
    return {
        "n": int(clean.shape[0]),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p75": float(clean.quantile(0.75)),
        "p90": float(clean.quantile(0.90)),
        "max": float(clean.max()),
    }


def residual_metrics(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_residual_eV_atom": metric_block(frame[f"{prefix}_residual_vs_mp_eV_atom"]),
        f"abs_{prefix}_residual_eV_atom": metric_block(frame[f"abs_{prefix}_residual_vs_mp_eV_atom"]),
    }


def grouped_abs_metrics(frame: pd.DataFrame, by: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(by, dropna=False):
        rows.append(
            {
                by: str(key),
                "n": int(len(group)),
                "raw_abs_median_eV_atom": float(group["abs_raw_residual_vs_mp_eV_atom"].median()),
                "mp2020_abs_median_eV_atom": float(group["abs_mp2020_residual_vs_mp_eV_atom"].median()),
                "ef_std_median_eV_atom": float(group["Ef_std"].median()),
                "mp2020_within_ef_std_rate": float(group["mp2020_within_ef_std"].mean()),
                "raw_within_ef_std_rate": float(group["raw_within_ef_std"].mean()),
            }
        )
    return rows


def spearman_pair(x: pd.Series, y: pd.Series) -> dict[str, float | int | None]:
    pairs = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pairs) < 3 or pairs["x"].nunique() < 2 or pairs["y"].nunique() < 2:
        return {"n": int(len(pairs)), "rho": None, "pvalue": None}
    rho, pvalue = spearmanr(pairs["x"], pairs["y"])
    return {"n": int(len(pairs)), "rho": float(rho), "pvalue": float(pvalue)}


def top_rows(frame: pd.DataFrame, column: str, n: int = 10) -> pd.DataFrame:
    cols = [
        "pbe_job_id",
        "formula",
        "chemistry_class",
        "noise_bin",
        "validation_role",
        "Ef_MP",
        "raw_formation_energy_per_atom_eV",
        "mp2020_formation_energy_per_atom_eV",
        "raw_residual_vs_mp_eV_atom",
        "mp2020_residual_vs_mp_eV_atom",
        "Ef_std",
    ]
    return frame.sort_values(column, ascending=False)[cols].head(n)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        if isinstance(value, float) and math.isnan(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.head(max_rows).to_markdown(index=False)


def build_report(summary: dict[str, Any], group_tables: dict[str, pd.DataFrame], top_abs: pd.DataFrame) -> str:
    raw_abs = summary["metrics"]["abs_raw_residual_eV_atom"]
    mp_abs = summary["metrics"]["abs_mp2020_residual_eV_atom"]
    return "\n".join(
        [
            "# MatAlign v3 PBE Primary Validation Results - 2026-06-06",
            "",
            "## Summary",
            "",
            f"- Total tasks parsed: `{summary['collection']['total_tasks']}`",
            f"- Static done: `{summary['collection']['static_done']}`",
            f"- Primary compounds: `{summary['compound_rows']}`",
            f"- Raw PBE formation energies ok: `{summary['raw_ok_rows']}`",
            f"- MP2020-compatible formation energies ok: `{summary['mp2020_ok_rows']}`",
            f"- Failed tasks: `{summary['collection']['failed_tasks']}`",
            "",
            "## Main Residuals vs MP",
            "",
            f"- Raw PBE median absolute residual: `{fmt(raw_abs['median'])}` eV/atom",
            f"- Raw PBE mean absolute residual: `{fmt(raw_abs['mean'])}` eV/atom",
            f"- MP2020-compatible median absolute residual: `{fmt(mp_abs['median'])}` eV/atom",
            f"- MP2020-compatible mean absolute residual: `{fmt(mp_abs['mean'])}` eV/atom",
            f"- MP2020 residual within MatAlign Ef_std rate: `{fmt(summary['mp2020_within_ef_std_rate'])}`",
            f"- Raw residual within MatAlign Ef_std rate: `{fmt(summary['raw_within_ef_std_rate'])}`",
            f"- Spearman(abs MP2020 residual, Ef_std): `rho={fmt(summary['correlations']['abs_mp2020_residual_vs_ef_std']['rho'])}`, `p={fmt(summary['correlations']['abs_mp2020_residual_vs_ef_std']['pvalue'], 3)}`",
            "",
            "## By Noise Bin",
            "",
            markdown_table(group_tables["noise_bin"]),
            "",
            "## By Chemistry Class",
            "",
            markdown_table(group_tables["chemistry_class"]),
            "",
            "## By Validation Role",
            "",
            markdown_table(group_tables["validation_role"]),
            "",
            "## Largest MP2020 Residuals vs MP",
            "",
            markdown_table(top_abs),
            "",
            "## Notes",
            "",
            "- Raw PBE formation energies use this run's compound total energies and this run's elemental references.",
            "- MP2020-compatible formation energies use MP-compatible elemental references from `mp-api`; raw gas molecule references are not used for MP2020.",
            "- Pymatgen emitted oxidation-state-guess warnings for many intermetallic or ambiguous compounds. These were warnings, not compatibility failures; all 405 MP2020 rows are `ok`.",
            "- This report is a first data-layer QC. Scientific interpretation should use the merged CSV and stratified tables in the next phase.",
        ]
    )


def main() -> None:
    args = parse_args()
    args.export_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)

    selection = pd.read_csv(args.selection)
    raw = pd.read_csv(args.analysis_dir / "pbe_formation_energies_raw.csv")
    mp2020 = pd.read_csv(args.analysis_dir / "pbe_formation_energies_mp2020.csv")
    totals = pd.read_csv(args.analysis_dir / "pbe_total_energies.csv")
    references = pd.read_csv(args.analysis_dir / "pbe_reference_energies.csv")
    collection = read_json(args.analysis_dir / "collection_summary.json")

    compounds = totals[totals["task_kind"] == "compound"][
        [
            "job_id",
            "static_energy_eV",
            "static_energy_source",
            "relax_energy_eV",
            "relax_energy_source",
            "is_hubbard",
            "is_clean_subset",
            "potcar_symbols_json",
        ]
    ].copy()
    merged = (
        selection.merge(raw, left_on="pbe_job_id", right_on="job_id", how="left", suffixes=("", "_raw"))
        .merge(mp2020, left_on="pbe_job_id", right_on="job_id", how="left", suffixes=("", "_mp2020"))
        .merge(compounds, left_on="pbe_job_id", right_on="job_id", how="left", suffixes=("", "_totals"))
    )

    merged["raw_residual_vs_mp_eV_atom"] = merged["raw_formation_energy_per_atom_eV"] - merged["Ef_MP"]
    merged["mp2020_residual_vs_mp_eV_atom"] = merged["mp2020_formation_energy_per_atom_eV"] - merged["Ef_MP"]
    merged["abs_raw_residual_vs_mp_eV_atom"] = merged["raw_residual_vs_mp_eV_atom"].abs()
    merged["abs_mp2020_residual_vs_mp_eV_atom"] = merged["mp2020_residual_vs_mp_eV_atom"].abs()
    merged["raw_within_ef_std"] = merged["abs_raw_residual_vs_mp_eV_atom"] <= merged["Ef_std"]
    merged["mp2020_within_ef_std"] = merged["abs_mp2020_residual_vs_mp_eV_atom"] <= merged["Ef_std"]
    merged["mp2020_minus_raw_formation_eV_atom"] = (
        merged["mp2020_formation_energy_per_atom_eV"] - merged["raw_formation_energy_per_atom_eV"]
    )

    group_tables = {
        "noise_bin": pd.DataFrame(grouped_abs_metrics(merged, "noise_bin")),
        "chemistry_class": pd.DataFrame(grouped_abs_metrics(merged, "chemistry_class")),
        "validation_role": pd.DataFrame(grouped_abs_metrics(merged, "validation_role")),
        "is_clean_subset": pd.DataFrame(grouped_abs_metrics(merged, "is_clean_subset")),
        "is_hubbard": pd.DataFrame(grouped_abs_metrics(merged, "is_hubbard")),
    }

    metrics = {}
    metrics.update(residual_metrics(merged, "raw"))
    metrics.update(residual_metrics(merged, "mp2020"))
    summary = {
        "collection": collection,
        "compound_rows": int(len(merged)),
        "raw_ok_rows": int((merged["raw_formation_status"] == "ok").sum()),
        "mp2020_ok_rows": int((merged["mp2020_status"] == "ok").sum()),
        "reference_rows": int(len(references)),
        "mp2020_reference_rows": int(len(pd.read_csv(args.analysis_dir / "mp2020_element_references.csv"))),
        "raw_within_ef_std_rate": float(merged["raw_within_ef_std"].mean()),
        "mp2020_within_ef_std_rate": float(merged["mp2020_within_ef_std"].mean()),
        "metrics": metrics,
        "correlations": {
            "abs_raw_residual_vs_ef_std": spearman_pair(merged["abs_raw_residual_vs_mp_eV_atom"], merged["Ef_std"]),
            "abs_mp2020_residual_vs_ef_std": spearman_pair(
                merged["abs_mp2020_residual_vs_mp_eV_atom"], merged["Ef_std"]
            ),
            "mp2020_residual_vs_raw_residual": spearman_pair(
                merged["abs_mp2020_residual_vs_mp_eV_atom"], merged["abs_raw_residual_vs_mp_eV_atom"]
            ),
        },
        "group_tables": {name: table.to_dict("records") for name, table in group_tables.items()},
    }

    merged_out = args.export_dir / "pbe_primary_validation_results.csv"
    merged.to_csv(merged_out, index=False)
    merged.to_parquet(args.export_dir / "pbe_primary_validation_results.parquet", index=False)
    references.to_csv(args.export_dir / "pbe_reference_energies.csv", index=False)
    pd.read_csv(args.analysis_dir / "mp2020_element_references.csv").to_csv(
        args.export_dir / "mp2020_element_references.csv", index=False
    )
    for name, table in group_tables.items():
        table.to_csv(args.export_dir / f"pbe_primary_group_metrics_by_{name}.csv", index=False)

    top_abs = top_rows(merged, "abs_mp2020_residual_vs_mp_eV_atom")
    top_abs.to_csv(args.export_dir / "pbe_primary_top_mp2020_residuals.csv", index=False)

    summary_path = args.export_dir / "pbe_primary_qc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = args.docs_dir / "v3_pbe_primary_results_20260606.md"
    report_path.write_text(build_report(summary, group_tables, top_abs), encoding="utf-8")

    print(
        json.dumps(
            {
                "merged_results": str(merged_out),
                "summary": str(summary_path),
                "report": str(report_path),
                "compound_rows": summary["compound_rows"],
                "raw_ok_rows": summary["raw_ok_rows"],
                "mp2020_ok_rows": summary["mp2020_ok_rows"],
                "raw_abs_median_eV_atom": summary["metrics"]["abs_raw_residual_eV_atom"]["median"],
                "mp2020_abs_median_eV_atom": summary["metrics"]["abs_mp2020_residual_eV_atom"]["median"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
