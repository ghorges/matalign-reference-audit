from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DATA_ROOT = (Path(__file__).resolve().parents[2] / "data").resolve()
D1_DIR = DATA_ROOT / "processed" / "clean_reference_analysis"
OUT_DIR = DATA_ROOT / "processed" / "database_relative_model_checks"
DATABASES = ["MP", "OQMD", "AFLOW", "JARVIS"]
CleanCols = [
    "pbe_job_id",
    "matalign_id",
    "id_MP",
    "formula",
    "reduced_formula",
    "elements_str",
    "noise_bin",
    "validation_role",
    "chemistry_class",
    "Ef_std",
    "raw_formation_energy_per_atom_eV",
    "magnetic_heusler_review",
    *[f"Ef_{db}" for db in DATABASES],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D2.1 database-relative model error analysis.")
    parser.add_argument("--analysis-dir", type=Path, default=D1_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def median_or_nan(values: pd.Series) -> float:
    clean = numeric(values).dropna()
    return float(clean.median()) if len(clean) else math.nan


def mean_or_nan(values: pd.Series) -> float:
    clean = numeric(values).dropna()
    return float(clean.mean()) if len(clean) else math.nan


def wilcoxon_less(left: pd.Series, right: pd.Series) -> float | None:
    pairs = pd.DataFrame({"left": numeric(left), "right": numeric(right)}).dropna()
    if len(pairs) < 5:
        return None
    diff = pairs["left"] - pairs["right"]
    if np.allclose(diff, 0):
        return 1.0
    try:
        return float(wilcoxon(diff, alternative="less", zero_method="wilcox").pvalue)
    except ValueError:
        return None


def attach_clean_metadata(preds: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    clean = clean[[col for col in CleanCols if col in clean.columns]].drop_duplicates("pbe_job_id")
    merged = preds.merge(clean, on="pbe_job_id", how="left", suffixes=("", "_clean"))
    for col in clean.columns:
        if col == "pbe_job_id":
            continue
        clean_col = f"{col}_clean"
        if clean_col in merged.columns:
            merged[col] = merged[col].combine_first(merged[clean_col]) if col in merged.columns else merged[clean_col]
            merged = merged.drop(columns=[clean_col])
    return merged


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    clean = pd.read_csv(args.analysis_dir / "clean_intermetallic_master.csv")
    frontier_path = args.out_dir / "frontier_model_predictions_clean.csv"
    if frontier_path.exists():
        preds = pd.read_csv(frontier_path)
    else:
        preds = pd.read_csv(args.analysis_dir / "model_predictions_clean_joined.csv")
    preds = preds[preds["formation_status"].fillna("ok").astype(str).str.startswith("ok")].copy()
    merged = attach_clean_metadata(preds, clean)
    pred_col = "model_mp_reference_formation_energy_per_atom_eV"
    for db in DATABASES:
        target = f"Ef_{db}"
        merged[f"model_error_vs_{db}_eV_atom"] = numeric(merged[pred_col]) - numeric(merged[target])
        merged[f"abs_model_error_vs_{db}_eV_atom"] = merged[f"model_error_vs_{db}_eV_atom"].abs()
        merged[f"within_ef_std_vs_{db}"] = merged[f"abs_model_error_vs_{db}_eV_atom"] <= numeric(merged["Ef_std"])

    db_values = merged[[f"Ef_{db}" for db in DATABASES]].apply(pd.to_numeric, errors="coerce")
    merged["db_consensus_median_eV_atom"] = db_values.median(axis=1, skipna=True)
    merged["db_consensus_mean_eV_atom"] = db_values.mean(axis=1, skipna=True)
    merged["abs_model_error_vs_consensus_median_eV_atom"] = (
        numeric(merged[pred_col]) - merged["db_consensus_median_eV_atom"]
    ).abs()
    merged["within_ef_std_vs_consensus_median"] = (
        merged["abs_model_error_vs_consensus_median_eV_atom"] <= numeric(merged["Ef_std"])
    )

    summary_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    for model, group in merged.groupby("model"):
        mp_abs = group["abs_model_error_vs_MP_eV_atom"]
        for db in DATABASES:
            err = group[f"model_error_vs_{db}_eV_atom"]
            abs_err = group[f"abs_model_error_vs_{db}_eV_atom"]
            valid = pd.DataFrame({"abs": numeric(abs_err), "err": numeric(err), "floor": numeric(group["Ef_std"])}).dropna()
            summary_rows.append(
                {
                    "model": model,
                    "target_database": db,
                    "n": int(len(valid)),
                    "median_abs_error_eV_atom": median_or_nan(valid["abs"]),
                    "mean_abs_error_eV_atom": mean_or_nan(valid["abs"]),
                    "median_signed_error_eV_atom": median_or_nan(valid["err"]),
                    "mean_signed_error_eV_atom": mean_or_nan(valid["err"]),
                    "within_floor_rate": float((valid["abs"] <= valid["floor"]).mean()) if len(valid) else math.nan,
                }
            )
            if db != "MP":
                other_abs = group[f"abs_model_error_vs_{db}_eV_atom"]
                pairwise_rows.append(
                    {
                        "model": model,
                        "comparison": f"abs_error_MP < abs_error_{db}",
                        "n": int(pd.DataFrame({"mp": numeric(mp_abs), "other": numeric(other_abs)}).dropna().shape[0]),
                        "median_abs_error_mp_eV_atom": median_or_nan(mp_abs),
                        "median_abs_error_other_eV_atom": median_or_nan(other_abs),
                        "median_other_minus_mp_eV_atom": median_or_nan(numeric(other_abs) - numeric(mp_abs)),
                        "wilcoxon_p_mp_closer": wilcoxon_less(mp_abs, other_abs),
                        "mp_closer_material_rate": float((numeric(mp_abs) < numeric(other_abs)).mean()),
                    }
                )

        consensus_abs = merged.loc[group.index, "abs_model_error_vs_consensus_median_eV_atom"]
        valid_cons = pd.DataFrame({"abs": numeric(consensus_abs), "floor": numeric(group["Ef_std"])}).dropna()
        summary_rows.append(
            {
                "model": model,
                "target_database": "CONSENSUS_MEDIAN",
                "n": int(len(valid_cons)),
                "median_abs_error_eV_atom": median_or_nan(valid_cons["abs"]),
                "mean_abs_error_eV_atom": mean_or_nan(valid_cons["abs"]),
                "median_signed_error_eV_atom": math.nan,
                "mean_signed_error_eV_atom": math.nan,
                "within_floor_rate": float((valid_cons["abs"] <= valid_cons["floor"]).mean()) if len(valid_cons) else math.nan,
            }
        )

    nearest_counts = []
    abs_cols = [f"abs_model_error_vs_{db}_eV_atom" for db in DATABASES]
    for model, group in merged.groupby("model"):
        nearest = group[abs_cols].astype(float).idxmin(axis=1).str.replace("abs_model_error_vs_", "", regex=False).str.replace("_eV_atom", "", regex=False)
        for db, count in nearest.value_counts().sort_index().items():
            nearest_counts.append(
                {
                    "model": model,
                    "nearest_database": db,
                    "n": int(count),
                    "rate": float(count / len(group)) if len(group) else math.nan,
                }
            )

    summary = pd.DataFrame(summary_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    nearest_table = pd.DataFrame(nearest_counts)

    merged.to_csv(args.out_dir / "database_relative_model_errors.csv", index=False)
    summary.to_csv(args.out_dir / "database_relative_summary.csv", index=False)
    pairwise.to_csv(args.out_dir / "database_relative_mp_closeness_tests.csv", index=False)
    nearest_table.to_csv(args.out_dir / "database_relative_nearest_database_counts.csv", index=False)

    payload = {
        "models": sorted(merged["model"].dropna().unique().tolist()),
        "rows": int(len(merged)),
        "rows_by_model": {str(k): int(v) for k, v in merged["model"].value_counts().sort_index().items()},
        "mp_closeness_tests": pairwise.to_dict("records"),
    }
    (args.out_dir / "database_relative_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
