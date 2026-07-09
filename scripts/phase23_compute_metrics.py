from __future__ import annotations

import ast
import math

import numpy as np
import pandas as pd

from materials_fairness_audit.config import load_settings
from materials_fairness_audit.io_utils import read_table, write_json, write_table
from materials_fairness_audit.metrics import gini, performance_disparity_ratio, stable_metrics


def normalize_elements(value: object) -> list[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    return []


def main() -> None:
    settings = load_settings()
    merged = read_table(settings.paths.processed / "merged_predictions.parquet")
    mapping = read_table(settings.paths.processed / "element_material_mapping.parquet")
    usable = read_table(settings.paths.processed / "element_statistics.csv")
    usable_symbols = set(usable.loc[usable["is_usable"], "symbol"])

    model_cols = [
        column
        for column in merged.columns
        if column not in {"material_id", "formula", "dft_ehull", "e_form_dft", "dft_stable"}
    ]
    rows = []
    merged = merged.merge(mapping[["material_id", "elements"]], on="material_id", how="left")
    merged["elements"] = merged["elements"].map(normalize_elements)

    for model_col in model_cols:
        for symbol in usable_symbols:
            subset = merged[merged["elements"].map(lambda values: symbol in values)]
            if subset.empty:
                rows.append(
                    {
                        "model_key": model_col,
                        "symbol": symbol,
                        "n_samples": 0,
                        "F1": math.nan,
                        "DAF": math.nan,
                        "FPR": math.nan,
                        "FNR": math.nan,
                        "Accuracy": math.nan,
                        "MAE": math.nan,
                        "ME": math.nan,
                        "RMSE": math.nan,
                        "TP": 0.0,
                        "FP": 0.0,
                        "TN": 0.0,
                        "FN": 0.0,
                    }
                )
                continue
            form_error = (subset[model_col] - subset["e_form_dft"]).abs()
            clean_subset = subset.loc[form_error <= settings.analysis.max_abs_form_energy_error].copy()
            each_pred = clean_subset["dft_ehull"] + clean_subset[model_col] - clean_subset["e_form_dft"]
            metrics = stable_metrics(clean_subset["dft_ehull"], each_pred, threshold=0.0)
            rows.append({"model_key": model_col, "symbol": symbol, "n_samples": len(subset), **metrics})

    per_element = pd.DataFrame(rows)
    write_table(per_element, settings.paths.processed / "element_error_by_model.csv")

    summary = (
        per_element.groupby("symbol")[["MAE", "ME", "RMSE", "F1", "FPR", "FNR", "DAF"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["symbol"] + [f"{metric}_{stat}" for metric, stat in summary.columns.tolist()[1:]]
    global_rows = []
    for model_col in model_cols:
        form_error = (merged[model_col] - merged["e_form_dft"]).abs()
        clean_merged = merged.loc[form_error <= settings.analysis.max_abs_form_energy_error].copy()
        each_pred = clean_merged["dft_ehull"] + clean_merged[model_col] - clean_merged["e_form_dft"]
        global_rows.append(
            {
                "model_key": model_col,
                **stable_metrics(clean_merged["dft_ehull"], each_pred, threshold=0.0),
            }
        )
    global_frame = pd.DataFrame(global_rows)
    global_baselines = {
        "MAE_global": float(global_frame["MAE"].mean()),
        "F1_global": float(global_frame["F1"].mean()),
        "DAF_global": float(global_frame["DAF"].mean()),
    }
    summary["MAE_ratio"] = summary["MAE_mean"] / global_baselines["MAE_global"]
    summary["F1_ratio"] = summary["F1_mean"] / global_baselines["F1_global"]
    write_table(summary, settings.paths.processed / "element_error_summary.csv")
    write_json(global_baselines, settings.paths.processed / "global_baselines.json")
    write_table(global_frame, settings.paths.processed / "global_metrics_by_model.csv")

    fairness_by_model = (
        per_element.groupby("model_key")["MAE"]
        .agg(
            gini_mae=lambda values: gini(values),
            pdr_mae=lambda values: performance_disparity_ratio(values),
            mean_mae="mean",
        )
        .reset_index()
    )
    fairness_by_model["eeog_placeholder"] = math.nan
    write_table(fairness_by_model, settings.paths.processed / "fairness_metrics_by_model.csv")
    print(f"Computed per-element metrics for {len(model_cols)} models.")


if __name__ == "__main__":
    main()
