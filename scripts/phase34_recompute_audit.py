from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import statsmodels.api as sm

from materials_fairness_audit.config import load_settings
from materials_fairness_audit.elements import element_property_record, parse_formula_features
from materials_fairness_audit.io_utils import read_table, write_json, write_table
from materials_fairness_audit.metrics import gini, performance_disparity_ratio, stable_metrics


def normalize_elements(value: object) -> list[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
        if isinstance(parsed, list):
            return parsed
    return []


def load_train_element_counts(mp_entries_path: Path) -> Counter[str]:
    with gzip.open(mp_entries_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    counts: Counter[str] = Counter()
    entries = payload.get("entry", {})
    for entry in entries.values():
        if not isinstance(entry, dict):
            continue
        composition = entry.get("composition", {})
        if isinstance(composition, dict):
            counts.update(composition.keys())
    return counts


def bh_adjust(p_values: pd.Series) -> pd.Series:
    valid = p_values.dropna().sort_values()
    if valid.empty:
        return pd.Series(index=p_values.index, dtype=float)
    m = len(valid)
    adjusted = pd.Series(index=valid.index, dtype=float)
    running_min = 1.0
    for rank, (index, value) in enumerate(valid.iloc[::-1].items(), start=1):
        corrected = min(1.0, value * m / (m - rank + 1))
        running_min = min(running_min, corrected)
        adjusted[index] = running_min
    return adjusted.reindex(p_values.index)


def dataset_token_count(value: object) -> int:
    if not isinstance(value, str) or not value.strip():
        return 0
    return len([token.strip() for token in value.split(";") if token.strip()])


def sequential_r2(design: pd.DataFrame, target: pd.Series, blocks: list[list[str]]) -> dict[str, float]:
    results: dict[str, float] = {}
    active: list[str] = []
    previous_r2 = 0.0
    for name, block in zip(("training_data", "dft_uncertainty", "physical_complexity"), blocks, strict=False):
        active.extend(block)
        model = sm.OLS(target, sm.add_constant(design[active], has_constant="add")).fit()
        current_r2 = float(model.rsquared)
        results[f"{name}_r2"] = current_r2
        results[f"{name}_increment"] = max(0.0, current_r2 - previous_r2)
        previous_r2 = current_r2
    results["unexplained"] = max(0.0, 1.0 - previous_r2)
    return results


def classify_root_cause(row: pd.Series, train_median: float, ef_median: float, difficulty_median: float) -> str:
    scarcity = max(0.0, train_median - float(row["log_n_train_elem"]))
    dft_noise = max(0.0, float(row["ef_std_avg_e"]) - ef_median)
    physical = max(0.0, float(row["dft_difficulty_score"]) - difficulty_median)
    if float(row["residual_from_dft_uncertainty"]) <= 0 and float(row["ef_std_avg_e"]) >= ef_median:
        return "dft_noise_dominated"
    scores = {
        "training_data": scarcity,
        "dft_uncertainty": dft_noise,
        "physical_complexity": physical,
    }
    return max(scores, key=scores.get)


def main() -> None:
    settings = load_settings()
    settings.paths.ensure()

    matbench = read_table(settings.paths.processed_matbench / "matbench_merged_predictions.parquet")
    model_meta = read_table(settings.paths.processed_matbench / "model_metadata.csv")
    matalign = read_table(settings.paths.processed_matalign / "matalign_full.parquet")

    features = matbench["formula"].map(parse_formula_features)
    mapping = pd.DataFrame(
        {
            "material_id": matbench["material_id"],
            "formula": matbench["formula"],
            "reduced_formula": [item.reduced_formula for item in features],
            "elements": [list(item.elements) for item in features],
            "n_elements": [item.n_elements for item in features],
            "has_3d_transition_metal": [item.has_3d_transition_metal for item in features],
            "has_4f_lanthanide": [item.has_4f_lanthanide for item in features],
            "has_5d_heavy_element": [item.has_5d_heavy_element for item in features],
            "avg_electronegativity": [item.avg_electronegativity for item in features],
            "std_electronegativity": [item.std_electronegativity for item in features],
            "has_plus_u_element": [item.has_plus_u_element for item in features],
        }
    )
    write_table(mapping, settings.paths.processed_audit / "wbm_element_mapping.parquet")

    test_counts = Counter(symbol for symbols in mapping["elements"] for symbol in symbols)
    mp_entries_candidates = sorted(settings.paths.raw_wbm.glob("*mp-computed-structure-entries*.json.gz"))
    train_counts = load_train_element_counts(mp_entries_candidates[0]) if mp_entries_candidates else Counter()

    matalign["elements"] = matalign["elements"].map(normalize_elements)
    matalign_counts = Counter(symbol for symbols in matalign["elements"] for symbol in symbols)
    ef_uncertainty = (
        matalign[["Ef_std", "elements"]]
        .explode("elements")
        .dropna(subset=["elements"])
        .groupby("elements")["Ef_std"]
        .mean()
        .to_dict()
    )

    all_symbols = sorted(set(test_counts) | set(train_counts) | set(matalign_counts))
    element_stats = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "n_test_elem": test_counts.get(symbol, 0),
                "n_train_elem": train_counts.get(symbol, 0),
                "n_matalign_elem": matalign_counts.get(symbol, 0),
                "ef_std_avg_e": ef_uncertainty.get(symbol, math.nan),
                **element_property_record(symbol),
            }
            for symbol in all_symbols
        ]
    )
    element_stats["train_test_ratio"] = element_stats["n_train_elem"] / element_stats["n_test_elem"].replace({0: pd.NA})
    element_stats["is_usable"] = element_stats["n_test_elem"] >= settings.analysis.minimum_element_test_count
    element_stats["dft_difficulty_score"] = (
        element_stats[["is_plus_u", "is_f_block", "has_heavy_soc"]].astype(int).sum(axis=1)
        + element_stats["n_common_oxidation_states"].fillna(0).clip(upper=4)
        + element_stats["period"].fillna(0).clip(upper=7) / 7.0
    )
    write_table(element_stats, settings.paths.processed_audit / "element_statistics.csv")

    usable_symbols = sorted(element_stats.loc[element_stats["is_usable"], "symbol"])
    merged = matbench.merge(mapping[["material_id", "elements"]], on="material_id", how="left")
    merged["elements"] = merged["elements"].map(normalize_elements)
    symbol_masks = {symbol: merged["elements"].map(lambda values, s=symbol: s in values).to_numpy() for symbol in usable_symbols}

    model_cols = [
        column
        for column in merged.columns
        if column not in {"material_id", "formula", "dft_ehull", "e_form_dft", "dft_stable", "elements"}
    ]

    per_rows = []
    global_rows = []
    for model_col in model_cols:
        form_error = (merged[model_col] - merged["e_form_dft"]).abs().to_numpy()
        clean_mask = form_error <= settings.analysis.max_abs_form_energy_error
        clean = merged.loc[clean_mask].copy()
        predicted_ehull = clean["dft_ehull"] + clean[model_col] - clean["e_form_dft"]
        global_metrics = stable_metrics(clean["dft_ehull"], predicted_ehull, threshold=0.0)
        global_rows.append({"model_key": model_col, **global_metrics})

        for symbol in usable_symbols:
            subset = merged.loc[clean_mask & symbol_masks[symbol]].copy()
            if subset.empty:
                per_rows.append({"model_key": model_col, "symbol": symbol, "n_samples": 0})
                continue
            each_pred = subset["dft_ehull"] + subset[model_col] - subset["e_form_dft"]
            metrics = stable_metrics(subset["dft_ehull"], each_pred, threshold=0.0)
            per_rows.append({"model_key": model_col, "symbol": symbol, "n_samples": len(subset), **metrics})

    per_element = pd.DataFrame(per_rows)
    global_frame = pd.DataFrame(global_rows)
    global_baselines = {
        "MAE_global": float(global_frame["MAE"].mean()),
        "F1_global": float(global_frame["F1"].mean()),
        "DAF_global": float(global_frame["DAF"].mean()),
    }
    global_frame.to_csv(settings.paths.processed_audit / "global_metrics_by_model.csv", index=False)
    write_json(global_baselines, settings.paths.processed_audit / "global_baselines.json")

    per_element = per_element.merge(
        global_frame[["model_key", "MAE", "F1", "FPR", "FNR"]].rename(
            columns={
                "MAE": "MAE_global",
                "F1": "F1_global",
                "FPR": "FPR_global",
                "FNR": "FNR_global",
            }
        ),
        on="model_key",
        how="left",
    )
    per_element["EEOG"] = (per_element["FPR"] - per_element["FPR_global"]).abs() + (
        per_element["FNR"] - per_element["FNR_global"]
    ).abs()
    per_element["MAE_ratio"] = per_element["MAE"] / per_element["MAE_global"]
    per_element["F1_ratio"] = per_element["F1"] / per_element["F1_global"]
    write_table(per_element, settings.paths.processed_audit / "element_error_by_model.csv")

    summary = (
        per_element.groupby("symbol")[["MAE", "ME", "RMSE", "F1", "FPR", "FNR", "DAF", "EEOG", "MAE_ratio", "F1_ratio"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["symbol"] + [f"{metric}_{stat}" for metric, stat in summary.columns.tolist()[1:]]
    summary["MAE_ratio"] = summary["MAE_mean"] / global_baselines["MAE_global"]
    summary["F1_ratio"] = summary["F1_mean"] / global_baselines["F1_global"]

    significance_rows = []
    for symbol, subset in per_element.groupby("symbol"):
        valid = subset.dropna(subset=["MAE", "MAE_global"])
        p_value = math.nan
        statistic = math.nan
        if len(valid) >= 3 and not np.allclose(valid["MAE"].to_numpy(), valid["MAE_global"].to_numpy()):
            statistic, p_value = wilcoxon(valid["MAE"], valid["MAE_global"], alternative="greater")
        significance_rows.append(
            {
                "symbol": symbol,
                "n_models": int(len(valid)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
    significance = pd.DataFrame(significance_rows)
    significance["p_adj_bh"] = bh_adjust(significance["p_value"])
    significance["is_significant"] = significance["p_adj_bh"] < 0.05
    write_table(significance, settings.paths.processed_audit / "element_significance.csv")

    summary = summary.merge(significance[["symbol", "p_value", "p_adj_bh", "is_significant"]], on="symbol", how="left")
    write_table(summary, settings.paths.processed_audit / "element_error_summary.csv")

    fairness_by_model = (
        per_element.groupby("model_key")
        .agg(
            pdr_mae=("MAE", performance_disparity_ratio),
            gini_f1=("F1", gini),
            mean_eeog=("EEOG", "mean"),
            mean_mae=("MAE", "mean"),
            mean_f1=("F1", "mean"),
        )
        .reset_index()
    )
    write_table(fairness_by_model, settings.paths.processed_audit / "fairness_metrics_by_model.csv")

    joined = summary.merge(element_stats, on="symbol", how="left")
    joined["log_n_train_elem"] = np.log1p(joined["n_train_elem"].fillna(0))
    write_table(joined, settings.paths.processed_audit / "training_frequency_vs_error.csv")

    regression_cols = [
        "MAE_mean",
        "log_n_train_elem",
        "ef_std_avg_e",
        "is_plus_u",
        "is_f_block",
        "n_common_oxidation_states",
        "period",
        "has_heavy_soc",
        "dft_difficulty_score",
    ]
    regression = joined[regression_cols].dropna().copy()
    predictors = [
        "log_n_train_elem",
        "ef_std_avg_e",
        "is_plus_u",
        "is_f_block",
        "n_common_oxidation_states",
        "period",
        "has_heavy_soc",
    ]
    design = regression[predictors].astype(float)
    standardized = (design - design.mean()) / design.std(ddof=0).replace(0, 1.0)
    ols = sm.OLS(regression["MAE_mean"], sm.add_constant(standardized, has_constant="add")).fit()

    seq_r2 = sequential_r2(
        standardized,
        regression["MAE_mean"],
        [
            ["log_n_train_elem"],
            ["ef_std_avg_e"],
            ["is_plus_u", "is_f_block", "n_common_oxidation_states", "period", "has_heavy_soc"],
        ],
    )

    kfold = KFold(n_splits=min(5, len(standardized)), shuffle=True, random_state=settings.analysis.bootstrap_seed)
    cv_scores = []
    for train_index, test_index in kfold.split(standardized):
        model = LinearRegression()
        model.fit(standardized.iloc[train_index], regression["MAE_mean"].iloc[train_index])
        predictions = model.predict(standardized.iloc[test_index])
        cv_scores.append(r2_score(regression["MAE_mean"].iloc[test_index], predictions))

    dft_design = sm.add_constant(joined[["ef_std_avg_e"]].fillna(joined["ef_std_avg_e"].median()), has_constant="add")
    dft_model = sm.OLS(joined["MAE_mean"], dft_design, missing="drop").fit()
    joined["pred_from_dft_uncertainty"] = dft_model.predict(dft_design)
    joined["residual_from_dft_uncertainty"] = joined["MAE_mean"] - joined["pred_from_dft_uncertainty"]
    train_median = float(joined["log_n_train_elem"].median())
    ef_median = float(joined["ef_std_avg_e"].median())
    difficulty_median = float(joined["dft_difficulty_score"].median())
    joined["root_cause_label"] = joined.apply(
        lambda row: classify_root_cause(row, train_median, ef_median, difficulty_median),
        axis=1,
    )
    write_table(
        joined[
            [
                "symbol",
                "MAE_mean",
                "ef_std_avg_e",
                "log_n_train_elem",
                "dft_difficulty_score",
                "pred_from_dft_uncertainty",
                "residual_from_dft_uncertainty",
                "root_cause_label",
            ]
        ],
        settings.paths.processed_audit / "residual_analysis.csv",
    )

    write_json(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "spearman_training_frequency_vs_mae": {
                "rho": spearmanr(joined["log_n_train_elem"], joined["MAE_mean"], nan_policy="omit").statistic,
                "p_value": spearmanr(joined["log_n_train_elem"], joined["MAE_mean"], nan_policy="omit").pvalue,
            },
            "spearman_dft_uncertainty_vs_mae": {
                "rho": spearmanr(joined["ef_std_avg_e"], joined["MAE_mean"], nan_policy="omit").statistic,
                "p_value": spearmanr(joined["ef_std_avg_e"], joined["MAE_mean"], nan_policy="omit").pvalue,
            },
            "spearman_difficulty_vs_mae": {
                "rho": spearmanr(joined["dft_difficulty_score"], joined["MAE_mean"], nan_policy="omit").statistic,
                "p_value": spearmanr(joined["dft_difficulty_score"], joined["MAE_mean"], nan_policy="omit").pvalue,
            },
            "ols_r_squared": float(ols.rsquared),
            "ols_adj_r_squared": float(ols.rsquared_adj),
            "ols_params": {key: float(value) for key, value in ols.params.items()},
            "ols_pvalues": {key: float(value) for key, value in ols.pvalues.items()},
            "cv_r2_mean": float(np.mean(cv_scores)),
            "cv_r2_std": float(np.std(cv_scores)),
        },
        settings.paths.processed_audit / "root_cause_regression.json",
    )
    write_json(seq_r2, settings.paths.processed_audit / "variance_decomposition.json")

    matrix = per_element.pivot(index="symbol", columns="model_key", values="MAE_ratio")
    write_table(matrix.reset_index(), settings.paths.processed_audit / "element_model_matrix.csv")

    tau_rows = []
    model_keys = list(matrix.columns)
    for left_index, left_key in enumerate(model_keys):
        for right_key in model_keys[left_index + 1 :]:
            pair = matrix[[left_key, right_key]].dropna()
            if len(pair) < 2:
                continue
            tau, p_value = kendalltau(pair[left_key], pair[right_key])
            tau_rows.append(
                {
                    "model_key_left": left_key,
                    "model_key_right": right_key,
                    "kendall_tau": tau,
                    "p_value": p_value,
                }
            )
    tau_frame = pd.DataFrame(tau_rows)
    write_table(tau_frame, settings.paths.processed_audit / "model_ranking_correlation.csv")

    difficulty = (
        per_element.groupby("symbol")["MAE_ratio"]
        .agg(frac_models_high_error=lambda values: float((values > 1.5).mean()), n_models="size")
        .reset_index()
    )
    write_table(
        difficulty.loc[difficulty["frac_models_high_error"] >= 0.8].sort_values("frac_models_high_error", ascending=False),
        settings.paths.processed_audit / "universal_difficult_elements.csv",
    )

    fairness_with_meta = fairness_by_model.merge(
        model_meta[["model_key", "training_set", "model_type", "model_family"]],
        on="model_key",
        how="left",
    )
    fairness_with_meta["dataset_token_count"] = fairness_with_meta["training_set"].map(dataset_token_count)
    median_token_count = fairness_with_meta["dataset_token_count"].median()
    fairness_with_meta["training_data_group"] = np.where(
        fairness_with_meta["dataset_token_count"] >= median_token_count,
        "higher_data",
        "lower_data",
    )
    write_table(fairness_with_meta, settings.paths.processed_audit / "fairness_by_model.csv")
    write_table(
        fairness_with_meta.groupby(["training_data_group", "model_type"])[["pdr_mae", "gini_f1", "mean_eeog"]]
        .mean()
        .reset_index(),
        settings.paths.processed_audit / "fairness_by_group.csv",
    )

    train_total = joined["n_train_elem"].sum()
    test_total = joined["n_test_elem"].sum()
    divergence = joined[["symbol", "n_train_elem", "n_test_elem", "train_test_ratio", "MAE_mean"]].copy()
    divergence["train_fraction"] = divergence["n_train_elem"] / train_total
    divergence["test_fraction"] = divergence["n_test_elem"] / test_total
    midpoint = 0.5 * (divergence["train_fraction"] + divergence["test_fraction"])
    eps = 1e-12
    divergence["js_contrib"] = 0.5 * divergence["test_fraction"] * np.log(
        (divergence["test_fraction"] + eps) / (midpoint + eps)
    ) + 0.5 * divergence["train_fraction"] * np.log((divergence["train_fraction"] + eps) / (midpoint + eps))
    write_table(divergence, settings.paths.processed_audit / "train_test_distribution_divergence.csv")

    print(
        json.dumps(
            {
                "usable_elements": len(usable_symbols),
                "model_count": len(model_cols),
                "matalign_rows": int(len(matalign)),
                "global_baselines": global_baselines,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
