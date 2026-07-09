from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
import statsmodels.api as sm
from itertools import combinations

from materials_fairness_audit.config import load_settings
from materials_fairness_audit.io_utils import read_table, write_json, write_table


def dataset_token_count(value: object) -> int:
    if not isinstance(value, str) or not value.strip():
        return 0
    return len([token.strip() for token in value.split(";") if token.strip()])


def main() -> None:
    settings = load_settings()
    element_summary = read_table(settings.paths.processed / "element_error_summary.csv")
    element_stats = read_table(settings.paths.processed / "element_statistics.csv")
    per_model = read_table(settings.paths.processed / "element_error_by_model.csv")
    model_meta = read_table(settings.paths.processed / "model_metadata.csv")
    global_metrics = read_table(settings.paths.processed / "global_metrics_by_model.csv")
    fairness_by_model = read_table(settings.paths.processed / "fairness_metrics_by_model.csv")

    joined = element_summary.merge(element_stats, on="symbol", how="left")
    write_table(joined, settings.paths.processed / "training_frequency_vs_error.csv")
    freq_correlation = spearmanr(joined["n_train_elem"], joined["MAE_mean"], nan_policy="omit")

    regression_frame = joined[
        ["MAE_mean", "n_train_elem", "is_plus_u", "is_f_block", "n_common_oxidation_states", "period", "has_heavy_soc"]
    ].dropna()
    regression_frame["log_n_train_elem"] = regression_frame["n_train_elem"].map(lambda value: np.log1p(value))
    design = sm.add_constant(
        regression_frame[
            ["log_n_train_elem", "is_plus_u", "is_f_block", "n_common_oxidation_states", "period", "has_heavy_soc"]
        ].astype(float)
    )
    model = sm.OLS(regression_frame["MAE_mean"], design).fit()

    residual_design = sm.add_constant(joined["n_train_elem"].map(lambda value: np.log1p(value)).fillna(0.0))
    residual_model = sm.OLS(joined["MAE_mean"], residual_design, missing="drop").fit()
    residual_input = sm.add_constant(joined["n_train_elem"].map(lambda value: np.log1p(value)).fillna(0.0), has_constant="add")
    joined["residual_from_train_frequency"] = joined["MAE_mean"] - residual_model.predict(residual_input)
    write_table(
        joined[
            [
                "symbol",
                "n_test_elem",
                "n_train_elem",
                "train_test_ratio",
                "MAE_mean",
                "residual_from_train_frequency",
            ]
        ],
        settings.paths.processed / "residual_analysis.csv",
    )

    matrix = per_model.pivot(index="symbol", columns="model_key", values="MAE")
    tau_rows = []
    model_keys = list(matrix.columns)
    for left_index, left_key in enumerate(model_keys):
        for right_key in model_keys[left_index + 1:]:
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

    write_table(pd.DataFrame(tau_rows), settings.paths.processed / "model_ranking_correlation_matrix.csv")
    write_table(matrix.reset_index(), settings.paths.processed / "element_model_error_matrix.csv")

    baseline_joined = per_model.merge(
        global_metrics[["model_key", "MAE"]].rename(columns={"MAE": "MAE_global"}),
        on="model_key",
        how="left",
    )
    baseline_joined["is_high_error"] = baseline_joined["MAE"] > (1.5 * baseline_joined["MAE_global"])
    difficulty = (
        baseline_joined.groupby("symbol")["is_high_error"]
        .agg(frac_models_high_error="mean", n_models_high_error="sum")
        .reset_index()
    )
    universal = difficulty.loc[difficulty["frac_models_high_error"] >= 0.8].sort_values(
        "frac_models_high_error", ascending=False
    )
    model_specific = difficulty.loc[
        (difficulty["frac_models_high_error"] > 0.0) & (difficulty["frac_models_high_error"] <= 0.2)
    ].sort_values("frac_models_high_error", ascending=True)
    write_table(universal, settings.paths.processed / "universal_difficult_elements.csv")
    write_table(model_specific, settings.paths.processed / "model_specific_difficult_elements.csv")

    fairness_with_meta = fairness_by_model.merge(
        model_meta[["model_key", "training_set", "model_type"]],
        on="model_key",
        how="left",
    )
    write_table(fairness_with_meta, settings.paths.processed / "fairness_by_training_data.csv")
    write_table(fairness_with_meta, settings.paths.processed / "fairness_by_architecture.csv")

    eps = 1e-12
    train_total = joined["n_train_elem"].sum()
    test_total = joined["n_test_elem"].sum()
    divergence = joined[["symbol", "n_train_elem", "n_test_elem", "train_test_ratio", "MAE_mean"]].copy()
    divergence["train_fraction"] = divergence["n_train_elem"] / train_total
    divergence["test_fraction"] = divergence["n_test_elem"] / test_total
    divergence["kl_test_to_train_contrib"] = divergence["test_fraction"] * np.log(
        (divergence["test_fraction"] + eps) / (divergence["train_fraction"] + eps)
    )
    midpoint = 0.5 * (divergence["train_fraction"] + divergence["test_fraction"])
    divergence["js_contrib"] = 0.5 * divergence["test_fraction"] * np.log(
        (divergence["test_fraction"] + eps) / (midpoint + eps)
    ) + 0.5 * divergence["train_fraction"] * np.log(
        (divergence["train_fraction"] + eps) / (midpoint + eps)
    )
    write_table(divergence, settings.paths.processed / "train_test_distribution_divergence.csv")

    paired = fairness_with_meta.merge(
        global_metrics[["model_key", "MAE", "F1", "DAF"]].rename(
            columns={"MAE": "global_mae", "F1": "global_f1", "DAF": "global_daf"}
        ),
        on="model_key",
        how="left",
    ).merge(
        model_meta[["model_key", "model_family"]],
        on="model_key",
        how="left",
    )
    paired["dataset_token_count"] = paired["training_set"].map(dataset_token_count)
    natural_rows = []
    for family, family_df in paired.groupby("model_family"):
        family_df = family_df.dropna(subset=["training_set"])
        if len(family_df) < 2:
            continue
        records = family_df.to_dict("records")
        for left, right in combinations(records, 2):
            if left["training_set"] == right["training_set"]:
                continue
            higher = left if left["dataset_token_count"] >= right["dataset_token_count"] else right
            lower = right if higher is left else left
            natural_rows.append(
                {
                    "model_family": family,
                    "higher_data_model": higher["model_key"],
                    "lower_data_model": lower["model_key"],
                    "higher_training_set": higher["training_set"],
                    "lower_training_set": lower["training_set"],
                    "higher_dataset_token_count": higher["dataset_token_count"],
                    "lower_dataset_token_count": lower["dataset_token_count"],
                    "delta_gini_mae": higher["gini_mae"] - lower["gini_mae"],
                    "delta_mean_mae": higher["mean_mae"] - lower["mean_mae"],
                    "delta_global_mae": higher["global_mae"] - lower["global_mae"],
                    "delta_global_f1": higher["global_f1"] - lower["global_f1"],
                    "delta_global_daf": higher["global_daf"] - lower["global_daf"],
                }
            )
    write_table(pd.DataFrame(natural_rows), settings.paths.processed / "natural_experiment_results.csv")

    write_json(
        {
            "spearman_training_frequency_vs_mae": {"rho": freq_correlation.statistic, "p_value": freq_correlation.pvalue},
            "ols_r_squared": model.rsquared,
            "ols_adj_r_squared": model.rsquared_adj,
            "ols_params": model.params.to_dict(),
            "ols_pvalues": model.pvalues.to_dict(),
            "model_count": int(len(fairness_by_model)),
        },
        settings.paths.processed / "regression_results.json",
    )
    print("Saved Phase 4-6 analysis outputs.")


if __name__ == "__main__":
    main()
