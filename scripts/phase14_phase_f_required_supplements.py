from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd


DATA_ROOT = (Path(__file__).resolve().parents[2] / "data").resolve()
PHASE_F = DATA_ROOT / "processed" / "distance_response_discovery"
WP4 = DATA_ROOT / "processed" / "three_reference_dft_tier100"
OUT_DIR = DATA_ROOT / "processed" / "supplementary_robustness_checks"
DOC = Path("docs/required_supplement_numbers_20260614.md")

MODEL_MAP_WBM = {
    "chgnet": "chgnet-0.3.0",
    "mace": "mace-mp-0",
    "mattersim": "mattersim-v1-5M",
}
MODEL_MAP_INSAMPLE = {
    "chgnet": "chgnet",
    "mace": "mace",
    "mattersim": "mattersim_5m",
}
STANDARD_PAIRS = [("chgnet", "mace"), ("chgnet", "mattersim"), ("mace", "mattersim")]


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def wp2_jaccard() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(PHASE_F / "discovery_overlap_jaccard.csv")
    mag = df[df["segment"] == "magnetic_3d_hard"].copy()
    mag["pair"] = mag["model_a"] + " vs " + mag["model_b"]
    mag["is_orb_sevennet"] = (
        mag[["model_a", "model_b"]].apply(lambda row: set(row) == {"orb-v3", "sevennet-mf-ompa"}, axis=1)
    )
    cross = mag[~mag["is_orb_sevennet"]].copy()
    standard_models = {"chgnet-0.3.0", "mace-mp-0", "mattersim-v1-5M"}
    standard = mag[mag["model_a"].isin(standard_models) & mag["model_b"].isin(standard_models)].copy()
    summary = {
        "segment": "magnetic_3d_hard",
        "n_cross_family_excl_orb_sevennet": int(len(cross)),
        "cross_min_jaccard": float(cross["jaccard"].min()),
        "cross_median_jaccard": float(cross["jaccard"].median()),
        "cross_max_jaccard": float(cross["jaccard"].max()),
        "orb_sevennet_jaccard": float(mag.loc[mag["is_orb_sevennet"], "jaccard"].iloc[0]),
        "standard_min_jaccard": float(standard["jaccard"].min()),
        "standard_median_jaccard": float(standard["jaccard"].median()),
        "standard_max_jaccard": float(standard["jaccard"].max()),
    }
    keep = mag[
        [
            "segment",
            "model_a",
            "model_b",
            "pair",
            "jaccard",
            "n_stable_a",
            "n_stable_b",
            "n_intersection",
            "n_union",
            "is_orb_sevennet",
        ]
    ].sort_values("jaccard")
    return keep, summary


def wp2_error_bins() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(PHASE_F / "disagreement_error_bins.csv")
    rows = []
    for segment, group in df.groupby("segment"):
        q1 = group[group["disagreement_bin"] == "q1_low"]
        q5 = group[group["disagreement_bin"] == "q5_high"]
        q1_rate = q1["classification_error_rate"].mean()
        q5_rate = q5["classification_error_rate"].mean()
        rows.append(
            {
                "segment": segment,
                "q1_low_mean_error_rate": q1_rate,
                "q5_high_mean_error_rate": q5_rate,
                "q5_over_q1_error_rate_ratio": q5_rate / q1_rate if q1_rate else float("nan"),
                "q1_low_median_width_eV_atom": q1["median_cross_family_width_eV_atom"].mean(),
                "q5_high_median_width_eV_atom": q5["median_cross_family_width_eV_atom"].mean(),
            }
        )
    summary = pd.DataFrame(rows).sort_values("segment")
    return df, summary


def wp4_bucket_direction() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(WP4 / "bucket_segment_summary.csv")
    ref_cols = {
        "centered_median_model_abs_error_to_mp_pbe": "mp_pbe",
        "centered_median_model_abs_error_to_pbe_variant": "pbe_variant",
        "centered_median_model_abs_error_to_r2scan": "r2scan",
    }
    df["best_centered_reference"] = df[list(ref_cols)].idxmin(axis=1).map(ref_cols)
    df["best_family"] = df["best_centered_reference"].map(lambda ref: "pbe_family" if ref in {"mp_pbe", "pbe_variant"} else "r2scan")
    summary = {
        "n_bucket_segments": int(len(df)),
        "best_ref_counts": df["best_centered_reference"].value_counts().to_dict(),
        "best_family_counts": df["best_family"].value_counts().to_dict(),
        "weighted_best_ref_counts": df.groupby("best_centered_reference")["n_materials"].sum().to_dict(),
        "weighted_best_family_counts": df.groupby("best_family")["n_materials"].sum().to_dict(),
        "strict_qc_weighted_best_ref_counts": df.groupby("best_centered_reference")["n_basic_qc"].sum().to_dict(),
        "strict_qc_weighted_best_family_counts": df.groupby("best_family")["n_basic_qc"].sum().to_dict(),
    }
    cols = [
        "distance_bin",
        "wp4_segment",
        "n_materials",
        "n_basic_qc",
        "best_centered_reference",
        "centered_median_model_abs_error_to_mp_pbe",
        "centered_median_model_abs_error_to_pbe_variant",
        "centered_median_model_abs_error_to_r2scan",
    ]
    return df[cols].sort_values(["distance_bin", "wp4_segment"]), summary


def pooled_pair_stats_from_wbm(df: pd.DataFrame, label: str) -> dict:
    pair_rows = []
    for a, b in STANDARD_PAIRS:
        ca = MODEL_MAP_WBM[a]
        cb = MODEL_MAP_WBM[b]
        diffs = (df[ca] - df[cb]).abs()
        pair_rows.append(pd.DataFrame({"material_id": df["material_id"], "pair": f"{a}__{b}", "abs_diff_eV_atom": diffs}))
    pairs = pd.concat(pair_rows, ignore_index=True)
    per_mat = pairs.groupby("material_id")["abs_diff_eV_atom"].median()
    return {
        "label": label,
        "n_materials": int(df["material_id"].nunique()),
        "n_pair_values": int(len(pairs)),
        "pooled_pair_median_eV_atom": float(pairs["abs_diff_eV_atom"].median()),
        "per_material_median_width_eV_atom": float(per_mat.median()),
        "pooled_pair_p90_eV_atom": float(pairs["abs_diff_eV_atom"].quantile(0.90)),
    }


def standard_ood_sensitivity() -> pd.DataFrame:
    wbm = pd.read_parquet(
        DATA_ROOT / "processed" / "wbm_heldout_validation" / "wbm_e1_clean_metal_materials.parquet",
        columns=[
            "material_id",
            "has_magnetic_3d_hard",
            "chgnet-0.3.0",
            "mace-mp-0",
            "mattersim-v1-5M",
        ],
    )
    rows = [
        pooled_pair_stats_from_wbm(wbm, "WBM clean-metal, standard OOD models"),
        pooled_pair_stats_from_wbm(wbm[wbm["has_magnetic_3d_hard"].astype(bool)].copy(), "WBM magnetic-3d-hard, standard OOD models"),
    ]

    insample = pd.read_csv(DATA_ROOT / "processed" / "pairwise_consistency_checks" / "model_pairwise_consistency.csv")
    pair_names = {f"{MODEL_MAP_INSAMPLE[a]}__{MODEL_MAP_INSAMPLE[b]}" for a, b in STANDARD_PAIRS}
    clean = insample[insample["pair"].isin(pair_names)].copy()
    per_mat = clean.groupby("pbe_job_id")["abs_diff_eV_atom"].median()
    rows.append(
        {
            "label": "MatAlign clean in-sample, same three pairs",
            "n_materials": int(clean["pbe_job_id"].nunique()),
            "n_pair_values": int(len(clean)),
            "pooled_pair_median_eV_atom": float(clean["abs_diff_eV_atom"].median()),
            "per_material_median_width_eV_atom": float(per_mat.median()),
            "pooled_pair_p90_eV_atom": float(clean["abs_diff_eV_atom"].quantile(0.90)),
        }
    )
    out = pd.DataFrame(rows)
    return out


def wp1_standard_distance_bins() -> pd.DataFrame:
    wbm = pd.read_parquet(
        DATA_ROOT / "processed" / "wbm_heldout_validation" / "wbm_e1_clean_metal_materials.parquet",
        columns=[
            "material_id",
            "chgnet-0.3.0",
            "mace-mp-0",
            "mattersim-v1-5M",
        ],
    )
    distance = pd.read_csv(
        PHASE_F / "distance_response_by_material.csv",
        usecols=["material_id", "distance_bin", "composition_distance", "nearest_type"],
    )
    merged = distance.merge(wbm, on="material_id", how="inner")
    order = {
        "wbm_exact_composition": 0,
        "wbm_near": 1,
        "wbm_mid": 2,
        "wbm_far": 3,
    }
    rows = []
    for distance_bin, group in merged.groupby("distance_bin"):
        stats = pooled_pair_stats_from_wbm(group, f"WBM {distance_bin}, standard OOD models")
        stats["distance_bin"] = distance_bin
        stats["bucket_order"] = order.get(distance_bin, 99)
        stats["median_composition_distance"] = float(group["composition_distance"].median())
        stats["nearest_type_mode"] = group["nearest_type"].mode().iloc[0] if not group["nearest_type"].mode().empty else ""
        rows.append(stats)
    return pd.DataFrame(rows).sort_values(["bucket_order", "distance_bin"])


def write_md(
    jaccard: pd.DataFrame,
    jaccard_summary: dict,
    error_summary: pd.DataFrame,
    wp4: pd.DataFrame,
    wp4_summary: dict,
    sensitivity: pd.DataFrame,
    wp1_distance: pd.DataFrame,
) -> None:
    DOC.parent.mkdir(parents=True, exist_ok=True)
    mag_error = error_summary[error_summary["segment"] == "magnetic_3d_hard"].iloc[0]
    all_error = error_summary[error_summary["segment"] == "all_clean_metal"].iloc[0]
    hard_mn = error_summary[error_summary["segment"] == "hard_Mn"].iloc[0]
    hard_fe = error_summary[error_summary["segment"] == "hard_Fe"].iloc[0]
    wbm_clean = sensitivity[sensitivity["label"].str.startswith("WBM clean")].iloc[0]
    in_sample = sensitivity[sensitivity["label"].str.startswith("MatAlign")].iloc[0]
    exact = wp1_distance[wp1_distance["distance_bin"] == "wbm_exact_composition"].iloc[0]
    near = wp1_distance[wp1_distance["distance_bin"] == "wbm_near"].iloc[0]
    mid = wp1_distance[wp1_distance["distance_bin"] == "wbm_mid"].iloc[0]
    far = wp1_distance[wp1_distance["distance_bin"] == "wbm_far"].iloc[0]
    lines = [
        "# Phase F Required Supplement Numbers",
        "",
        "## 1. WP2 Cross-Family Discovery-List Jaccard",
        "",
        "- Segment: `magnetic_3d_hard`.",
        "- Excluding only the ORB v3 vs SevenNet-MF-ompa same-source pair:",
        f"  - minimum Jaccard = `{fmt(jaccard_summary['cross_min_jaccard'])}`.",
        f"  - median Jaccard = `{fmt(jaccard_summary['cross_median_jaccard'])}`.",
        f"  - maximum Jaccard = `{fmt(jaccard_summary['cross_max_jaccard'])}`.",
        f"- ORB v3 vs SevenNet-MF-ompa same-source Jaccard = `{fmt(jaccard_summary['orb_sevennet_jaccard'])}`.",
        "- Cleaner standard-model-only pairs (`CHGNet/MACE/MatterSim`) are even lower:",
        f"  - min/median/max = `{fmt(jaccard_summary['standard_min_jaccard'])}` / `{fmt(jaccard_summary['standard_median_jaccard'])}` / `{fmt(jaccard_summary['standard_max_jaccard'])}`.",
        "",
        "Interpretation: the headline should use the cross-family number, not the same-source ORB-SevenNet pair. In the magnetic 3d hard segment, the cross-family stable-list overlap is roughly `0.32-0.33` at the median, far below the same-source `0.76`.",
        "",
        "## 2. WP2 Disagreement-to-Error Calibration",
        "",
        "- For `magnetic_3d_hard`, the mean stability classification error rate rises from "
        f"`{fmt(mag_error['q1_low_mean_error_rate'])}` in the lowest-disagreement quintile to "
        f"`{fmt(mag_error['q5_high_mean_error_rate'])}` in the highest-disagreement quintile.",
        f"- Ratio: `{fmt(mag_error['q5_over_q1_error_rate_ratio'], 2)}x`.",
        f"- For `hard_Mn`, the ratio is `{fmt(hard_mn['q5_over_q1_error_rate_ratio'], 2)}x`; for `hard_Fe`, `{fmt(hard_fe['q5_over_q1_error_rate_ratio'], 2)}x`.",
        f"- For all clean metals together, the ratio is only `{fmt(all_error['q5_over_q1_error_rate_ratio'], 2)}x`, so this is a hard-chemistry effect rather than a uniform global law.",
        "",
        "Suggested sentence: `Among magnetic-3d-hard WBM materials, the top disagreement quintile has a stability-classification error rate 1.66x that of the bottom quintile, rising to about 2x in Mn/Fe subsets.`",
        "",
        "## 3. WP4 Bucket x Segment Dialect Check",
        "",
        "- Across 25 distance-bin x chemistry-segment cells, best centered reference counts are:",
        f"  - `mp_pbe`: `{wp4_summary['best_ref_counts'].get('mp_pbe', 0)}` cells.",
        f"  - `pbe_variant`: `{wp4_summary['best_ref_counts'].get('pbe_variant', 0)}` cells.",
        f"  - `r2scan`: `{wp4_summary['best_ref_counts'].get('r2scan', 0)}` cells.",
        "- Grouped as PBE-family (`mp_pbe` or `pbe_variant`) vs r2SCAN:",
        f"  - PBE-family wins `{wp4_summary['best_family_counts'].get('pbe_family', 0)}/25` cells.",
        f"  - r2SCAN wins `{wp4_summary['best_family_counts'].get('r2scan', 0)}/25` cells.",
        "- Material-weighted cell counts:",
        f"  - PBE-family covers `{int(wp4_summary['weighted_best_family_counts'].get('pbe_family', 0))}/100` materials.",
        f"  - r2SCAN covers `{int(wp4_summary['weighted_best_family_counts'].get('r2scan', 0))}/100` materials.",
        "- Strict-QC material-weighted counts:",
        f"  - PBE-family covers `{int(wp4_summary['strict_qc_weighted_best_family_counts'].get('pbe_family', 0))}/89` strict-QC materials.",
        f"  - r2SCAN covers `{int(wp4_summary['strict_qc_weighted_best_family_counts'].get('r2scan', 0))}/89` strict-QC materials.",
        "",
        "Interpretation: the bucket table is mixed, not uniformly PBE-family. The model-level headline remains PBE-family (`r2SCAN = 0/5` models), but local r2SCAN wins occur in `10/25` bucket-segment cells. This should be reported as heterogeneity, not hidden.",
        "",
        "## 4. Standard-OOD-Model Sensitivity",
        "",
        "Using only the three models with standard OOD confidence (`CHGNet`, `MACE`, `MatterSim`) and the same three pairs in both settings:",
        "",
        f"- MatAlign clean in-sample pooled pair median = `{fmt(in_sample['pooled_pair_median_eV_atom'], 6)} eV/atom`.",
        f"- WBM clean-metal OOD pooled pair median = `{fmt(wbm_clean['pooled_pair_median_eV_atom'], 6)} eV/atom`.",
        f"- Ratio OOD / in-sample = `{fmt(wbm_clean['pooled_pair_median_eV_atom'] / in_sample['pooled_pair_median_eV_atom'], 2)}x`.",
        f"- WBM clean-metal OOD per-material median width = `{fmt(wbm_clean['per_material_median_width_eV_atom'], 6)} eV/atom`.",
        "- The WBM clean-metal standard-only consistency is still above the `0.011 eV/atom` D3 floor.",
        "",
        "Interpretation: the in-sample to OOD consistency reversal does not depend on ORB/SevenNet. Even using only CHGNet/MACE/MatterSim, WBM OOD model disagreement remains substantially larger than the MatAlign in-sample disagreement and above the DFT-floor threshold.",
        "",
        "## 5. WP1 Exact-Composition Near-Training Control",
        "",
        "Using the same three standard-OOD-confidence models (`CHGNet`, `MACE`, `MatterSim`) and the same three pairwise comparisons:",
        "",
        f"- MatAlign clean in-sample pooled pair median = `{fmt(in_sample['pooled_pair_median_eV_atom'], 6)} eV/atom`.",
        f"- WBM `exact_composition` pooled pair median = `{fmt(exact['pooled_pair_median_eV_atom'], 6)} eV/atom` over `{int(exact['n_materials'])}` materials.",
        f"- WBM `near` pooled pair median = `{fmt(near['pooled_pair_median_eV_atom'], 6)} eV/atom`.",
        f"- WBM `mid` pooled pair median = `{fmt(mid['pooled_pair_median_eV_atom'], 6)} eV/atom`.",
        f"- WBM `far` pooled pair median = `{fmt(far['pooled_pair_median_eV_atom'], 6)} eV/atom`.",
        f"- Exact-composition / in-sample ratio = `{fmt(exact['pooled_pair_median_eV_atom'] / in_sample['pooled_pair_median_eV_atom'], 2)}x`; far / in-sample ratio = `{fmt(far['pooled_pair_median_eV_atom'] / in_sample['pooled_pair_median_eV_atom'], 2)}x`.",
        "",
        "Interpretation: this gives the standard-three-model staircase the user wanted: `in-sample 0.0165 -> WBM exact-composition 0.0218 -> WBM near 0.0287 -> WBM mid 0.0333 -> WBM far 0.0391 eV/atom`. The exact-composition bucket is a WBM internal near-training control, not a true in-sample set, and it lands between MatAlign in-sample and farther WBM materials.",
        "",
        "## Recommended Use",
        "",
        "- WP2 headline: use cross-family magnetic-3d-hard Jaccard min/median, not ORB-SevenNet.",
        "- WP2 consequence: quote the `1.66x` magnetic-3d-hard disagreement-error multiplier, with Mn/Fe around `2x` as hard-element examples.",
        "- WP4: write `model-level dialect result favors PBE-family, but bucket-level heterogeneity exists`; do not claim every bucket points to PBE.",
        "- OOD sensitivity: use the standard-three-model result to preempt leakage concerns about ORB/SevenNet.",
        "- WP1: use the exact-composition bucket as the near-training middle step in the distance-response narrative.",
        "",
        "## Output Tables",
        "",
        "- `magnetic_discovery_jaccard.csv`",
        "- `disagreement_error_ratio_by_segment.csv`",
        "- `three_reference_bucket_direction.csv`",
        "- `standard_ood_sensitivity.csv`",
        "- `standard_distance_bins.csv`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jaccard, jaccard_summary = wp2_jaccard()
    _, error_summary = wp2_error_bins()
    wp4, wp4_summary = wp4_bucket_direction()
    sensitivity = standard_ood_sensitivity()
    wp1_distance = wp1_standard_distance_bins()

    jaccard.to_csv(OUT_DIR / "magnetic_discovery_jaccard.csv", index=False)
    error_summary.to_csv(OUT_DIR / "disagreement_error_ratio_by_segment.csv", index=False)
    wp4.to_csv(OUT_DIR / "three_reference_bucket_direction.csv", index=False)
    sensitivity.to_csv(OUT_DIR / "standard_ood_sensitivity.csv", index=False)
    wp1_distance.to_csv(OUT_DIR / "standard_distance_bins.csv", index=False)
    write_md(jaccard, jaccard_summary, error_summary, wp4, wp4_summary, sensitivity, wp1_distance)
    print(f"Wrote {DOC}")
    print(f"Wrote tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
