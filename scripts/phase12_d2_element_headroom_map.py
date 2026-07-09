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
OUT_DIR = DATA_ROOT / "processed" / "v3_analysis_d2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D2.4 element-resolved headroom map.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def element_set(value: Any) -> set[str]:
    return {token for token in str(value).split() if token}


def chemistry_hint(element: str) -> str:
    if element in {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"}:
        return "3d_magnetic"
    if element in {"Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au"}:
        return "4d_5d"
    if element in {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Ac", "Th", "Pa", "U"}:
        return "f_block"
    return "main_group"


def bootstrap_median_ci(values: np.ndarray, *, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan, math.nan
    if len(values) == 1:
        val = float(values[0])
        return val, val, val
    medians = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        medians[idx] = np.median(rng.choice(values, size=len(values), replace=True))
    return float(np.median(values)), float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def wilcoxon_greater(error: pd.Series, floor: pd.Series) -> float | None:
    pairs = pd.DataFrame({"error": pd.to_numeric(error, errors="coerce"), "floor": pd.to_numeric(floor, errors="coerce")}).dropna()
    if len(pairs) < 5:
        return None
    diff = pairs["error"] - pairs["floor"]
    if np.allclose(diff, 0):
        return 1.0
    try:
        return float(wilcoxon(diff, alternative="greater", zero_method="wilcox").pvalue)
    except ValueError:
        return None


def ci_overlap(a_low: float, a_high: float, b_low: float, b_high: float) -> bool:
    if any(math.isnan(x) for x in [a_low, a_high, b_low, b_high]):
        return False
    return max(a_low, b_low) <= min(a_high, b_high)


def classify(row: dict[str, Any]) -> str:
    if row["n"] < 5:
        return "insufficient_n"
    pvalue = row["wilcoxon_error_greater_than_floor_p"]
    if row["median_ci_overlap"] and (pvalue is None or pvalue >= 0.05):
        return "saturated_indistinguishable_from_floor"
    if pvalue is not None and pvalue < 0.05 and row["model_abs_error_median_eV_atom"] > row["ef_std_median_eV_atom"]:
        return "above_floor_headroom"
    if pvalue is not None and pvalue >= 0.05 and row["model_abs_error_median_eV_atom"] <= row["ef_std_median_eV_atom"]:
        return "below_floor_or_saturated"
    return "inconclusive"


def summarize(frame: pd.DataFrame, *, model: str, subset: str, element: str, rng: np.random.Generator, n_boot: int) -> dict[str, Any]:
    err = pd.to_numeric(frame["abs_model_error_vs_Ef_MP_eV_atom"], errors="coerce").to_numpy(float)
    floor = pd.to_numeric(frame["Ef_std"], errors="coerce").to_numpy(float)
    err_med, err_low, err_high = bootstrap_median_ci(err, rng=rng, n_boot=n_boot)
    floor_med, floor_low, floor_high = bootstrap_median_ci(floor, rng=rng, n_boot=n_boot)
    row = {
        "model": model,
        "subset": subset,
        "element": element,
        "chemistry_hint": chemistry_hint(element),
        "n": int(len(frame)),
        "model_abs_error_median_eV_atom": err_med,
        "model_abs_error_ci95_low": err_low,
        "model_abs_error_ci95_high": err_high,
        "ef_std_median_eV_atom": floor_med,
        "ef_std_ci95_low": floor_low,
        "ef_std_ci95_high": floor_high,
        "median_ci_overlap": ci_overlap(err_low, err_high, floor_low, floor_high),
        "wilcoxon_error_greater_than_floor_p": wilcoxon_greater(frame["abs_model_error_vs_Ef_MP_eV_atom"], frame["Ef_std"]),
        "within_floor_rate": float((frame["abs_model_error_vs_Ef_MP_eV_atom"] <= frame["Ef_std"]).mean()) if len(frame) else math.nan,
    }
    row["saturation_call"] = classify(row)
    return row


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    pred_path = args.out_dir / "frontier_model_predictions_clean.csv"
    if not pred_path.exists():
        raise SystemExit(f"Missing {pred_path}; run phase12_d2_frontier_model_summary.py first.")
    preds = pd.read_csv(pred_path)
    if "elements_str" not in preds.columns:
        raise SystemExit("frontier_model_predictions_clean.csv must contain elements_str.")

    labels_path = args.out_dir / "training_leakage_labels.csv"
    if labels_path.exists() and "held_out_exact" not in preds.columns:
        labels = pd.read_csv(labels_path)
        preds = preds.merge(labels[["pbe_job_id", "held_out_exact"]], on="pbe_job_id", how="left")

    heldout_available = "held_out_exact" in preds.columns and preds["held_out_exact"].notna().any() and preds["held_out_exact"].astype(bool).sum() >= 30
    subset_defs = {"clean_all": preds}
    if heldout_available:
        subset_defs["held_out_exact_clean_all"] = preds[preds["held_out_exact"].astype(bool)].copy()

    rows = []
    for subset_name, subset in subset_defs.items():
        for model, group in subset.groupby("model"):
            elements = sorted(set().union(*[element_set(value) for value in group["elements_str"].dropna()]))
            for element in elements:
                sub = group[group["elements_str"].map(lambda value: element in element_set(value))].copy()
                rows.append(summarize(sub, model=model, subset=subset_name, element=element, rng=rng, n_boot=args.bootstrap))
    table = pd.DataFrame(rows)
    class_summary = (
        table.groupby(["subset", "model", "chemistry_hint", "saturation_call"])
        .agg(n_elements=("element", "count"), median_element_n=("n", "median"))
        .reset_index()
    )
    table.to_csv(args.out_dir / "element_headroom_map_d2.csv", index=False)
    class_summary.to_csv(args.out_dir / "element_class_headroom_summary_d2.csv", index=False)
    payload = {
        "rows": int(len(table)),
        "heldout_available": bool(heldout_available),
        "models": sorted(table["model"].dropna().unique().tolist()) if not table.empty else [],
    }
    (args.out_dir / "element_headroom_map_d2_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
