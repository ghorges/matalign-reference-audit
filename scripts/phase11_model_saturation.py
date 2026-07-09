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
ANALYSIS_DIR = DATA_ROOT / "processed" / "v3_analysis"
DOCS_DIR = Path("docs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MatAlign v3 model-vs-noise-floor saturation.")
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def load_model_predictions(analysis_dir: Path) -> pd.DataFrame:
    model_dir = analysis_dir / "model_outputs"
    frames = []
    for path in sorted(model_dir.glob("*_compound_predictions.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def element_set(value: Any) -> set[str]:
    return {token for token in str(value).split() if token}


def bootstrap_median_ci(values: np.ndarray, *, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan, math.nan
    if len(values) == 1:
        val = float(values[0])
        return val, val, val
    medians = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        medians[i] = np.median(sample)
    return float(np.median(values)), float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def wilcoxon_greater(error: pd.Series, floor: pd.Series) -> float | None:
    pairs = pd.DataFrame({"error": error, "floor": floor}).dropna()
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
    overlap = bool(row["median_ci_overlap"])
    pvalue = row["wilcoxon_error_greater_than_floor_p"]
    if overlap and (pvalue is None or pvalue >= 0.05):
        return "saturated_indistinguishable_from_floor"
    if pvalue is not None and pvalue < 0.05 and row["model_abs_error_median_eV_atom"] > row["ef_std_median_eV_atom"]:
        return "above_floor_headroom"
    if pvalue is not None and pvalue >= 0.05 and row["model_abs_error_median_eV_atom"] <= row["ef_std_median_eV_atom"]:
        return "below_floor_or_saturated"
    return "inconclusive"


def summarize_subset(
    frame: pd.DataFrame,
    *,
    model: str,
    label: str,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, Any]:
    errors = pd.to_numeric(frame["abs_model_error_vs_Ef_MP_eV_atom"], errors="coerce").to_numpy(dtype=float)
    floors = pd.to_numeric(frame["Ef_std"], errors="coerce").to_numpy(dtype=float)
    err_med, err_low, err_high = bootstrap_median_ci(errors, rng=rng, n_boot=n_boot)
    floor_med, floor_low, floor_high = bootstrap_median_ci(floors, rng=rng, n_boot=n_boot)
    pvalue = wilcoxon_greater(frame["abs_model_error_vs_Ef_MP_eV_atom"], frame["Ef_std"])
    row = {
        "model": model,
        "subset": label,
        "n": int(len(frame)),
        "model_abs_error_median_eV_atom": err_med,
        "model_abs_error_ci95_low": err_low,
        "model_abs_error_ci95_high": err_high,
        "ef_std_median_eV_atom": floor_med,
        "ef_std_ci95_low": floor_low,
        "ef_std_ci95_high": floor_high,
        "median_ci_overlap": ci_overlap(err_low, err_high, floor_low, floor_high),
        "wilcoxon_error_greater_than_floor_p": pvalue,
        "within_floor_rate": float((frame["abs_model_error_vs_Ef_MP_eV_atom"] <= frame["Ef_std"]).mean())
        if len(frame)
        else math.nan,
    }
    row["saturation_call"] = classify(row)
    return row


def markdown_table(frame: pd.DataFrame, n: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.head(n).to_markdown(index=False)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    model_preds = load_model_predictions(args.analysis_dir)
    if model_preds.empty:
        raise SystemExit(f"No model prediction files found in {args.analysis_dir / 'model_outputs'}")

    clean = pd.read_csv(args.analysis_dir / "clean_intermetallic_master.csv")
    clean_cols = [
        "pbe_job_id",
        "elements_str",
        "magnetic_heusler_review",
        "clean_intermetallic_universe",
        "raw_formation_energy_per_atom_eV",
        "abs_raw_residual_vs_mp_eV_atom",
        "raw_residual_within_ef_std",
    ]
    preds = model_preds.merge(clean[clean_cols], on="pbe_job_id", how="left", suffixes=("", "_clean"))
    preds = preds[preds["formation_status"] == "ok"].copy()

    summary_rows = []
    for model, group in preds.groupby("model"):
        clean_group = group[group["clean_intermetallic_universe"].astype(bool)].copy()
        saturation = clean_group[
            (clean_group["noise_bin"] == "low") & (clean_group["validation_role"] == "saturation_probe")
        ].copy()
        no_mag = clean_group[~clean_group["magnetic_heusler_review"].astype(bool)].copy()
        summary_rows.extend(
            [
                summarize_subset(clean_group, model=model, label="clean_all", rng=rng, n_boot=args.bootstrap),
                summarize_subset(saturation, model=model, label="clean_low_saturation", rng=rng, n_boot=args.bootstrap),
                summarize_subset(no_mag, model=model, label="clean_excluding_magnetic_review", rng=rng, n_boot=args.bootstrap),
            ]
        )
    summary = pd.DataFrame(summary_rows)

    element_rows = []
    for model, group in preds.groupby("model"):
        elements = sorted(set().union(*[element_set(value) for value in group["elements_str"].dropna()]))
        for element in elements:
            sub = group[group["elements_str"].map(lambda value: element in element_set(value))].copy()
            row = summarize_subset(sub, model=model, label=f"element:{element}", rng=rng, n_boot=args.bootstrap)
            row["element"] = element
            row["chemistry_hint"] = (
                "3d_magnetic" if element in {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"} else
                "4d_5d" if element in {"Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au"} else
                "f_block" if element in {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Ac", "Th", "Pa", "U"} else
                "main_group"
            )
            element_rows.append(row)
    element_table = pd.DataFrame(element_rows)

    model_preds.to_csv(args.analysis_dir / "model_predictions_clean.csv", index=False)
    preds.to_csv(args.analysis_dir / "model_predictions_clean_joined.csv", index=False)
    summary.to_csv(args.analysis_dir / "model_noise_floor_summary.csv", index=False)
    element_table.to_csv(args.analysis_dir / "element_resolved_saturation.csv", index=False)
    payload = {
        "models": sorted(preds["model"].unique().tolist()),
        "prediction_rows": int(len(preds)),
        "summary": summary.to_dict("records"),
        "element_rows": int(len(element_table)),
        "go_no_go": {
            row["model"]: row["saturation_call"]
            for row in summary[summary["subset"] == "clean_low_saturation"].to_dict("records")
        },
    }
    (args.analysis_dir / "model_saturation_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report_path = args.docs_dir / "v3_analysis_phase_results_20260606.md"
    previous = report_path.read_text(encoding="utf-8") if report_path.exists() else "# MatAlign v3 Analysis Phase Results\n"
    addition = [
        "",
        "## Phase D Model Saturation Pilot",
        "",
        "Main metric is `model_mp_reference_formation_energy_per_atom_eV - Ef_MP`.",
        "",
        "### Model Noise-Floor Summary",
        "",
        markdown_table(summary, 20),
        "",
        "### Element-Resolved Saturation",
        "",
        markdown_table(
            element_table.sort_values(["model", "saturation_call", "element"])[
                [
                    "model",
                    "element",
                    "chemistry_hint",
                    "n",
                    "model_abs_error_median_eV_atom",
                    "ef_std_median_eV_atom",
                    "median_ci_overlap",
                    "wilcoxon_error_greater_than_floor_p",
                    "within_floor_rate",
                    "saturation_call",
                ]
            ],
            60,
        ),
        "",
        "### Model Pilot Boundary",
        "",
        "- CHGNet and MACE are a pilot only. If the saturation call is favorable, the paper still needs 3-5 model families for a general model-population claim.",
        "- `within_floor_rate` is auxiliary; around 50% is compatible with saturation because raw PBE itself lands near the floor at about that rate.",
    ]
    marker = "## Phase D Model Saturation Pilot"
    if marker in previous:
        previous = previous.split(marker)[0].rstrip()
    report_path.write_text(previous.rstrip() + "\n" + "\n".join(addition) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
