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
D1_DIR = DATA_ROOT / "processed" / "v3_analysis"
OUT_DIR = DATA_ROOT / "processed" / "v3_analysis_d2"
MODEL_OUTPUT_DIR = OUT_DIR / "model_outputs"
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
    "Ef_MP",
    "Ef_OQMD",
    "Ef_AFLOW",
    "Ef_JARVIS",
    "Ef_std",
    "raw_formation_energy_per_atom_eV",
    "magnetic_heusler_review",
    "clean_intermetallic_universe",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate D2 frontier model predictions and saturation calls.")
    parser.add_argument("--analysis-dir", type=Path, default=D1_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def load_predictions(analysis_dir: Path, out_dir: Path) -> pd.DataFrame:
    frames = []
    d1_path = analysis_dir / "model_predictions_clean_joined.csv"
    if d1_path.exists():
        d1 = pd.read_csv(d1_path)
        d1["source_phase"] = "D1_pilot"
        frames.append(d1)
    for path in sorted((out_dir / "model_outputs").glob("*_compound_predictions.csv")):
        model = path.name.replace("_compound_predictions.csv", "")
        frame = pd.read_csv(path)
        frame["model"] = frame.get("model", model)
        frame["source_phase"] = "D2_frontier"
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    preds = pd.concat(frames, ignore_index=True, sort=False)
    preds = preds[preds["formation_status"].fillna("ok").astype(str).str.startswith("ok")].copy()
    return preds


def attach_clean_metadata(preds: pd.DataFrame, analysis_dir: Path) -> pd.DataFrame:
    clean = pd.read_csv(analysis_dir / "clean_intermetallic_master.csv")
    clean = clean[[col for col in CleanCols if col in clean.columns]].drop_duplicates("pbe_job_id")
    if "pbe_job_id" not in preds.columns:
        return preds
    merged = preds.merge(clean, on="pbe_job_id", how="left", suffixes=("", "_clean"))
    for col in clean.columns:
        if col == "pbe_job_id":
            continue
        clean_col = f"{col}_clean"
        if clean_col in merged.columns:
            merged[col] = merged[col].combine_first(merged[clean_col]) if col in merged.columns else merged[clean_col]
            merged = merged.drop(columns=[clean_col])
    return merged


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


def summarize(frame: pd.DataFrame, *, model: str, subset: str, rng: np.random.Generator, n_boot: int) -> dict[str, Any]:
    err = pd.to_numeric(frame["abs_model_error_vs_Ef_MP_eV_atom"], errors="coerce").to_numpy(float)
    floor = pd.to_numeric(frame["Ef_std"], errors="coerce").to_numpy(float)
    err_med, err_low, err_high = bootstrap_median_ci(err, rng=rng, n_boot=n_boot)
    floor_med, floor_low, floor_high = bootstrap_median_ci(floor, rng=rng, n_boot=n_boot)
    row = {
        "model": model,
        "subset": subset,
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
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    preds = load_predictions(args.analysis_dir, args.out_dir)
    if preds.empty:
        raise SystemExit("No D1 or D2 model predictions found.")
    preds = attach_clean_metadata(preds, args.analysis_dir)
    leakage_path = args.out_dir / "training_leakage_labels.csv"
    if leakage_path.exists():
        labels = pd.read_csv(leakage_path)
        preds = preds.merge(labels[["pbe_job_id", "in_sample_exact", "held_out_exact", "held_out_strict"]], on="pbe_job_id", how="left")

    rows = []
    for model, group in preds.groupby("model"):
        clean_group = group[group["clean_intermetallic_universe"].fillna(True).astype(bool)].copy()
        low_sat = clean_group[(clean_group["noise_bin"] == "low") & (clean_group["validation_role"] == "saturation_probe")]
        rows.append(summarize(clean_group, model=model, subset="clean_all", rng=rng, n_boot=args.bootstrap))
        rows.append(summarize(low_sat, model=model, subset="clean_low_saturation", rng=rng, n_boot=args.bootstrap))
        if "held_out_exact" in clean_group.columns and clean_group["held_out_exact"].notna().any():
            held = clean_group[clean_group["held_out_exact"].astype(bool)]
            held_low = held[(held["noise_bin"] == "low") & (held["validation_role"] == "saturation_probe")]
            rows.append(summarize(held, model=model, subset="held_out_exact_clean_all", rng=rng, n_boot=args.bootstrap))
            rows.append(summarize(held_low, model=model, subset="held_out_exact_clean_low_saturation", rng=rng, n_boot=args.bootstrap))
    summary = pd.DataFrame(rows)

    failures = []
    for path in sorted((args.out_dir / "model_outputs").glob("*_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("compound_ok", 0) < payload.get("compound_rows", 0):
            failures.append(payload)
    for path in sorted((args.out_dir / "model_outputs").glob("*_failures.json")):
        failures.append(json.loads(path.read_text(encoding="utf-8")))
    failure_frame = pd.json_normalize(failures) if failures else pd.DataFrame(columns=["model", "status", "reason"])

    preds.to_csv(args.out_dir / "frontier_model_predictions_clean.csv", index=False)
    summary.to_csv(args.out_dir / "frontier_model_noise_floor_summary.csv", index=False)
    failure_frame.to_csv(args.out_dir / "frontier_model_failures.csv", index=False)
    payload = {
        "models": sorted(preds["model"].dropna().unique().tolist()),
        "prediction_rows": int(len(preds)),
        "summary_rows": int(len(summary)),
        "failure_rows": int(len(failure_frame)),
    }
    (args.out_dir / "frontier_model_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
