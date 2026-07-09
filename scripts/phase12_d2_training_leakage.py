from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DATA_ROOT = (Path(__file__).resolve().parents[2] / "data").resolve()
D1_DIR = DATA_ROOT / "processed" / "clean_reference_analysis"
OUT_DIR = DATA_ROOT / "processed" / "database_relative_model_checks"
MPTRJ_CACHE = DATA_ROOT / "cache" / "mptrj_index"
MP2022_CSE = DATA_ROOT / "raw" / "wbm" / "40344436_2023-02-07-mp-computed-structure-entries.json.gz"
MP_ID_RE = re.compile(r"mp-[A-Za-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D2.2 model training leakage labels and held-out saturation.")
    parser.add_argument("--analysis-dir", type=Path, default=D1_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--mptrj-cache", type=Path, default=MPTRJ_CACHE)
    parser.add_argument("--mp2022-cse", type=Path, default=MP2022_CSE)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def load_id_set(path: Path) -> set[str] | None:
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    for col in ["material_id", "id_MP", "mp_id", "mpid"]:
        if col in frame.columns:
            return set(frame[col].dropna().astype(str))
    if len(frame.columns) == 1:
        return set(frame.iloc[:, 0].dropna().astype(str))
    return None


def find_mptrj_ids(cache_dir: Path) -> tuple[set[str] | None, Path | None]:
    candidates = [
        cache_dir / "mptrj_material_ids.csv",
        cache_dir / "mptrj_material_ids.parquet",
        cache_dir / "mptrj_index.csv",
        cache_dir / "mptrj_index.parquet",
    ]
    for path in candidates:
        ids = load_id_set(path)
        if ids:
            return ids, path
    return None, None


def build_mp2022_id_cache(raw_path: Path, cache_dir: Path) -> tuple[set[str], Path | None]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "mp2022_material_ids.csv"
    cached = load_id_set(out_path)
    if cached is not None:
        return cached, out_path
    if not raw_path.exists():
        return set(), None
    ids: set[str] = set()
    tail = ""
    with gzip.open(raw_path, "rt", encoding="utf-8", errors="ignore") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            text = tail + chunk
            ids.update(MP_ID_RE.findall(text))
            tail = text[-32:]
    pd.DataFrame({"material_id": sorted(ids)}).to_csv(out_path, index=False)
    return ids, out_path


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

    clean = pd.read_csv(args.analysis_dir / "clean_intermetallic_master.csv")
    preds = pd.read_csv(args.analysis_dir / "model_predictions_clean_joined.csv")
    preds = preds[preds["formation_status"] == "ok"].copy()

    mptrj_ids, mptrj_path = find_mptrj_ids(args.mptrj_cache)
    mp2022_ids, mp2022_path = build_mp2022_id_cache(args.mp2022_cse, args.mptrj_cache)
    exact_available = mptrj_ids is not None
    conservative_available = bool(mp2022_ids)

    labels = clean[
        [
            "pbe_job_id",
            "matalign_id",
            "id_MP",
            "reduced_formula",
            "spacegroup_number",
            "noise_bin",
            "validation_role",
            "chemistry_class",
            "Ef_std",
            "magnetic_heusler_review",
        ]
    ].copy()
    labels["in_sample_exact"] = labels["id_MP"].astype(str).map(lambda value: value in mptrj_ids) if exact_available else pd.NA
    labels["held_out_exact"] = (~labels["in_sample_exact"].astype(bool)) if exact_available else pd.NA
    labels["exact_match_status"] = "ok_mptrj_index" if exact_available else "blocked_no_mptrj_index"
    labels["in_sample_conservative"] = (
        labels["id_MP"].astype(str).map(lambda value: value in mp2022_ids) if conservative_available else pd.NA
    )
    if exact_available:
        labels["held_out_strict"] = (~labels["in_sample_exact"].astype(bool)) & (~labels["in_sample_conservative"].astype(bool))
    else:
        labels["held_out_strict"] = pd.NA
    labels["conservative_match_source"] = "mp2022_material_ids" if conservative_available else "missing_mp2022_cache"

    joined = preds.merge(labels, on="pbe_job_id", how="left", suffixes=("", "_leak"))
    summary_rows = []
    heldout_exact_n = int(labels["held_out_exact"].sum()) if exact_available else 0
    heldout_low_exact_n = int(((labels["held_out_exact"].astype(bool)) & (labels["noise_bin"] == "low") & (labels["validation_role"] == "saturation_probe")).sum()) if exact_available else 0
    for model, group in joined.groupby("model"):
        summary_rows.append(summarize(group, model=model, subset="clean_all_with_leakage_labels", rng=rng, n_boot=args.bootstrap))
        if exact_available:
            held = group[group["held_out_exact"].astype(bool)].copy()
            held_low = held[(held["noise_bin"] == "low") & (held["validation_role"] == "saturation_probe")].copy()
            summary_rows.append(summarize(held, model=model, subset="held_out_exact_clean_all", rng=rng, n_boot=args.bootstrap))
            summary_rows.append(summarize(held_low, model=model, subset="held_out_exact_clean_low_saturation", rng=rng, n_boot=args.bootstrap))
    heldout_summary = pd.DataFrame(summary_rows)
    payload = {
        "clean_rows": int(len(labels)),
        "exact_mptrj_index_available": bool(exact_available),
        "exact_mptrj_index_path": str(mptrj_path) if mptrj_path else None,
        "mp2022_index_path": str(mp2022_path) if mp2022_path else None,
        "in_sample_exact_count": int(labels["in_sample_exact"].sum()) if exact_available else None,
        "in_sample_exact_rate": float(labels["in_sample_exact"].mean()) if exact_available else None,
        "in_sample_conservative_count": int(labels["in_sample_conservative"].sum()) if conservative_available else None,
        "in_sample_conservative_rate": float(labels["in_sample_conservative"].mean()) if conservative_available else None,
        "heldout_exact_n": heldout_exact_n,
        "heldout_clean_low_saturation_n": heldout_low_exact_n,
        "heldout_underpowered": bool((not exact_available) or heldout_low_exact_n < 30),
        "blocked_reason": None if exact_available else "No local MPtrj material-id index was found and Figshare direct download is not a reliable scripted source in this environment.",
    }

    labels.to_csv(args.out_dir / "training_leakage_labels.csv", index=False)
    heldout_summary.to_csv(args.out_dir / "heldout_saturation_summary.csv", index=False)
    (args.out_dir / "training_leakage_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
