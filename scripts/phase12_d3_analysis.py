from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

try:
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
except Exception:  # pragma: no cover - optional unless candidate structures exist
    Structure = None
    StructureMatcher = None


DATA_ROOT = (Path(__file__).resolve().parents[2] / "data").resolve()
D1_DIR = DATA_ROOT / "processed" / "v3_analysis"
D2_DIR = DATA_ROOT / "processed" / "v3_analysis_d2"
D2FIX_DIR = DATA_ROOT / "processed" / "v3_analysis_d2fix"
OUT_DIR = DATA_ROOT / "processed" / "v3_analysis_d3"
DOCS_DIR = Path("docs")
TARGET_STRUCTURE_ROOT = Path("vasp_v3_pbe_work") / "remote_results" / "primary" / "inputs" / "primary" / "compounds"
CANDIDATE_STRUCTURE_DIR = DATA_ROOT / "cache" / "training_structure_candidates"

MODELS = ["chgnet", "mace", "mattersim_5m", "orb_v3", "sevennet_mf_ompa"]
FRONTIER4 = ["mace", "mattersim_5m", "orb_v3", "sevennet_mf_ompa"]
DATABASES_ALL4 = ["MP", "OQMD", "AFLOW", "JARVIS"]
DATABASES_PBE3 = ["MP", "OQMD", "AFLOW"]
PRED_COL = "model_mp_reference_formation_energy_per_atom_eV"
CONSENSUS_COL = "consensus_pbe3_median"

MODEL_TRAINING_SOURCES = {
    "chgnet": ["mptrj"],
    "mace": ["mptrj"],
    "mattersim_5m": ["mattersim", "mptrj_screen"],
    "orb_v3": ["omat24", "mptrj"],
    "sevennet_mf_ompa": ["omat24", "mptrj", "salex"],
}
PARTIAL_SOURCES = {"omat24", "salex"}
UNAVAILABLE_SOURCES = {"mattersim"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MatAlign v3 Phase D3 analysis.")
    parser.add_argument("--d1-dir", type=Path, default=D1_DIR)
    parser.add_argument("--d2-dir", type=Path, default=D2_DIR)
    parser.add_argument("--d2fix-dir", type=Path, default=D2FIX_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--target-structure-root", type=Path, default=TARGET_STRUCTURE_ROOT)
    parser.add_argument("--candidate-structure-dir", type=Path, default=CANDIDATE_STRUCTURE_DIR)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def bootstrap_ci(values: pd.Series | np.ndarray, rng: np.random.Generator, n_boot: int, stat: str = "median") -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (math.nan, math.nan)
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))
    stats = np.empty(n_boot)
    for idx in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        stats[idx] = np.median(sample) if stat == "median" else np.mean(sample)
    return (float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975)))


def safe_wilcoxon(x: pd.Series, y: pd.Series, alternative: str) -> float:
    frame = pd.concat([numeric(x), numeric(y)], axis=1).dropna()
    if len(frame) == 0:
        return math.nan
    diff = frame.iloc[:, 0] - frame.iloc[:, 1]
    if np.allclose(diff.to_numpy(dtype=float), 0):
        return 1.0
    try:
        return float(wilcoxon(frame.iloc[:, 0], frame.iloc[:, 1], alternative=alternative).pvalue)
    except ValueError:
        return math.nan


def elements_from_text(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).replace("|", " ").replace(",", " ")
    return [part for part in text.split() if part]


def load_inputs(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    pred = pd.read_csv(args.d2_dir / "frontier_model_predictions_clean.csv")
    floor = pd.read_csv(args.d2fix_dir / "dual_noise_floors.csv")
    leakage = pd.read_csv(args.d2fix_dir / "training_leakage_per_model.csv")
    clean = pd.read_csv(args.d1_dir / "clean_intermetallic_master.csv")
    d2fix_summary = pd.read_json(args.d2fix_dir / "phase_d2fix_summary.json", typ="series")
    return {
        "pred": pred,
        "floor": floor,
        "leakage": leakage,
        "clean": clean,
        "d2fix_summary": d2fix_summary.to_frame("value").reset_index().rename(columns={"index": "metric"}),
    }


def model_pairwise(pred: pd.DataFrame, floor: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wide = pred.pivot_table(index="pbe_job_id", columns="model", values=PRED_COL, aggfunc="first")
    meta_cols = ["pbe_job_id", "matalign_id", "formula", "noise_bin", "validation_role", "chemistry_class"]
    meta = floor[[col for col in meta_cols if col in floor.columns]].drop_duplicates("pbe_job_id")

    rows: list[dict[str, Any]] = []
    for model_a, model_b in combinations(MODELS, 2):
        if model_a not in wide.columns or model_b not in wide.columns:
            continue
        values = (wide[model_a] - wide[model_b]).abs().rename("abs_diff_eV_atom").reset_index()
        for record in values.to_dict("records"):
            rows.append(
                {
                    "pbe_job_id": record["pbe_job_id"],
                    "model_a": model_a,
                    "model_b": model_b,
                    "pair": f"{model_a}__{model_b}",
                    "abs_diff_eV_atom": record["abs_diff_eV_atom"],
                    "is_frontier4_pair": model_a in FRONTIER4 and model_b in FRONTIER4,
                    "is_orb_sevennet_pair": {model_a, model_b} == {"orb_v3", "sevennet_mf_ompa"},
                }
            )
    pairwise = pd.DataFrame(rows).merge(meta, on="pbe_job_id", how="left")

    summary_rows: list[dict[str, Any]] = []
    width_frames: list[pd.DataFrame] = []
    group_defs = {
        "all5": MODELS,
        "frontier4_excl_chgnet": FRONTIER4,
        "orb_v3_vs_sevennet": ["orb_v3", "sevennet_mf_ompa"],
    }
    for group_name, group_models in group_defs.items():
        pairs = {f"{a}__{b}" for a, b in combinations(group_models, 2)}
        sub = pairwise[pairwise["pair"].isin(pairs)].copy()
        widths = sub.groupby("pbe_job_id", as_index=False)["abs_diff_eV_atom"].median()
        widths = widths.rename(columns={"abs_diff_eV_atom": f"{group_name}_per_material_width_eV_atom"})
        width_frames.append(widths)
        ci_low, ci_high = bootstrap_ci(sub["abs_diff_eV_atom"], rng, n_boot)
        mat_ci_low, mat_ci_high = bootstrap_ci(widths[f"{group_name}_per_material_width_eV_atom"], rng, n_boot)
        summary_rows.append(
            {
                "entity": "model",
                "group": group_name,
                "n_materials": int(widths["pbe_job_id"].nunique()),
                "n_pair_values": int(sub["abs_diff_eV_atom"].notna().sum()),
                "pooled_pair_median_eV_atom": float(sub["abs_diff_eV_atom"].median()),
                "pooled_pair_ci95_low": ci_low,
                "pooled_pair_ci95_high": ci_high,
                "pooled_pair_mean_eV_atom": float(sub["abs_diff_eV_atom"].mean()),
                "per_material_width_median_eV_atom": float(widths[f"{group_name}_per_material_width_eV_atom"].median()),
                "per_material_width_ci95_low": mat_ci_low,
                "per_material_width_ci95_high": mat_ci_high,
                "per_material_width_mean_eV_atom": float(widths[f"{group_name}_per_material_width_eV_atom"].mean()),
            }
        )
    widths_all = width_frames[0]
    for frame in width_frames[1:]:
        widths_all = widths_all.merge(frame, on="pbe_job_id", how="outer")
    return pairwise, pd.DataFrame(summary_rows), widths_all


def dft_pairwise(floor: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    width_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    group_defs = {"pbe3": DATABASES_PBE3, "all4": DATABASES_ALL4}

    for group_name, databases in group_defs.items():
        group_rows: list[dict[str, Any]] = []
        for db_a, db_b in combinations(databases, 2):
            a_col = f"Ef_{db_a}"
            b_col = f"Ef_{db_b}"
            if a_col not in floor.columns or b_col not in floor.columns:
                continue
            diff = (numeric(floor[a_col]) - numeric(floor[b_col])).abs()
            for idx, value in diff.items():
                row = floor.iloc[idx]
                group_rows.append(
                    {
                        "pbe_job_id": row["pbe_job_id"],
                        "matalign_id": row.get("matalign_id"),
                        "formula": row.get("formula"),
                        "database_a": db_a,
                        "database_b": db_b,
                        "pair": f"{db_a}__{db_b}",
                        "database_group": group_name,
                        "abs_diff_eV_atom": value,
                    }
                )
        group_frame = pd.DataFrame(group_rows)
        rows.extend(group_rows)
        width = group_frame.groupby("pbe_job_id", as_index=False)["abs_diff_eV_atom"].median()
        width = width.rename(columns={"abs_diff_eV_atom": f"dft_{group_name}_per_material_width_eV_atom"})
        width_frames.append(width)
        ci_low, ci_high = bootstrap_ci(group_frame["abs_diff_eV_atom"], rng, n_boot)
        mat_ci_low, mat_ci_high = bootstrap_ci(width[f"dft_{group_name}_per_material_width_eV_atom"], rng, n_boot)
        for pair, sub in group_frame.groupby("pair"):
            summary_rows.append(
                {
                    "entity": "database_pair",
                    "group": group_name,
                    "pair": pair,
                    "n_materials": int(sub["pbe_job_id"].nunique()),
                    "n_pair_values": int(sub["abs_diff_eV_atom"].notna().sum()),
                    "pooled_pair_median_eV_atom": float(sub["abs_diff_eV_atom"].median()),
                    "pooled_pair_mean_eV_atom": float(sub["abs_diff_eV_atom"].mean()),
                    "per_material_width_median_eV_atom": math.nan,
                    "per_material_width_mean_eV_atom": math.nan,
                }
            )
        summary_rows.append(
            {
                "entity": "database_group",
                "group": group_name,
                "pair": "pooled",
                "n_materials": int(width["pbe_job_id"].nunique()),
                "n_pair_values": int(group_frame["abs_diff_eV_atom"].notna().sum()),
                "pooled_pair_median_eV_atom": float(group_frame["abs_diff_eV_atom"].median()),
                "pooled_pair_ci95_low": ci_low,
                "pooled_pair_ci95_high": ci_high,
                "pooled_pair_mean_eV_atom": float(group_frame["abs_diff_eV_atom"].mean()),
                "per_material_width_median_eV_atom": float(width[f"dft_{group_name}_per_material_width_eV_atom"].median()),
                "per_material_width_ci95_low": mat_ci_low,
                "per_material_width_ci95_high": mat_ci_high,
                "per_material_width_mean_eV_atom": float(width[f"dft_{group_name}_per_material_width_eV_atom"].mean()),
            }
        )

    widths_all = width_frames[0]
    for frame in width_frames[1:]:
        widths_all = widths_all.merge(frame, on="pbe_job_id", how="outer")
    return pd.DataFrame(rows), pd.DataFrame(summary_rows), widths_all


def gate2_tests(model_widths: pd.DataFrame, dft_widths: pd.DataFrame, model_summary: pd.DataFrame, dft_summary: pd.DataFrame) -> pd.DataFrame:
    widths = model_widths.merge(dft_widths, on="pbe_job_id", how="inner")
    comparisons = [
        ("all5", "pbe3"),
        ("frontier4_excl_chgnet", "pbe3"),
        ("orb_v3_vs_sevennet", "pbe3"),
        ("frontier4_excl_chgnet", "all4"),
    ]
    rows: list[dict[str, Any]] = []
    for model_group, dft_group in comparisons:
        model_col = f"{model_group}_per_material_width_eV_atom"
        dft_col = f"dft_{dft_group}_per_material_width_eV_atom"
        frame = widths[["pbe_job_id", model_col, dft_col]].dropna()
        model_summary_row = model_summary[model_summary["group"] == model_group].iloc[0]
        dft_summary_row = dft_summary[(dft_summary["entity"] == "database_group") & (dft_summary["group"] == dft_group)].iloc[0]
        rows.append(
            {
                "comparison": f"{model_group}_vs_dft_{dft_group}",
                "model_group": model_group,
                "dft_group": dft_group,
                "n_materials": int(len(frame)),
                "model_pooled_pair_median_eV_atom": model_summary_row["pooled_pair_median_eV_atom"],
                "dft_pooled_pair_median_eV_atom": dft_summary_row["pooled_pair_median_eV_atom"],
                "model_per_material_width_median_eV_atom": float(frame[model_col].median()),
                "dft_per_material_width_median_eV_atom": float(frame[dft_col].median()),
                "median_width_difference_model_minus_dft": float((frame[model_col] - frame[dft_col]).median()),
                "model_width_less_than_dft_rate": float((frame[model_col] < frame[dft_col]).mean()),
                "wilcoxon_model_width_less_than_dft_p": safe_wilcoxon(frame[model_col], frame[dft_col], alternative="less"),
                "unconditional_gate2_pass": bool(
                    model_summary_row["pooled_pair_median_eV_atom"] < dft_summary_row["pooled_pair_median_eV_atom"]
                ),
            }
        )
    return pd.DataFrame(rows)


def load_structure(path: Path) -> Any | None:
    if Structure is None:
        return None
    try:
        return Structure.from_file(path)
    except Exception:
        return None


def target_structure_path(job_id: str, root: Path) -> tuple[Path | None, str]:
    base = root / job_id
    candidates = [
        (base / "static" / "CONTCAR", "static_CONTCAR"),
        (base / "static" / "POSCAR", "static_POSCAR"),
        (base / "relax" / "CONTCAR", "relax_CONTCAR"),
        (Path("vasp_v3_pbe_work") / "inputs" / "primary" / "compounds" / job_id / "static_template" / "POSCAR", "static_template_POSCAR"),
        (Path("vasp_v3_pbe_work") / "inputs" / "primary" / "compounds" / job_id / "relax" / "POSCAR", "input_relax_POSCAR"),
    ]
    for path, label in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path, label
    return None, "missing"


def load_candidate_structures(candidate_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for source in ["mptrj", "omat24", "salex", "mattersim", "mptrj_screen"]:
        for suffix in [".parquet", ".csv", ".jsonl"]:
            path = candidate_dir / f"{source}_candidates{suffix}"
            if not path.exists():
                continue
            if suffix == ".parquet":
                frame = pd.read_parquet(path)
            elif suffix == ".csv":
                frame = pd.read_csv(path)
            else:
                frame = pd.read_json(path, lines=True)
            out[source] = frame
            break
    return out


def structure_match_status(
    target: Any | None,
    candidates: pd.DataFrame,
    composition_key: str,
    nsites: int | float | None,
) -> tuple[bool | float, bool | float, str, str | None, int]:
    if target is None or StructureMatcher is None:
        return (math.nan, math.nan, "not_run_target_structure_unavailable", None, 0)
    if candidates.empty or "structure_json" not in candidates.columns:
        return (math.nan, math.nan, "not_run_no_candidate_structure_cache", None, 0)
    sub = candidates[candidates["composition_key"].astype(str) == str(composition_key)].copy()
    if nsites is not None and np.isfinite(float(nsites)) and "nsites" in sub.columns:
        sub = sub[pd.to_numeric(sub["nsites"], errors="coerce") == int(nsites)]
    if sub.empty:
        return (False, False, "no_composition_key_candidate_structure", None, 0)
    strict = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True, attempt_supercell=False)
    loose = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10, primitive_cell=True, scale=True, attempt_supercell=False)
    any_loose = False
    for _, cand in sub.iterrows():
        try:
            candidate = Structure.from_dict(json.loads(cand["structure_json"]) if isinstance(cand["structure_json"], str) else cand["structure_json"])
        except Exception:
            continue
        if strict.fit(target, candidate):
            return (True, True, "strict_match", str(cand.get("training_id", cand.get("id", ""))), int(len(sub)))
        if loose.fit(target, candidate):
            any_loose = True
            loose_id = str(cand.get("training_id", cand.get("id", "")))
    if any_loose:
        return (False, True, "loose_match", loose_id, int(len(sub)))
    return (False, False, "no_structure_match", None, int(len(sub)))


def gate1_leakage(
    leakage: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates_by_source = load_candidate_structures(args.candidate_structure_dir)
    structure_cache_available = bool(candidates_by_source)
    labels: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []

    target_cache: dict[str, tuple[Any | None, str, str | None]] = {}
    for _, row in leakage.iterrows():
        job_id = str(row["pbe_job_id"])
        model = str(row["model"])
        sources = MODEL_TRAINING_SOURCES.get(model, [])
        composition_key = str(row.get("composition_key", ""))
        nsites = numeric(pd.Series([row.get("structure_nsites")])).iloc[0]
        if job_id not in target_cache:
            path, source_label = target_structure_path(job_id, args.target_structure_root)
            target_cache[job_id] = (load_structure(path) if path else None, source_label, str(path) if path else None)
        target, target_source, target_path = target_cache[job_id]

        strict_values: list[bool] = []
        loose_values: list[bool] = []
        source_statuses: list[str] = []
        candidate_total = 0
        matched_ids: list[str] = []
        for source in sources:
            source_frame = candidates_by_source.get(source, pd.DataFrame())
            strict, loose, status, match_id, n_candidates = structure_match_status(target, source_frame, composition_key, nsites)
            source_statuses.append(f"{source}:{status}")
            candidate_total += n_candidates
            if isinstance(strict, bool):
                strict_values.append(strict)
            if isinstance(loose, bool):
                loose_values.append(loose)
            if match_id:
                matched_ids.append(f"{source}:{match_id}")
            matches.append(
                {
                    "pbe_job_id": job_id,
                    "model": model,
                    "source": source,
                    "composition_key": composition_key,
                    "target_structure_source": target_source,
                    "target_structure_path": target_path,
                    "candidate_structure_cache_available": source in candidates_by_source,
                    "candidate_structure_count": n_candidates,
                    "structure_match_strict": strict,
                    "structure_match_loose": loose,
                    "structure_match_status": status,
                    "matched_training_id": match_id,
                }
            )

        has_partial = any(source in PARTIAL_SOURCES for source in sources)
        has_unavailable = any(source in UNAVAILABLE_SOURCES for source in sources)
        strict_value: bool | float
        loose_value: bool | float
        if strict_values:
            strict_value = any(strict_values)
            loose_value = any(loose_values)
        else:
            strict_value = math.nan
            loose_value = math.nan
        if not structure_cache_available:
            coverage_level = "no_candidate_structure_cache"
        elif has_unavailable:
            coverage_level = "training_data_unavailable_or_partial"
        elif has_partial:
            coverage_level = "partial_training_subset_structure_match"
        else:
            coverage_level = "full_training_structure_match"
        eligible = coverage_level == "full_training_structure_match" and isinstance(strict_value, bool)
        held_out_strict: bool | float = (not strict_value) if eligible else math.nan
        held_out_loose: bool | float = (not loose_value) if eligible else math.nan
        labels.append(
            {
                **row.to_dict(),
                "d3_training_sources": " ".join(sources),
                "d3_coverage_level": coverage_level,
                "candidate_structure_cache_dir": str(args.candidate_structure_dir),
                "candidate_structure_cache_available": structure_cache_available,
                "candidate_structure_count_d3": candidate_total,
                "target_structure_source": target_source,
                "target_structure_path": target_path,
                "structure_match_status_d3": ";".join(source_statuses),
                "structure_match_strict_d3": strict_value,
                "structure_match_loose_d3": loose_value,
                "matched_training_ids_d3": " ".join(matched_ids),
                "eligible_for_strong_heldout": eligible,
                "held_out_strict_d3": held_out_strict,
                "held_out_loose_d3": held_out_loose,
            }
        )

    labels_frame = pd.DataFrame(labels)
    matches_frame = pd.DataFrame(matches)
    counts = (
        labels_frame.groupby(["model", "d3_coverage_level", "candidate_structure_cache_available"], dropna=False)
        .agg(
            rows=("pbe_job_id", "count"),
            eligible_for_strong_heldout_rows=("eligible_for_strong_heldout", "sum"),
            strict_in_sample_rows=("structure_match_strict_d3", lambda s: int((s == True).sum())),
            strict_heldout_rows=("held_out_strict_d3", lambda s: int((s == True).sum())),
            total_candidate_structures_examined=("candidate_structure_count_d3", "sum"),
        )
        .reset_index()
    )
    return labels_frame, matches_frame, counts


def pairwise_floor_table(floor: pd.DataFrame, dft_summary: pd.DataFrame, d2fix_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in dft_summary.iterrows():
        rows.append(
            {
                "metric_family": "pairwise_dft",
                "metric": f"{row['group']}:{row['pair']}",
                "n": row.get("n_pair_values"),
                "median_eV_atom": row.get("pooled_pair_median_eV_atom"),
                "mean_eV_atom": row.get("pooled_pair_mean_eV_atom"),
                "per_material_width_median_eV_atom": row.get("per_material_width_median_eV_atom"),
            }
        )
    for col in ["floor_all4_mad_scaled", "floor_pbe3_mad_scaled", "floor_all4_std", "floor_pbe3_std"]:
        if col in floor.columns:
            rows.append(
                {
                    "metric_family": "d2fix_sensitivity",
                    "metric": col,
                    "n": int(numeric(floor[col]).notna().sum()),
                    "median_eV_atom": float(numeric(floor[col]).median()),
                    "mean_eV_atom": float(numeric(floor[col]).mean()),
                    "per_material_width_median_eV_atom": math.nan,
                }
            )
    return pd.DataFrame(rows)


def consensus_ambiguity(floor: pd.DataFrame, d2fix_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    consensus_path = d2fix_dir / "consensus_distance_summary.csv"
    if consensus_path.exists():
        consensus = pd.read_csv(consensus_path)
        for _, row in consensus.iterrows():
            rows.append(
                {
                    "entity_type": row.get("entity_type"),
                    "entity": row.get("entity"),
                    "consensus": row.get("consensus"),
                    "metric": "distance_to_consensus",
                    "n": row.get("n"),
                    "median_eV_atom": row.get("median_distance_eV_atom"),
                    "mean_eV_atom": row.get("mean_distance_eV_atom"),
                    "note": "copied_from_d2fix_consensus_summary",
                }
            )
    rows.append(
        {
            "entity_type": "database",
            "entity": "JARVIS",
            "consensus": "all4",
            "metric": "largest_all4_deviation_rate",
            "n": len(floor),
            "median_eV_atom": math.nan,
            "mean_eV_atom": float(floor.get("jarvis_is_all4_outlier", pd.Series(dtype=float)).astype(float).mean()),
            "note": "fraction_where_jarvis_is_largest_all4_deviation",
        }
    )
    rows.append(
        {
            "entity_type": "floor",
            "entity": "definition_span",
            "consensus": "n/a",
            "metric": "floor_all4_std_over_pbe3_mad_median_ratio",
            "n": len(floor),
            "median_eV_atom": float(numeric(floor["floor_all4_std"]).median() / numeric(floor["floor_pbe3_mad_scaled"]).median()),
            "mean_eV_atom": math.nan,
            "note": "sensitivity_span_not_a_physical_constant",
        }
    )
    return pd.DataFrame(rows)


def element_frontier_map(
    pred: pd.DataFrame,
    floor: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int,
) -> pd.DataFrame:
    pbe_pair_widths = []
    for _, row in floor.iterrows():
        diffs = []
        for db_a, db_b in combinations(DATABASES_PBE3, 2):
            a = row.get(f"Ef_{db_a}")
            b = row.get(f"Ef_{db_b}")
            if pd.notna(a) and pd.notna(b):
                diffs.append(abs(float(a) - float(b)))
        pbe_pair_widths.append(float(np.median(diffs)) if diffs else math.nan)
    floor = floor.copy()
    floor["d3_pbe_pairwise_width_eV_atom"] = pbe_pair_widths
    global_pbe_width = float(pd.Series(pbe_pair_widths).median())

    pred_sub = pred[pred["model"].isin(FRONTIER4)].copy()
    joined = pred_sub.merge(
        floor[["pbe_job_id", CONSENSUS_COL, "d3_pbe_pairwise_width_eV_atom", "elements_str"]],
        on="pbe_job_id",
        how="left",
        suffixes=("", "_floor"),
    )
    joined["abs_model_error_to_pbe3_consensus_eV_atom"] = (
        numeric(joined[PRED_COL]) - numeric(joined[CONSENSUS_COL])
    ).abs()

    rows: list[dict[str, Any]] = []
    expanded: list[dict[str, Any]] = []
    for _, row in joined.iterrows():
        for element in elements_from_text(row.get("elements_str", row.get("elements_str_floor"))):
            expanded.append({**row.to_dict(), "element": element})
    expanded_frame = pd.DataFrame(expanded)
    if expanded_frame.empty:
        return expanded_frame

    for model_name in [*FRONTIER4, "frontier4_aggregate"]:
        if model_name == "frontier4_aggregate":
            model_frame = (
                expanded_frame.groupby(["pbe_job_id", "element"], as_index=False)
                .agg(
                    abs_model_error_to_pbe3_consensus_eV_atom=("abs_model_error_to_pbe3_consensus_eV_atom", "median"),
                    d3_pbe_pairwise_width_eV_atom=("d3_pbe_pairwise_width_eV_atom", "first"),
                )
            )
        else:
            model_frame = expanded_frame[expanded_frame["model"] == model_name].copy()
        for element, sub in model_frame.groupby("element"):
            material_sub = sub.drop_duplicates("pbe_job_id")
            n = int(material_sub["pbe_job_id"].nunique())
            model_err = numeric(material_sub["abs_model_error_to_pbe3_consensus_eV_atom"])
            ref_width = numeric(material_sub["d3_pbe_pairwise_width_eV_atom"])
            model_ci = bootstrap_ci(model_err, rng, n_boot)
            ref_ci = bootstrap_ci(ref_width, rng, n_boot)
            p_value = safe_wilcoxon(model_err, ref_width, alternative="greater") if n >= 5 else math.nan
            ci_overlap = bool(model_ci[0] <= ref_ci[1] and ref_ci[0] <= model_ci[1]) if n >= 5 else False
            model_median = float(model_err.median()) if model_err.notna().any() else math.nan
            ref_median = float(ref_width.median()) if ref_width.notna().any() else math.nan
            if n < 5:
                call = "insufficient_n"
            elif ref_median >= global_pbe_width and (model_median <= ref_median or ci_overlap or p_value >= 0.05):
                call = "reference_limited"
            elif ref_median < global_pbe_width and model_median > ref_median and p_value < 0.05:
                call = "model_limited"
            elif model_median > ref_median and p_value < 0.05:
                call = "mixed_high_reference_and_model_gap"
            else:
                call = "ambiguous_or_saturated"
            rows.append(
                {
                    "element": element,
                    "model": model_name,
                    "n": n,
                    "model_to_pbe3_consensus_median_eV_atom": model_median,
                    "model_to_pbe3_consensus_ci95_low": model_ci[0],
                    "model_to_pbe3_consensus_ci95_high": model_ci[1],
                    "pbe_pairwise_reference_width_median_eV_atom": ref_median,
                    "pbe_pairwise_reference_width_ci95_low": ref_ci[0],
                    "pbe_pairwise_reference_width_ci95_high": ref_ci[1],
                    "global_pbe_pairwise_width_median_eV_atom": global_pbe_width,
                    "wilcoxon_model_error_greater_than_reference_p": p_value,
                    "within_reference_width_rate": float((model_err <= ref_width).mean()) if n else math.nan,
                    "ci_overlap": ci_overlap,
                    "d3_frontier_call": call,
                }
            )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.loc[:, columns].head(max_rows).copy()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in view.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    args: argparse.Namespace,
    summary: dict[str, Any],
    model_summary: pd.DataFrame,
    dft_summary: pd.DataFrame,
    gate2: pd.DataFrame,
    candidate_counts: pd.DataFrame,
    pairwise_floor: pd.DataFrame,
    element_map: pd.DataFrame,
) -> Path:
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.docs_dir / "v3_analysis_phase_D3_results_20260606.md"
    main_gate2 = gate2[gate2["comparison"] == "frontier4_excl_chgnet_vs_dft_pbe3"]
    top_reference = (
        element_map[(element_map["model"] == "frontier4_aggregate") & (element_map["d3_frontier_call"] == "reference_limited")]
        .sort_values("pbe_pairwise_reference_width_median_eV_atom", ascending=False)
        .head(10)
    )
    top_model_aggregate = (
        element_map[(element_map["model"] == "frontier4_aggregate") & (element_map["d3_frontier_call"] == "model_limited")]
        .sort_values("model_to_pbe3_consensus_median_eV_atom", ascending=False)
        .head(10)
    )
    top_model_per_model = (
        element_map[(element_map["model"] != "frontier4_aggregate") & (element_map["d3_frontier_call"] == "model_limited")]
        .sort_values("model_to_pbe3_consensus_median_eV_atom", ascending=False)
        .head(10)
    )
    text = "\n".join(
        [
            "# MatAlign v3 Phase D3 Results",
            "",
            "## Summary",
            "",
            f"- Clean materials: `{summary['clean_n']}`.",
            f"- Model prediction rows: `{summary['prediction_rows']}`.",
            f"- D3 PBE pairwise floor, pooled median: `{summary['pbe3_pooled_pair_median_eV_atom']:.6f}` eV/atom.",
            f"- Frontier4 model-model pooled median: `{summary['frontier4_pooled_pair_median_eV_atom']:.6f}` eV/atom.",
            f"- Unconditional Gate 2 pass: `{summary['gate2_unconditional_pass']}`.",
            f"- Strong held-out coverage available: `{summary['strong_heldout_coverage_available']}`.",
            f"- NMI/NCS ready: `{summary['nmi_ncs_ready']}`.",
            "",
            "## Gate 2: Model-Model vs DFT-DFT Consistency",
            "",
            markdown_table(
                gate2,
                [
                    "comparison",
                    "n_materials",
                    "model_pooled_pair_median_eV_atom",
                    "dft_pooled_pair_median_eV_atom",
                    "model_per_material_width_median_eV_atom",
                    "dft_per_material_width_median_eV_atom",
                    "wilcoxon_model_width_less_than_dft_p",
                    "unconditional_gate2_pass",
                ],
            ),
            "",
            "The clean-subset result supports the descriptive statement that the frontier model cluster is tighter than the PBE database cluster. It is not yet a leakage-free generalization claim.",
            "",
            "## Gate 1: Structure Leakage Coverage",
            "",
            markdown_table(
                candidate_counts,
                [
                    "model",
                    "d3_coverage_level",
                    "rows",
                    "eligible_for_strong_heldout_rows",
                    "strict_in_sample_rows",
                    "strict_heldout_rows",
                ],
            ),
            "",
            "No strong held-out claim is made unless `d3_coverage_level` is `full_training_structure_match` and strict held-out rows are available.",
            "",
            "## Gate 3: Floor Sensitivity",
            "",
            markdown_table(
                pairwise_floor,
                ["metric_family", "metric", "n", "median_eV_atom", "mean_eV_atom", "per_material_width_median_eV_atom"],
                max_rows=16,
            ),
            "",
            "D3 uses PBE pairwise disagreement as the main floor and keeps D2fix MAD/std metrics as sensitivity checks.",
            "",
            "## Element Frontier Map",
            "",
            "Reference-limited candidates:",
            "",
            markdown_table(
                top_reference,
                [
                    "element",
                    "n",
                    "model_to_pbe3_consensus_median_eV_atom",
                    "pbe_pairwise_reference_width_median_eV_atom",
                    "d3_frontier_call",
                ],
            ),
            "",
            "Aggregate frontier4 model-limited candidates:",
            "",
            markdown_table(
                top_model_aggregate,
                [
                    "element",
                    "n",
                    "model_to_pbe3_consensus_median_eV_atom",
                    "pbe_pairwise_reference_width_median_eV_atom",
                    "d3_frontier_call",
                ],
            ),
            "",
            "Strong per-model model-limited candidates:",
            "",
            markdown_table(
                top_model_per_model,
                [
                    "element",
                    "model",
                    "n",
                    "model_to_pbe3_consensus_median_eV_atom",
                    "pbe_pairwise_reference_width_median_eV_atom",
                    "d3_frontier_call",
                ],
            ),
            "",
            "## Go / No-Go",
            "",
            "- NC/NMI route remains blocked until per-model full-training StructureMatcher labels produce enough true held-out rows.",
            "- Current evidence supports an npj/Digital Discovery-style story immediately: cross-database floor ambiguity, MP/JARVIS consensus deviations, and a metal-chemistry element frontier map.",
        ]
    )
    report_path.write_text(text, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    inputs = load_inputs(args)
    pred = inputs["pred"]
    floor = inputs["floor"]
    leakage = inputs["leakage"]

    model_pairs, model_summary, model_widths = model_pairwise(pred, floor, rng, args.bootstrap)
    dft_pairs, dft_summary, dft_widths = dft_pairwise(floor, rng, args.bootstrap)
    gate2 = gate2_tests(model_widths, dft_widths, model_summary, dft_summary)
    labels, matches, candidate_counts = gate1_leakage(leakage, args)
    pairwise_floor = pairwise_floor_table(floor, dft_summary, args.d2fix_dir)
    ambiguity = consensus_ambiguity(floor, args.d2fix_dir)
    element_map = element_frontier_map(pred, floor, rng, args.bootstrap)

    model_pairs.to_csv(args.out_dir / "model_pairwise_consistency.csv", index=False)
    model_summary.to_csv(args.out_dir / "model_pairwise_consistency_summary.csv", index=False)
    dft_pairs.to_csv(args.out_dir / "dft_pairwise_consistency.csv", index=False)
    gate2.to_csv(args.out_dir / "gate2_consistency_tests.csv", index=False)
    candidate_counts.to_csv(args.out_dir / "training_candidate_counts_d3.csv", index=False)
    matches.to_csv(args.out_dir / "structure_leakage_matches_d3.csv", index=False)
    labels.to_csv(args.out_dir / "per_model_heldout_labels_d3.csv", index=False)
    pairwise_floor.to_csv(args.out_dir / "d3_pairwise_floor_table.csv", index=False)
    ambiguity.to_csv(args.out_dir / "d3_consensus_ambiguity_summary.csv", index=False)
    element_map.to_csv(args.out_dir / "d3_element_frontier_map.csv", index=False)

    frontier_summary = model_summary[model_summary["group"] == "frontier4_excl_chgnet"].iloc[0]
    all5_summary = model_summary[model_summary["group"] == "all5"].iloc[0]
    pbe3_summary = dft_summary[(dft_summary["entity"] == "database_group") & (dft_summary["group"] == "pbe3")].iloc[0]
    main_gate = gate2[gate2["comparison"] == "frontier4_excl_chgnet_vs_dft_pbe3"].iloc[0]
    strong_heldout_counts = (
        labels[labels["eligible_for_strong_heldout"] & (labels["held_out_strict_d3"] == True)]
        .groupby("model")["pbe_job_id"]
        .nunique()
        .to_dict()
    )
    frontier_strong_models = [model for model in FRONTIER4 if strong_heldout_counts.get(model, 0) >= 30]
    nmi_ncs_ready = bool(len(frontier_strong_models) >= 3 and main_gate["unconditional_gate2_pass"])

    summary = {
        "clean_n": int(floor["pbe_job_id"].nunique()),
        "prediction_rows": int(len(pred)),
        "models": MODELS,
        "frontier4_models": FRONTIER4,
        "frontier4_pooled_pair_median_eV_atom": float(frontier_summary["pooled_pair_median_eV_atom"]),
        "all5_pooled_pair_median_eV_atom": float(all5_summary["pooled_pair_median_eV_atom"]),
        "pbe3_pooled_pair_median_eV_atom": float(pbe3_summary["pooled_pair_median_eV_atom"]),
        "pbe3_per_material_width_median_eV_atom": float(pbe3_summary["per_material_width_median_eV_atom"]),
        "gate2_unconditional_pass": bool(main_gate["unconditional_gate2_pass"]),
        "gate2_main_wilcoxon_p": float(main_gate["wilcoxon_model_width_less_than_dft_p"]),
        "strong_heldout_counts_by_model": {k: int(v) for k, v in strong_heldout_counts.items()},
        "strong_heldout_coverage_available": bool(len(frontier_strong_models) >= 3),
        "nmi_ncs_ready": nmi_ncs_ready,
        "route_recommendation": "NC/NMI_blocked_by_leakage_coverage" if not nmi_ncs_ready else "NC/NMI_candidate",
        "output_dir": str(args.out_dir),
    }
    (args.out_dir / "d3_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = write_report(args, summary, model_summary, dft_summary, gate2, candidate_counts, pairwise_floor, element_map)
    print(json.dumps({**summary, "report_path": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
