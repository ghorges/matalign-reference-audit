from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
from scipy.stats import wilcoxon


DATA_ROOT = (Path(__file__).resolve().parents[2] / "data").resolve()
D1_DIR = DATA_ROOT / "processed" / "clean_reference_analysis"
D2_DIR = DATA_ROOT / "processed" / "database_relative_model_checks"
OUT_DIR = DATA_ROOT / "processed" / "dual_noise_floor_checks"
TRAINING_INDEX_DIR = DATA_ROOT / "cache" / "training_indices"
MP_DOWNLOADS = Path("vasp_uniform_pbe_work") / "mp_downloads" / "all" / "mp_summary_all.jsonl"
DOCS_DIR = Path("docs")
MAD_SCALE = 1.4826
DATABASES = ["MP", "OQMD", "AFLOW", "JARVIS"]
PBE_DATABASES = ["MP", "OQMD", "AFLOW"]
MODELS = ["chgnet", "mace", "mattersim_5m", "orb_v3", "sevennet_mf_ompa"]
MODEL_TRAINING_SOURCES = {
    "chgnet": ["mptrj"],
    "mace": ["mptrj"],
    "mattersim_5m": ["mattersim", "mptrj_screen"],
    "orb_v3": ["omat24", "mptrj"],
    "sevennet_mf_ompa": ["omat24", "mptrj", "salex"],
}
FULL_COVERAGE_SOURCES = {"mptrj"}
MP_NUMERIC_RE = re.compile(r"^mp-\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MatAlign v3 D2-fix analysis.")
    parser.add_argument("--analysis-dir", type=Path, default=D1_DIR)
    parser.add_argument("--d2-dir", type=Path, default=D2_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--training-index-dir", type=Path, default=TRAINING_INDEX_DIR)
    parser.add_argument("--mp-summary-jsonl", type=Path, default=MP_DOWNLOADS)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def numeric(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return frame.apply(pd.to_numeric, errors="coerce") if isinstance(frame, pd.DataFrame) else pd.to_numeric(frame, errors="coerce")


def composition_key_from_formula(formula: str) -> str:
    comp = Composition(str(formula))
    counts = {Element(symbol).Z: int(round(amount)) for symbol, amount in comp.as_dict().items()}
    return composition_key_from_counts(counts)


def composition_key_from_counts(counts: dict[int | str, int]) -> str:
    clean = {int(z): int(v) for z, v in counts.items() if int(v) != 0}
    gcd = 0
    for value in clean.values():
        gcd = math.gcd(gcd, abs(value))
    if gcd > 1:
        clean = {z: value // gcd for z, value in clean.items()}
    return ";".join(f"{z}:{clean[z]}" for z in sorted(clean))


def parse_elements(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value)
    if "|" in text:
        return [part for part in text.split("|") if part]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    try:
        return sorted(str(el) for el in Composition(text).elements)
    except Exception:
        return []


def bootstrap_median_ci(values: pd.Series | np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if len(arr) == 0:
        return math.nan, math.nan, math.nan
    if len(arr) == 1:
        val = float(arr[0])
        return val, val, val
    med = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        med[idx] = np.median(rng.choice(arr, size=len(arr), replace=True))
    return float(np.median(arr)), float(np.quantile(med, 0.025)), float(np.quantile(med, 0.975))


def wilcoxon_greater(left: pd.Series, right: pd.Series) -> float | None:
    pairs = pd.DataFrame({"left": numeric(left), "right": numeric(right)}).dropna()
    if len(pairs) < 5:
        return None
    diff = pairs["left"] - pairs["right"]
    if np.allclose(diff, 0):
        return 1.0
    try:
        return float(wilcoxon(diff, alternative="greater", zero_method="wilcox").pvalue)
    except ValueError:
        return None


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


def ci_overlap(a_low: float, a_high: float, b_low: float, b_high: float) -> bool:
    if any(math.isnan(v) for v in [a_low, a_high, b_low, b_high]):
        return False
    return max(a_low, b_low) <= min(a_high, b_high)


def saturation_call(row: dict[str, Any]) -> str:
    if int(row["n"]) < 5:
        return "insufficient_n"
    pvalue = row.get("wilcoxon_error_greater_than_floor_p")
    err = row["model_abs_error_median_eV_atom"]
    floor = row["floor_median_eV_atom"]
    if err <= floor and (pvalue is None or pvalue >= 0.05):
        return "below_floor_or_saturated"
    if row.get("median_ci_overlap") and (pvalue is None or pvalue >= 0.05):
        return "saturated_indistinguishable_from_floor"
    if pvalue is not None and pvalue < 0.05 and err > floor:
        return "above_floor_headroom"
    return "inconclusive"


def load_mp_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["id_MP", "mp_task_ids", "mp_api_material_id_status"])
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            material_id = str(payload.get("material_id", ""))
            rows.append(
                {
                    "id_MP": material_id,
                    "mp_task_ids": "|".join(str(v) for v in (payload.get("task_ids") or [])),
                    "mp_api_material_id_status": "accepted_by_mp_api",
                    "mp_api_formula": payload.get("formula_pretty"),
                    "mp_api_spacegroup_number": (payload.get("symmetry") or {}).get("number"),
                }
            )
    return pd.DataFrame(rows).drop_duplicates("id_MP")


def load_training_index(index_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for source in ["mptrj", "omat24", "salex", "mattersim", "mptrj_screen"]:
        payload: dict[str, Any] = {"ids": set(), "keys": {}, "coverage": "unavailable", "path": None}
        id_path = index_dir / f"{source}_ids.csv"
        key_path = index_dir / f"{source}_composition_keys.csv"
        if id_path.exists():
            ids = pd.read_csv(id_path)
            for col in ["material_id", "mp_id", "task_id", "source_id", "id"]:
                if col in ids.columns:
                    payload["ids"].update(ids[col].dropna().astype(str).tolist())
            payload["coverage"] = "id_index"
            payload["path"] = str(id_path)
        if key_path.exists():
            keys = pd.read_csv(key_path)
            key_col = "composition_key"
            count_col = "count"
            if key_col in keys.columns:
                payload["keys"] = {
                    str(row[key_col]): int(row[count_col]) if count_col in keys.columns and pd.notna(row[count_col]) else 1
                    for _, row in keys.iterrows()
                }
                payload["coverage"] = "full_key_index" if source in FULL_COVERAGE_SOURCES else "subset_key_index"
                payload["path"] = str(key_path)
        out[source] = payload
    # Backward-compatible local MPtrj material-id cache.
    old_mptrj = DATA_ROOT / "cache" / "mptrj_index" / "mptrj_material_ids.csv"
    if old_mptrj.exists() and not out["mptrj"]["ids"]:
        frame = pd.read_csv(old_mptrj)
        out["mptrj"]["ids"].update(frame.iloc[:, 0].dropna().astype(str).tolist())
        out["mptrj"]["coverage"] = "id_index_old_numeric"
        out["mptrj"]["path"] = str(old_mptrj)
    if out["mptrj"]["coverage"] != "unavailable" and out["mptrj_screen"]["coverage"] == "unavailable":
        out["mptrj_screen"] = {
            "ids": set(out["mptrj"]["ids"]),
            "keys": dict(out["mptrj"]["keys"]),
            "coverage": f"screen_alias:{out['mptrj']['coverage']}",
            "path": out["mptrj"]["path"],
        }
    return out


def compute_dual_floors(clean: pd.DataFrame) -> pd.DataFrame:
    floor = clean[
        [
            "pbe_job_id",
            "matalign_id",
            "id_MP",
            "formula",
            "reduced_formula",
            "elements_str",
            "noise_bin",
            "validation_role",
            "chemistry_class",
            "spacegroup_number",
            "structure_nsites",
            *[f"Ef_{db}" for db in DATABASES],
        ]
    ].copy()
    all4_cols = [f"Ef_{db}" for db in DATABASES]
    pbe3_cols = [f"Ef_{db}" for db in PBE_DATABASES]
    all4 = numeric(floor[all4_cols])
    pbe3 = numeric(floor[pbe3_cols])
    floor["consensus_all4_median"] = all4.median(axis=1, skipna=True)
    floor["consensus_pbe3_median"] = pbe3.median(axis=1, skipna=True)
    all4_abs = all4.sub(floor["consensus_all4_median"], axis=0).abs()
    pbe3_abs = pbe3.sub(floor["consensus_pbe3_median"], axis=0).abs()
    floor["floor_all4_mad_scaled"] = all4_abs.median(axis=1, skipna=True) * MAD_SCALE
    floor["floor_pbe3_mad_scaled"] = pbe3_abs.median(axis=1, skipna=True) * MAD_SCALE
    floor["floor_all4_std"] = all4.std(axis=1, ddof=0, skipna=True)
    floor["floor_pbe3_std"] = pbe3.std(axis=1, ddof=0, skipna=True)
    for db in DATABASES:
        floor[f"{db}_abs_dev_from_all4_consensus"] = (numeric(floor[f"Ef_{db}"]) - floor["consensus_all4_median"]).abs()
    floor["all4_outlier_database"] = floor[[f"{db}_abs_dev_from_all4_consensus" for db in DATABASES]].idxmax(axis=1).str.replace("_abs_dev_from_all4_consensus", "", regex=False)
    floor["jarvis_is_all4_outlier"] = floor["all4_outlier_database"].eq("JARVIS")
    floor["composition_key"] = floor["reduced_formula"].map(composition_key_from_formula)
    return floor


def build_leakage_labels(floor: pd.DataFrame, mp_summary: pd.DataFrame, training_index: dict[str, dict[str, Any]]) -> pd.DataFrame:
    materials = floor.merge(mp_summary, on="id_MP", how="left")
    materials["mp_api_material_id_status"] = materials["mp_api_material_id_status"].fillna("not_checked_or_missing")
    rows: list[dict[str, Any]] = []
    for rec in materials.to_dict("records"):
        current_ids = {str(rec.get("id_MP", ""))}
        current_ids.update(x for x in str(rec.get("mp_task_ids", "")).split("|") if x)
        id_namespace = "numeric_mp_id" if any(MP_NUMERIC_RE.match(value) for value in current_ids) else "mp_api_new_id_space"
        for model in MODELS:
            sources = MODEL_TRAINING_SOURCES[model]
            exact_sources: list[str] = []
            key_sources: list[str] = []
            key_count = 0
            source_coverages: list[str] = []
            for source in sources:
                info = training_index.get(source, {"ids": set(), "keys": {}, "coverage": "unavailable"})
                source_coverages.append(f"{source}:{info.get('coverage', 'unavailable')}")
                if current_ids & set(info.get("ids", set())):
                    exact_sources.append(source)
                source_count = int(info.get("keys", {}).get(rec["composition_key"], 0))
                if source_count:
                    key_sources.append(source)
                    key_count += source_count
            any_full_key_source = any(
                training_index.get(source, {}).get("coverage") == "full_key_index" for source in sources
            )
            exact_match = bool(exact_sources)
            composition_upper_bound = bool(key_sources)
            if exact_match:
                in_sample_strict: bool | float = True
                in_sample_loose: bool | float = True
                held_out_strict: bool | float = False
                held_out_loose: bool | float = False
                match_status = "exact_id_or_task_match"
                coverage_level = "exact_id_verified"
            elif composition_upper_bound:
                in_sample_strict = math.nan
                in_sample_loose = True
                held_out_strict = math.nan
                held_out_loose = False
                match_status = "composition_key_candidate_upper_bound"
                coverage_level = "upper_bound_only_no_structure_match"
            elif any_full_key_source:
                in_sample_strict = False
                in_sample_loose = False
                held_out_strict = True
                held_out_loose = True
                match_status = "no_exact_or_composition_candidate_in_full_index"
                coverage_level = "held_out_by_full_composition_key"
            else:
                in_sample_strict = math.nan
                in_sample_loose = math.nan
                held_out_strict = math.nan
                held_out_loose = math.nan
                match_status = "training_structure_index_unavailable"
                coverage_level = "blocked_by_training_data"
            rows.append(
                {
                    "pbe_job_id": rec["pbe_job_id"],
                    "matalign_id": rec["matalign_id"],
                    "model": model,
                    "id_MP": rec["id_MP"],
                    "id_namespace_status": id_namespace,
                    "mp_api_material_id_status": rec["mp_api_material_id_status"],
                    "training_sources": "|".join(sources),
                    "source_coverages": "|".join(source_coverages),
                    "exact_match_sources": "|".join(exact_sources),
                    "composition_candidate_sources": "|".join(key_sources),
                    "composition_candidate_count": key_count,
                    "in_sample_strict": in_sample_strict,
                    "in_sample_loose": in_sample_loose,
                    "held_out_strict": held_out_strict,
                    "held_out_loose": held_out_loose,
                    "underpowered": bool(pd.isna(held_out_strict)),
                    "coverage_level": coverage_level,
                    "match_status": match_status,
                    "structure_match_status": "not_run_no_candidate_structure_cache",
                    "structure_match_strict": math.nan,
                    "structure_match_loose": math.nan,
                    "reduced_formula": rec["reduced_formula"],
                    "composition_key": rec["composition_key"],
                    "spacegroup_number": rec["spacegroup_number"],
                    "structure_nsites": rec["structure_nsites"],
                    "noise_bin": rec["noise_bin"],
                    "validation_role": rec["validation_role"],
                    "chemistry_class": rec["chemistry_class"],
                }
            )
    return pd.DataFrame(rows)


def summarize_saturation(frame: pd.DataFrame, model: str, subset: str, floor_col: str, rng: np.random.Generator, n_boot: int) -> dict[str, Any]:
    err = numeric(frame["abs_model_error_vs_Ef_MP_eV_atom"])
    floor = numeric(frame[floor_col])
    err_med, err_low, err_high = bootstrap_median_ci(err, rng, n_boot)
    floor_med, floor_low, floor_high = bootstrap_median_ci(floor, rng, n_boot)
    row = {
        "model": model,
        "subset": subset,
        "floor_metric": floor_col,
        "n": int(len(pd.DataFrame({"err": err, "floor": floor}).dropna())),
        "model_abs_error_median_eV_atom": err_med,
        "model_abs_error_ci95_low": err_low,
        "model_abs_error_ci95_high": err_high,
        "floor_median_eV_atom": floor_med,
        "floor_ci95_low": floor_low,
        "floor_ci95_high": floor_high,
        "median_ci_overlap": ci_overlap(err_low, err_high, floor_low, floor_high),
        "wilcoxon_error_greater_than_floor_p": wilcoxon_greater(err, floor),
        "within_floor_rate": float((err <= floor).mean()) if len(frame) else math.nan,
    }
    row["saturation_call"] = saturation_call(row)
    return row


def compute_saturation(preds: pd.DataFrame, floor: pd.DataFrame, leakage: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    floor_cols = [
        "floor_all4_mad_scaled",
        "floor_pbe3_mad_scaled",
        "floor_all4_std",
        "floor_pbe3_std",
        "consensus_all4_median",
        "consensus_pbe3_median",
    ]
    merged = preds.merge(floor[["pbe_job_id", *floor_cols]], on="pbe_job_id", how="left", suffixes=("", "_d2fix"))
    merged = merged.drop(columns=[col for col in ["held_out_strict", "held_out_loose", "coverage_level", "match_status"] if col in merged.columns])
    merged = merged.merge(
        leakage[["pbe_job_id", "model", "held_out_strict", "held_out_loose", "coverage_level", "match_status"]],
        on=["pbe_job_id", "model"],
        how="left",
    )
    subsets = {
        "clean_all": lambda g: g["clean_intermetallic_universe"].fillna(True).astype(bool),
        "clean_low_saturation": lambda g: (g["noise_bin"] == "low") & (g["validation_role"] == "saturation_probe") & g["clean_intermetallic_universe"].fillna(True).astype(bool),
        "held_out_strict_clean_all": lambda g: g["held_out_strict"].eq(True),
        "held_out_strict_clean_low_saturation": lambda g: g["held_out_strict"].eq(True) & (g["noise_bin"] == "low") & (g["validation_role"] == "saturation_probe"),
        "held_out_loose_clean_all": lambda g: g["held_out_loose"].eq(True),
        "held_out_loose_clean_low_saturation": lambda g: g["held_out_loose"].eq(True) & (g["noise_bin"] == "low") & (g["validation_role"] == "saturation_probe"),
    }
    rows: list[dict[str, Any]] = []
    for model, group in merged.groupby("model"):
        for subset, mask_func in subsets.items():
            sub = group[mask_func(group)].copy()
            for floor_col in ["floor_all4_mad_scaled", "floor_pbe3_mad_scaled", "floor_all4_std", "floor_pbe3_std"]:
                rows.append(summarize_saturation(sub, str(model), subset, floor_col, rng, n_boot))
    return pd.DataFrame(rows), merged


def compute_consensus_distance(preds: pd.DataFrame, floor: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    details: list[pd.DataFrame] = []
    material_floor = floor[["pbe_job_id", "consensus_all4_median", "consensus_pbe3_median", *[f"Ef_{db}" for db in DATABASES]]]
    pred = preds.merge(material_floor, on="pbe_job_id", how="left", suffixes=("", "_d2fix"))
    for consensus_name, consensus_col in [("all4", "consensus_all4_median"), ("pbe3", "consensus_pbe3_median")]:
        for model, group in pred.groupby("model"):
            frame = group[["pbe_job_id", "model", consensus_col]].copy()
            frame["entity_type"] = "model"
            frame["entity"] = str(model)
            frame["consensus"] = consensus_name
            frame["distance_eV_atom"] = (numeric(group["model_mp_reference_formation_energy_per_atom_eV"]) - numeric(group[consensus_col])).abs()
            details.append(frame[["pbe_job_id", "entity_type", "entity", "consensus", "distance_eV_atom"]])
        for db in DATABASES:
            if consensus_name == "pbe3" and db == "JARVIS":
                continue
            frame = floor[["pbe_job_id", consensus_col]].copy()
            frame["entity_type"] = "database"
            frame["entity"] = db
            frame["consensus"] = consensus_name
            frame["distance_eV_atom"] = (numeric(floor[f"Ef_{db}"]) - numeric(floor[consensus_col])).abs()
            details.append(frame[["pbe_job_id", "entity_type", "entity", "consensus", "distance_eV_atom"]])
    detail = pd.concat(details, ignore_index=True)
    rows: list[dict[str, Any]] = []
    for (entity_type, entity, consensus), group in detail.groupby(["entity_type", "entity", "consensus"]):
        med, low, high = bootstrap_median_ci(group["distance_eV_atom"], rng, n_boot)
        mp = detail[(detail["entity_type"] == "database") & (detail["entity"] == "MP") & (detail["consensus"] == consensus)]
        merged = group[["pbe_job_id", "distance_eV_atom"]].merge(
            mp[["pbe_job_id", "distance_eV_atom"]],
            on="pbe_job_id",
            suffixes=("", "_mp"),
        )
        rows.append(
            {
                "entity_type": entity_type,
                "entity": entity,
                "consensus": consensus,
                "n": int(len(group)),
                "median_distance_eV_atom": med,
                "ci95_low": low,
                "ci95_high": high,
                "mean_distance_eV_atom": float(numeric(group["distance_eV_atom"]).mean()),
                "wilcoxon_p_entity_closer_than_mp": wilcoxon_less(merged["distance_eV_atom"], merged["distance_eV_atom_mp"]) if entity_type == "model" else None,
                "entity_closer_than_mp_rate": float((numeric(merged["distance_eV_atom"]) < numeric(merged["distance_eV_atom_mp"])).mean()) if entity_type == "model" else None,
            }
        )
    return pd.DataFrame(rows), detail


def compute_element_headroom(merged_preds: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    expanded_rows: list[dict[str, Any]] = []
    for rec in merged_preds.to_dict("records"):
        for element in parse_elements(rec.get("elements_str") or rec.get("formula")):
            payload = dict(rec)
            payload["element"] = element
            expanded_rows.append(payload)
    expanded = pd.DataFrame(expanded_rows)
    rows: list[dict[str, Any]] = []
    subsets = {
        "clean_all": expanded.index == expanded.index,
        "clean_low_saturation": (expanded["noise_bin"] == "low") & (expanded["validation_role"] == "saturation_probe"),
    }
    for subset, mask in subsets.items():
        sub = expanded[mask].copy()
        for (model, element), group in sub.groupby(["model", "element"]):
            for floor_col in ["floor_all4_mad_scaled", "floor_pbe3_mad_scaled"]:
                row = summarize_saturation(group, str(model), subset, floor_col, rng, n_boot)
                row["element"] = element
                if row["n"] < 5:
                    row["saturation_call"] = "insufficient_n"
                rows.append(row)
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, n: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.head(n).to_markdown(index=False)


def write_report(
    out_dir: Path,
    docs_dir: Path,
    floor: pd.DataFrame,
    saturation: pd.DataFrame,
    leakage: pd.DataFrame,
    consensus: pd.DataFrame,
    element: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    clean_low = saturation[
        (saturation["subset"] == "clean_low_saturation")
        & (saturation["floor_metric"].isin(["floor_all4_mad_scaled", "floor_pbe3_mad_scaled"]))
    ]
    leakage_summary = (
        leakage.groupby(["model", "coverage_level"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["model", "coverage_level"])
    )
    consensus_focus = consensus[
        (consensus["consensus"] == "all4") & (consensus["entity"].isin(["MP", "OQMD", "AFLOW", "JARVIS", *MODELS]))
    ].sort_values(["entity_type", "median_distance_eV_atom"])
    report = [
        "# MatAlign v3 Phase D2-fix Results",
        "",
        "## Summary",
        "",
        "- No VASP or model inference was added in this phase.",
        "- The previous `id_MP` join is no longer treated as held-out evidence: current MP API IDs are valid but incompatible with the old numeric MPtrj ID index.",
        "- Main noise-floor statistic is `1.4826 * MAD-to-median`; population std is reported only as a sensitivity check.",
        "- Strong held-out saturation claims are blocked unless per-model training coverage is sufficient.",
        "",
        "## Dual Noise Floors",
        "",
        f"- clean rows: `{summary['clean_rows']}`.",
        f"- all4 MAD-scaled median: `{summary['floor_all4_mad_scaled_median']:.6f}` eV/atom.",
        f"- pbe3 MAD-scaled median: `{summary['floor_pbe3_mad_scaled_median']:.6f}` eV/atom.",
        f"- all4 std median: `{summary['floor_all4_std_median']:.6f}` eV/atom.",
        f"- pbe3 std median: `{summary['floor_pbe3_std_median']:.6f}` eV/atom.",
        f"- JARVIS is largest all4 deviation for `{summary['jarvis_outlier_rate']:.3f}` of clean materials.",
        "",
        "## Dual-Floor Saturation",
        "",
        md_table(clean_low.sort_values(["floor_metric", "model"]), 40),
        "",
        "## Leakage Recheck",
        "",
        "- `mp-bglum`-style IDs are accepted by MP API, but they do not directly match old numeric MPtrj IDs.",
        "- Rows marked `upper_bound_only_no_structure_match` are not true held-out evidence.",
        "- Rows marked `blocked_by_training_data` must not be used for strong held-out claims.",
        "",
        md_table(leakage_summary, 40),
        "",
        "## Consensus Distance",
        "",
        md_table(consensus_focus, 40),
        "",
        "## Element Headroom",
        "",
        md_table(element.sort_values(["floor_metric", "model", "saturation_call", "element"]), 80),
        "",
        "## Go / No-Go",
        "",
        f"- NMI/NCS readiness: `{summary['nmi_ncs_ready']}`.",
        f"- Reason: `{summary['go_no_go_reason']}`.",
    ]
    (docs_dir / "dual_noise_floor_checks_results_20260606.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    clean = pd.read_csv(args.analysis_dir / "clean_intermetallic_master.csv")
    preds = pd.read_csv(args.d2_dir / "frontier_model_predictions_clean.csv")
    preds = preds[preds["formation_status"].fillna("ok").astype(str).str.startswith("ok")].copy()

    floor = compute_dual_floors(clean)
    mp_summary = load_mp_summary(args.mp_summary_jsonl)
    training_index = load_training_index(args.training_index_dir)
    leakage = build_leakage_labels(floor, mp_summary, training_index)
    saturation, merged_preds = compute_saturation(preds, floor, leakage, rng, args.bootstrap)
    consensus, consensus_detail = compute_consensus_distance(preds, floor, rng, args.bootstrap)
    element = compute_element_headroom(merged_preds, rng, args.bootstrap)

    strict_heldout = leakage[leakage["held_out_strict"].eq(True)]
    strict_counts = strict_heldout.groupby("model")["pbe_job_id"].nunique().to_dict()
    ready_models = [
        model
        for model in MODELS
        if strict_counts.get(model, 0) >= 30
        and not saturation[
            (saturation["model"] == model)
            & (saturation["subset"] == "held_out_strict_clean_low_saturation")
            & (saturation["floor_metric"] == "floor_pbe3_mad_scaled")
            & (saturation["saturation_call"].isin(["below_floor_or_saturated", "saturated_indistinguishable_from_floor"]))
        ].empty
    ]
    summary = {
        "clean_rows": int(len(clean)),
        "prediction_rows": int(len(preds)),
        "rows_by_model": {str(k): int(v) for k, v in preds["model"].value_counts().sort_index().items()},
        "clean_low_saturation_rows": int(((floor["noise_bin"] == "low") & (floor["validation_role"] == "saturation_probe")).sum()),
        "floor_all4_mad_scaled_median": float(floor["floor_all4_mad_scaled"].median()),
        "floor_pbe3_mad_scaled_median": float(floor["floor_pbe3_mad_scaled"].median()),
        "floor_all4_std_median": float(floor["floor_all4_std"].median()),
        "floor_pbe3_std_median": float(floor["floor_pbe3_std"].median()),
        "jarvis_outlier_rate": float(floor["jarvis_is_all4_outlier"].mean()),
        "strict_heldout_counts_by_model": {str(k): int(v) for k, v in strict_counts.items()},
        "pbe3_heldout_ready_models": ready_models,
        "nmi_ncs_ready": bool(len(ready_models) >= 2),
        "go_no_go_reason": "Needs per-model true held-out evidence under pbe3 MAD-scaled floor; current staged leakage labels are insufficient unless strict held-out counts are available.",
        "training_index_sources": {
            source: {"coverage": info.get("coverage"), "path": info.get("path"), "id_count": len(info.get("ids", [])), "key_count": len(info.get("keys", {}))}
            for source, info in training_index.items()
        },
    }

    floor.to_csv(args.out_dir / "dual_noise_floors.csv", index=False)
    saturation.to_csv(args.out_dir / "dual_floor_saturation_summary.csv", index=False)
    leakage.to_csv(args.out_dir / "training_leakage_per_model.csv", index=False)
    consensus.to_csv(args.out_dir / "consensus_distance_summary.csv", index=False)
    consensus_detail.to_csv(args.out_dir / "consensus_distance_details.csv", index=False)
    element.to_csv(args.out_dir / "element_headroom_dual_floor.csv", index=False)
    (args.out_dir / "dual_noise_floor_checks_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(args.out_dir, args.docs_dir, floor, saturation, leakage, consensus, element, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
