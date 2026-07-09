from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element, Structure
from scipy.stats import kendalltau, spearmanr


DATA_ROOT = (Path(__file__).resolve().parents[2] / "data").resolve()
OUT_DIR = DATA_ROOT / "processed" / "distance_response_discovery"
DOCS_DIR = Path("docs")

FLOOR_EVA = 0.011
STABILITY_THRESHOLD = 0.0
SEED = 20260610

ANION_OR_H = set("O N S Se Cl F Br I Te P As H".split())
HALIDES = set("F Cl Br I".split())
CHALCOGENIDES = set("S Se Te".split())
MP_PLUS_U_ELEMENTS = set("V Cr Mn Fe Co Ni Cu Mo W U Np Pu".split())
MP_PLUS_U_ANIONS = {"O", "F"}
MAGNETIC_3D_HARD = set("Mn Cr V Fe".split())
TRANSITION_3D = set("Sc Ti V Cr Mn Fe Co Ni Cu Zn".split())
TRANSITION_4D_5D = set(
    "Y Zr Nb Mo Tc Ru Rh Pd Ag Cd Hf Ta W Re Os Ir Pt Au Hg Rf Db Sg Bh Hs Mt Ds Rg Cn".split()
)
LANTHANIDES = set("La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split())
ACTINIDES = set("Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split())

MODEL_KEYS = [
    "chgnet-0.3.0",
    "mace-mp-0",
    "mattersim-v1-5M",
    "orb-v3",
    "sevennet-mf-ompa",
]
MODEL_DISPLAY = {
    "chgnet-0.3.0": "CHGNet-0.3.0",
    "mace-mp-0": "MACE-MP-0",
    "mattersim-v1-5M": "MatterSim-v1-5M",
    "orb-v3": "ORB-v3",
    "sevennet-mf-ompa": "SevenNet-MF-ompa",
}
FRONTIER4 = ["mace-mp-0", "mattersim-v1-5M", "orb-v3", "sevennet-mf-ompa"]
ORB_SEVENNET_PAIR = ("orb-v3", "sevennet-mf-ompa")
CROSS_FAMILY_EXCL_ORB_SEVENNET_PAIRS = [
    ("mace-mp-0", "mattersim-v1-5M"),
    ("mace-mp-0", "orb-v3"),
    ("mace-mp-0", "sevennet-mf-ompa"),
    ("mattersim-v1-5M", "orb-v3"),
    ("mattersim-v1-5M", "sevennet-mf-ompa"),
]
PAIR_GROUPS = {
    "all5": list(combinations(MODEL_KEYS, 2)),
    "frontier4": list(combinations(FRONTIER4, 2)),
    "cross_family_excl_orb_sevennet": CROSS_FAMILY_EXCL_ORB_SEVENNET_PAIRS,
    "orb_sevennet": [ORB_SEVENNET_PAIR],
}
MODEL_TRAINING_SOURCES = {
    "chgnet-0.3.0": ["mptrj"],
    "mace-mp-0": ["mptrj"],
    "mattersim-v1-5M": [],
    "orb-v3": ["mptrj", "omat24", "salex"],
    "sevennet-mf-ompa": ["mptrj", "omat24", "salex"],
}
LOWER_OOD_CONFIDENCE = {"orb-v3", "sevennet-mf-ompa"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MatAlign Phase F NC/NMI upgrade analyses.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--floor", type=float, default=FLOOR_EVA)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--max-sites", type=int, default=80)
    return parser.parse_args()


def parse_elements(formula: str) -> list[str]:
    try:
        return sorted(str(element) for element in Composition(str(formula)).elements)
    except Exception:
        return []


def composition_counts_from_formula(formula: str) -> dict[int, int]:
    comp = Composition(str(formula))
    counts: dict[int, int] = {}
    for symbol, amount in comp.get_el_amt_dict().items():
        counts[Element(symbol).Z] = int(round(float(amount)))
    return reduce_counts(counts)


def reduce_counts(counts: dict[int, int]) -> dict[int, int]:
    gcd = 0
    for count in counts.values():
        gcd = math.gcd(gcd, int(count))
    if gcd > 1:
        return {number: int(count // gcd) for number, count in counts.items()}
    return {number: int(count) for number, count in counts.items()}


def composition_key_from_counts(counts: dict[int, int]) -> str:
    reduced = reduce_counts(counts)
    return ";".join(f"{number}:{reduced[number]}" for number in sorted(reduced))


def parse_composition_key(key: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    for part in str(key).split(";"):
        if not part:
            continue
        number, count = part.split(":")
        counts[int(number)] = int(count)
    return counts


def composition_key_from_formula(formula: str) -> str:
    return composition_key_from_counts(composition_counts_from_formula(formula))


def normalized_counts(counts: dict[int, int]) -> dict[int, float]:
    total = float(sum(counts.values()))
    if total <= 0:
        return {}
    return {number: count / total for number, count in counts.items()}


def stoich_l1_half(left: dict[int, int], right: dict[int, int]) -> float:
    a = normalized_counts(left)
    b = normalized_counts(right)
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(key, 0.0) - b.get(key, 0.0)) for key in keys)


def element_jaccard_distance(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 1.0
    return 1.0 - len(left & right) / len(union)


def mp_hubbard_proxy(elements: list[str]) -> bool:
    return bool(set(elements) & MP_PLUS_U_ELEMENTS) and bool(set(elements) & MP_PLUS_U_ANIONS)


def element_group_flags(elements: list[str]) -> dict[str, bool]:
    element_set = set(elements)
    has_hard = bool(element_set & MAGNETIC_3D_HARD)
    has_3d = bool(element_set & TRANSITION_3D)
    has_4d5d = bool(element_set & TRANSITION_4D_5D)
    has_lanth = bool(element_set & LANTHANIDES)
    has_act = bool(element_set & ACTINIDES)
    family_hits = int(has_3d) + int(has_4d5d) + int(has_lanth or has_act)
    return {
        "has_magnetic_3d_hard": has_hard,
        "has_3d": has_3d,
        "has_other_3d": has_3d and not has_hard,
        "has_4d_5d": has_4d5d,
        "has_lanthanide": has_lanth,
        "has_actinide": has_act,
        "main_group_metal": family_hits == 0,
        "mixed": family_hits >= 2,
        "has_Mn": "Mn" in element_set,
        "has_Cr": "Cr" in element_set,
        "has_V": "V" in element_set,
        "has_Fe": "Fe" in element_set,
    }


def chemistry_class(elements: list[str]) -> str:
    element_set = set(elements)
    if "H" in element_set:
        return "hydride"
    if "O" in element_set:
        return "oxide"
    if element_set & HALIDES:
        return "halide"
    if element_set & CHALCOGENIDES:
        return "chalcogenide"
    if not (element_set & ANION_OR_H):
        return "intermetallic"
    return "other"


def primary_discovery_segment(row: pd.Series) -> str:
    for element in ["Mn", "Cr", "V", "Fe"]:
        if bool(row.get(f"has_{element}", False)):
            return f"hard_{element}"
    if bool(row.get("has_magnetic_3d_hard", False)):
        return "magnetic_3d_hard"
    if bool(row.get("main_group_metal", False)):
        return "easy_main_group_metal"
    if bool(row.get("has_4d_5d", False)):
        return "easy_4d_5d"
    if bool(row.get("has_lanthanide", False)):
        return "easy_lanthanide"
    return "easy_other_metal"


def load_wbm_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    matbench_path = args.data_root / "processed" / "matbench" / "matbench_merged_predictions.parquet"
    summary_path = (
        args.data_root
        / "cache"
        / "matbench-discovery"
        / "data"
        / "wbm"
        / "2023-12-13-wbm-summary.csv.gz"
    )
    meta_path = args.data_root / "processed" / "matbench" / "model_metadata.csv"

    matbench_cols = [
        "material_id",
        "formula",
        "dft_ehull",
        "e_form_dft",
        "dft_stable",
        *MODEL_KEYS,
    ]
    summary_cols = [
        "material_id",
        "n_sites",
        "e_form_per_atom_mp2020_corrected",
        "e_form_per_atom_uncorrected",
        "e_above_hull_mp2020_corrected_ppd_mp",
        "unique_prototype",
        "bandgap_pbe",
    ]
    matbench = pd.read_parquet(matbench_path, columns=matbench_cols)
    summary = pd.read_csv(summary_path, usecols=summary_cols)
    meta = pd.read_csv(meta_path)
    frame = matbench.merge(summary, on="material_id", how="left", validate="one_to_one")
    frame = frame.rename(columns={"e_form_per_atom_mp2020_corrected": "wbm_ref_mp2020_e_form"})
    return frame, meta


def add_material_tags(frame: pd.DataFrame, max_sites: int) -> pd.DataFrame:
    out = frame.copy()
    parsed_elements = out["formula"].map(parse_elements)
    out["elements"] = parsed_elements.map(lambda xs: " ".join(xs))
    out["n_elements"] = parsed_elements.map(len)
    out["composition_key"] = out["formula"].map(composition_key_from_formula)
    out["chemistry_class"] = parsed_elements.map(chemistry_class)
    out["has_anion_or_H"] = parsed_elements.map(lambda xs: bool(set(xs) & ANION_OR_H))
    out["is_hubbard"] = parsed_elements.map(mp_hubbard_proxy)
    flags = pd.DataFrame([element_group_flags(xs) for xs in parsed_elements], index=out.index)
    out = pd.concat([out, flags], axis=1)
    out["is_clean_metal_e1"] = (
        ~out["has_anion_or_H"].astype(bool)
        & ~out["is_hubbard"].astype(bool)
        & (pd.to_numeric(out["n_sites"], errors="coerce") <= max_sites)
        & out["wbm_ref_mp2020_e_form"].notna()
    )
    out["primary_discovery_segment"] = out.apply(primary_discovery_segment, axis=1)
    out["actual_stable"] = (
        pd.to_numeric(out["e_above_hull_mp2020_corrected_ppd_mp"], errors="coerce")
        <= STABILITY_THRESHOLD
    )
    for model in MODEL_KEYS:
        out[f"{model}_abs_error_mp2020"] = (out[model] - out["wbm_ref_mp2020_e_form"]).abs()
        out[f"{model}_pred_ehull_proxy"] = (
            out["e_above_hull_mp2020_corrected_ppd_mp"] + out[model] - out["e_form_dft"]
        )
        out[f"{model}_pred_stable_proxy"] = out[f"{model}_pred_ehull_proxy"] <= STABILITY_THRESHOLD
        out[f"{model}_stable_error_proxy"] = out[f"{model}_pred_stable_proxy"] != out["actual_stable"]
    return out


def pair_frame(frame: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.concat(
        [(frame[left] - frame[right]).abs().rename(f"{left}__{right}") for left, right in pairs],
        axis=1,
    )


def add_pairwise_widths(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for group, pairs in PAIR_GROUPS.items():
        pairs_df = pair_frame(out, pairs)
        out[f"{group}_per_material_width"] = pairs_df.median(axis=1, skipna=True)
        out[f"{group}_mean_pair_width"] = pairs_df.mean(axis=1, skipna=True)
    return out


def finite_array(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    sample_size = min(len(values), 20_000)
    if len(values) > sample_size:
        values = rng.choice(values, size=sample_size, replace=False)
    meds = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        meds[idx] = np.median(rng.choice(values, size=len(values), replace=True))
    return float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


class TrainingCompositionIndex:
    def __init__(self, train_keys: set[str]):
        self.train_keys = set(train_keys)
        self.key_counts = {key: parse_composition_key(key) for key in self.train_keys}
        self.key_elements = {key: set(counts) for key, counts in self.key_counts.items()}
        self.elementset_to_keys: dict[tuple[int, ...], list[str]] = defaultdict(list)
        self.element_to_elementsets: dict[int, set[tuple[int, ...]]] = defaultdict(set)
        for key, elements in self.key_elements.items():
            elementset = tuple(sorted(elements))
            self.elementset_to_keys[elementset].append(key)
            for number in elementset:
                self.element_to_elementsets[number].add(elementset)
        self.elementset_freq = {
            elementset: len(keys) for elementset, keys in self.elementset_to_keys.items()
        }
        self.element_freq = {
            number: len(elementsets) for number, elementsets in self.element_to_elementsets.items()
        }

    def nearest(self, key: str) -> dict[str, Any]:
        counts = parse_composition_key(key)
        elements = set(counts)
        elementset = tuple(sorted(elements))
        if key in self.train_keys:
            return {
                "composition_distance": 0.0,
                "nearest_type": "exact_composition",
                "nearest_training_key": key,
                "element_jaccard_distance": 0.0,
                "stoich_l1_half": 0.0,
                "candidate_elementsets_checked": 1,
            }

        same_element_keys = self.elementset_to_keys.get(elementset, [])
        if same_element_keys:
            best_key = min(
                same_element_keys, key=lambda train_key: stoich_l1_half(counts, self.key_counts[train_key])
            )
            stoich = stoich_l1_half(counts, self.key_counts[best_key])
            return {
                "composition_distance": float(0.5 * stoich),
                "nearest_type": "same_element_set",
                "nearest_training_key": best_key,
                "element_jaccard_distance": 0.0,
                "stoich_l1_half": float(stoich),
                "candidate_elementsets_checked": len(same_element_keys),
            }

        ordered_elements = sorted(elements, key=lambda number: self.element_freq.get(number, 10**9))
        candidate_sets: set[tuple[int, ...]] | None = None
        for number in ordered_elements:
            sets_for_element = set(self.element_to_elementsets.get(number, set()))
            if not sets_for_element:
                continue
            if candidate_sets is None:
                candidate_sets = sets_for_element
            else:
                intersection = candidate_sets & sets_for_element
                if intersection:
                    candidate_sets = intersection
        if not candidate_sets:
            candidate_sets = set().union(
                *(self.element_to_elementsets.get(number, set()) for number in ordered_elements)
            )
        if not candidate_sets:
            return {
                "composition_distance": 1.0,
                "nearest_type": "no_shared_training_element",
                "nearest_training_key": None,
                "element_jaccard_distance": 1.0,
                "stoich_l1_half": 1.0,
                "candidate_elementsets_checked": 0,
            }

        ranked_sets = sorted(
            candidate_sets,
            key=lambda candidate: (
                element_jaccard_distance(elements, set(candidate)),
                self.elementset_freq.get(candidate, 10**9),
                len(candidate),
            ),
        )[:250]
        best: tuple[float, float, float, str] | None = None
        for candidate in ranked_sets:
            jaccard = element_jaccard_distance(elements, set(candidate))
            for train_key in self.elementset_to_keys[candidate][:250]:
                stoich = stoich_l1_half(counts, self.key_counts[train_key])
                distance = 0.6 * jaccard + 0.4 * stoich
                if best is None or distance < best[0]:
                    best = (distance, jaccard, stoich, train_key)
        if best is None:
            return {
                "composition_distance": 1.0,
                "nearest_type": "candidate_search_empty",
                "nearest_training_key": None,
                "element_jaccard_distance": 1.0,
                "stoich_l1_half": 1.0,
                "candidate_elementsets_checked": len(ranked_sets),
            }
        distance, jaccard, stoich, best_key = best
        return {
            "composition_distance": float(distance),
            "nearest_type": "nearest_element_overlap",
            "nearest_training_key": best_key,
            "element_jaccard_distance": float(jaccard),
            "stoich_l1_half": float(stoich),
            "candidate_elementsets_checked": len(ranked_sets),
        }


def compute_distance_cache(clean: pd.DataFrame, train_keys: set[str], out_dir: Path) -> pd.DataFrame:
    cache_path = out_dir / "composition_distance_cache.parquet"
    unique_keys = pd.Series(clean["composition_key"].dropna().unique(), name="composition_key")
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if set(unique_keys).issubset(set(cached["composition_key"])):
            return cached[cached["composition_key"].isin(set(unique_keys))].copy()

    index = TrainingCompositionIndex(train_keys)
    rows: list[dict[str, Any]] = []
    for key in unique_keys:
        rec = index.nearest(str(key))
        rec["composition_key"] = str(key)
        rows.append(rec)
    distances = pd.DataFrame(rows)
    distances.to_parquet(cache_path, index=False)
    return distances


def add_distance_bins(clean: pd.DataFrame) -> pd.DataFrame:
    out = clean.copy()
    out["distance_bin"] = "wbm_exact_composition"
    non_exact = out["composition_distance"] > 0
    if non_exact.any():
        ranks = out.loc[non_exact, "composition_distance"].rank(method="first")
        labels = pd.qcut(ranks, q=3, labels=["wbm_near", "wbm_mid", "wbm_far"])
        out.loc[non_exact, "distance_bin"] = labels.astype(str).to_numpy()
    return out


def wp1_distance_response(
    clean: pd.DataFrame,
    d3fix_dir: Path,
    out_dir: Path,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    response_cols = [
        "material_id",
        "formula",
        "elements",
        "composition_key",
        "nearest_training_key",
        "nearest_type",
        "composition_distance",
        "element_jaccard_distance",
        "stoich_l1_half",
        "distance_bin",
        "primary_discovery_segment",
        "has_magnetic_3d_hard",
        "n_sites",
        "cross_family_excl_orb_sevennet_per_material_width",
        "all5_per_material_width",
        "frontier4_per_material_width",
        "orb_sevennet_per_material_width",
        *[f"{model}_abs_error_mp2020" for model in MODEL_KEYS],
    ]
    response = clean[response_cols].copy()
    response.to_csv(out_dir / "distance_response_by_material.csv", index=False)

    rows: list[dict[str, Any]] = []
    metric_cols = [
        "cross_family_excl_orb_sevennet_per_material_width",
        "all5_per_material_width",
        "frontier4_per_material_width",
        "mattersim-v1-5M_abs_error_mp2020",
        "mace-mp-0_abs_error_mp2020",
        "chgnet-0.3.0_abs_error_mp2020",
    ]
    for bin_name, sub in response.groupby("distance_bin", dropna=False):
        for metric in metric_cols:
            values = finite_array(sub[metric])
            lo, hi = bootstrap_median_ci(values, rng, n_boot)
            rows.append(
                {
                    "source": "WBM_clean_metal",
                    "distance_bin": bin_name,
                    "n_materials": int(len(sub)),
                    "metric": metric,
                    "median": float(np.median(values)) if len(values) else math.nan,
                    "mean": float(np.mean(values)) if len(values) else math.nan,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )

    for metric in metric_cols:
        valid = response[["composition_distance", metric]].dropna()
        if len(valid) >= 3:
            rho, rho_p = spearmanr(valid["composition_distance"], valid[metric])
            tau, tau_p = kendalltau(valid["composition_distance"], valid[metric])
        else:
            rho = rho_p = tau = tau_p = math.nan
        rows.append(
            {
                "source": "WBM_clean_metal_trend",
                "distance_bin": "all_wbm_clean",
                "n_materials": int(len(valid)),
                "metric": metric,
                "median": math.nan,
                "mean": math.nan,
                "ci95_low": math.nan,
                "ci95_high": math.nan,
                "spearman_rho": float(rho) if np.isfinite(rho) else math.nan,
                "spearman_p": float(rho_p) if np.isfinite(rho_p) else math.nan,
                "kendall_tau": float(tau) if np.isfinite(tau) else math.nan,
                "kendall_p": float(tau_p) if np.isfinite(tau_p) else math.nan,
            }
        )

    cross_summary_path = d3fix_dir / "cross_family_pairwise_summary.csv"
    if cross_summary_path.exists():
        cross_summary = pd.read_csv(cross_summary_path)
        for _, row in cross_summary.iterrows():
            rows.append(
                {
                    "source": "MatAlign_in_sample_anchor",
                    "distance_bin": "distance_0_in_sample",
                    "n_materials": 221,
                    "metric": str(row.get("comparison", "cross_family_pairwise")),
                    "median": float(row.get("pooled_pair_median_eV_atom", math.nan)),
                    "mean": math.nan,
                    "ci95_low": float(row.get("bootstrap_ci_low", math.nan))
                    if "bootstrap_ci_low" in row
                    else math.nan,
                    "ci95_high": float(row.get("bootstrap_ci_high", math.nan))
                    if "bootstrap_ci_high" in row
                    else math.nan,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "distance_bucket_summary.csv", index=False)
    return response, summary


def segment_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    base = frame["is_clean_metal_e1"].astype(bool)
    return {
        "all_clean_metal": base,
        "magnetic_3d_hard": base & frame["has_magnetic_3d_hard"].astype(bool),
        "hard_Mn": base & frame["has_Mn"].astype(bool),
        "hard_Cr": base & frame["has_Cr"].astype(bool),
        "hard_V": base & frame["has_V"].astype(bool),
        "hard_Fe": base & frame["has_Fe"].astype(bool),
    }


def wp2_discovery_cost(clean: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    overlap_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for segment, mask in segment_masks(clean).items():
        sub = clean.loc[mask].copy()
        stable_sets = {
            model: set(sub.loc[sub[f"{model}_pred_stable_proxy"].fillna(False), "material_id"])
            for model in MODEL_KEYS
        }
        for left, right in combinations(MODEL_KEYS, 2):
            a = stable_sets[left]
            b = stable_sets[right]
            union = a | b
            overlap_rows.append(
                {
                    "segment": segment,
                    "model_a": left,
                    "model_b": right,
                    "n_segment": int(len(sub)),
                    "n_stable_a": int(len(a)),
                    "n_stable_b": int(len(b)),
                    "n_intersection": int(len(a & b)),
                    "n_union": int(len(union)),
                    "jaccard": float(len(a & b) / len(union)) if union else math.nan,
                    "stability_threshold_eV_atom": STABILITY_THRESHOLD,
                    "predicted_ehull_method": "each_true + model_e_form - e_form_dft proxy",
                }
            )

        width = sub["cross_family_excl_orb_sevennet_per_material_width"].rank(method="first")
        labels = pd.qcut(width, q=5, labels=["q1_low", "q2", "q3", "q4", "q5_high"])
        sub = sub.assign(disagreement_bin=labels.astype(str))
        for bin_name, bin_df in sub.groupby("disagreement_bin", dropna=False):
            for model in MODEL_KEYS:
                pred_stable = bin_df[f"{model}_pred_stable_proxy"].fillna(False)
                actual_stable = bin_df["actual_stable"].fillna(False)
                false_positive = pred_stable & ~actual_stable
                predicted_count = int(pred_stable.sum())
                error_rows.append(
                    {
                        "segment": segment,
                        "disagreement_bin": bin_name,
                        "model": model,
                        "n_materials": int(len(bin_df)),
                        "median_cross_family_width_eV_atom": float(
                            bin_df["cross_family_excl_orb_sevennet_per_material_width"].median()
                        ),
                        "actual_stable_rate": float(actual_stable.mean()) if len(bin_df) else math.nan,
                        "predicted_stable_rate": float(pred_stable.mean()) if len(bin_df) else math.nan,
                        "classification_error_rate": float(
                            bin_df[f"{model}_stable_error_proxy"].mean()
                        )
                        if len(bin_df)
                        else math.nan,
                        "false_positive_rate_among_predicted_stable": float(
                            false_positive.sum() / predicted_count
                        )
                        if predicted_count
                        else math.nan,
                        "predicted_ehull_method": "each_true + model_e_form - e_form_dft proxy",
                    }
                )

    overlap = pd.DataFrame(overlap_rows)
    errors = pd.DataFrame(error_rows)
    overlap.to_csv(out_dir / "discovery_overlap_jaccard.csv", index=False)
    errors.to_csv(out_dir / "disagreement_error_bins.csv", index=False)
    return overlap, errors


def pairwise_summary_for_frame(frame: pd.DataFrame, group_col: str, out_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [("all_wbm", frame)]
    groups.extend((str(name), sub) for name, sub in frame.groupby(group_col, dropna=False))
    for segment, sub in groups:
        for pair_group, pairs in PAIR_GROUPS.items():
            pf = pair_frame(sub, pairs)
            pooled = pf.to_numpy(dtype=float).ravel()
            pooled = pooled[np.isfinite(pooled)]
            per_material = pf.median(axis=1, skipna=True).dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "segment": segment,
                    "pair_group": pair_group,
                    "n_materials": int(len(sub)),
                    "n_pairs": int(len(pairs)),
                    "n_pair_values": int(len(pooled)),
                    "pooled_pair_median_eV_atom": float(np.median(pooled)) if len(pooled) else math.nan,
                    "pooled_pair_mean_eV_atom": float(np.mean(pooled)) if len(pooled) else math.nan,
                    "per_material_width_median_eV_atom": float(np.median(per_material))
                    if len(per_material)
                    else math.nan,
                    "floor_reference_eV_atom": FLOOR_EVA,
                    "uses_reference_energy": False,
                    "interpretation": "model-model consistency only",
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False)
    return summary


def load_wbm_structures_for_ids(raw_path: Path, material_ids: set[str]) -> dict[str, dict[str, Any]]:
    with bz2.open(raw_path, "rt") as handle:
        data = json.load(handle)
    id_map = data["material_id"]
    structures = data["initial_structure"]
    result: dict[str, dict[str, Any]] = {}
    for idx, material_id in id_map.items():
        if material_id in material_ids:
            result[material_id] = structures[idx]
    return result


def wbm_structure_coverage(raw_path: Path, target_ids: set[str], out_dir: Path) -> dict[str, Any]:
    with bz2.open(raw_path, "rt") as handle:
        data = json.load(handle)
    available_ids = set(data["material_id"].values())
    missing_ids = sorted(target_ids - available_ids)
    pd.DataFrame({"material_id": missing_ids}).to_csv(
        out_dir / "missing_wbm_init_structures.csv", index=False
    )
    return {
        "target_ids": int(len(target_ids)),
        "available_ids": int(len(target_ids) - len(missing_ids)),
        "missing_ids": int(len(missing_ids)),
        "coverage_rate": float((len(target_ids) - len(missing_ids)) / len(target_ids))
        if target_ids
        else math.nan,
    }


def md5sum(path: Path, chunk_size: int = 1024 * 1024 * 8) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wp4_selection_manifest(
    clean: pd.DataFrame,
    args: argparse.Namespace,
    out_dir: Path,
    rng: np.random.Generator,
) -> pd.DataFrame:
    pool = clean.copy()
    pool["wp4_segment"] = pool.apply(primary_discovery_segment, axis=1)
    target_segments = [
        "hard_Mn",
        "hard_Cr",
        "hard_V",
        "hard_Fe",
        "easy_main_group_metal",
        "easy_4d_5d",
    ]
    pool["wp4_segment"] = np.where(
        pool["wp4_segment"].isin(target_segments), pool["wp4_segment"], "easy_other_metal"
    )
    selected_indices: list[int] = []
    per_cell = 10
    for (distance_bin, segment), sub in pool.groupby(["distance_bin", "wp4_segment"], dropna=False):
        if len(sub) == 0:
            continue
        take = min(per_cell, len(sub))
        ranked = sub.sort_values(
            ["cross_family_excl_orb_sevennet_per_material_width", "composition_distance"],
            ascending=[False, False],
        )
        selected_indices.extend(list(ranked.head(take).index))

    selected = pool.loc[sorted(set(selected_indices))].copy()
    if len(selected) < 300:
        remaining = pool.drop(index=selected.index, errors="ignore").sort_values(
            ["cross_family_excl_orb_sevennet_per_material_width", "composition_distance"],
            ascending=[False, False],
        )
        selected = pd.concat([selected, remaining.head(300 - len(selected))], axis=0)
    selected = selected.sort_values(
        ["wp4_segment", "distance_bin", "cross_family_excl_orb_sevennet_per_material_width"],
        ascending=[True, True, False],
    ).head(300)
    selected = selected.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    selected["tier_n100"] = selected["selection_rank"] <= 100
    selected["tier_n200"] = selected["selection_rank"] <= 200
    selected["tier_n300"] = selected["selection_rank"] <= 300

    raw_struct_path = (
        args.data_root / "raw" / "wbm" / "40344466_2022-10-19-wbm-init-structs.json.bz2"
    )
    structures = load_wbm_structures_for_ids(raw_struct_path, set(selected["material_id"]))
    selected["structure_json"] = selected["material_id"].map(
        lambda material_id: json.dumps(structures.get(material_id, {}), separators=(",", ":"))
    )
    selected["structure_source"] = "WBM initial_structure JSON"
    selected["recommended_protocol"] = (
        "Run MP-PBE, PBE-variant, and r2SCAN static/relax protocol; compute r2SCAN elemental "
        "references before forming r2SCAN formation energies."
    )

    cols = [
        "selection_rank",
        "tier_n100",
        "tier_n200",
        "tier_n300",
        "material_id",
        "formula",
        "elements",
        "composition_key",
        "nearest_training_key",
        "nearest_type",
        "composition_distance",
        "distance_bin",
        "wp4_segment",
        "n_sites",
        "cross_family_excl_orb_sevennet_per_material_width",
        "mattersim-v1-5M_abs_error_mp2020",
        "e_above_hull_mp2020_corrected_ppd_mp",
        "structure_source",
        "recommended_protocol",
        "structure_json",
    ]
    manifest = selected[cols].copy()
    manifest.to_csv(out_dir / "three_reference_dft_selection_manifest.csv", index=False)
    manifest.to_parquet(out_dir / "three_reference_dft_selection_manifest.parquet", index=False)
    return manifest


def load_training_key_sets(args: argparse.Namespace) -> dict[str, set[str]]:
    base = args.data_root / "cache" / "training_indices"
    files = {
        "mptrj": base / "mptrj_composition_keys.csv",
        "omat24": base / "omat24_composition_keys.csv",
        "salex": base / "salex_composition_keys.csv",
    }
    key_sets: dict[str, set[str]] = {}
    for source, path in files.items():
        if path.exists():
            key_sets[source] = set(pd.read_csv(path, usecols=["composition_key"])["composition_key"].astype(str))
        else:
            key_sets[source] = set()
    return key_sets


def wp6_coverage_association(clean: pd.DataFrame, full: pd.DataFrame, key_sets: dict[str, set[str]], out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODEL_KEYS:
        sources = MODEL_TRAINING_SOURCES[model]
        if sources:
            union_keys = set().union(*(key_sets.get(source, set()) for source in sources))
            coverage_clean = clean["composition_key"].isin(union_keys)
            coverage_all = full["composition_key"].isin(union_keys)
            coverage_level = "composition_key_proxy"
        else:
            union_keys = set()
            coverage_clean = pd.Series(False, index=clean.index)
            coverage_all = pd.Series(False, index=full.index)
            coverage_level = "training_data_unavailable"
        valid = clean[["composition_distance", f"{model}_abs_error_mp2020"]].dropna()
        if len(valid) >= 3:
            rho, rho_p = spearmanr(valid["composition_distance"], valid[f"{model}_abs_error_mp2020"])
        else:
            rho = rho_p = math.nan
        other_pairs = [
            (left, right)
            for left, right in combinations(MODEL_KEYS, 2)
            if model in (left, right)
        ]
        model_pair_width = pair_frame(clean, other_pairs).to_numpy(dtype=float).ravel()
        model_pair_width = model_pair_width[np.isfinite(model_pair_width)]
        rows.append(
            {
                "model": model,
                "training_sources_used_for_proxy": ";".join(sources) if sources else "unavailable",
                "coverage_level": coverage_level,
                "n_training_composition_keys_proxy": int(len(union_keys)),
                "exact_composition_coverage_rate_clean": float(coverage_clean.mean()) if len(clean) else math.nan,
                "exact_composition_coverage_rate_all_wbm": float(coverage_all.mean()) if len(full) else math.nan,
                "median_abs_error_clean_eV_atom": float(clean[f"{model}_abs_error_mp2020"].median()),
                "median_pairwise_width_to_other_models_clean_eV_atom": float(np.median(model_pair_width))
                if len(model_pair_width)
                else math.nan,
                "error_vs_distance_spearman_rho": float(rho) if np.isfinite(rho) else math.nan,
                "error_vs_distance_spearman_p": float(rho_p) if np.isfinite(rho_p) else math.nan,
                "ood_confidence": "lower" if model in LOWER_OOD_CONFIDENCE else "standard",
            }
        )
    table = pd.DataFrame(rows)
    valid_assoc = table[table["coverage_level"] == "composition_key_proxy"].copy()
    if len(valid_assoc) >= 3:
        rho, pval = spearmanr(
            valid_assoc["exact_composition_coverage_rate_clean"],
            valid_assoc["median_abs_error_clean_eV_atom"],
        )
        table["coverage_vs_error_spearman_rho_across_models"] = float(rho)
        table["coverage_vs_error_spearman_p_across_models"] = float(pval)
    else:
        table["coverage_vs_error_spearman_rho_across_models"] = math.nan
        table["coverage_vs_error_spearman_p_across_models"] = math.nan
    table.to_csv(out_dir / "training_coverage_association.csv", index=False)
    return table


def write_guidelines(out_dir: Path) -> Path:
    path = out_dir / "memory_probe_guidelines.md"
    text = """# Phase F Memory Probe and Evaluation Guidelines

## Memorization Probe

Use model-model agreement as a no-new-DFT diagnostic for benchmark memorization.

1. Select at least three model families and exclude homologous pairs from the main statistic.
2. Compute per-material pairwise formation-energy width on an in-sample anchor and on the target OOD set.
3. Estimate each OOD material's distance to the training distribution using exact composition keys first, then composition-nearest neighbors, then structure checks only for candidate near-duplicates.
4. Bin OOD materials by training distance and test whether model-model width increases monotonically with distance.
5. Interpret a tight in-sample width but widening OOD width as evidence of shared training-label memory, not physical saturation.

## Evaluation Rules

- Report structure-level leakage audits for each model family whenever training structures are available.
- Report reference uncertainty as a range and sensitivity layer, not as a single noise-floor scalar.
- Require OOD and cross-functional validation before making saturation claims.
- Report headroom element by element; do not collapse magnetic 3d, f-block, and simple metals into one global median.

## Discovery Use

Model disagreement can be used as a zero-cost uncertainty proxy. Materials in high-disagreement bins should be deprioritized for expensive DFT follow-up unless their scientific value justifies the risk.
"""
    path.write_text(text, encoding="utf-8")
    return path


def write_report(
    args: argparse.Namespace,
    summary: dict[str, Any],
    wp1_summary: pd.DataFrame,
    wp2_overlap: pd.DataFrame,
    wp2_errors: pd.DataFrame,
    wp3: pd.DataFrame,
    wp4: pd.DataFrame,
    wp6: pd.DataFrame,
) -> Path:
    report_path = args.docs_dir / "distance_response_discovery_results_20260610.md"
    args.docs_dir.mkdir(parents=True, exist_ok=True)

    trend = wp1_summary[wp1_summary["source"] == "WBM_clean_metal_trend"].copy()
    width_trend = trend[
        trend["metric"] == "cross_family_excl_orb_sevennet_per_material_width"
    ]
    width_rho = (
        float(width_trend.iloc[0]["spearman_rho"]) if not width_trend.empty else math.nan
    )

    overlap_key = wp2_overlap[
        (wp2_overlap["segment"] == "magnetic_3d_hard")
        & (wp2_overlap["model_a"] == "orb-v3")
        & (wp2_overlap["model_b"] == "sevennet-mf-ompa")
    ]
    homologous_jaccard = (
        float(overlap_key.iloc[0]["jaccard"]) if not overlap_key.empty else math.nan
    )

    cross_wbm = wp3[
        (wp3["segment"] == "all_wbm")
        & (wp3["pair_group"] == "cross_family_excl_orb_sevennet")
    ]
    full_cross = (
        float(cross_wbm.iloc[0]["pooled_pair_median_eV_atom"]) if not cross_wbm.empty else math.nan
    )

    text = [
        "# MatAlign Phase F NC/NMI Upgrade Results",
        "",
        "## Summary",
        "",
        f"- WBM total rows: `{summary['wbm_rows']}`.",
        f"- WBM clean-metal rows: `{summary['clean_metal_rows']}`.",
        f"- WP1 distance trend Spearman rho for cross-family width: `{width_rho:.4g}`.",
        f"- WP3 full-WBM cross-family pooled median width: `{full_cross:.4g}` eV/atom.",
        f"- WP4 DFT manifest rows: `{len(wp4)}` with tier flags for 100/200/300 material runs.",
        f"- WP4 manifest ready for inspection: `{summary['wp4_manifest_ready']}`.",
        f"- Remote DFT launched in this phase: `{summary['go_wp4_remote_dft']}`.",
        f"- WBM init-structure coverage on clean-metal rows: `{summary['wbm_init_structure_coverage']['coverage_rate']:.6f}`.",
        f"- MPtrj MD5 verified: `{summary['mptrj_raw_zip_md5_verified']}`.",
        "",
        "## WP1 Distance Response",
        "",
        "The low-cost distance metric uses exact composition-key membership first, same-element stoichiometry second, and bounded element-overlap nearest-neighbor search third. It is not a full structure-leakage proof; structure matching is reserved for selected candidates.",
        "",
        wp1_summary[
            (wp1_summary["source"] == "WBM_clean_metal")
            & (wp1_summary["metric"] == "cross_family_excl_orb_sevennet_per_material_width")
        ][["distance_bin", "n_materials", "median", "ci95_low", "ci95_high"]].to_markdown(index=False),
        "",
        "## WP2 Discovery-List Consequences",
        "",
        "Predicted hull distances use the Matbench Discovery proxy `each_true + model_e_form - e_form_dft`. This is a discovery-list proxy, not a recomputed convex hull.",
        "",
        f"- ORB-SevenNet magnetic-3d hard stable-list Jaccard: `{homologous_jaccard:.4g}`.",
        "- See `discovery_overlap_jaccard.csv` for all model pairs and segments.",
        "- See `disagreement_error_bins.csv` for disagreement-bin classification error rates.",
        "",
        "## WP3 Full Chemistry Consistency",
        "",
        "Full-chemistry analysis reports model-model agreement only. It does not use corrected or raw reference energies for strong error claims.",
        "",
        wp3[
            (wp3["segment"].isin(["all_wbm", "intermetallic", "oxide", "halide", "chalcogenide", "hydride", "other"]))
            & (wp3["pair_group"] == "cross_family_excl_orb_sevennet")
        ][["segment", "n_materials", "pooled_pair_median_eV_atom", "per_material_width_median_eV_atom"]].to_markdown(index=False),
        "",
        "## WP4 DFT Selection Manifest",
        "",
        "The manifest is a candidate list only. No VASP/r2SCAN jobs were submitted in this phase.",
        "",
        wp4.groupby(["distance_bin", "wp4_segment"]).size().reset_index(name="n").head(30).to_markdown(index=False),
        "",
        "## WP6 Coverage Association",
        "",
        "Coverage is a composition-key proxy. MatterSim full training data is unavailable locally, and ORB/SevenNet remain lower-OOD-confidence models.",
        "",
        wp6[
            [
                "model",
                "training_sources_used_for_proxy",
                "coverage_level",
                "exact_composition_coverage_rate_clean",
                "median_abs_error_clean_eV_atom",
                "error_vs_distance_spearman_rho",
                "ood_confidence",
            ]
        ].to_markdown(index=False),
        "",
        "## Interpretation Discipline",
        "",
        "- Error claims remain limited to clean-metal WBM and MP2020/WBM-mouth reference space.",
        "- Full-WBM chemistry claims are consistency-only.",
        "- `model error < DFT floor` is not interpreted as models being more accurate than DFT.",
        "- WP4 is not complete until MP-PBE, PBE-variant, r2SCAN, and r2SCAN elemental references are actually computed.",
    ]
    report_path.write_text("\n".join(text), encoding="utf-8")
    return report_path


def append_worklog(summary: dict[str, Any], report_path: Path) -> None:
    path = Path("WORKLOG.md")
    if not path.exists():
        return
    existing = path.read_text(encoding="utf-8")
    if "## 2026-06-10 Phase F NC/NMI Upgrade" in existing:
        return
    entry = f"""

## 2026-06-10 Phase F NC/NMI Upgrade

### 已完成

- 按 `distance_response_discovery_plan.md` 执行第一轮零 DFT 分析。
- 新增 Phase F 本地分析脚本，完成 WP1/WP2/WP3/WP6，并生成 WP4 DFT 选材 manifest。
- 没有提交远端 VASP/r2SCAN 任务，没有覆盖 D3fix/E1 已冻结结果。

### 产出

- Output directory: `processed/distance_response_discovery`
- 阶段报告：`{report_path.as_posix()}`
- 关键表：`distance_response_by_material.csv`、`distance_bucket_summary.csv`、`discovery_overlap_jaccard.csv`、`disagreement_error_bins.csv`、`full_chemistry_consistency.csv`、`three_reference_dft_selection_manifest.csv/parquet`、`training_coverage_association.csv`。

### 结果

- WBM 总数 `{summary['wbm_rows']}`，clean-metal `{summary['clean_metal_rows']}`。
- WP4 manifest 已生成 `{summary['wp4_manifest_rows']}` 条候选，并带 100/200/300 三档 tier。
- 当前 `go_wp4_remote_dft={summary['go_wp4_remote_dft']}`；本阶段只冻结选材，不启动算力。

### 下一步

- 人工检查 WP1 距离桶和 WP4 manifest 的化学覆盖后，再决定是否开 MP-PBE/PBE-variant/r2SCAN 三套远端 DFT。
- 若继续冲 NMI/NC，下一阶段应优先把 WP1 hero curve、WP2 discovery-list 分歧和 WP4 r2SCAN 结果整合进 manuscript。
"""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    full, meta = load_wbm_frame(args)
    full = add_material_tags(full, args.max_sites)
    full = add_pairwise_widths(full)
    clean = full[full["is_clean_metal_e1"]].copy()

    key_sets = load_training_key_sets(args)
    raw_wbm_struct_path = (
        args.data_root / "raw" / "wbm" / "40344466_2022-10-19-wbm-init-structs.json.bz2"
    )
    wbm_struct_coverage = wbm_structure_coverage(
        raw_wbm_struct_path, set(clean["material_id"]), args.out_dir
    )
    distances = compute_distance_cache(clean, key_sets["mptrj"], args.out_dir)
    clean = clean.merge(distances, on="composition_key", how="left", validate="many_to_one")
    clean = add_distance_bins(clean)
    full = full.merge(
        clean[["material_id", "composition_distance", "distance_bin"]], on="material_id", how="left"
    )

    d3fix_dir = args.data_root / "processed" / "leakage_controlled_consistency"
    wp1_response, wp1_summary = wp1_distance_response(
        clean, d3fix_dir, args.out_dir, rng, args.bootstrap
    )
    wp2_overlap, wp2_errors = wp2_discovery_cost(clean, args.out_dir)
    wp3 = pairwise_summary_for_frame(
        full, "chemistry_class", args.out_dir / "full_chemistry_consistency.csv"
    )
    wp4 = wp4_selection_manifest(clean, args, args.out_dir, rng)
    wp6 = wp6_coverage_association(clean, full, key_sets, args.out_dir)
    guideline_path = write_guidelines(args.out_dir)

    width_trend = wp1_summary[
        (wp1_summary["source"] == "WBM_clean_metal_trend")
        & (wp1_summary["metric"] == "cross_family_excl_orb_sevennet_per_material_width")
    ]
    width_rho = (
        float(width_trend.iloc[0]["spearman_rho"]) if not width_trend.empty else math.nan
    )
    manifest_ready = bool(np.isfinite(width_rho) and width_rho > 0)
    mptrj_zip = args.data_root / "raw" / "mptrj" / "2024-09-03-mp-trj.extxyz.zip"
    mptrj_md5 = md5sum(mptrj_zip)
    summary = {
        "wbm_rows": int(len(full)),
        "clean_metal_rows": int(len(clean)),
        "wbm_init_structure_coverage": wbm_struct_coverage,
        "mptrj_raw_zip_exists": bool(mptrj_zip.exists()),
        "mptrj_raw_zip_md5": mptrj_md5,
        "mptrj_raw_zip_md5_verified": mptrj_md5 == "7f433171e4e5f2ef9304dccd42d5488f",
        "wp1_distance_response_rows": int(len(wp1_response)),
        "wp4_manifest_rows": int(len(wp4)),
        "wp4_tier_counts": {
            "n100": int(wp4["tier_n100"].sum()),
            "n200": int(wp4["tier_n200"].sum()),
            "n300": int(wp4["tier_n300"].sum()),
        },
        "wp1_cross_family_width_spearman_rho": width_rho,
        "wp4_manifest_ready": manifest_ready,
        "go_wp4_remote_dft": False,
        "go_wp4_reason": "Manifest is ready for human inspection; remote DFT was intentionally not launched."
        if manifest_ready
        else "Distance-response trend not positive; treat WP4 as optional stress test.",
        "memory_probe_guidelines": str(guideline_path),
    }
    (args.out_dir / "distance_response_discovery_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report_path = write_report(args, summary, wp1_summary, wp2_overlap, wp2_errors, wp3, wp4, wp6)
    append_worklog(summary, report_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
