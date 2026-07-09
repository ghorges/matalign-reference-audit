from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Composition


DATA_ROOT = (Path(__file__).resolve().parents[2] / "data").resolve()
OUT_DIR = DATA_ROOT / "processed" / "v3_wbm_validation"
DOCS_DIR = Path("docs")
FLOOR_EVA = 0.011

ANION_OR_H = set("O N S Se Cl F Br I Te P As H".split())
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
STANDARD_OOD_MODELS = ["chgnet-0.3.0", "mace-mp-0", "mattersim-v1-5M"]
FRONTIER4 = ["mace-mp-0", "mattersim-v1-5M", "orb-v3", "sevennet-mf-ompa"]
LOWER_OOD_CONFIDENCE = {"orb-v3", "sevennet-mf-ompa"}
ORB_SEVENNET_PAIR = ("orb-v3", "sevennet-mf-ompa")
CROSS_FAMILY_EXCL_ORB_SEVENNET_PAIRS = [
    ("mace-mp-0", "mattersim-v1-5M"),
    ("mace-mp-0", "orb-v3"),
    ("mace-mp-0", "sevennet-mf-ompa"),
    ("mattersim-v1-5M", "orb-v3"),
    ("mattersim-v1-5M", "sevennet-mf-ompa"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MatAlign v3 Phase E1 WBM validation.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--floor", type=float, default=FLOOR_EVA)
    parser.add_argument("--max-sites", type=int, default=80)
    return parser.parse_args()


def parse_elements(formula: str) -> list[str]:
    try:
        return sorted(str(element) for element in Composition(str(formula)).elements)
    except Exception:
        return []


def mp_hubbard_proxy(elements: list[str]) -> bool:
    # MP +U is only relevant for selected elements in oxide/fluoride-like chemistries.
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


def n_sites_bin(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    n = float(value)
    if n <= 4:
        return "n_sites_1_4"
    if n <= 8:
        return "n_sites_5_8"
    if n <= 16:
        return "n_sites_9_16"
    if n <= 32:
        return "n_sites_17_32"
    return "n_sites_33_80"


def load_wbm_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    matbench_path = args.data_root / "processed" / "matbench" / "matbench_merged_predictions.parquet"
    summary_path = args.data_root / "cache" / "matbench-discovery" / "data" / "wbm" / "2023-12-13-wbm-summary.csv.gz"
    meta_path = args.data_root / "processed" / "matbench" / "model_metadata.csv"

    matbench_cols = ["material_id", "formula", "dft_ehull", "e_form_dft", "dft_stable", *MODEL_KEYS]
    matbench = pd.read_parquet(matbench_path, columns=matbench_cols)
    summary_cols = [
        "material_id",
        "n_sites",
        "e_form_per_atom_mp2020_corrected",
        "e_form_per_atom_uncorrected",
        "e_above_hull_mp2020_corrected_ppd_mp",
        "unique_prototype",
        "bandgap_pbe",
    ]
    summary = pd.read_csv(summary_path, usecols=summary_cols)
    meta = pd.read_csv(meta_path)

    frame = matbench.merge(summary, on="material_id", how="left", validate="one_to_one")
    frame = frame.rename(columns={"e_form_per_atom_mp2020_corrected": "wbm_ref_mp2020_e_form"})
    return frame, meta


def add_material_tags(frame: pd.DataFrame, max_sites: int) -> pd.DataFrame:
    out = frame.copy()
    elements = out["formula"].map(parse_elements)
    out["elements"] = elements.map(lambda xs: " ".join(xs))
    out["n_elements"] = elements.map(len)
    out["has_anion_or_H"] = elements.map(lambda xs: bool(set(xs) & ANION_OR_H))
    out["is_hubbard"] = elements.map(mp_hubbard_proxy)
    flags = pd.DataFrame([element_group_flags(xs) for xs in elements], index=out.index)
    out = pd.concat([out, flags], axis=1)
    out["n_sites_bin"] = out["n_sites"].map(n_sites_bin)
    out["is_clean_metal_e1"] = (
        ~out["has_anion_or_H"].astype(bool)
        & ~out["is_hubbard"].astype(bool)
        & (pd.to_numeric(out["n_sites"], errors="coerce") <= max_sites)
        & out["wbm_ref_mp2020_e_form"].notna()
    )
    out["ref_delta_e_form_dft_minus_mp2020"] = out["e_form_dft"] - out["wbm_ref_mp2020_e_form"]
    for model in MODEL_KEYS:
        out[f"{model}_abs_error_mp2020"] = (out[model] - out["wbm_ref_mp2020_e_form"]).abs()
    return out


def segment_masks(frame: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    base = frame["is_clean_metal_e1"].astype(bool)
    masks: dict[tuple[str, str], pd.Series] = {
        ("global", "all_clean_metal"): base,
        ("element_family", "main_group_metal"): base & frame["main_group_metal"].astype(bool),
        ("element_family", "magnetic_3d_hard"): base & frame["has_magnetic_3d_hard"].astype(bool),
        ("element_family", "other_3d"): base & frame["has_other_3d"].astype(bool),
        ("element_family", "4d_5d"): base & frame["has_4d_5d"].astype(bool),
        ("element_family", "lanthanide"): base & frame["has_lanthanide"].astype(bool),
        ("element_family", "mixed"): base & frame["mixed"].astype(bool),
        ("hard_element", "hard_Mn"): base & frame["has_Mn"].astype(bool),
        ("hard_element", "hard_Cr"): base & frame["has_Cr"].astype(bool),
        ("hard_element", "hard_V"): base & frame["has_V"].astype(bool),
        ("hard_element", "hard_Fe"): base & frame["has_Fe"].astype(bool),
    }
    for label in ["n_sites_1_4", "n_sites_5_8", "n_sites_9_16", "n_sites_17_32", "n_sites_33_80"]:
        masks[("n_sites", label)] = base & (frame["n_sites_bin"] == label)
    return masks


def finite(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)


def grouped_model_errors(frame: pd.DataFrame, floor: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (segment_type, segment), mask in segment_masks(frame).items():
        sub = frame.loc[mask].copy()
        for model in MODEL_KEYS:
            errors = finite(sub[f"{model}_abs_error_mp2020"])
            rows.append(
                {
                    "segment_type": segment_type,
                    "segment": segment,
                    "model": model,
                    "ood_confidence": "lower" if model in LOWER_OOD_CONFIDENCE else "standard",
                    "n_materials": int(len(sub)),
                    "n_predictions": int(len(errors)),
                    "coverage_rate": float(len(errors) / len(sub)) if len(sub) else math.nan,
                    "median_abs_error_eV_atom": float(np.median(errors)) if len(errors) else math.nan,
                    "mae_eV_atom": float(np.mean(errors)) if len(errors) else math.nan,
                    "rmse_eV_atom": float(np.sqrt(np.mean(errors**2))) if len(errors) else math.nan,
                    "p25_abs_error_eV_atom": float(np.quantile(errors, 0.25)) if len(errors) else math.nan,
                    "p75_abs_error_eV_atom": float(np.quantile(errors, 0.75)) if len(errors) else math.nan,
                    "within_0p011_rate": float(np.mean(errors <= floor)) if len(errors) else math.nan,
                    "reference_column": "wbm_ref_mp2020_e_form",
                }
            )
    return pd.DataFrame(rows)


def pairwise_values(sub: pd.DataFrame, pairs: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    pair_arrays = []
    for left, right in pairs:
        pair_arrays.append((sub[left] - sub[right]).abs().rename(f"{left}__{right}"))
    pair_frame = pd.concat(pair_arrays, axis=1)
    pooled = pair_frame.to_numpy(dtype=float).ravel()
    pooled = pooled[np.isfinite(pooled)]
    per_material = pair_frame.median(axis=1, skipna=True).dropna().to_numpy(dtype=float)
    return pooled, per_material


def pairwise_summary(frame: pd.DataFrame, floor: float) -> pd.DataFrame:
    pair_groups = {
        "all5": list(combinations(MODEL_KEYS, 2)),
        "frontier4": list(combinations(FRONTIER4, 2)),
        "cross_family_excl_orb_sevennet": CROSS_FAMILY_EXCL_ORB_SEVENNET_PAIRS,
        "orb_sevennet": [ORB_SEVENNET_PAIR],
    }
    rows: list[dict[str, Any]] = []
    for (segment_type, segment), mask in segment_masks(frame).items():
        sub = frame.loc[mask].copy()
        for group, pairs in pair_groups.items():
            pooled, per_material = pairwise_values(sub, pairs)
            rows.append(
                {
                    "segment_type": segment_type,
                    "segment": segment,
                    "pair_group": group,
                    "n_materials": int(len(sub)),
                    "n_pairs": int(len(pairs)),
                    "n_pair_values": int(len(pooled)),
                    "pooled_pair_median_eV_atom": float(np.median(pooled)) if len(pooled) else math.nan,
                    "pooled_pair_mean_eV_atom": float(np.mean(pooled)) if len(pooled) else math.nan,
                    "per_material_width_median_eV_atom": float(np.median(per_material)) if len(per_material) else math.nan,
                    "per_material_width_mean_eV_atom": float(np.mean(per_material)) if len(per_material) else math.nan,
                    "pair_values_within_floor_rate": float(np.mean(pooled <= floor)) if len(pooled) else math.nan,
                    "per_material_width_within_floor_rate": float(np.mean(per_material <= floor)) if len(per_material) else math.nan,
                    "floor_eV_atom": floor,
                    "tight_vs_floor": bool(np.median(pooled) <= floor) if len(pooled) else False,
                }
            )
    return pd.DataFrame(rows)


def ood_caveats(meta: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    clean = frame[frame["is_clean_metal_e1"]].copy()
    for model in MODEL_KEYS:
        rec = meta[meta["model_key"] == model].iloc[0].to_dict()
        if model == "orb-v3":
            confidence = "lower"
            caveat = "Training includes Alex/OMat24/MPtrj; Alexandria-style prototype overlap with WBM is a known caveat."
        elif model == "sevennet-mf-ompa":
            confidence = "lower"
            caveat = "Training includes OMat24/sAlex/MPtrj; WBM near-duplicate risk is not fully excluded."
        else:
            confidence = "standard"
            caveat = "Uses public WBM prediction; no additional full-training structure exclusion claimed in E1."
        rows.append(
            {
                "model": model,
                "model_name": rec.get("model_name"),
                "training_set": rec.get("training_set"),
                "trained_for_benchmark": bool(rec.get("trained_for_benchmark")),
                "openness": rec.get("openness"),
                "ood_confidence": confidence,
                "clean_metal_rows": int(len(clean)),
                "clean_metal_predictions": int(clean[model].notna().sum()),
                "prediction_coverage_rate": float(clean[model].notna().mean()) if len(clean) else math.nan,
                "leakage_caveat": caveat,
            }
        )
    return pd.DataFrame(rows)


def decide_go_nogo(grouped: pd.DataFrame, pairwise: pd.DataFrame, floor: float) -> dict[str, Any]:
    def model_median(segment: str, model: str) -> float:
        row = grouped[(grouped["segment"] == segment) & (grouped["model"] == model)]
        if row.empty:
            return math.nan
        return float(row.iloc[0]["median_abs_error_eV_atom"])

    def segment_n(segment: str) -> int:
        row = grouped[grouped["segment"] == segment]
        return int(row.iloc[0]["n_materials"]) if not row.empty else 0

    decision_segment = "magnetic_3d_hard" if segment_n("magnetic_3d_hard") >= 30 else "all_clean_metal"
    standard_medians = {model: model_median(decision_segment, model) for model in STANDARD_OOD_MODELS}
    all_medians = {model: model_median(decision_segment, model) for model in MODEL_KEYS}
    top_standard_model, top_standard = min(
        ((model, value) for model, value in standard_medians.items() if np.isfinite(value)),
        key=lambda item: item[1],
    )
    top_all_model, top_all = min(
        ((model, value) for model, value in all_medians.items() if np.isfinite(value)),
        key=lambda item: item[1],
    )
    pair_row = pairwise[
        (pairwise["segment"] == decision_segment)
        & (pairwise["pair_group"] == "cross_family_excl_orb_sevennet")
    ].iloc[0]
    cross_pooled = float(pair_row["pooled_pair_median_eV_atom"])
    cross_tight = bool(cross_pooled <= floor)

    if top_standard <= floor:
        go_e2 = True
        go_level = "go"
        reason = (
            f"{decision_segment}: best standard-OOD model {top_standard_model} has median error "
            f"{top_standard:.4f} <= floor {floor:.4f}."
        )
        recommended_n = 200 if cross_tight else 100
    elif top_standard <= 0.025:
        if cross_tight:
            go_e2 = True
            go_level = "edge_go"
            reason = (
                f"{decision_segment}: best standard-OOD model is edge ({top_standard:.4f}), "
                f"but cross-family model width remains tight ({cross_pooled:.4f} <= {floor:.4f})."
            )
            recommended_n = 100
        else:
            go_e2 = False
            go_level = "edge_stop"
            reason = (
                f"{decision_segment}: best standard-OOD model is edge ({top_standard:.4f}) "
                f"and cross-family width is already scattered ({cross_pooled:.4f} > {floor:.4f})."
            )
            recommended_n = 0
    elif top_standard > 0.03:
        go_e2 = False
        go_level = "stop"
        reason = (
            f"{decision_segment}: standard-OOD models are far above floor; best is "
            f"{top_standard_model} at {top_standard:.4f} > 0.0300."
        )
        recommended_n = 0
    else:
        go_e2 = False
        go_level = "stop"
        reason = (
            f"{decision_segment}: standard-OOD models do not justify E2; best is "
            f"{top_standard_model} at {top_standard:.4f}."
        )
        recommended_n = 0

    if top_all_model in LOWER_OOD_CONFIDENCE and top_standard > floor:
        reason += (
            f" Best all-model median is {top_all_model} at {top_all:.4f}, "
            "but that model has lower OOD confidence and is not used alone for go."
        )

    return {
        "go_e2": go_e2,
        "go_level": go_level,
        "go_reason": reason,
        "recommended_e2_n": recommended_n,
        "decision_segment": decision_segment,
        "decision_segment_n": segment_n(decision_segment),
        "floor_eV_atom": floor,
        "top_standard_ood_model": top_standard_model,
        "top_standard_ood_median_abs_error_eV_atom": top_standard,
        "top_all_model": top_all_model,
        "top_all_median_abs_error_eV_atom": top_all,
        "cross_family_excl_orb_sevennet_pooled_median_eV_atom": cross_pooled,
        "cross_family_excl_orb_sevennet_tight_vs_floor": cross_tight,
    }


def compact_group_table(grouped: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "all_clean_metal",
        "main_group_metal",
        "magnetic_3d_hard",
        "hard_Mn",
        "hard_Cr",
        "hard_V",
        "hard_Fe",
        "other_3d",
        "4d_5d",
        "lanthanide",
        "mixed",
    ]
    sub = grouped[grouped["segment"].isin(wanted)].copy()
    pivot = sub.pivot_table(
        index=["segment", "n_materials"],
        columns="model",
        values="median_abs_error_eV_atom",
        aggfunc="first",
    ).reset_index()
    for model in MODEL_KEYS:
        if model in pivot:
            pivot[model] = pivot[model].map(lambda value: round(float(value), 5) if pd.notna(value) else value)
    return pivot


def write_report(
    args: argparse.Namespace,
    summary: dict[str, Any],
    grouped: pd.DataFrame,
    pairwise: pd.DataFrame,
    caveats: pd.DataFrame,
) -> Path:
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    path = args.docs_dir / "v3_phase_E_wbm_validation_results_20260608.md"
    ref = summary["reference_sanity"]
    group_table = compact_group_table(grouped)
    pair_table = pairwise[
        (pairwise["segment"].isin(["all_clean_metal", "magnetic_3d_hard"]))
        & (pairwise["pair_group"].isin(["cross_family_excl_orb_sevennet", "frontier4", "all5", "orb_sevennet"]))
    ][
        [
            "segment",
            "pair_group",
            "n_materials",
            "pooled_pair_median_eV_atom",
            "per_material_width_median_eV_atom",
            "floor_eV_atom",
            "tight_vs_floor",
        ]
    ].copy()
    for col in ["pooled_pair_median_eV_atom", "per_material_width_median_eV_atom", "floor_eV_atom"]:
        pair_table[col] = pair_table[col].map(lambda value: round(float(value), 6) if pd.notna(value) else value)

    text = "\n".join(
        [
            "# MatAlign v3 Phase E1 WBM Validation Results",
            "",
            "## Summary",
            "",
            f"- WBM rows: `{summary['wbm_rows']}`.",
            f"- Clean-metal E1 rows: `{summary['clean_metal_rows']}`.",
            "- Main reference: `wbm_ref_mp2020_e_form = e_form_per_atom_mp2020_corrected`.",
            "- `e_form_dft` is used only for sanity checking, not for the main error metric.",
            f"- Reference sanity median |e_form_dft - mp2020_ref|: `{ref['median_abs_delta_eV_atom']:.6g}` eV/atom.",
            f"- Go/no-go: `{summary['go_nogo']['go_level']}`; go_e2=`{summary['go_nogo']['go_e2']}`; recommended_e2_n=`{summary['go_nogo']['recommended_e2_n']}`.",
            f"- Go reason: {summary['go_nogo']['go_reason']}",
            "",
            "## Grouped Model Errors",
            "",
            "Median absolute error against `wbm_ref_mp2020_e_form`; this is an MP2020/WBM-mouth validation metric, not saturation evidence.",
            "",
            group_table.to_markdown(index=False),
            "",
            "## Model-Model Consistency vs Floor",
            "",
            "The `0.011 eV/atom` floor is imported from D3fix as an E1 routing heuristic only.",
            "",
            pair_table.to_markdown(index=False),
            "",
            "## OOD Caveats",
            "",
            caveats[["model", "training_set", "trained_for_benchmark", "openness", "ood_confidence", "prediction_coverage_rate", "leakage_caveat"]].to_markdown(index=False),
            "",
            "## Interpretation Rules",
            "",
            "- Do not claim WBM E1 proves saturation; WBM reference and most models share MP-like training/evaluation口径.",
            "- Edge cases require the model-model consistency gate; if models already scatter on WBM, E2 should not consume weeks of DFT.",
            "- ORB v3 and SevenNet-MF-ompa have lower OOD confidence because large OMat24/sAlex/Alexandria-style sources may contain WBM near-duplicates.",
            "- If E2 is launched, r2SCAN needs its own elemental reference energies so PBE to r2SCAN systematic offsets are not mislabeled as model error.",
        ]
    )
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame, meta = load_wbm_frame(args)
    frame = add_material_tags(frame, args.max_sites)
    clean = frame[frame["is_clean_metal_e1"]].copy()

    material_cols = [
        "material_id",
        "formula",
        "elements",
        "n_sites",
        "n_sites_bin",
        "n_elements",
        "wbm_ref_mp2020_e_form",
        "e_form_dft",
        "ref_delta_e_form_dft_minus_mp2020",
        "dft_ehull",
        "e_above_hull_mp2020_corrected_ppd_mp",
        "has_anion_or_H",
        "is_hubbard",
        "is_clean_metal_e1",
        "main_group_metal",
        "has_magnetic_3d_hard",
        "has_other_3d",
        "has_4d_5d",
        "has_lanthanide",
        "mixed",
        *MODEL_KEYS,
        *[f"{model}_abs_error_mp2020" for model in MODEL_KEYS],
    ]
    clean[material_cols].to_parquet(args.out_dir / "wbm_e1_clean_metal_materials.parquet", index=False)
    clean[material_cols].head(1000).to_csv(args.out_dir / "wbm_e1_clean_metal_materials_preview.csv", index=False)

    grouped = grouped_model_errors(frame, args.floor)
    grouped.to_csv(args.out_dir / "wbm_e1_grouped_model_errors.csv", index=False)
    pairwise = pairwise_summary(frame, args.floor)
    pairwise.to_csv(args.out_dir / "wbm_e1_pairwise_vs_floor.csv", index=False)
    caveats = ood_caveats(meta, frame)
    caveats.to_csv(args.out_dir / "wbm_e1_ood_caveats.csv", index=False)

    ref_delta = finite(clean["ref_delta_e_form_dft_minus_mp2020"].abs())
    go_nogo = decide_go_nogo(grouped, pairwise, args.floor)
    summary = {
        "wbm_rows": int(len(frame)),
        "clean_metal_rows": int(len(clean)),
        "reference_column": "wbm_ref_mp2020_e_form",
        "reference_source_column": "e_form_per_atom_mp2020_corrected",
        "non_main_reference_column": "e_form_dft",
        "reference_sanity": {
            "median_abs_delta_eV_atom": float(np.median(ref_delta)) if len(ref_delta) else math.nan,
            "p95_abs_delta_eV_atom": float(np.quantile(ref_delta, 0.95)) if len(ref_delta) else math.nan,
            "max_abs_delta_eV_atom": float(np.max(ref_delta)) if len(ref_delta) else math.nan,
        },
        "model_prediction_coverage": {
            model: {
                "n_predictions": int(clean[model].notna().sum()),
                "coverage_rate": float(clean[model].notna().mean()) if len(clean) else math.nan,
            }
            for model in MODEL_KEYS
        },
        "hubbard_rule": "MP +U proxy = plus-U element with O/F; E1 clean-metal excludes O/F/H and therefore main subset is non-Hubbard.",
        "go_nogo": go_nogo,
    }
    (args.out_dir / "wbm_e1_go_nogo.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = write_report(args, summary, grouped, pairwise, caveats)
    print(json.dumps({**summary, "report_path": str(report), "output_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
