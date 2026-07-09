from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Composition


DATA_ROOT = (Path(__file__).resolve().parents[2] / "data").resolve()
OUT_DIR = DATA_ROOT / "processed" / "uniform_pbe_selection"

CHEMISTRY_ORDER = ["main_group", "3d_transition", "4d_5d_transition", "f_block"]
NOISE_ORDER = ["low", "mid", "high"]

TARGET_GRID = {
    ("main_group", "low"): 30,
    ("main_group", "mid"): 30,
    ("main_group", "high"): 30,
    ("3d_transition", "low"): 35,
    ("3d_transition", "mid"): 35,
    ("3d_transition", "high"): 40,
    ("4d_5d_transition", "low"): 30,
    ("4d_5d_transition", "mid"): 35,
    ("4d_5d_transition", "high"): 35,
    ("f_block", "low"): 30,
    ("f_block", "mid"): 35,
    ("f_block", "high"): 40,
}

BACKUP_PER_CELL = 20
R2SCAN_PER_CELL = 5
ACTINIDE_STRETCH_ROWS = 60

LANTHANIDES = set("La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split())
ACTINIDES = set("Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split())
TRANSITION_3D = set("Sc Ti V Cr Mn Fe Co Ni Cu Zn".split())
TRANSITION_4D_5D = set(
    "Y Zr Nb Mo Tc Ru Rh Pd Ag Cd Hf Ta W Re Os Ir Pt Au Hg Rf Db Sg Bh Hs Mt Ds Rg Cn".split()
)
PLUS_U_ELEMENTS = set("V Cr Mn Fe Co Ni Cu Mo W U Np Pu".split())
HIGH_VALUE_HARD_ELEMENTS = set("Pu Np U Mn Fe Cr V N".split())

SOURCE_PAIRS = [
    ("aflow", "jarvis"),
    ("aflow", "mp"),
    ("aflow", "oqmd"),
    ("jarvis", "mp"),
    ("jarvis", "oqmd"),
    ("mp", "oqmd"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a MatAlign v3 PBE validation set from existing MatAlign artifacts."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-sites-primary", type=int, default=80)
    parser.add_argument("--min-databases-primary", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260604)
    return parser.parse_args()


def as_element_list(value: Any) -> list[str]:
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    if isinstance(value, str):
        raw = value.strip().strip("[]")
        if not raw:
            return []
        return [part.strip().strip("'\"") for part in raw.split(",") if part.strip()]
    return []


def nsites_from_jarvis_json(value: Any) -> float:
    if not isinstance(value, str) or not value:
        return math.nan
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return math.nan
    elements = payload.get("elements") or []
    return float(len(elements))


def formula_order(formula: str) -> str:
    try:
        n_elements = len(Composition(formula).elements)
    except Exception:
        return "unknown"
    if n_elements == 1:
        return "elemental"
    if n_elements == 2:
        return "binary"
    if n_elements == 3:
        return "ternary"
    if n_elements == 4:
        return "quaternary"
    return "quinary_plus"


def reduced_formula_natoms(formula: str) -> float:
    try:
        return float(Composition(formula).num_atoms)
    except Exception:
        return math.nan


def chemistry_class(elements: list[str]) -> str:
    element_set = set(elements)
    if element_set & (LANTHANIDES | ACTINIDES):
        return "f_block"
    if element_set & TRANSITION_4D_5D:
        return "4d_5d_transition"
    if element_set & TRANSITION_3D:
        return "3d_transition"
    return "main_group"


def source_id(row: pd.Series, source: str) -> str | None:
    value = row.get(f"id_{source.upper()}")
    if pd.isna(value):
        return None
    return str(value)


def load_pair_lookup(pair_dir: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for left, right in SOURCE_PAIRS:
        path = pair_dir / f"{left}_{right}.csv"
        if not path.exists():
            continue
        cols = [
            "left_source",
            "left_source_id",
            "right_source",
            "right_source_id",
            "match_rule",
            "ambiguity_count",
            "volume_rel_diff",
        ]
        pairs = pd.read_csv(path, usecols=cols)
        for rec in pairs.to_dict("records"):
            key = (
                str(rec["left_source"]),
                str(rec["left_source_id"]),
                str(rec["right_source"]),
                str(rec["right_source_id"]),
            )
            lookup[key] = {
                "match_rule": rec["match_rule"],
                "ambiguity_count": int(rec["ambiguity_count"]),
                "volume_rel_diff": float(rec["volume_rel_diff"]),
            }
    return lookup


def pair_stats(row: pd.Series, lookup: dict[tuple[str, str, str, str], dict[str, Any]]) -> pd.Series:
    records: list[dict[str, Any]] = []
    for left, right in SOURCE_PAIRS:
        left_id = source_id(row, left)
        right_id = source_id(row, right)
        if left_id is None or right_id is None:
            continue
        rec = lookup.get((left, left_id, right, right_id))
        if rec is not None:
            records.append(rec)

    if not records:
        return pd.Series(
            {
                "n_pair_links": 0,
                "n_exact_key_pairs": 0,
                "max_pair_ambiguity": math.nan,
                "max_pair_volume_rel_diff": math.nan,
                "median_pair_volume_rel_diff": math.nan,
            }
        )

    ambiguities = [rec["ambiguity_count"] for rec in records]
    volume_diffs = [rec["volume_rel_diff"] for rec in records if pd.notna(rec["volume_rel_diff"])]
    exact_pairs = sum(1 for rec in records if rec["match_rule"] == "exact_key")
    return pd.Series(
        {
            "n_pair_links": len(records),
            "n_exact_key_pairs": exact_pairs,
            "max_pair_ambiguity": max(ambiguities),
            "max_pair_volume_rel_diff": max(volume_diffs) if volume_diffs else math.nan,
            "median_pair_volume_rel_diff": float(np.median(volume_diffs)) if volume_diffs else math.nan,
        }
    )


def add_element_features(candidates: pd.DataFrame, audit_dir: Path) -> pd.DataFrame:
    stats = pd.read_csv(audit_dir / "training_frequency_vs_error.csv").set_index("symbol")

    def summarize(elements: list[str], col: str, default: float = math.nan) -> float:
        values = [float(stats.loc[sym, col]) for sym in elements if sym in stats.index and pd.notna(stats.loc[sym, col])]
        if not values:
            return default
        return float(np.mean(values))

    def min_summarize(elements: list[str], col: str, default: float = math.nan) -> float:
        values = [float(stats.loc[sym, col]) for sym in elements if sym in stats.index and pd.notna(stats.loc[sym, col])]
        if not values:
            return default
        return float(np.min(values))

    out = candidates.copy()
    out["element_mae_ratio_mean"] = out["element_list"].map(lambda x: summarize(x, "MAE_ratio"))
    out["element_noise_floor_mean"] = out["element_list"].map(lambda x: summarize(x, "ef_std_avg_e"))
    out["element_train_count_min"] = out["element_list"].map(lambda x: min_summarize(x, "n_train_elem"))
    out["element_train_log_mean"] = out["element_list"].map(lambda x: summarize(x, "log_n_train_elem"))
    out["contains_high_value_hard_element"] = out["element_list"].map(
        lambda x: bool(set(x) & HIGH_VALUE_HARD_ELEMENTS)
    )
    out["contains_plus_u_element"] = out["element_list"].map(lambda x: bool(set(x) & PLUS_U_ELEMENTS))
    out["contains_lanthanide"] = out["element_list"].map(lambda x: bool(set(x) & LANTHANIDES))
    out["contains_actinide"] = out["element_list"].map(lambda x: bool(set(x) & ACTINIDES))
    return out


def assign_noise_bins(candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    out = candidates.copy()
    thresholds: dict[str, dict[str, float]] = {}
    out["noise_bin"] = ""
    for chem_class, idx in out.groupby("chemistry_class").groups.items():
        sub = out.loc[idx]
        q33 = float(sub["Ef_std"].quantile(0.33))
        q67 = float(sub["Ef_std"].quantile(0.67))
        thresholds[chem_class] = {"q33": q33, "q67": q67}
        out.loc[idx, "noise_bin"] = np.select(
            [sub["Ef_std"] <= q33, sub["Ef_std"] >= q67],
            ["low", "high"],
            default="mid",
        )
    return out, thresholds


def assign_validation_role(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    difficulty_cut = float(out["element_mae_ratio_mean"].quantile(0.67))
    roles = []
    for row in out.itertuples(index=False):
        if row.noise_bin == "high":
            roles.append("protocol_noise_probe")
        elif row.contains_high_value_hard_element or row.element_mae_ratio_mean >= difficulty_cut:
            roles.append("headroom_probe")
        elif row.noise_bin == "low":
            roles.append("saturation_probe")
        else:
            roles.append("bridge_probe")
    out["validation_role"] = roles
    return out


def priority_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    max_ambiguity = out["max_pair_ambiguity"].fillna(99)
    max_vol_diff = out["max_pair_volume_rel_diff"].fillna(99)
    out["high_confidence_match"] = (
        (out["n_pair_links"] >= (out["n_databases"] - 1).clip(lower=1))
        & (max_ambiguity <= 2)
        & (out["n_exact_key_pairs"] >= 1)
    )
    out["match_confidence_score"] = (
        out["n_databases"] * 5
        + out["n_pair_links"] * 2
        + out["n_exact_key_pairs"]
        - np.log1p(max_ambiguity)
        - np.minimum(max_vol_diff, 1.0)
    )
    out["structure_size_score"] = -out["structure_nsites"]
    out["hard_element_score"] = out["contains_high_value_hard_element"].astype(int)
    return out


def select_grid(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_parts = []
    backup_parts = []
    used_ids: set[str] = set()

    for chem_class in CHEMISTRY_ORDER:
        for noise_bin in NOISE_ORDER:
            target = TARGET_GRID[(chem_class, noise_bin)]
            cell = candidates[
                (candidates["chemistry_class"] == chem_class)
                & (candidates["noise_bin"] == noise_bin)
                & (~candidates["matalign_id"].isin(used_ids))
            ].copy()

            if noise_bin == "high":
                cell["noise_priority"] = cell["Ef_std"]
            elif noise_bin == "low":
                cell["noise_priority"] = -cell["Ef_std"]
            else:
                median = float(cell["Ef_std"].median()) if len(cell) else 0.0
                cell["noise_priority"] = -(cell["Ef_std"] - median).abs()

            cell = cell.sort_values(
                [
                    "high_confidence_match",
                    "n_databases",
                    "match_confidence_score",
                    "hard_element_score",
                    "noise_priority",
                    "structure_size_score",
                    "matalign_id",
                ],
                ascending=[False, False, False, False, False, False, True],
            )
            chosen = cell.head(target).copy()
            chosen["selection_set"] = "primary"
            chosen["selection_stratum"] = f"{chem_class}/{noise_bin}"
            selected_parts.append(chosen)
            used_ids.update(chosen["matalign_id"].tolist())

            backup = cell[~cell["matalign_id"].isin(used_ids)].head(BACKUP_PER_CELL).copy()
            backup["selection_set"] = "backup"
            backup["selection_stratum"] = f"{chem_class}/{noise_bin}"
            backup_parts.append(backup)

    primary = pd.concat(selected_parts, ignore_index=True)
    backup = pd.concat(backup_parts, ignore_index=True)
    primary.insert(0, "pbe_job_id", [f"pbe-primary-{idx:04d}" for idx in range(1, len(primary) + 1)])
    backup.insert(0, "pbe_job_id", [f"pbe-backup-{idx:04d}" for idx in range(1, len(backup) + 1)])
    return primary, backup


def choose_r2scan_subset(primary: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for chem_class in CHEMISTRY_ORDER:
        for noise_bin in NOISE_ORDER:
            cell = primary[
                (primary["chemistry_class"] == chem_class) & (primary["noise_bin"] == noise_bin)
            ].copy()
            if cell.empty:
                continue
            cell = cell.sort_values(
                [
                    "n_databases",
                    "high_confidence_match",
                    "contains_high_value_hard_element",
                    "Ef_std",
                    "structure_nsites",
                ],
                ascending=[False, False, False, False, True],
            )
            parts.append(cell.head(R2SCAN_PER_CELL))
    subset = pd.concat(parts, ignore_index=True)
    subset = subset.drop_duplicates("matalign_id").copy()
    subset.insert(1, "r2scan_job_id", [f"r2scan-validation-{idx:03d}" for idx in range(1, len(subset) + 1)])
    return subset


def choose_actinide_stretch_candidates(
    candidates: pd.DataFrame, primary: pd.DataFrame
) -> pd.DataFrame:
    primary_ids = set(primary["matalign_id"])
    stretch = candidates[candidates["contains_actinide"]].copy()
    if stretch.empty:
        return stretch
    stretch["in_primary"] = stretch["matalign_id"].isin(primary_ids)
    stretch["contains_pu_or_np"] = stretch["element_list"].map(lambda x: bool(set(x) & {"Pu", "Np"}))
    stretch = stretch.sort_values(
        [
            "in_primary",
            "contains_pu_or_np",
            "n_databases",
            "high_confidence_match",
            "Ef_std",
            "structure_nsites",
            "matalign_id",
        ],
        ascending=[True, False, False, False, False, True, True],
    ).head(ACTINIDE_STRETCH_ROWS)
    stretch.insert(
        0,
        "stretch_job_id",
        [f"actinide-stretch-{idx:03d}" for idx in range(1, len(stretch) + 1)],
    )
    stretch.insert(
        1,
        "pbe_job_id",
        [f"pbe-actinide-stretch-{idx:03d}" for idx in range(1, len(stretch) + 1)],
    )
    stretch["selection_set"] = np.where(stretch["in_primary"], "primary", "actinide_stretch")
    stretch["selection_stratum"] = stretch["chemistry_class"] + "/" + stretch["noise_bin"]
    return stretch


def output_columns(include_structure: bool = False) -> list[str]:
    cols = [
        "pbe_job_id",
        "matalign_id",
        "selection_set",
        "selection_stratum",
        "validation_role",
        "chemistry_class",
        "noise_bin",
        "reduced_formula",
        "formula_order",
        "elements_str",
        "n_databases",
        "sources_present_str",
        "spacegroup_number",
        "structure_nsites",
        "reduced_formula_natoms",
        "id_JARVIS",
        "id_MP",
        "id_OQMD",
        "id_AFLOW",
        "Ef_std",
        "Eg_std",
        "Ef_MP",
        "Ef_JARVIS",
        "Ef_OQMD",
        "Ef_AFLOW",
        "Eg_MP",
        "Eg_JARVIS",
        "Eg_OQMD",
        "Eg_AFLOW",
        "element_mae_ratio_mean",
        "element_noise_floor_mean",
        "element_train_count_min",
        "contains_high_value_hard_element",
        "contains_plus_u_element",
        "contains_lanthanide",
        "contains_actinide",
        "high_confidence_match",
        "n_pair_links",
        "n_exact_key_pairs",
        "max_pair_ambiguity",
        "max_pair_volume_rel_diff",
        "median_pair_volume_rel_diff",
    ]
    if include_structure:
        cols.append("structure_json")
    return cols


def element_reference_table(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    counts = Counter()
    for elems in primary["element_list"]:
        counts.update(elems)
    for symbol, count in sorted(counts.items()):
        rows.append(
            {
                "element": symbol,
                "n_primary_materials": count,
                "is_plus_u_candidate": symbol in PLUS_U_ELEMENTS,
                "is_lanthanide": symbol in LANTHANIDES,
                "is_actinide": symbol in ACTINIDES,
                "is_high_value_hard_element": symbol in HIGH_VALUE_HARD_ELEMENTS,
            }
        )
    return pd.DataFrame(rows)


def write_structure_json_files(primary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    struct_dir = out_dir / "jarvis_structure_json"
    struct_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for row in primary.itertuples(index=False):
        path = struct_dir / f"{row.pbe_job_id}_{row.matalign_id}.json"
        path.write_text(row.structure_json, encoding="utf-8")
        manifest_rows.append(
            {
                "pbe_job_id": row.pbe_job_id,
                "matalign_id": row.matalign_id,
                "structure_source": "jarvis",
                "structure_source_id": row.id_JARVIS,
                "structure_json_path": str(path),
            }
        )
    return pd.DataFrame(manifest_rows)


def main() -> None:
    args = parse_args()
    data_root: Path = args.data_root
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    matalign = pd.read_parquet(data_root / "processed" / "matalign" / "matalign_full.parquet")
    jarvis = pd.read_parquet(
        data_root / "processed" / "databases" / "jarvis_data.parquet",
        columns=["source_id", "formula", "structure_json", "has_structure"],
    ).rename(columns={"formula": "jarvis_formula"})

    candidates = matalign[matalign["id_JARVIS"].notna()].merge(
        jarvis, left_on="id_JARVIS", right_on="source_id", how="left"
    )
    candidates["element_list"] = candidates["elements"].map(as_element_list)
    candidates["elements_str"] = candidates["element_list"].map(lambda x: " ".join(x))
    candidates["sources_present_str"] = candidates["sources_present"].map(
        lambda x: " ".join(as_element_list(x))
    )
    candidates["structure_nsites"] = candidates["structure_json"].map(nsites_from_jarvis_json)
    candidates["formula_order"] = candidates["reduced_formula"].map(formula_order)
    candidates["reduced_formula_natoms"] = candidates["reduced_formula"].map(reduced_formula_natoms)
    candidates["chemistry_class"] = candidates["element_list"].map(chemistry_class)

    candidates = candidates[
        (candidates["n_databases"] >= args.min_databases_primary)
        & candidates["Ef_std"].notna()
        & candidates["structure_json"].notna()
        & candidates["structure_nsites"].notna()
        & (candidates["structure_nsites"] <= args.max_sites_primary)
        & (candidates["formula_order"] != "elemental")
    ].copy()

    pair_lookup = load_pair_lookup(data_root / "processed" / "matalign" / "matalign_pairs")
    pair_feature_frame = candidates.apply(pair_stats, axis=1, lookup=pair_lookup)
    candidates = pd.concat([candidates.reset_index(drop=True), pair_feature_frame.reset_index(drop=True)], axis=1)
    candidates = add_element_features(candidates, data_root / "processed" / "audit")
    candidates, noise_thresholds = assign_noise_bins(candidates)
    candidates = assign_validation_role(candidates)
    candidates = priority_frame(candidates)

    primary, backup = select_grid(candidates)
    r2scan = choose_r2scan_subset(primary)
    actinide_stretch = choose_actinide_stretch_candidates(candidates, primary)
    reference = element_reference_table(primary)
    structure_manifest = write_structure_json_files(primary, out_dir)

    primary_csv = primary[output_columns()].copy()
    backup_csv = backup[output_columns()].copy()
    r2scan_csv = r2scan[["r2scan_job_id"] + output_columns()].copy()
    stretch_cols = [
        "stretch_job_id",
        "in_primary",
        "contains_pu_or_np",
        *output_columns(),
    ]
    actinide_stretch_csv = actinide_stretch[stretch_cols].copy()

    primary_csv.to_csv(out_dir / "pbe_validation_primary.csv", index=False)
    backup_csv.to_csv(out_dir / "pbe_validation_backup.csv", index=False)
    r2scan_csv.to_csv(out_dir / "r2scan_validation_subset.csv", index=False)
    actinide_stretch_csv.to_csv(out_dir / "actinide_stretch_candidates.csv", index=False)
    primary[output_columns(include_structure=True)].to_parquet(
        out_dir / "pbe_validation_primary_with_structures.parquet", index=False
    )
    backup[output_columns(include_structure=True)].to_parquet(
        out_dir / "pbe_validation_backup_with_structures.parquet", index=False
    )
    reference.to_csv(out_dir / "element_reference_jobs.csv", index=False)
    structure_manifest.to_csv(out_dir / "structure_manifest.csv", index=False)

    stratum_summary = (
        primary.groupby(["chemistry_class", "noise_bin", "validation_role"], dropna=False)
        .size()
        .reset_index(name="n_primary")
        .sort_values(["chemistry_class", "noise_bin", "validation_role"])
    )
    stratum_summary.to_csv(out_dir / "selection_stratum_summary.csv", index=False)

    summary = {
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "candidate_rows_after_hard_filters": int(len(candidates)),
        "primary_rows": int(len(primary)),
        "backup_rows": int(len(backup)),
        "r2scan_subset_rows": int(len(r2scan)),
        "actinide_stretch_rows": int(len(actinide_stretch)),
        "unique_elements_primary": int(reference["element"].nunique()),
        "primary_n_databases_distribution": {
            str(k): int(v) for k, v in primary["n_databases"].value_counts().sort_index().items()
        },
        "primary_chemistry_class_distribution": {
            str(k): int(v) for k, v in primary["chemistry_class"].value_counts().sort_index().items()
        },
        "primary_noise_bin_distribution": {
            str(k): int(v) for k, v in primary["noise_bin"].value_counts().sort_index().items()
        },
        "primary_validation_role_distribution": {
            str(k): int(v) for k, v in primary["validation_role"].value_counts().sort_index().items()
        },
        "primary_structure_nsites": {
            "median": float(primary["structure_nsites"].median()),
            "p95": float(primary["structure_nsites"].quantile(0.95)),
            "max": float(primary["structure_nsites"].max()),
        },
        "primary_ef_std": {
            "median": float(primary["Ef_std"].median()),
            "p95": float(primary["Ef_std"].quantile(0.95)),
            "min": float(primary["Ef_std"].min()),
            "max": float(primary["Ef_std"].max()),
        },
        "high_confidence_match_share": float(primary["high_confidence_match"].mean()),
        "contains_plus_u_share": float(primary["contains_plus_u_element"].mean()),
        "contains_actinide_count": int(primary["contains_actinide"].sum()),
        "noise_thresholds_by_chemistry_class": noise_thresholds,
        "outputs": {
            "primary_csv": str(out_dir / "pbe_validation_primary.csv"),
            "primary_structures_parquet": str(out_dir / "pbe_validation_primary_with_structures.parquet"),
            "backup_csv": str(out_dir / "pbe_validation_backup.csv"),
            "backup_structures_parquet": str(out_dir / "pbe_validation_backup_with_structures.parquet"),
            "r2scan_subset_csv": str(out_dir / "r2scan_validation_subset.csv"),
            "actinide_stretch_candidates_csv": str(out_dir / "actinide_stretch_candidates.csv"),
            "element_reference_jobs_csv": str(out_dir / "element_reference_jobs.csv"),
            "structure_manifest_csv": str(out_dir / "structure_manifest.csv"),
            "stratum_summary_csv": str(out_dir / "selection_stratum_summary.csv"),
        },
    }
    (out_dir / "pbe_selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
