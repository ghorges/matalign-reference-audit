from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mp_api.client import MPRester
from scipy.stats import spearmanr


DATA_ROOT = (Path(__file__).resolve().parents[2] / "data").resolve()
VALIDATION_DIR = DATA_ROOT / "processed" / "uniform_pbe_validation"
OUT_DIR = DATA_ROOT / "processed" / "clean_reference_analysis"
LOCAL_RESULTS_ROOT = Path("vasp_uniform_pbe_work") / "remote_results" / "primary"
DOCS_DIR = Path("docs")

ANION_ELEMENTS = {"O", "N", "S", "Se", "Cl", "F", "Br", "I", "Te", "P", "As"}
MAG_RE = re.compile(r"mag=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MatAlign v3 analysis master tables.")
    parser.add_argument("--validation-dir", type=Path, default=VALIDATION_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--local-results-root", type=Path, default=LOCAL_RESULTS_ROOT)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--api-key-env", default="MP_API_KEY")
    parser.add_argument("--skip-mp-magnetism", action="store_true")
    return parser.parse_args()


def element_set(value: Any) -> set[str]:
    if pd.isna(value):
        return set()
    return {token for token in str(value).split() if token}


def read_static_mag(results_root: Path, job_id: str) -> float:
    oszicar = results_root / "inputs" / "primary" / "compounds" / job_id / "static" / "OSZICAR"
    if not oszicar.exists():
        return math.nan
    matches = MAG_RE.findall(oszicar.read_text(encoding="utf-8", errors="ignore"))
    return float(matches[-1]) if matches else math.nan


def load_mp_magnetism(material_ids: list[str], api_key: str | None) -> pd.DataFrame:
    fields = [
        "material_id",
        "ordering",
        "total_magnetization",
        "total_magnetization_normalized_formula_units",
        "total_magnetization_normalized_vol",
        "num_magnetic_sites",
        "num_unique_magnetic_sites",
    ]
    empty = pd.DataFrame(columns=["material_id", *fields[1:], "mp_magnetism_status"])
    if not api_key or not material_ids:
        empty["mp_magnetism_status"] = []
        return empty
    rows: list[dict[str, Any]] = []
    try:
        with MPRester(api_key, mute_progress_bars=True) as mpr:
            docs = mpr.materials.magnetism.search(
                material_ids=material_ids,
                fields=fields,
                all_fields=False,
            )
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "material_id": material_id,
                    "mp_magnetism_status": f"mp_api_failed:{type(exc).__name__}:{exc}",
                }
                for material_id in material_ids
            ]
        )
    for doc in docs:
        payload = doc.model_dump() if hasattr(doc, "model_dump") else doc.dict() if hasattr(doc, "dict") else dict(doc)
        payload["material_id"] = str(payload.get("material_id", ""))
        payload["mp_magnetism_status"] = "ok"
        rows.append(payload)
    by_id = {row["material_id"]: row for row in rows}
    for material_id in material_ids:
        if material_id not in by_id:
            rows.append({"material_id": material_id, "mp_magnetism_status": "not_returned"})
    return pd.DataFrame(rows)


def metric(values: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"n": 0, "median": None, "mean": None, "p25": None, "p75": None}
    return {
        "n": int(len(clean)),
        "median": float(clean.median()),
        "mean": float(clean.mean()),
        "p25": float(clean.quantile(0.25)),
        "p75": float(clean.quantile(0.75)),
    }


def spearman_payload(x: pd.Series, y: pd.Series) -> dict[str, float | int | None]:
    pairs = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pairs) < 3 or pairs["x"].nunique() < 2 or pairs["y"].nunique() < 2:
        return {"n": int(len(pairs)), "rho": None, "pvalue": None}
    rho, pvalue = spearmanr(pairs["x"], pairs["y"])
    return {"n": int(len(pairs)), "rho": float(rho), "pvalue": float(pvalue)}


def build_element_offsets(validation_dir: Path, out_dir: Path) -> pd.DataFrame:
    raw_refs = pd.read_csv(validation_dir / "pbe_reference_energies.csv")
    mp_refs = pd.read_csv(validation_dir / "mp2020_element_references.csv")
    offsets = raw_refs.merge(mp_refs, on="element", how="left", suffixes=("_raw", "_mp"))
    offsets["B_i_raw_minus_mp_eV_atom"] = (
        offsets["reference_energy_per_atom_eV"] - offsets["energy_per_atom"]
    )
    offsets["abs_B_i_eV_atom"] = offsets["B_i_raw_minus_mp_eV_atom"].abs()
    offsets["reference_scope"] = offsets["element"].map(lambda el: "gas_raw" if el in {"H", "N", "O", "F", "Cl"} else "solid")
    cols = [
        "element",
        "reference_job_id",
        "reference_scope",
        "reference_energy_per_atom_eV",
        "energy_per_atom",
        "B_i_raw_minus_mp_eV_atom",
        "abs_B_i_eV_atom",
        "entry_id",
        "formula",
        "run_type",
    ]
    offsets[cols].sort_values("abs_B_i_eV_atom", ascending=False).to_csv(
        out_dir / "element_reference_offsets_Bi.csv", index=False
    )
    return offsets


def weighted_b_offset(row: pd.Series, b_map: dict[str, float]) -> float:
    comp = json.loads(row["composition_json"]) if "composition_json" in row else None
    if comp is None:
        comp = {el: 1.0 for el in element_set(row["elements_str"])}
    denom = sum(float(v) for v in comp.values())
    if not denom:
        return math.nan
    missing = set(comp) - set(b_map)
    if missing:
        return math.nan
    return sum(float(amount) * b_map[element] for element, amount in comp.items()) / denom


def markdown_table(frame: pd.DataFrame, n: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.head(n).to_markdown(index=False)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(args.validation_dir / "pbe_primary_validation_results.csv")
    totals = pd.read_csv(Path("vasp_uniform_pbe_work") / "analysis" / "primary_full" / "pbe_total_energies.csv")
    totals_comp = totals[totals["task_kind"] == "compound"][["job_id", "composition_json"]].copy()
    master = master.merge(totals_comp, left_on="pbe_job_id", right_on="job_id", how="left", suffixes=("", "_composition"))

    master["element_set"] = master["elements_str"].map(lambda value: " ".join(sorted(element_set(value))))
    master["has_anion_protocol_sensitive_element"] = master["elements_str"].map(
        lambda value: bool(element_set(value) & ANION_ELEMENTS)
    )
    master["has_hydrogen"] = master["elements_str"].map(lambda value: "H" in element_set(value))
    master["clean_intermetallic_universe"] = (
        ~master["has_anion_protocol_sensitive_element"]
        & ~master["has_hydrogen"]
        & ~master["is_hubbard"].astype(bool)
    )
    master["regime"] = np.select(
        [
            master["has_hydrogen"],
            master["clean_intermetallic_universe"],
        ],
        [
            "hydride",
            "clean_intermetallic",
        ],
        default="anion_or_U_bridge",
    )
    master["magnetic_heusler_review"] = (
        master["clean_intermetallic_universe"]
        & master["elements_str"].str.contains("Mn", regex=False, na=False)
        & (master["abs_raw_residual_vs_mp_eV_atom"] > 0.15)
        & (master["abs_mp2020_residual_vs_mp_eV_atom"] > 0.15)
    )
    master["review_flag"] = np.where(master["magnetic_heusler_review"], "magnetic_heusler_review", "")
    master["raw_residual_within_ef_std"] = master["abs_raw_residual_vs_mp_eV_atom"] <= master["Ef_std"]
    master["mp2020_residual_within_ef_std"] = master["abs_mp2020_residual_vs_mp_eV_atom"] <= master["Ef_std"]
    master["vasp_static_total_magnetization_muB"] = master["pbe_job_id"].map(
        lambda job_id: read_static_mag(args.local_results_root, str(job_id))
    )

    offsets = build_element_offsets(args.validation_dir, args.out_dir)
    b_map = dict(zip(offsets["element"], offsets["B_i_raw_minus_mp_eV_atom"], strict=False))
    master["weighted_B_i_raw_minus_mp_eV_atom"] = master.apply(lambda row: weighted_b_offset(row, b_map), axis=1)
    master["mp2020_minus_raw_minus_weighted_B_i_eV_atom"] = (
        master["mp2020_minus_raw_formation_eV_atom"] - master["weighted_B_i_raw_minus_mp_eV_atom"]
    )

    magnetic = master[master["magnetic_heusler_review"]].copy()
    if not args.skip_mp_magnetism:
        api_key = os.environ.get(args.api_key_env)
        mp_mag = load_mp_magnetism(sorted(magnetic["id_MP"].dropna().astype(str).unique()), api_key)
        if not mp_mag.empty:
            magnetic = magnetic.merge(mp_mag, left_on="id_MP", right_on="material_id", how="left")
    magnetic["magnetism_review_recommendation"] = np.where(
        magnetic["vasp_static_total_magnetization_muB"].abs() < 0.5,
        "check_mp_magnetic_state_or_exclude_from_keystone",
        "magnetic_state_nonzero; compare_with_mp_magnetism",
    )

    clean = master[master["clean_intermetallic_universe"]].copy()
    clean_excluding_magnetic_review = clean[~clean["magnetic_heusler_review"]].copy()

    summary = {
        "rows_total": int(len(master)),
        "regime_counts": {str(k): int(v) for k, v in master["regime"].value_counts().sort_index().items()},
        "clean_intermetallic_universe_rows": int(len(clean)),
        "magnetic_heusler_review_rows": int(master["magnetic_heusler_review"].sum()),
        "raw_abs_residual_clean": metric(clean["abs_raw_residual_vs_mp_eV_atom"]),
        "ef_std_clean": metric(clean["Ef_std"]),
        "raw_within_ef_std_clean_count": int(clean["raw_residual_within_ef_std"].sum()),
        "raw_within_ef_std_clean_rate": float(clean["raw_residual_within_ef_std"].mean()),
        "spearman_abs_raw_vs_ef_std_clean": spearman_payload(clean["abs_raw_residual_vs_mp_eV_atom"], clean["Ef_std"]),
        "spearman_abs_raw_vs_ef_std_all": spearman_payload(
            master["abs_raw_residual_vs_mp_eV_atom"], master["Ef_std"]
        ),
        "clean_excluding_magnetic_review": {
            "rows": int(len(clean_excluding_magnetic_review)),
            "raw_abs_residual": metric(clean_excluding_magnetic_review["abs_raw_residual_vs_mp_eV_atom"]),
            "ef_std": metric(clean_excluding_magnetic_review["Ef_std"]),
            "spearman_abs_raw_vs_ef_std": spearman_payload(
                clean_excluding_magnetic_review["abs_raw_residual_vs_mp_eV_atom"],
                clean_excluding_magnetic_review["Ef_std"],
            ),
        },
    }

    master.to_csv(args.out_dir / "analysis_master.csv", index=False)
    master.to_parquet(args.out_dir / "analysis_master.parquet", index=False)
    clean.to_csv(args.out_dir / "clean_intermetallic_master.csv", index=False)
    clean.to_parquet(args.out_dir / "clean_intermetallic_master.parquet", index=False)
    magnetic.to_csv(args.out_dir / "magnetic_heusler_review.csv", index=False)
    (args.out_dir / "clean_reference_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    group_rows = []
    for keys, group in clean.groupby(["noise_bin", "validation_role"], dropna=False):
        group_rows.append(
            {
                "noise_bin": keys[0],
                "validation_role": keys[1],
                "n": int(len(group)),
                "raw_abs_median_eV_atom": float(group["abs_raw_residual_vs_mp_eV_atom"].median()),
                "ef_std_median_eV_atom": float(group["Ef_std"].median()),
                "within_ef_std_rate": float(group["raw_residual_within_ef_std"].mean()),
            }
        )
    pd.DataFrame(group_rows).to_csv(args.out_dir / "clean_noise_floor_group_metrics.csv", index=False)

    report = [
        "# MatAlign v3 Analysis Phase A-C Results - 2026-06-06",
        "",
        "## Gates",
        "",
        f"- Total rows: `{summary['rows_total']}`",
        f"- Clean intermetallic universe: `{summary['clean_intermetallic_universe_rows']}`",
        f"- Magnetic Heusler review rows: `{summary['magnetic_heusler_review_rows']}`",
        f"- Clean raw median abs residual: `{summary['raw_abs_residual_clean']['median']:.6f} eV/atom`",
        f"- Clean Ef_std median: `{summary['ef_std_clean']['median']:.6f} eV/atom`",
        f"- Clean raw within Ef_std: `{summary['raw_within_ef_std_clean_count']}/{summary['clean_intermetallic_universe_rows']}`",
        f"- Clean Spearman abs(raw residual) vs Ef_std: `{summary['spearman_abs_raw_vs_ef_std_clean']['rho']:.6f}`",
        f"- All-row Spearman abs(raw residual) vs Ef_std: `{summary['spearman_abs_raw_vs_ef_std_all']['rho']:.6f}`",
        "",
        "## Clean Group Metrics",
        "",
        markdown_table(pd.DataFrame(group_rows), 20),
        "",
        "## Magnetic Heusler Review",
        "",
        markdown_table(
            magnetic[
                [
                    "pbe_job_id",
                    "formula",
                    "id_MP",
                    "noise_bin",
                    "abs_raw_residual_vs_mp_eV_atom",
                    "abs_mp2020_residual_vs_mp_eV_atom",
                    "Ef_std",
                    "vasp_static_total_magnetization_muB",
                    "ordering",
                    "total_magnetization",
                    "total_magnetization_normalized_formula_units",
                    "mp_magnetism_status",
                    "magnetism_review_recommendation",
                ]
                if "mp_magnetism_status" in magnetic.columns
                else [
                    "pbe_job_id",
                    "formula",
                    "id_MP",
                    "noise_bin",
                    "abs_raw_residual_vs_mp_eV_atom",
                    "abs_mp2020_residual_vs_mp_eV_atom",
                    "Ef_std",
                    "vasp_static_total_magnetization_muB",
                    "magnetism_review_recommendation",
                ]
            ],
            20,
        ),
        "",
        "## Interpretation Boundary",
        "",
        "- Q1 noise-floor claims are restricted to `clean_intermetallic_universe` and raw PBE residuals.",
        "- `magnetic_heusler_review` is a review flag inside the clean universe, not a different top-level regime; this preserves the planned `n=221` keystone universe while enabling exclusion sensitivity.",
        "- MP2020 closeness to MP is treated as constructive consistency, not independent accuracy evidence.",
    ]
    (args.docs_dir / "clean_reference_analysis_results_20260606.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
