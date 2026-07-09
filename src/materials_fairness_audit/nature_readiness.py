from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .quality_reports import categorize_formula_order


SOURCE_LABELS = {
    "mp": "MP",
    "jarvis": "JARVIS",
    "oqmd": "OQMD",
    "aflow": "AFLOW",
}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
    return []


def _source_values(row: pd.Series, metric_prefix: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for source, label in SOURCE_LABELS.items():
        value = row.get(f"{metric_prefix}_{label}")
        if pd.notna(value):
            values[source] = float(value)
    return values


def build_formula_uncertainty(matalign: pd.DataFrame) -> pd.DataFrame:
    """Aggregate MatAlign uncertainty by reduced formula for WBM-level joins."""

    required = {"reduced_formula", "matalign_id", "n_databases", "Ef_std", "Eg_std"}
    missing = sorted(required - set(matalign.columns))
    if missing:
        raise ValueError(f"MatAlign table is missing required columns: {missing}")

    grouped = matalign.groupby("reduced_formula", dropna=True)
    summary = grouped.agg(
        matalign_matches=("matalign_id", "count"),
        max_n_databases=("n_databases", "max"),
        ef_std_median=("Ef_std", "median"),
        ef_std_mean=("Ef_std", "mean"),
        ef_std_p90=("Ef_std", lambda values: float(pd.Series(values).quantile(0.90))),
        eg_std_median=("Eg_std", "median"),
        eg_std_mean=("Eg_std", "mean"),
        eg_std_p90=("Eg_std", lambda values: float(pd.Series(values).quantile(0.90))),
    )
    return summary.reset_index().sort_values(
        ["matalign_matches", "max_n_databases", "reduced_formula"],
        ascending=[False, False, True],
    )


def build_leave_one_source_errors(matalign: pd.DataFrame) -> pd.DataFrame:
    """Estimate cross-source predictability by holding out one database at a time.

    For each aligned material with at least three databases, the held-out source
    value is compared against the mean of the remaining sources. This is not a
    model benchmark; it is a data-resource validation probe for cross-database
    consistency.
    """

    rows: list[dict[str, Any]] = []
    for row in matalign.itertuples(index=False):
        series = pd.Series(row._asdict())
        elements = _as_list(series.get("elements"))
        formula_order = categorize_formula_order(elements)
        for metric_prefix, metric_name in (("Ef", "formation_energy"), ("Eg", "band_gap")):
            values = _source_values(series, metric_prefix)
            if len(values) < 3:
                continue
            source_std = series.get(f"{metric_prefix}_std")
            for heldout_source, heldout_value in values.items():
                train_values = [
                    value for source, value in values.items() if source != heldout_source
                ]
                if len(train_values) < 2:
                    continue
                predicted = float(np.mean(train_values))
                error = heldout_value - predicted
                rows.append(
                    {
                        "matalign_id": series["matalign_id"],
                        "reduced_formula": series["reduced_formula"],
                        "spacegroup_number": series["spacegroup_number"],
                        "metric": metric_name,
                        "heldout_source": heldout_source,
                        "n_train_sources": len(train_values),
                        "n_databases": int(series["n_databases"]),
                        "formula_order": formula_order,
                        "heldout_value": heldout_value,
                        "consensus_prediction": predicted,
                        "error": error,
                        "abs_error": abs(error),
                        "source_std": float(source_std) if pd.notna(source_std) else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def summarize_leave_one_errors(errors: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if errors.empty:
        return pd.DataFrame(columns=[*group_columns, "n_rows"])

    def rmse(values: pd.Series) -> float:
        return float(np.sqrt(np.mean(np.square(values.astype(float)))))

    summary = (
        errors.groupby(group_columns, dropna=False)
        .agg(
            n_rows=("abs_error", "size"),
            mean_abs_error=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
            p90_abs_error=("abs_error", lambda values: float(values.quantile(0.90))),
            p95_abs_error=("abs_error", lambda values: float(values.quantile(0.95))),
            bias_mean=("error", "mean"),
            rmse=("error", rmse),
            source_std_median=("source_std", "median"),
        )
        .reset_index()
    )
    return summary.sort_values(group_columns).reset_index(drop=True)


def attach_uncertainty_bins(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    bins = pd.Series(pd.NA, index=frame.index, dtype="object")
    valid = values.notna()
    if not valid.any():
        return bins
    ranks = values.loc[valid].rank(pct=True, method="first")
    bins.loc[valid & (ranks <= 1 / 3)] = "low"
    bins.loc[valid & (ranks > 1 / 3) & (ranks <= 2 / 3)] = "mid"
    bins.loc[valid & (ranks > 2 / 3)] = "high"
    return bins


def build_uncertainty_aware_leaderboard(
    *,
    wbm: pd.DataFrame,
    per_material: pd.DataFrame,
    formula_uncertainty: pd.DataFrame,
    model_keys: list[str],
    uncertainty_floor: float = 0.025,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_wbm = {"material_id", "e_form_dft", *model_keys}
    missing_wbm = sorted(required_wbm - set(wbm.columns))
    if missing_wbm:
        raise ValueError(f"WBM table is missing required columns: {missing_wbm}")
    required_per_material = {"material_id", "reduced_formula"}
    missing_per_material = sorted(required_per_material - set(per_material.columns))
    if missing_per_material:
        raise ValueError(f"Composition-distance table is missing columns: {missing_per_material}")

    material_features = per_material[
        [
            column
            for column in (
                "material_id",
                "reduced_formula",
                "nearest_train_distance",
                "seen_in_training_formula",
            )
            if column in per_material.columns
        ]
    ].drop_duplicates("material_id")
    merged = wbm[["material_id", "e_form_dft", *model_keys]].merge(
        material_features,
        on="material_id",
        how="left",
    )
    merged = merged.merge(formula_uncertainty, on="reduced_formula", how="left")
    merged["has_matalign_uncertainty"] = merged["ef_std_median"].notna()
    merged["ef_uncertainty_bin"] = attach_uncertainty_bins(merged, "ef_std_median")

    target = pd.to_numeric(merged["e_form_dft"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for model_key in model_keys:
        prediction = pd.to_numeric(merged[model_key], errors="coerce")
        valid = target.notna() & prediction.notna()
        abs_error = (prediction - target).abs()
        matched = valid & merged["has_matalign_uncertainty"]
        tolerance = pd.to_numeric(merged["ef_std_median"], errors="coerce").clip(lower=uncertainty_floor)

        low = matched & (merged["ef_uncertainty_bin"] == "low")
        mid = matched & (merged["ef_uncertainty_bin"] == "mid")
        high = matched & (merged["ef_uncertainty_bin"] == "high")

        matched_errors = abs_error.loc[matched]
        matched_tolerance = tolerance.loc[matched]
        row = {
            "model_key": model_key,
            "n_total": int(valid.sum()),
            "n_uncertainty_matched": int(matched.sum()),
            "uncertainty_coverage": float(matched.sum() / valid.sum()) if valid.sum() else math.nan,
            "standard_mae_all": float(abs_error.loc[valid].mean()) if valid.any() else math.nan,
            "standard_mae_matched": float(matched_errors.mean())
            if not matched_errors.empty
            else math.nan,
            "excess_mae_matched": float(
                (matched_errors - matched_tolerance).clip(lower=0).mean()
            )
            if not matched_errors.empty
            else math.nan,
            "uncertainty_normalized_mae": float((matched_errors / matched_tolerance).mean())
            if not matched_errors.empty
            else math.nan,
            "tolerance_hit_rate": float((matched_errors <= matched_tolerance).mean())
            if not matched_errors.empty
            else math.nan,
            "mae_low_uncertainty": float(abs_error.loc[low].mean()) if low.any() else math.nan,
            "mae_mid_uncertainty": float(abs_error.loc[mid].mean()) if mid.any() else math.nan,
            "mae_high_uncertainty": float(abs_error.loc[high].mean()) if high.any() else math.nan,
        }
        row["high_minus_low_mae"] = row["mae_high_uncertainty"] - row["mae_low_uncertainty"]
        rows.append(row)

    leaderboard = pd.DataFrame(rows)
    for metric, rank_column in (
        ("standard_mae_all", "standard_rank_all"),
        ("standard_mae_matched", "standard_rank_matched"),
        ("excess_mae_matched", "excess_rank"),
        ("uncertainty_normalized_mae", "normalized_rank"),
    ):
        leaderboard[rank_column] = leaderboard[metric].rank(method="min").astype("Int64")
    leaderboard["rank_shift_excess_vs_standard"] = (
        leaderboard["excess_rank"] - leaderboard["standard_rank_all"]
    ).astype("Int64")
    leaderboard["rank_shift_normalized_vs_standard"] = (
        leaderboard["normalized_rank"] - leaderboard["standard_rank_all"]
    ).astype("Int64")

    matched_materials = int(merged["has_matalign_uncertainty"].sum())
    summary = {
        "n_models": int(len(model_keys)),
        "n_materials": int(len(merged)),
        "n_materials_with_matalign_uncertainty": matched_materials,
        "uncertainty_coverage": float(matched_materials / len(merged)) if len(merged) else 0.0,
        "uncertainty_floor_eV": uncertainty_floor,
        "standard_vs_excess_rank_spearman": float(
            leaderboard["standard_rank_all"].corr(leaderboard["excess_rank"], method="spearman")
        ),
        "standard_vs_normalized_rank_spearman": float(
            leaderboard["standard_rank_all"].corr(
                leaderboard["normalized_rank"],
                method="spearman",
            )
        ),
        "mean_abs_rank_shift_excess": float(
            leaderboard["rank_shift_excess_vs_standard"].abs().mean()
        ),
        "mean_abs_rank_shift_normalized": float(
            leaderboard["rank_shift_normalized_vs_standard"].abs().mean()
        ),
        "top5_standard": leaderboard.sort_values("standard_rank_all")["model_key"]
        .head(5)
        .tolist(),
        "top5_excess": leaderboard.sort_values("excess_rank")["model_key"].head(5).tolist(),
        "top5_normalized": leaderboard.sort_values("normalized_rank")["model_key"]
        .head(5)
        .tolist(),
    }
    return leaderboard.sort_values("standard_rank_all").reset_index(drop=True), summary


def audit_publication_readiness(export_dir: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add_check(check_id: str, passed: bool, severity: str, detail: str) -> None:
        checks.append(
            {
                "check_id": check_id,
                "passed": bool(passed),
                "severity": severity,
                "detail": detail,
            }
        )

    croissant_path = export_dir / "matalign.croissant.json"
    datasheet_path = export_dir / "datasheet.md"
    reproducibility_path = export_dir / "reproducibility.md"
    artifact_path = export_dir / "release_artifacts.csv"

    add_check("croissant_exists", croissant_path.exists(), "blocker", str(croissant_path))
    add_check("datasheet_exists", datasheet_path.exists(), "blocker", str(datasheet_path))
    add_check("reproducibility_exists", reproducibility_path.exists(), "major", str(reproducibility_path))
    add_check("artifact_manifest_exists", artifact_path.exists(), "blocker", str(artifact_path))

    if croissant_path.exists():
        croissant = json.loads(croissant_path.read_text(encoding="utf-8"))
        distributions = croissant.get("distribution", [])
        content_urls = [str(item.get("contentUrl", "")) for item in distributions]
        local_urls = [
            url for url in content_urls if re.match(r"^[A-Za-z]:\\", url) or url.startswith("/")
        ]
        add_check(
            "croissant_no_local_content_urls",
            not local_urls,
            "blocker",
            f"{len(local_urls)} local contentUrl values found",
        )
        license_text = str(croissant.get("license", "")).strip().lower()
        vague_license = not license_text or "see upstream" in license_text
        add_check(
            "croissant_specific_license",
            not vague_license,
            "blocker",
            f"license={croissant.get('license', '')!r}",
        )
        has_record_sets = bool(croissant.get("recordSet") or croissant.get("recordSets"))
        add_check(
            "croissant_has_record_sets",
            has_record_sets,
            "major",
            "Croissant should expose record-level schemas before public release",
        )

    if artifact_path.exists():
        artifacts = pd.read_csv(artifact_path)
        has_sha256 = "sha256" in artifacts.columns and artifacts["sha256"].notna().all()
        add_check(
            "artifacts_have_checksums",
            has_sha256,
            "major",
            "release_artifacts.csv should include SHA256 checksums for citable release files",
        )

    if reproducibility_path.exists():
        reproducibility = reproducibility_path.read_text(encoding="utf-8").lower()
        mentions_aflow = "phase1_download_aflow" in reproducibility or "aflow" in reproducibility
        add_check(
            "reproducibility_mentions_aflow_raw_collection",
            mentions_aflow,
            "major",
            "AFLOW full raw collection/cached input should be documented for reviewers",
        )

    failed_blockers = [check for check in checks if not check["passed"] and check["severity"] == "blocker"]
    failed_major = [check for check in checks if not check["passed"] and check["severity"] == "major"]
    status = "blocked" if failed_blockers else "needs_major_cleanup" if failed_major else "ready"
    return {
        "status": status,
        "failed_blocker_count": len(failed_blockers),
        "failed_major_count": len(failed_major),
        "checks": checks,
        "recommended_next_actions": [
            "Replace local Windows contentUrl values with relative release paths or public DOI URLs.",
            "Write explicit provenance and license terms for each upstream database-derived artifact.",
            "Add recordSet/field schemas and SHA256 checksums to the Croissant/release metadata.",
            "Document AFLOW full raw collection and cached-data assumptions in reproducibility notes.",
        ],
    }


def build_markdown_summary(
    *,
    leave_one_source: pd.DataFrame,
    leaderboard_summary: dict[str, Any],
    publication_report: dict[str, Any],
) -> str:
    ef = leave_one_source.loc[leave_one_source["metric"] == "formation_energy"]
    eg = leave_one_source.loc[leave_one_source["metric"] == "band_gap"]
    ef_mae = float(ef["mean_abs_error"].mean()) if not ef.empty else math.nan
    eg_mae = float(eg["mean_abs_error"].mean()) if not eg.empty else math.nan
    return (
        "# Nature-Level Data Readiness Addendum\n\n"
        "This addendum records data-layer checks added after the initial MatAlign V2 draft.\n\n"
        "## Leave-One-Database Validation\n\n"
        f"- Mean source-level formation-energy holdout MAE: `{ef_mae:.4f}` eV/atom\n"
        f"- Mean source-level band-gap holdout MAE: `{eg_mae:.4f}` eV\n"
        "- Interpretation: this measures whether remaining databases predict a held-out "
        "source for the same aligned material. It is a data-resource validation probe, "
        "not an ML model benchmark.\n\n"
        "## Uncertainty-Aware Leaderboard\n\n"
        f"- WBM materials with MatAlign uncertainty proxy: "
        f"`{leaderboard_summary['n_materials_with_matalign_uncertainty']:,}` / "
        f"`{leaderboard_summary['n_materials']:,}`\n"
        f"- Standard-vs-excess rank Spearman: "
        f"`{leaderboard_summary['standard_vs_excess_rank_spearman']:.4f}`\n"
        f"- Standard-vs-normalized rank Spearman: "
        f"`{leaderboard_summary['standard_vs_normalized_rank_spearman']:.4f}`\n"
        f"- Mean absolute excess-rank shift: "
        f"`{leaderboard_summary['mean_abs_rank_shift_excess']:.2f}`\n\n"
        "## Publication Readiness\n\n"
        f"- Release metadata status: `{publication_report['status']}`\n"
        f"- Failed blocker checks: `{publication_report['failed_blocker_count']}`\n"
        f"- Failed major checks: `{publication_report['failed_major_count']}`\n"
    )
