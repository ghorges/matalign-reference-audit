from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import Settings
from .download import (
    clone_or_update_repo,
    download_file,
    file_is_readable,
    fetch_figshare_article,
    save_figshare_manifest,
    select_article_files,
)
from .io_utils import read_table, write_json, write_table


@dataclass(slots=True)
class ModelMetadataRecord:
    model_family: str
    yaml_path: str
    model_name: str | None
    model_key: str | None
    model_version: str | None
    model_type: str | None
    training_set: str | None
    trained_for_benchmark: bool | None
    openness: str | None
    n_estimators: int | None
    model_params: str | None
    pred_file: str | None
    pred_file_url: str | None
    pred_col: str | None
    pred_file_exists: bool
    full_test_mae: float | None
    full_test_f1: float | None
    full_test_daf: float | None


def _flatten_training_set(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return "; ".join(map(str, value))
    return str(value)


def collect_model_metadata(repo_dir: Path) -> pd.DataFrame:
    records: list[ModelMetadataRecord] = []
    for yml_path in sorted(repo_dir.joinpath("models").rglob("*.yml")):
        payload = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "model_key" not in payload:
            continue
        metrics = payload.get("metrics", {})
        discovery = metrics.get("discovery", {}) if isinstance(metrics, dict) else {}
        if not isinstance(discovery, dict):
            discovery = {}
        full_test = discovery.get("full_test_set", {})
        if not isinstance(full_test, dict):
            full_test = {}
        records.append(
            ModelMetadataRecord(
                model_family=yml_path.parent.name,
                yaml_path=str(yml_path.relative_to(repo_dir)),
                model_name=payload.get("model_name"),
                model_key=payload.get("model_key"),
                model_version=payload.get("model_version"),
                model_type=payload.get("model_type"),
                training_set=_flatten_training_set(payload.get("training_set")),
                trained_for_benchmark=payload.get("trained_for_benchmark"),
                openness=payload.get("openness"),
                n_estimators=payload.get("n_estimators"),
                model_params=str(payload.get("model_params")) if payload.get("model_params") is not None else None,
                pred_file=discovery.get("pred_file"),
                pred_file_url=discovery.get("pred_file_url"),
                pred_col=discovery.get("pred_col"),
                pred_file_exists=bool(
                    discovery.get("pred_file")
                    and (repo_dir / str(discovery.get("pred_file"))).exists()
                    and file_is_readable(repo_dir / str(discovery.get("pred_file")))
                ),
                full_test_mae=full_test.get("MAE"),
                full_test_f1=full_test.get("F1"),
                full_test_daf=full_test.get("DAF"),
            )
        )
    return pd.DataFrame(asdict(record) for record in records)


def download_article_bundle(
    article_id: int,
    destination: Path,
    suffixes: tuple[str, ...],
    *,
    max_files: int | None = None,
) -> list[dict[str, Any]]:
    article = fetch_figshare_article(article_id)
    files = select_article_files(article, suffixes)
    if max_files is not None:
        files = files[:max_files]

    results: list[dict[str, Any]] = []
    for item in files:
        target = destination / f"{item['id']}_{item['name']}"
        download_file(item["download_url"], target, item.get("supplied_md5"))
        results.append(
            {
                "article_id": article_id,
                "file_id": item.get("id"),
                "name": item["name"],
                "size": item.get("size"),
                "download_url": item.get("download_url"),
                "path": str(target),
            }
        )
    return results


def download_named_article_files(
    article_id: int,
    destination: Path,
    required_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    article = fetch_figshare_article(article_id)
    name_to_item = {item["name"]: item for item in article.get("files", [])}
    results: list[dict[str, Any]] = []
    for name in required_names:
        if name not in name_to_item:
            continue
        item = name_to_item[name]
        target = destination / f"{item['id']}_{item['name']}"
        download_file(item["download_url"], target, item.get("supplied_md5"))
        results.append(
            {
                "article_id": article_id,
                "file_id": item.get("id"),
                "name": item["name"],
                "size": item.get("size"),
                "download_url": item.get("download_url"),
                "path": str(target),
            }
        )
    return results


def download_prediction_files_from_metadata(
    settings: Settings,
    model_metadata: pd.DataFrame,
    *,
    max_files: int | None = None,
) -> list[dict[str, Any]]:
    candidates = model_metadata.dropna(subset=["pred_file", "pred_file_url"]).copy()
    candidates = candidates.loc[~candidates["pred_file_exists"]]
    if max_files is not None:
        candidates = candidates.head(max_files)

    results: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        relative_path = Path(str(row["pred_file"]))
        target = settings.paths.official_repo / relative_path
        download_file(str(row["pred_file_url"]), target)
        results.append(
            {
                "model_key": row["model_key"],
                "pred_file": str(relative_path),
                "pred_file_url": row["pred_file_url"],
                "path": str(target),
            }
        )
    return results


def infer_column(df: pd.DataFrame, aliases: Iterable[str]) -> str:
    lower_map = {column.lower(): column for column in df.columns}
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    raise KeyError(f"None of the aliases {tuple(aliases)} were found in {list(df.columns)}")


def normalize_prediction_frame(path: Path, settings: Settings, explicit_pred_col: str | None = None) -> pd.DataFrame:
    frame = read_table(path)
    material_col = infer_column(frame, settings.columns.material_id)
    pred_col = explicit_pred_col or infer_column(frame, settings.columns.prediction)
    return frame[[material_col, pred_col]].rename(
        columns={material_col: "material_id", pred_col: path.stem.replace(".csv", "")}
    )


def find_reference_table(settings: Settings) -> Path | None:
    for candidate in sorted(settings.paths.official_repo.glob("data/wbm/*.csv.gz")):
        return candidate
    raw_wbm_dir = settings.paths.raw / "wbm"
    if raw_wbm_dir.exists():
        for candidate in sorted(raw_wbm_dir.glob("**/*")):
            if candidate.is_file() and "".join(candidate.suffixes).lower().endswith(settings.phase0.reference_suffixes):
                return candidate
    return None


def merge_prediction_tables(settings: Settings, model_metadata: pd.DataFrame) -> Path | None:
    reference_path = find_reference_table(settings)
    if reference_path is None:
        return None

    reference = read_table(reference_path)
    material_col = infer_column(reference, settings.columns.material_id)
    formula_col = infer_column(reference, settings.columns.formula)
    dft_col = infer_column(reference, settings.columns.dft_ehull)
    e_form_col = infer_column(reference, settings.columns.dft_e_form)
    merged = reference[[material_col, formula_col, dft_col]].rename(
        columns={material_col: "material_id", formula_col: "formula", dft_col: "dft_ehull"}
    )
    merged["e_form_dft"] = reference[e_form_col]

    downloaded_prediction_paths = []
    for _, row in model_metadata.dropna(subset=["pred_file"]).iterrows():
        candidate = settings.paths.official_repo / str(row["pred_file"])
        if candidate.exists() and file_is_readable(candidate):
            downloaded_prediction_paths.append((str(row["model_key"]), candidate, row.get("pred_col")))

    for model_key, path, pred_col in downloaded_prediction_paths:
        try:
            frame = normalize_prediction_frame(path, settings, explicit_pred_col=pred_col)
        except Exception:
            continue
        frame = frame.rename(columns={frame.columns[1]: model_key})
        merged = merged.merge(frame, on="material_id", how="left")

    ordered_cols = ["material_id", "formula", "dft_ehull", "e_form_dft"]
    other_cols = [column for column in merged.columns if column not in ordered_cols]
    merged = merged[ordered_cols + other_cols]
    merged["dft_stable"] = merged["dft_ehull"] <= 0
    output_path = settings.paths.processed / "merged_predictions.parquet"
    write_table(merged, output_path)
    return output_path


def run_phase0(settings: Settings, *, max_files: int | None = None, dry_run: bool = False) -> dict[str, Any]:
    settings.paths.ensure()
    manifest_dir = settings.paths.raw / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    save_figshare_manifest(settings.phase0.prediction_article_id, manifest_dir)
    save_figshare_manifest(settings.phase0.wbm_article_id, manifest_dir)

    official_repo = clone_or_update_repo(settings)
    model_metadata = collect_model_metadata(official_repo)
    model_metadata_path = settings.paths.processed / "model_metadata.csv"
    write_table(model_metadata, model_metadata_path)

    downloads: dict[str, list[dict[str, Any]]] = {"predictions": [], "wbm": [], "direct_prediction_files": []}
    if not dry_run:
        downloads["wbm"] = download_named_article_files(
            settings.phase0.wbm_article_id,
            settings.paths.raw / "wbm",
            settings.phase0.reference_target_files,
        )
        downloads["direct_prediction_files"] = download_prediction_files_from_metadata(
            settings,
            model_metadata,
            max_files=max_files,
        )
        model_metadata = collect_model_metadata(official_repo)
        write_table(model_metadata, model_metadata_path)

    merged_path = merge_prediction_tables(settings, model_metadata)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "model_count": int(len(model_metadata)),
        "predictions_with_local_files": int(model_metadata["pred_file_exists"].sum()),
        "model_metadata_path": str(model_metadata_path),
        "official_repo": str(official_repo),
        "downloads": downloads,
        "merged_predictions_path": str(merged_path) if merged_path else None,
    }
    write_json(report, settings.paths.processed / "phase0_report.json")
    return report
