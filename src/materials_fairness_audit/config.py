from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "project.toml"


@dataclass(slots=True)
class Paths:
    repo_root: Path
    data_root: Path
    raw: Path
    interim: Path
    processed: Path
    tables: Path
    cache: Path
    exports: Path
    official_repo: Path
    raw_predictions: Path
    raw_wbm: Path
    raw_aflow: Path
    processed_matbench: Path
    processed_databases: Path
    processed_matalign: Path
    processed_audit: Path
    export_release: Path

    def ensure(self) -> None:
        for path in (
            self.data_root,
            self.raw,
            self.interim,
            self.processed,
            self.tables,
            self.cache,
            self.exports,
            self.raw_predictions,
            self.raw_wbm,
            self.raw_aflow,
            self.processed_matbench,
            self.processed_databases,
            self.processed_matalign,
            self.processed_audit,
            self.export_release,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class Phase0Settings:
    prediction_article_id: int
    wbm_article_id: int
    matbench_repo_url: str
    matbench_repo_ref: str
    missing_rate_warn: float
    missing_rate_drop: float
    prediction_suffixes: tuple[str, ...]
    reference_suffixes: tuple[str, ...]
    reference_target_files: tuple[str, ...]


@dataclass(slots=True)
class AnalysisSettings:
    minimum_element_test_count: int
    stability_thresholds: tuple[float, ...]
    bootstrap_samples: int
    bootstrap_seed: int
    max_abs_form_energy_error: float


@dataclass(slots=True)
class DatabaseSettings:
    mp_chunk_size: int
    oqmd_page_limit: int
    oqmd_stability_max: float
    formation_energy_min: float
    formation_energy_max: float
    band_gap_min: float
    volume_min: float
    volume_max: float
    aflow_min_records: int
    aflow_page_size: int
    aflow_batch_pages: int


@dataclass(slots=True)
class MatchingSettings:
    volume_rel_tol: float
    structure_ltol: float
    structure_stol: float
    structure_angle_tol: float
    anchor_priority: tuple[str, ...]


@dataclass(slots=True)
class PublishSettings:
    hf_dataset_id: str
    croissant_filename: str
    datasheet_markdown: Path


@dataclass(slots=True)
class ColumnAliases:
    material_id: tuple[str, ...]
    formula: tuple[str, ...]
    dft_ehull: tuple[str, ...]
    dft_e_form: tuple[str, ...]
    prediction: tuple[str, ...]


@dataclass(slots=True)
class Settings:
    project_name: str
    timezone: str
    paths: Paths
    phase0: Phase0Settings
    analysis: AnalysisSettings
    databases: DatabaseSettings
    matching: MatchingSettings
    publish: PublishSettings
    columns: ColumnAliases
    mp_api_key: str | None
    config_path: Path


def _load_env_file(repo_root: Path) -> None:
    env_path = repo_root / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_settings(config_path: str | Path | None = None) -> Settings:
    resolved_config = Path(config_path) if config_path else DEFAULT_CONFIG
    resolved_config = resolved_config.resolve()
    repo_root = resolved_config.parents[1]
    _load_env_file(repo_root)

    with resolved_config.open("rb") as handle:
        data = tomllib.load(handle)

    project = data["project"]
    path_block = data["paths"]
    data_root = _resolve_path(repo_root, path_block["data_root"])
    paths = Paths(
        repo_root=repo_root,
        data_root=data_root,
        raw=_resolve_path(data_root, path_block["raw"]),
        interim=_resolve_path(data_root, path_block["interim"]),
        processed=_resolve_path(data_root, path_block["processed"]),
        tables=_resolve_path(data_root, path_block["tables"]),
        cache=_resolve_path(data_root, path_block["cache"]),
        exports=_resolve_path(data_root, path_block["exports"]),
        official_repo=_resolve_path(repo_root, path_block["official_repo"]),
        raw_predictions=_resolve_path(data_root, path_block["raw_predictions"]),
        raw_wbm=_resolve_path(data_root, path_block["raw_wbm"]),
        raw_aflow=_resolve_path(data_root, path_block["raw_aflow"]),
        processed_matbench=_resolve_path(data_root, path_block["processed_matbench"]),
        processed_databases=_resolve_path(data_root, path_block["processed_databases"]),
        processed_matalign=_resolve_path(data_root, path_block["processed_matalign"]),
        processed_audit=_resolve_path(data_root, path_block["processed_audit"]),
        export_release=_resolve_path(data_root, path_block.get("export_release", "exports/release")),
    )

    phase0_block = data["phase0"]
    analysis_block = data["analysis"]
    databases_block = data["databases"]
    matching_block = data["matching"]
    publish_block = data["publish"]
    column_block = data["columns"]

    return Settings(
        project_name=project["name"],
        timezone=project["timezone"],
        paths=paths,
        phase0=Phase0Settings(
            prediction_article_id=phase0_block["prediction_article_id"],
            wbm_article_id=phase0_block["wbm_article_id"],
            matbench_repo_url=phase0_block["matbench_repo_url"],
            matbench_repo_ref=phase0_block["matbench_repo_ref"],
            missing_rate_warn=phase0_block["missing_rate_warn"],
            missing_rate_drop=phase0_block["missing_rate_drop"],
            prediction_suffixes=tuple(phase0_block["prediction_suffixes"]),
            reference_suffixes=tuple(phase0_block["reference_suffixes"]),
            reference_target_files=tuple(phase0_block["reference_target_files"]),
        ),
        analysis=AnalysisSettings(
            minimum_element_test_count=analysis_block["minimum_element_test_count"],
            stability_thresholds=tuple(analysis_block["stability_thresholds"]),
            bootstrap_samples=analysis_block["bootstrap_samples"],
            bootstrap_seed=analysis_block["bootstrap_seed"],
            max_abs_form_energy_error=analysis_block["max_abs_form_energy_error"],
        ),
        databases=DatabaseSettings(
            mp_chunk_size=databases_block["mp_chunk_size"],
            oqmd_page_limit=databases_block["oqmd_page_limit"],
            oqmd_stability_max=databases_block["oqmd_stability_max"],
            formation_energy_min=databases_block["formation_energy_min"],
            formation_energy_max=databases_block["formation_energy_max"],
            band_gap_min=databases_block["band_gap_min"],
            volume_min=databases_block["volume_min"],
            volume_max=databases_block["volume_max"],
            aflow_min_records=databases_block["aflow_min_records"],
            aflow_page_size=databases_block["aflow_page_size"],
            aflow_batch_pages=databases_block["aflow_batch_pages"],
        ),
        matching=MatchingSettings(
            volume_rel_tol=matching_block["volume_rel_tol"],
            structure_ltol=matching_block["structure_ltol"],
            structure_stol=matching_block["structure_stol"],
            structure_angle_tol=matching_block["structure_angle_tol"],
            anchor_priority=tuple(matching_block["anchor_priority"]),
        ),
        publish=PublishSettings(
            hf_dataset_id=publish_block["hf_dataset_id"],
            croissant_filename=publish_block["croissant_filename"],
            datasheet_markdown=_resolve_path(repo_root, publish_block["datasheet_markdown"]),
        ),
        columns=ColumnAliases(
            material_id=tuple(column_block["material_id"]),
            formula=tuple(column_block["formula"]),
            dft_ehull=tuple(column_block["dft_ehull"]),
            dft_e_form=tuple(column_block["dft_e_form"]),
            prediction=tuple(column_block["prediction"]),
        ),
        mp_api_key=os.getenv("MP_API_KEY"),
        config_path=resolved_config,
    )
