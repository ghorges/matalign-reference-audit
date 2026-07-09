from __future__ import annotations

from collections.abc import Iterable
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
from pymatgen.core import Composition
from pymatgen.symmetry.groups import SpaceGroup

from .config import Settings
from .elements import composition_to_symbols
from .io_utils import read_table, write_json, write_table


STANDARD_COLUMNS = [
    "source",
    "source_id",
    "formula",
    "reduced_formula",
    "formation_energy_per_atom",
    "band_gap",
    "volume_per_atom",
    "spacegroup_number",
    "elements",
    "has_structure",
    "structure_payload_path",
    "structure_json",
]


def _payload_to_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    if hasattr(payload, "dict"):
        return payload.dict()
    return {}


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        return value
    return None


def _parse_spacegroup_number(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("number", "spacegroup_number", "int_number", "symbol"):
            parsed = _parse_spacegroup_number(value.get(key))
            if parsed is not None:
                return parsed
        return None
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    try:
        return int(SpaceGroup(text).int_number)
    except Exception:
        return None


def _normalize_formula(formula: Any) -> tuple[str | None, str | None, list[str]]:
    if formula is None:
        return None, None, []
    try:
        composition = Composition(str(formula))
    except Exception:
        return None, None, []
    return str(formula), composition.reduced_formula, list(composition_to_symbols(str(formula)))


def _serialize_structure(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "as_dict"):
        return json.dumps(value.as_dict(), ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return None


def _apply_quality_filters(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned = cleaned.dropna(
        subset=[
            "source_id",
            "formula",
            "reduced_formula",
            "formation_energy_per_atom",
            "band_gap",
            "volume_per_atom",
            "spacegroup_number",
        ]
    )
    cleaned = cleaned.loc[
        cleaned["formation_energy_per_atom"].between(
            settings.databases.formation_energy_min,
            settings.databases.formation_energy_max,
        )
    ]
    cleaned = cleaned.loc[cleaned["band_gap"] >= settings.databases.band_gap_min]
    cleaned = cleaned.loc[
        cleaned["volume_per_atom"].between(settings.databases.volume_min, settings.databases.volume_max)
    ]
    cleaned["elements"] = cleaned["elements"].map(lambda value: value if isinstance(value, list) else [])
    cleaned["has_structure"] = cleaned["structure_json"].notna()
    cleaned["structure_payload_path"] = cleaned["structure_payload_path"].fillna(pd.NA)
    return cleaned[STANDARD_COLUMNS]


def normalize_database_frame(frame: pd.DataFrame, source: str, settings: Settings) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["source"] = source
    normalized["formation_energy_per_atom"] = normalized["formation_energy_per_atom"].map(_to_float)
    normalized["band_gap"] = normalized["band_gap"].map(_to_float)
    normalized["volume_per_atom"] = normalized["volume_per_atom"].map(_to_float)
    normalized["spacegroup_number"] = normalized["spacegroup_number"].map(_parse_spacegroup_number)

    formula_data = normalized["formula"].map(_normalize_formula)
    normalized["formula"] = [item[0] for item in formula_data]
    normalized["reduced_formula"] = [item[1] for item in formula_data]
    normalized["elements"] = [item[2] for item in formula_data]

    if "structure_payload_path" not in normalized.columns:
        normalized["structure_payload_path"] = pd.NA
    if "structure_json" not in normalized.columns:
        normalized["structure_json"] = pd.NA

    normalized = _apply_quality_filters(normalized, settings)
    normalized["source_id"] = normalized["source_id"].astype(str)
    normalized = normalized.drop_duplicates(subset=["source", "source_id"]).reset_index(drop=True)
    return normalized


def mp_to_frame(records: Iterable[Any], settings: Settings) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for doc in records:
        payload = _payload_to_dict(doc)
        symmetry = payload.get("symmetry") or {}
        volume = payload.get("volume")
        nsites = payload.get("nsites") or payload.get("num_sites")
        volume_per_atom = _to_float(volume)
        if volume_per_atom is not None and nsites:
            volume_per_atom = volume_per_atom / float(nsites)
        rows.append(
            {
                "source_id": payload.get("material_id"),
                "formula": _coalesce(payload.get("formula_pretty"), payload.get("composition_reduced")),
                "formation_energy_per_atom": payload.get("formation_energy_per_atom"),
                "band_gap": payload.get("band_gap"),
                "volume_per_atom": volume_per_atom,
                "spacegroup_number": symmetry.get("number") if isinstance(symmetry, dict) else symmetry,
                "structure_payload_path": None,
                "structure_json": _serialize_structure(payload.get("structure")),
            }
        )
    return normalize_database_frame(pd.DataFrame(rows), "mp", settings)


def jarvis_to_frame(records: Iterable[dict[str, Any]], settings: Settings) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in records:
        atoms = entry.get("atoms") or {}
        formula = _coalesce(
            entry.get("formula"),
            atoms.get("formula"),
            atoms.get("composition"),
            atoms.get("reduced_formula"),
        )
        natoms = entry.get("natoms")
        if natoms is None:
            try:
                natoms = sum(Composition(str(formula)).as_dict().values())
            except Exception:
                natoms = None
        volume = _coalesce(entry.get("volume"), atoms.get("volume"))
        if volume is None and atoms.get("lattice_mat"):
            try:
                volume = abs(np.linalg.det(np.array(atoms["lattice_mat"], dtype=float)))
            except Exception:
                volume = None
        volume_per_atom = _to_float(volume)
        if natoms and volume_per_atom is not None:
            volume_per_atom = volume_per_atom / float(natoms)
        rows.append(
            {
                "source_id": entry.get("jid") or entry.get("id"),
                "formula": formula,
                "formation_energy_per_atom": _coalesce(
                    entry.get("formation_energy_peratom"),
                    entry.get("formation_energy_per_atom"),
                ),
                "band_gap": _coalesce(
                    entry.get("optb88vdw_bandgap"),
                    entry.get("bandgap"),
                    entry.get("band_gap"),
                ),
                "volume_per_atom": volume_per_atom,
                "spacegroup_number": _coalesce(
                    entry.get("spacegroup_number"),
                    entry.get("spg_number"),
                    entry.get("spacegroup"),
                ),
                "structure_payload_path": None,
                "structure_json": _serialize_structure(atoms if atoms else None),
            }
        )
    return normalize_database_frame(pd.DataFrame(rows), "jarvis", settings)


def oqmd_records_to_frame(records: Iterable[dict[str, Any]], settings: Settings) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in records:
        natoms = entry.get("natoms")
        volume = _to_float(entry.get("volume"))
        volume_per_atom = volume / float(natoms) if volume is not None and natoms else None
        rows.append(
            {
                "source_id": entry.get("entry_id"),
                "formula": entry.get("name") or entry.get("composition"),
                "formation_energy_per_atom": entry.get("delta_e"),
                "band_gap": entry.get("band_gap"),
                "volume_per_atom": volume_per_atom,
                "spacegroup_number": entry.get("spacegroup"),
                "structure_payload_path": None,
                "structure_json": None,
            }
        )
    return normalize_database_frame(pd.DataFrame(rows), "oqmd", settings)


def aflow_to_frame(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "source_id": frame.get("auid"),
            "formula": frame.get("compound"),
            "formation_energy_per_atom": frame.get("enthalpy_formation_atom"),
            "band_gap": frame.get("Egap"),
            "volume_per_atom": frame.get("volume_atom"),
            "spacegroup_number": frame.get("spacegroup_relax"),
            "structure_payload_path": None,
            "structure_json": None,
        }
    )
    return normalize_database_frame(rows, "aflow", settings)


def get_aflow_raw_files(raw_dir: Path, interim_dir: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if interim_dir is not None:
        manifest_path = interim_dir / "aflow_full_batches.csv"
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)
            candidates.extend(Path(path) for path in manifest["batch_file"].tolist())

    full_dir = raw_dir / "full"
    if full_dir.exists():
        candidates.extend(full_dir.glob("aflow_full_batch_*.jsonl.gz"))

    unique_files = sorted({path.resolve() for path in candidates if path.exists()})
    if unique_files:
        return unique_files

    return sorted(path for path in raw_dir.rglob("*") if path.is_file())


def load_aflow_raw_files(raw_dir: Path, interim_dir: Path | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in get_aflow_raw_files(raw_dir, interim_dir):
        suffixes = "".join(path.suffixes).lower()
        if suffixes.endswith(".parquet"):
            frames.append(pd.read_parquet(path))
        elif suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
            frames.append(pd.read_csv(path))
        elif suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz"):
            frames.append(pd.read_json(path, lines=True))
        elif suffixes.endswith(".json") or suffixes.endswith(".json.gz"):
            frames.append(pd.read_json(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_normalized_databases(processed_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(processed_dir.glob("*_data.parquet")):
        source = path.stem.removesuffix("_data")
        frames[source] = read_table(path)
    return frames


def write_database_summary(frames: dict[str, pd.DataFrame], settings: Settings) -> Path:
    rows = []
    for source, frame in sorted(frames.items()):
        rows.append(
            {
                "source": source,
                "n_rows": int(len(frame)),
                "n_with_structure": int(frame["has_structure"].sum()) if "has_structure" in frame else 0,
                "formation_energy_min": float(frame["formation_energy_per_atom"].min()) if len(frame) else math.nan,
                "formation_energy_max": float(frame["formation_energy_per_atom"].max()) if len(frame) else math.nan,
                "band_gap_nonzero_fraction": float((frame["band_gap"] > 0).mean()) if len(frame) else math.nan,
                "volume_median": float(frame["volume_per_atom"].median()) if len(frame) else math.nan,
            }
        )
    summary = pd.DataFrame(rows)
    output = settings.paths.processed_databases / "database_summary.csv"
    write_table(summary, output)
    return output


def write_database_report(payload: dict[str, Any], settings: Settings) -> Path:
    output = settings.paths.processed_databases / "database_report.json"
    write_json(payload, output)
    return output


def fetch_oqmd_pages(
    settings: Settings,
    *,
    max_pages: int | None = None,
    start_offset: int = 0,
    page_limit: int | None = None,
) -> tuple[list[dict[str, Any]], int, int, bool]:
    base_url = "https://oqmd.org/oqmdapi/formationenergy"
    session = requests.Session()
    session.trust_env = False
    offset = start_offset
    page_count = 0
    records: list[dict[str, Any]] = []
    has_more = True
    limit = page_limit or settings.databases.oqmd_page_limit
    while True:
        params = {
            "limit": limit,
            "offset": offset,
            "fields": "entry_id,name,delta_e,band_gap,spacegroup,volume,natoms,stability",
        }
        response = None
        for attempt in range(5):
            try:
                response = session.get(base_url, params=params, timeout=120)
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
        assert response is not None
        payload = response.json()
        page_rows = payload.get("data", [])
        filtered_rows = [
            row
            for row in page_rows
            if _to_float(row.get("stability")) is not None
            and float(row["stability"]) <= settings.databases.oqmd_stability_max
        ]
        records.extend(filtered_rows)
        page_count += 1
        if page_count % 10 == 0:
            print(
                f"OQMD progress: fetched {page_count} pages at limit={limit} / "
                f"{len(records):,} filtered rows / next_offset={offset + limit}"
            )
        offset += limit
        if max_pages is not None and page_count >= max_pages:
            break
        if not payload.get("links", {}).get("next") and len(page_rows) < limit:
            has_more = False
            break
    return records, offset, page_count, has_more
