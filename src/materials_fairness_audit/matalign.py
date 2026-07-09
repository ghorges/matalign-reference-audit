from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .config import MatchingSettings, Settings
from .io_utils import write_json, write_table


@dataclass(slots=True)
class MatchRecord:
    left_source: str
    left_source_id: str
    right_source: str
    right_source_id: str
    reduced_formula: str
    spacegroup_number: int
    match_tier: str
    match_rule: str
    ambiguity_count: int
    volume_rel_diff: float | None


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _node_id(source: str, source_id: str) -> str:
    return f"{source}:{source_id}"


def _volume_rel_diff(left: pd.Series, right: pd.Series) -> float | None:
    left_volume = left.get("volume_per_atom")
    right_volume = right.get("volume_per_atom")
    if pd.isna(left_volume) or pd.isna(right_volume):
        return None
    denom = max(abs(float(left_volume)), abs(float(right_volume)), 1e-12)
    return abs(float(left_volume) - float(right_volume)) / denom


def _resolve_ambiguous_group(
    left_group: pd.DataFrame,
    right_group: pd.DataFrame,
    settings: MatchingSettings,
) -> list[MatchRecord]:
    candidate_pairs: list[tuple[float, pd.Series, pd.Series]] = []
    for _, left_row in left_group.iterrows():
        for _, right_row in right_group.iterrows():
            diff = _volume_rel_diff(left_row, right_row)
            if diff is None or diff <= settings.volume_rel_tol:
                candidate_pairs.append((diff if diff is not None else 0.0, left_row, right_row))

    candidate_pairs.sort(key=lambda item: item[0])
    used_left: set[str] = set()
    used_right: set[str] = set()
    matches: list[MatchRecord] = []
    ambiguity = len(left_group) * len(right_group)
    for diff, left_row, right_row in candidate_pairs:
        left_id = str(left_row["source_id"])
        right_id = str(right_row["source_id"])
        if left_id in used_left or right_id in used_right:
            continue
        used_left.add(left_id)
        used_right.add(right_id)
        matches.append(
            MatchRecord(
                left_source=str(left_row["source"]),
                left_source_id=left_id,
                right_source=str(right_row["source"]),
                right_source_id=right_id,
                reduced_formula=str(left_row["reduced_formula"]),
                spacegroup_number=int(left_row["spacegroup_number"]),
                match_tier="tier1",
                match_rule="volume_rel_tol",
                ambiguity_count=ambiguity,
                volume_rel_diff=diff,
            )
        )
    return matches


def build_pair_matches(
    left: pd.DataFrame,
    right: pd.DataFrame,
    settings: MatchingSettings,
) -> pd.DataFrame:
    key_cols = ["reduced_formula", "spacegroup_number"]
    left_valid = left.dropna(subset=key_cols).copy()
    right_valid = right.dropna(subset=key_cols).copy()
    common_keys = set(map(tuple, left_valid[key_cols].drop_duplicates().itertuples(index=False, name=None))) & set(
        map(tuple, right_valid[key_cols].drop_duplicates().itertuples(index=False, name=None))
    )

    rows: list[MatchRecord] = []
    for reduced_formula, spacegroup_number in sorted(common_keys):
        left_group = left_valid.loc[
            (left_valid["reduced_formula"] == reduced_formula)
            & (left_valid["spacegroup_number"] == spacegroup_number)
        ]
        right_group = right_valid.loc[
            (right_valid["reduced_formula"] == reduced_formula)
            & (right_valid["spacegroup_number"] == spacegroup_number)
        ]
        if len(left_group) == 1 and len(right_group) == 1:
            left_row = left_group.iloc[0]
            right_row = right_group.iloc[0]
            rows.append(
                MatchRecord(
                    left_source=str(left_row["source"]),
                    left_source_id=str(left_row["source_id"]),
                    right_source=str(right_row["source"]),
                    right_source_id=str(right_row["source_id"]),
                    reduced_formula=str(reduced_formula),
                    spacegroup_number=int(spacegroup_number),
                    match_tier="tier1",
                    match_rule="exact_key",
                    ambiguity_count=1,
                    volume_rel_diff=_volume_rel_diff(left_row, right_row),
                )
            )
        else:
            rows.extend(_resolve_ambiguous_group(left_group, right_group, settings))

    if not rows:
        return pd.DataFrame(
            columns=[
                "left_source",
                "left_source_id",
                "right_source",
                "right_source_id",
                "reduced_formula",
                "spacegroup_number",
                "match_tier",
                "match_rule",
                "ambiguity_count",
                "volume_rel_diff",
            ]
        )
    return pd.DataFrame([asdict(record) for record in rows])


def build_matalign_table(
    frames: dict[str, pd.DataFrame],
    pair_tables: dict[str, pd.DataFrame],
    settings: Settings,
) -> pd.DataFrame:
    union_find = UnionFind()
    by_node: dict[str, dict[str, Any]] = {}
    for source, frame in frames.items():
        for _, row in frame.iterrows():
            node = _node_id(source, str(row["source_id"]))
            union_find.add(node)
            by_node[node] = row.to_dict()

    for pair_frame in pair_tables.values():
        for _, row in pair_frame.iterrows():
            union_find.union(
                _node_id(str(row["left_source"]), str(row["left_source_id"])),
                _node_id(str(row["right_source"]), str(row["right_source_id"])),
            )

    components: dict[str, list[str]] = defaultdict(list)
    for node in by_node:
        components[union_find.find(node)].append(node)

    source_upper = {"mp": "MP", "jarvis": "JARVIS", "oqmd": "OQMD", "aflow": "AFLOW"}
    rows: list[dict[str, Any]] = []
    matalign_index = 0
    for nodes in components.values():
        component_rows = [by_node[node] for node in nodes]
        present_sources = {row["source"] for row in component_rows}
        if len(present_sources) < 2:
            continue
        matalign_index += 1
        anchor = None
        for preferred in settings.matching.anchor_priority:
            anchor = next((row for row in component_rows if row["source"] == preferred), None)
            if anchor is not None:
                break
        if anchor is None:
            anchor = component_rows[0]

        element_union = sorted({symbol for row in component_rows for symbol in row.get("elements", [])})
        row_payload: dict[str, Any] = {
            "matalign_id": f"matalign-{matalign_index:07d}",
            "anchor_source": anchor["source"],
            "anchor_source_id": anchor["source_id"],
            "reduced_formula": anchor["reduced_formula"],
            "spacegroup_number": anchor["spacegroup_number"],
            "n_databases": len(present_sources),
            "elements": element_union,
            "sources_present": sorted(present_sources),
        }
        ef_values = []
        eg_values = []
        for source_key, source_label in source_upper.items():
            source_row = next((row for row in component_rows if row["source"] == source_key), None)
            row_payload[f"Ef_{source_label}"] = source_row["formation_energy_per_atom"] if source_row else math.nan
            row_payload[f"Eg_{source_label}"] = source_row["band_gap"] if source_row else math.nan
            row_payload[f"volume_{source_label}"] = source_row["volume_per_atom"] if source_row else math.nan
            row_payload[f"id_{source_label}"] = source_row["source_id"] if source_row else pd.NA
            if source_row is not None:
                ef_values.append(float(source_row["formation_energy_per_atom"]))
                eg_values.append(float(source_row["band_gap"]))
        row_payload["Ef_std"] = float(pd.Series(ef_values, dtype=float).std(ddof=0)) if len(ef_values) > 1 else 0.0
        row_payload["Eg_std"] = float(pd.Series(eg_values, dtype=float).std(ddof=0)) if len(eg_values) > 1 else 0.0
        rows.append(row_payload)
    return pd.DataFrame(rows)


def write_pair_tables(pair_tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pair_name, frame in pair_tables.items():
        write_table(frame, output_dir / f"{pair_name}.csv")


def write_matching_statistics(pair_tables: dict[str, pd.DataFrame], matalign: pd.DataFrame, settings: Settings) -> Path:
    stats = {
        "pair_counts": {name: int(len(frame)) for name, frame in sorted(pair_tables.items())},
        "matalign_rows": int(len(matalign)),
        "n_databases_distribution": matalign["n_databases"].value_counts().sort_index().to_dict()
        if not matalign.empty
        else {},
        "ef_std_median": float(matalign["Ef_std"].median()) if not matalign.empty else math.nan,
        "eg_std_median": float(matalign["Eg_std"].median()) if not matalign.empty else math.nan,
    }
    output = settings.paths.processed_matalign / "matching_statistics.json"
    write_json(stats, output)
    return output
