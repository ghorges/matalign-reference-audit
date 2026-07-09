from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import pandas as pd


MATCH_KEY_COLUMNS = ["reduced_formula", "spacegroup_number"]


def categorize_formula_order(elements: Any) -> str:
    if elements is None:
        return "unknown"
    if isinstance(elements, (str, bytes)):
        return "unknown"
    if hasattr(elements, "tolist"):
        elements = elements.tolist()
    if not isinstance(elements, Iterable):
        return "unknown"
    count = len({str(symbol) for symbol in elements if symbol})
    if count <= 0:
        return "unknown"
    if count == 1:
        return "elemental"
    if count == 2:
        return "binary"
    if count == 3:
        return "ternary"
    if count == 4:
        return "quaternary"
    return "quinary_plus"


def categorize_oqmd_entry_origin(path: str | None) -> str:
    if not path:
        return "unknown"
    text = str(path).strip().lower().replace("\\", "/")
    if "/libraries/icsd/" in text:
        return "icsd"
    if "/libraries/prototypes/elements/" in text:
        return "element_prototype"
    if "/libraries/prototypes/" in text:
        return "prototype"
    if "/elements/" in text:
        return "element"
    if "/enum/" in text:
        return "enum"
    if "/oqmd/" in text:
        return "oqmd"
    return "other"


def compute_pair_overlap_details(left: pd.DataFrame, right: pd.DataFrame, pair_frame: pd.DataFrame) -> pd.DataFrame:
    left_counts = (
        left.dropna(subset=MATCH_KEY_COLUMNS)
        .groupby(MATCH_KEY_COLUMNS)
        .size()
        .rename("left_size")
        .reset_index()
    )
    right_counts = (
        right.dropna(subset=MATCH_KEY_COLUMNS)
        .groupby(MATCH_KEY_COLUMNS)
        .size()
        .rename("right_size")
        .reset_index()
    )
    details = left_counts.merge(right_counts, on=MATCH_KEY_COLUMNS, how="inner")
    if details.empty:
        return pd.DataFrame(
            columns=[
                *MATCH_KEY_COLUMNS,
                "left_size",
                "right_size",
                "candidate_pairs",
                "matched_pairs",
                "matched_left_ids",
                "matched_right_ids",
                "unresolved_left",
                "unresolved_right",
                "is_ambiguous",
            ]
        )

    details["candidate_pairs"] = details["left_size"] * details["right_size"]
    details["is_ambiguous"] = (details["left_size"] > 1) | (details["right_size"] > 1)

    if pair_frame.empty:
        details["matched_pairs"] = 0
        details["matched_left_ids"] = 0
        details["matched_right_ids"] = 0
    else:
        matched = (
            pair_frame.groupby(MATCH_KEY_COLUMNS, as_index=False)
            .agg(
                matched_pairs=("left_source_id", "size"),
                matched_left_ids=("left_source_id", "nunique"),
                matched_right_ids=("right_source_id", "nunique"),
            )
        )
        details = details.merge(matched, on=MATCH_KEY_COLUMNS, how="left")
        for column in ("matched_pairs", "matched_left_ids", "matched_right_ids"):
            details[column] = details[column].fillna(0).astype(int)

    details["unresolved_left"] = (details["left_size"] - details["matched_left_ids"]).clip(lower=0)
    details["unresolved_right"] = (details["right_size"] - details["matched_right_ids"]).clip(lower=0)
    return details.sort_values(["candidate_pairs", "left_size", "right_size"], ascending=False).reset_index(drop=True)


def summarize_pair_overlap(
    details: pd.DataFrame,
    pair_frame: pd.DataFrame,
    *,
    left_source: str,
    right_source: str,
    left_total_rows: int,
    right_total_rows: int,
) -> dict[str, Any]:
    if details.empty:
        return {
            "pair": f"{left_source}_{right_source}",
            "left_source": left_source,
            "right_source": right_source,
            "left_total_rows": int(left_total_rows),
            "right_total_rows": int(right_total_rows),
            "common_keys": 0,
            "singleton_keys": 0,
            "ambiguous_keys": 0,
            "candidate_pairs": 0,
            "matched_pairs": 0,
            "matched_pair_fraction_vs_candidates": 0.0,
            "unresolved_left_records": 0,
            "unresolved_right_records": 0,
            "volume_rel_diff_median": math.nan,
            "volume_rel_diff_p95": math.nan,
            "match_rule_counts": {},
        }

    volume_series = pd.to_numeric(pair_frame.get("volume_rel_diff"), errors="coerce") if not pair_frame.empty else pd.Series(dtype=float)
    valid_volume = volume_series.dropna()
    candidate_pairs = int(details["candidate_pairs"].sum())
    matched_pairs = int(len(pair_frame))
    return {
        "pair": f"{left_source}_{right_source}",
        "left_source": left_source,
        "right_source": right_source,
        "left_total_rows": int(left_total_rows),
        "right_total_rows": int(right_total_rows),
        "common_keys": int(len(details)),
        "singleton_keys": int((~details["is_ambiguous"]).sum()),
        "ambiguous_keys": int(details["is_ambiguous"].sum()),
        "candidate_pairs": candidate_pairs,
        "matched_pairs": matched_pairs,
        "matched_pair_fraction_vs_candidates": 0.0 if candidate_pairs == 0 else matched_pairs / candidate_pairs,
        "unresolved_left_records": int(details["unresolved_left"].sum()),
        "unresolved_right_records": int(details["unresolved_right"].sum()),
        "volume_rel_diff_median": float(valid_volume.median()) if not valid_volume.empty else math.nan,
        "volume_rel_diff_p95": float(valid_volume.quantile(0.95)) if not valid_volume.empty else math.nan,
        "match_rule_counts": {
            str(key): int(value) for key, value in pair_frame.get("match_rule", pd.Series(dtype=object)).value_counts().items()
        },
    }
