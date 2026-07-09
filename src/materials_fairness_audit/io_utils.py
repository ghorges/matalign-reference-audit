from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path)
    if suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz"):
        return pd.read_json(path, lines=True)
    if suffixes.endswith(".json") or suffixes.endswith(".json.gz"):
        return pd.read_json(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        df.to_parquet(path, index=False)
        return
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported write format: {path}")


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
