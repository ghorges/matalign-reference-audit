from __future__ import annotations

from itertools import combinations
import json

from materials_fairness_audit.config import load_settings
from materials_fairness_audit.databases import load_normalized_databases
from materials_fairness_audit.io_utils import write_table
from materials_fairness_audit.matalign import (
    build_matalign_table,
    build_pair_matches,
    write_matching_statistics,
    write_pair_tables,
)


def main() -> None:
    settings = load_settings()
    settings.paths.ensure()

    frames = load_normalized_databases(settings.paths.processed_databases)
    usable_sources = {source: frame for source, frame in frames.items() if not frame.empty}
    if len(usable_sources) < 2:
        raise RuntimeError("Need at least two normalized database tables before building MatAlign.")

    pair_tables = {}
    for left_source, right_source in combinations(sorted(usable_sources), 2):
        pair_name = f"{left_source}_{right_source}"
        pair_tables[pair_name] = build_pair_matches(
            usable_sources[left_source],
            usable_sources[right_source],
            settings.matching,
        )

    pair_dir = settings.paths.processed_matalign / "matalign_pairs"
    write_pair_tables(pair_tables, pair_dir)

    matalign = build_matalign_table(usable_sources, pair_tables, settings)
    matalign_path = settings.paths.processed_matalign / "matalign_full.parquet"
    write_table(matalign, matalign_path)
    stats_path = write_matching_statistics(pair_tables, matalign, settings)
    print(
        json.dumps(
            {
                "matalign_path": str(matalign_path),
                "matching_statistics_path": str(stats_path),
                "pair_counts": {name: int(len(frame)) for name, frame in pair_tables.items()},
                "matalign_rows": int(len(matalign)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
