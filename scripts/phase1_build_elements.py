from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path

import pandas as pd

from materials_fairness_audit.config import load_settings
from materials_fairness_audit.elements import element_property_record, parse_formula_features
from materials_fairness_audit.io_utils import read_table, write_table


def load_train_element_counts(mp_entries_path: Path) -> Counter[str]:
    with gzip.open(mp_entries_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    counts: Counter[str] = Counter()
    entries = payload.get("entry", {})
    for entry in entries.values():
        if not isinstance(entry, dict):
            continue
        composition = entry.get("composition", {})
        if isinstance(composition, dict):
            counts.update(composition.keys())
    return counts


def main() -> None:
    settings = load_settings()
    merged_path = settings.paths.processed / "merged_predictions.parquet"
    frame = read_table(merged_path)

    features = frame["formula"].map(parse_formula_features)
    feature_frame = pd.DataFrame(
        {
            "material_id": frame["material_id"],
            "formula": frame["formula"],
            "elements": [list(item.elements) for item in features],
            "n_elements": [item.n_elements for item in features],
            "has_3d_transition_metal": [item.has_3d_transition_metal for item in features],
            "has_4f_lanthanide": [item.has_4f_lanthanide for item in features],
            "has_5d_heavy_element": [item.has_5d_heavy_element for item in features],
            "avg_electronegativity": [item.avg_electronegativity for item in features],
            "std_electronegativity": [item.std_electronegativity for item in features],
            "has_plus_u_element": [item.has_plus_u_element for item in features],
        }
    )
    write_table(feature_frame, settings.paths.processed / "element_material_mapping.parquet")

    test_counts = Counter(symbol for symbols in feature_frame["elements"] for symbol in symbols)
    mp_entries_candidates = sorted((settings.paths.raw / "wbm").glob("*mp-computed-structure-entries*.json.gz"))
    train_counts = load_train_element_counts(mp_entries_candidates[0]) if mp_entries_candidates else Counter()

    all_symbols = sorted(set(test_counts) | set(train_counts))
    statistics = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "n_test_elem": test_counts.get(symbol, 0),
                "n_train_elem": train_counts.get(symbol, 0),
                **element_property_record(symbol),
            }
            for symbol in all_symbols
        ]
    )
    statistics = statistics.sort_values(["n_test_elem", "symbol"], ascending=[False, True]).reset_index(drop=True)
    statistics["train_test_ratio"] = statistics["n_train_elem"] / statistics["n_test_elem"].replace({0: pd.NA})
    statistics["is_usable"] = statistics["n_test_elem"] >= settings.analysis.minimum_element_test_count
    write_table(statistics, settings.paths.processed / "element_statistics.csv")
    print(f"Saved {len(statistics)} element rows.")


if __name__ == "__main__":
    main()
