from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from mp_api.client import MPRester


WORK_DIR = Path("vasp_v3_pbe_work")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Materials Project elemental reference energies for MP2020-compatible formation energies."
    )
    parser.add_argument("--work-dir", type=Path, default=WORK_DIR)
    parser.add_argument("--totals", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--elements", default="", help="Optional comma/space separated element list.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--raw-output", type=Path, default=None)
    return parser.parse_args()


def load_elements(args: argparse.Namespace) -> list[str]:
    if args.elements.strip():
        tokens = args.elements.replace(",", " ").split()
        return sorted(set(tokens))

    source = args.totals or args.work_dir / "analysis" / "pbe_total_energies.csv"
    if not source.exists():
        source = args.manifest or args.work_dir / "manifests" / "task_manifest.csv"
    if not source.exists():
        raise FileNotFoundError(source)

    df = pd.read_csv(source)
    if "composition_json" not in df.columns:
        raise ValueError(f"{source} does not contain composition_json")
    if "task_kind" in df.columns:
        df = df[df["task_kind"] == "compound"].copy()

    elements: set[str] = set()
    for value in df["composition_json"].dropna():
        elements.update(json.loads(value).keys())
    return sorted(elements)


def entry_to_record(element: str, entries: list) -> tuple[dict, list[dict]]:
    if not entries:
        raise RuntimeError(f"No MP entries returned for element {element}")
    entries = sorted(entries, key=lambda entry: (float(entry.energy_per_atom), str(entry.entry_id)))
    selected = entries[0]
    records = []
    for entry in entries:
        records.append(
            {
                "element": element,
                "entry_id": str(entry.entry_id),
                "formula": entry.composition.reduced_formula,
                "energy_per_atom": float(entry.energy_per_atom),
                "energy_above_hull": entry.data.get("energy_above_hull"),
                "run_type": entry.parameters.get("run_type"),
                "selected": str(entry.entry_id) == str(selected.entry_id),
            }
        )
    selected_record = {
        "element": element,
        "energy_per_atom": float(selected.energy_per_atom),
        "entry_id": str(selected.entry_id),
        "formula": selected.composition.reduced_formula,
        "n_entries": len(entries),
        "energy_above_hull": selected.data.get("energy_above_hull"),
        "run_type": selected.parameters.get("run_type"),
        "selection_policy": "lowest_mp2020_compatible_energy_per_atom",
    }
    return selected_record, records


def main() -> None:
    args = parse_args()
    if not os.environ.get("MP_API_KEY"):
        raise RuntimeError("Missing MP_API_KEY environment variable.")

    output = args.output or args.work_dir / "analysis" / "mp2020_element_references.csv"
    raw_output = args.raw_output or args.work_dir / "analysis" / "mp2020_element_reference_candidates.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output.parent.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    candidate_rows = []
    elements = load_elements(args)
    with MPRester() as mpr:
        for element in elements:
            entries = mpr.get_entries_in_chemsys(
                [element],
                compatible_only=True,
                property_data=["energy_above_hull"],
            )
            selected, candidates = entry_to_record(element, entries)
            selected_rows.append(selected)
            candidate_rows.extend(candidates)

    pd.DataFrame(selected_rows).sort_values("element").to_csv(output, index=False)
    with raw_output.open("w", encoding="utf-8") as fh:
        for row in candidate_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "elements": len(selected_rows),
                "output": str(output),
                "raw_output": str(raw_output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
