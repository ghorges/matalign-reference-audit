from __future__ import annotations

import csv
from dataclasses import dataclass, field
import gzip
import io
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd
from pymatgen.core import Composition

from .io_utils import write_table


DEFAULT_OQMD_CALCULATION_PRIORITY = (
    "static|static",
    "standard|standard",
    "fine_relax|fine_relax",
    "coarse_relax|coarse_relax",
    "initialize|initialize",
    "relaxation|relaxation",
    "relaxation_0|relaxation",
    "relaxation_1|relaxation",
    "relaxation_2|relaxation",
    "relaxation_3|relaxation",
    "standard|",
    "fine_relax|",
)


def rank_oqmd_calculation_combo(
    combo: str | None,
    priority: tuple[str, ...] = DEFAULT_OQMD_CALCULATION_PRIORITY,
) -> int:
    if combo is None:
        return len(priority) + 999
    rank_map = {value: index for index, value in enumerate(priority)}
    return rank_map.get(combo, len(priority) + 999)


def iter_mysql_insert_rows(values_sql: str) -> Iterator[str]:
    in_quote = False
    escape = False
    depth = 0
    start: int | None = None

    for index, char in enumerate(values_sql):
        if escape:
            escape = False
            continue
        if in_quote and char == "\\":
            escape = True
            continue
        if char == "'":
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == "(":
            if depth == 0:
                start = index + 1
            depth += 1
            continue
        if char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Malformed INSERT payload: unexpected closing parenthesis.")
            if depth == 0 and start is not None:
                yield values_sql[start:index]
                start = None

    if in_quote or depth != 0:
        raise ValueError("Malformed INSERT payload: unbalanced quotes or parentheses.")


def _unescape_mysql_string(text: str) -> str:
    output: list[str] = []
    index = 0
    replacements = {
        "0": "\0",
        "b": "\b",
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "Z": "\x1a",
        "\\": "\\",
        "'": "'",
        '"': '"',
    }
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            nxt = text[index + 1]
            output.append(replacements.get(nxt, nxt))
            index += 2
            continue
        output.append(char)
        index += 1
    return "".join(output)


def parse_mysql_value(raw: str | bytes) -> str | None:
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    if text == "NULL":
        return None
    if len(text) >= 2 and text[0] == "'" and text[-1] == "'":
        return _unescape_mysql_string(text[1:-1])
    return text


def parse_mysql_insert_row(row_sql: str) -> list[str | None]:
    reader = csv.reader(
        io.StringIO(row_sql),
        delimiter=",",
        quotechar="'",
        escapechar="\\",
        doublequote=False,
        strict=True,
    )
    row = next(reader)
    return [None if value == "NULL" else value for value in row]


def parse_selected_mysql_insert_rows(values_sql: str, field_indices: tuple[int, ...]) -> list[list[str | None]]:
    parser = _MysqlInsertValueParser(field_indices)
    parser.consume(values_sql.encode("utf-8"))
    parser.finish()
    return parser.rows


def build_reference_tables(
    reference_energies: pd.DataFrame,
    reducers: tuple[str, ...] = ("min", "median", "max"),
) -> dict[str, dict[str, dict[str, float]]]:
    tables: dict[str, dict[str, dict[str, float]]] = {}
    cleaned = reference_energies.copy()
    cleaned["value"] = pd.to_numeric(cleaned["value"], errors="coerce")
    cleaned = cleaned.dropna(subset=["fit_id", "element_id", "value"])

    for reducer in reducers:
        grouped = (
            cleaned.groupby(["fit_id", "element_id"], as_index=False)["value"]
            .agg(reducer)
            .rename(columns={"value": "reference_energy"})
        )
        fit_tables: dict[str, dict[str, float]] = {}
        for fit_id, frame in grouped.groupby("fit_id"):
            fit_tables[str(fit_id)] = dict(zip(frame["element_id"].astype(str), frame["reference_energy"].astype(float)))
        tables[reducer] = fit_tables

    return tables


def compute_reference_energy_per_atom(formula: str, element_energies: dict[str, float]) -> float | None:
    try:
        composition = Composition(formula)
    except Exception:
        return None

    total_atoms = composition.num_atoms
    if total_atoms <= 0:
        return None

    total_energy = 0.0
    for element, amount in composition.get_el_amt_dict().items():
        reference = element_energies.get(str(element))
        if reference is None:
            return None
        total_energy += float(amount) * float(reference)
    return total_energy / float(total_atoms)


def scan_gzip_for_patterns(
    gzip_path: Path,
    patterns: dict[str, bytes],
    *,
    chunk_size: int = 8_000_000,
    carry_size: int = 20_000,
) -> dict[str, int | None]:
    positions: dict[str, int | None] = {name: None for name in patterns}
    seen = 0
    carry = b""

    with gzip.open(gzip_path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            blob = carry + chunk
            base = seen - len(carry)
            for name, needle in patterns.items():
                if positions[name] is not None:
                    continue
                offset = blob.find(needle)
                if offset >= 0:
                    positions[name] = base + offset
            seen += len(chunk)
            carry = blob[-carry_size:]
            if all(position is not None for position in positions.values()):
                break

    return positions


def capture_gzip_snippets(
    gzip_path: Path,
    captures: dict[str, bytes],
    output_dir: Path,
    *,
    capture_size: int = 12_000,
    chunk_size: int = 8_000_000,
    carry_size: int = 20_000,
) -> dict[str, dict[str, str | int | None]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, str | int | None]] = {}
    seen = 0
    carry = b""

    with gzip.open(gzip_path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            blob = carry + chunk
            base = seen - len(carry)
            for name, needle in captures.items():
                if name in results:
                    continue
                offset = blob.find(needle)
                if offset < 0:
                    continue
                snippet = blob[offset : min(len(blob), offset + capture_size)].decode("utf-8", errors="replace")
                filename = f"{name}.sql"
                (output_dir / filename).write_text(snippet, encoding="utf-8")
                results[name] = {"position": base + offset, "filename": filename}
            seen += len(chunk)
            carry = blob[-carry_size:]
            if len(results) == len(captures):
                break

    for name in captures:
        results.setdefault(name, {"position": None, "filename": None})
    return results


@dataclass(frozen=True, slots=True)
class OqmdDumpTableSpec:
    table_name: str
    selected_columns: tuple[str, ...]
    field_indices: tuple[int, ...]
    output_stem: str | None = None
    row_filter: Callable[[list[str | None]], bool] | None = None

    @property
    def insert_prefix(self) -> bytes:
        return f"INSERT INTO `{self.table_name}` VALUES ".encode("utf-8")


@dataclass(slots=True)
class _TableChunkWriter:
    spec: OqmdDumpTableSpec
    output_dir: Path
    chunk_rows: int
    rows: list[list[str | None]] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    total_rows: int = 0
    chunk_index: int = 0

    def append(self, row: list[str | None]) -> None:
        self.rows.append(row)
        self.total_rows += 1
        if len(self.rows) >= self.chunk_rows:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = self.spec.output_stem or self.spec.table_name
        path = self.output_dir / f"{stem}_chunk_{self.chunk_index:05d}.parquet"
        frame = pd.DataFrame(self.rows, columns=self.spec.selected_columns)
        write_table(frame, path)
        self.files.append(str(path))
        print(
            f"[oqmd_dump] wrote {self.spec.table_name} chunk {self.chunk_index:05d} "
            f"with {len(frame):,} rows -> {path}"
        )
        self.chunk_index += 1
        self.rows = []

    def finalize(self) -> dict[str, Any]:
        self.flush()
        return {
            "table_name": self.spec.table_name,
            "selected_columns": list(self.spec.selected_columns),
            "chunk_files": self.files,
            "chunk_count": len(self.files),
            "row_count": self.total_rows,
        }


class _MysqlInsertValueParser:
    def __init__(self, field_indices: tuple[int, ...]) -> None:
        self._index_to_position = {field_index: position for position, field_index in enumerate(field_indices)}
        self._target_indices = set(field_indices)
        self._field_count = len(field_indices)
        self.rows: list[list[str | None]] = []
        self._reset_insert_state()

    def _reset_insert_state(self) -> None:
        self._inside_row = False
        self._in_quote = False
        self._escape = False
        self._field_index = 0
        self._capture_field = False
        self._field_buffer = bytearray()
        self._current_row: list[str | None] = [None] * self._field_count

    def _start_row(self) -> None:
        self._inside_row = True
        self._in_quote = False
        self._escape = False
        self._field_index = 0
        self._capture_field = 0 in self._target_indices
        self._field_buffer = bytearray()
        self._current_row = [None] * self._field_count

    def _finish_field(self) -> None:
        if self._capture_field:
            position = self._index_to_position[self._field_index]
            self._current_row[position] = parse_mysql_value(bytes(self._field_buffer))
        self._field_index += 1
        self._capture_field = self._field_index in self._target_indices
        self._field_buffer = bytearray()

    def _finish_row(self) -> None:
        self.rows.append(self._current_row.copy())
        self._inside_row = False
        self._in_quote = False
        self._escape = False
        self._field_index = 0
        self._capture_field = False
        self._field_buffer = bytearray()

    def consume(self, data: bytes) -> tuple[int, bool]:
        for index, byte in enumerate(data):
            if self._inside_row:
                if self._escape:
                    if self._capture_field:
                        self._field_buffer.append(byte)
                    self._escape = False
                    continue
                if self._in_quote:
                    if self._capture_field:
                        self._field_buffer.append(byte)
                    if byte == 92:
                        self._escape = True
                    elif byte == 39:
                        self._in_quote = False
                    continue
                if byte == 39:
                    if self._capture_field:
                        self._field_buffer.append(byte)
                    self._in_quote = True
                    continue
                if byte == 44:
                    self._finish_field()
                    continue
                if byte == 41:
                    self._finish_field()
                    self._finish_row()
                    continue
                if self._capture_field:
                    self._field_buffer.append(byte)
                continue

            if byte == 40:
                self._start_row()
                continue
            if byte == 59:
                return index + 1, True

        return len(data), False

    def finish(self) -> None:
        if self._inside_row or self._in_quote or self._escape:
            raise ValueError("Unterminated INSERT row while finalizing parser.")


class _TableStreamExtractor:
    def __init__(self, spec: OqmdDumpTableSpec, writer: _TableChunkWriter) -> None:
        self.spec = spec
        self.writer = writer
        self.parser = _MysqlInsertValueParser(spec.field_indices)

    def consume(self, data: bytes) -> tuple[int, bool]:
        consumed, finished = self.parser.consume(data)
        if self.parser.rows:
            for row in self.parser.rows:
                if self.spec.row_filter is None or self.spec.row_filter(row):
                    self.writer.append(row)
            self.parser.rows = []
        if finished:
            self.parser.finish()
        return consumed, finished


def extract_oqmd_tables(
    gzip_path: Path,
    specs: list[OqmdDumpTableSpec],
    output_root: Path,
    *,
    chunk_rows: int = 100_000,
    chunk_size: int = 8_000_000,
    carry_size: int | None = None,
) -> dict[str, dict[str, Any]]:
    if not specs:
        return {}

    output_root.mkdir(parents=True, exist_ok=True)
    prefix_to_spec = {spec.insert_prefix: spec for spec in specs}
    max_prefix_len = max(len(prefix) for prefix in prefix_to_spec)
    scan_carry_size = carry_size or max_prefix_len
    writers = {
        spec.table_name: _TableChunkWriter(
            spec=spec,
            output_dir=output_root / spec.table_name,
            chunk_rows=chunk_rows,
        )
        for spec in specs
    }

    active: _TableStreamExtractor | None = None
    carry = b""

    with gzip.open(gzip_path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            buffer = chunk if active is not None else carry + chunk
            base_index = 0

            while base_index < len(buffer):
                if active is None:
                    next_match: tuple[int, bytes, OqmdDumpTableSpec] | None = None
                    for prefix, spec in prefix_to_spec.items():
                        offset = buffer.find(prefix, base_index)
                        if offset < 0:
                            continue
                        if next_match is None or offset < next_match[0]:
                            next_match = (offset, prefix, spec)
                    if next_match is None:
                        carry = buffer[-scan_carry_size:] if len(buffer) > scan_carry_size else buffer
                        base_index = len(buffer)
                        break
                    offset, prefix, spec = next_match
                    print(f"[oqmd_dump] found INSERT section for table {spec.table_name}")
                    active = _TableStreamExtractor(spec, writers[spec.table_name])
                    base_index = offset + len(prefix)
                    carry = b""
                    continue

                consumed, finished = active.consume(buffer[base_index:])
                base_index += consumed
                if finished:
                    print(
                        f"[oqmd_dump] finished table {active.spec.table_name} "
                        f"with {active.writer.total_rows:,} rows parsed"
                    )
                    active = None

        if active is not None:
            raise ValueError(f"Reached EOF while still parsing INSERT for table {active.spec.table_name}.")

    return {table_name: writer.finalize() for table_name, writer in writers.items()}
