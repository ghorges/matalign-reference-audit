from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import time
import zipfile
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from pymatgen.core import Element, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.stats import wilcoxon

import phase12_d3_analysis as d3

try:
    import ase.io
except Exception:  # pragma: no cover - handled in runtime status
    ase = None


DATA_ROOT = (Path(__file__).resolve().parents[2] / "data").resolve()
D3_DIR = DATA_ROOT / "processed" / "pairwise_consistency_checks"
OUT_DIR = DATA_ROOT / "processed" / "leakage_controlled_consistency"
RAW_MPTRJ_DIR = DATA_ROOT / "raw" / "mptrj"
CANDIDATE_DIR = DATA_ROOT / "cache" / "training_structure_candidates"
DOCS_DIR = Path("docs")
TARGET_STRUCTURE_ROOT = Path("vasp_uniform_pbe_work") / "remote_results" / "primary" / "inputs" / "primary" / "compounds"

MPTRJ_URL = "https://ndownloader.figshare.com/files/49034296"
MPTRJ_FALLBACK_URL = "https://ndownloader.figshare.com/files/41619375"
MPTRJ_FILENAME = "2024-09-03-mp-trj.extxyz.zip"
MPTRJ_MD5 = "7f433171e4e5f2ef9304dccd42d5488f"
MPTRJ_SIZE_BYTES = 1_521_713_089
PBE3_FLOOR = 0.010902695625000902
CROSS_FAMILY_PAIRS = {
    "mace__orb_v3",
    "mace__sevennet_mf_ompa",
    "mace__mattersim_5m",
    "mattersim_5m__orb_v3",
    "mattersim_5m__sevennet_mf_ompa",
}
HOMOLOGOUS_PAIR = "orb_v3__sevennet_mf_ompa"
MACE_CHGNET = ["chgnet", "mace"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MatAlign v3 Phase D3-fix analysis.")
    parser.add_argument("--d3-dir", type=Path, default=D3_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--raw-mptrj-dir", type=Path, default=RAW_MPTRJ_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=CANDIDATE_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--target-structure-root", type=Path, default=TARGET_STRUCTURE_ROOT)
    parser.add_argument("--download-mptrj", action="store_true", help="Download MPtrj if the raw zip is missing or invalid.")
    parser.add_argument("--download-only", action="store_true", help="Only download/check MPtrj and exit.")
    parser.add_argument("--force-extract", action="store_true", help="Re-extract MPtrj candidate structures.")
    parser.add_argument("--max-frames", type=int, default=None, help="Debug cap for extxyz frames scanned.")
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def composition_key_from_atomic_numbers(numbers: list[int]) -> str:
    counts: dict[int, int] = {}
    for number in numbers:
        counts[int(number)] = counts.get(int(number), 0) + 1
    gcd = 0
    for count in counts.values():
        gcd = math.gcd(gcd, count)
    if gcd > 1:
        counts = {number: count // gcd for number, count in counts.items()}
    return ";".join(f"{number}:{counts[number]}" for number in sorted(counts))


def composition_key_from_symbols(symbols: list[str]) -> str:
    return composition_key_from_atomic_numbers([Element(symbol).Z for symbol in symbols])


def composition_key_from_structure(structure: Structure) -> str:
    return composition_key_from_atomic_numbers([Element(str(site.specie)).Z for site in structure])


def json_numpy_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def md5sum(path: Path, chunk_size: int = 1024 * 1024 * 8) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_with_resume(url: str, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl:
        start = time.time()
        total_size = MPTRJ_SIZE_BYTES if "49034296" in url else None
        if total_size is None:
            probe = requests.get(url, stream=True, timeout=60, allow_redirects=True)
            probe.raise_for_status()
            total_size = int(probe.headers.get("content-length", "0")) or None
            probe.close()
        if total_size is None:
            return {"status": "blocked_unknown_content_length", "path": str(path), "url": url}
        downloaded = tmp.stat().st_size if tmp.exists() else 0
        chunk_size = 16 * 1024 * 1024
        while downloaded < total_size:
            end = min(downloaded + chunk_size - 1, total_size - 1)
            expected = end - downloaded + 1
            chunk_path = tmp.with_suffix(tmp.suffix + ".chunk")
            completed: subprocess.CompletedProcess[bytes] | None = None
            for attempt in range(1, 9):
                chunk_path.unlink(missing_ok=True)
                cmd = [
                    curl,
                    "-L",
                    "--fail",
                    "--retry",
                    "5",
                    "--retry-delay",
                    "10",
                    "--range",
                    f"{downloaded}-{end}",
                    "-o",
                    str(chunk_path),
                    url,
                ]
                completed = subprocess.run(cmd, check=False)
                if completed.returncode == 0 and chunk_path.exists():
                    break
                print(f"range {downloaded}-{end} failed on attempt {attempt}; retrying", flush=True)
                time.sleep(30)
            if completed is None or completed.returncode != 0 or not chunk_path.exists():
                return {
                    "status": "curl_range_failed",
                    "path": str(path),
                    "partial_path": str(tmp),
                    "returncode": completed.returncode if completed else None,
                    "partial_bytes": downloaded,
                    "range": f"{downloaded}-{end}",
                }
            got = chunk_path.stat().st_size
            if got != expected:
                return {
                    "status": "curl_range_size_mismatch",
                    "path": str(path),
                    "partial_path": str(tmp),
                    "partial_bytes": downloaded,
                    "range": f"{downloaded}-{end}",
                    "expected_chunk_bytes": expected,
                    "got_chunk_bytes": got,
                }
            with tmp.open("ab") as out, chunk_path.open("rb") as src:
                shutil.copyfileobj(src, out, length=1024 * 1024 * 8)
            chunk_path.unlink(missing_ok=True)
            downloaded += got
            print(f"downloaded {downloaded}/{total_size} bytes ({downloaded / total_size:.1%})", flush=True)
        tmp.replace(path)
        return {
            "status": "downloaded",
            "path": str(path),
            "bytes": path.stat().st_size,
            "elapsed_sec": round(time.time() - start, 1),
            "download_tool": "curl",
        }
    downloaded = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    start = time.time()
    with requests.get(url, stream=True, timeout=60, headers=headers, allow_redirects=True) as response:
        if response.status_code == 416:
            tmp.replace(path)
            return {"status": "already_complete_by_range", "path": str(path), "bytes": path.stat().st_size}
        response.raise_for_status()
        if downloaded and response.status_code != 206:
            downloaded = 0
            tmp.unlink(missing_ok=True)
        mode = "ab" if downloaded else "wb"
        total = response.headers.get("content-length")
        with tmp.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    handle.write(chunk)
    tmp.replace(path)
    return {
        "status": "downloaded",
        "path": str(path),
        "bytes": path.stat().st_size,
        "elapsed_sec": round(time.time() - start, 1),
        "content_length_header": total,
    }


def ensure_mptrj(args: argparse.Namespace) -> dict[str, Any]:
    raw_path = args.raw_mptrj_dir / MPTRJ_FILENAME
    if raw_path.exists():
        digest = md5sum(raw_path)
        return {
            "status": "available_md5_ok" if digest == MPTRJ_MD5 else "available_md5_mismatch",
            "path": str(raw_path),
            "bytes": raw_path.stat().st_size,
            "md5": digest,
            "expected_md5": MPTRJ_MD5,
        }
    if not args.download_mptrj:
        partial = raw_path.with_suffix(raw_path.suffix + ".part")
        partial_bytes = partial.stat().st_size if partial.exists() else 0
        return {
            "status": "partial_download_present" if partial_bytes > 0 else "missing_download_not_requested",
            "path": str(raw_path),
            "partial_path": str(partial),
            "partial_bytes": partial_bytes,
            "url": MPTRJ_URL,
            "expected_md5": MPTRJ_MD5,
        }
    try:
        result = download_with_resume(MPTRJ_URL, raw_path)
        if not raw_path.exists():
            return {
                **result,
                "expected_md5": MPTRJ_MD5,
                "url": MPTRJ_URL,
            }
        digest = md5sum(raw_path)
        result |= {"md5": digest, "expected_md5": MPTRJ_MD5}
        result["status"] = "downloaded_md5_ok" if digest == MPTRJ_MD5 else "downloaded_md5_mismatch"
        return result
    except Exception as exc:
        fallback_path = args.raw_mptrj_dir / "2023-11-22-mp-trj.extxyz.zip"
        try:
            result = download_with_resume(MPTRJ_FALLBACK_URL, fallback_path)
            result |= {"md5": md5sum(fallback_path), "expected_md5": None, "fallback": True}
            return result
        except Exception as fallback_exc:
            return {
                "status": "download_failed",
                "path": str(raw_path),
                "url": MPTRJ_URL,
                "error": repr(exc),
                "fallback_error": repr(fallback_exc),
            }


def load_target_keys(labels: pd.DataFrame) -> pd.DataFrame:
    sub = labels[labels["model"].isin(MACE_CHGNET)].copy()
    cols = ["pbe_job_id", "composition_key", "structure_nsites", "composition_candidate_count"]
    targets = sub[cols].drop_duplicates("pbe_job_id")
    targets["structure_nsites"] = pd.to_numeric(targets["structure_nsites"], errors="coerce").astype("Int64")
    return targets


def quoted_extxyz_value(header: str, key: str) -> str | None:
    match = re.search(rf'{re.escape(key)}="([^"]*)"', header)
    if match:
        return match.group(1)
    match = re.search(rf"{re.escape(key)}=([^\s]+)", header)
    return match.group(1) if match else None


def structure_from_extxyz_frame(header: str, atom_lines: list[str]) -> Structure | None:
    lattice_text = quoted_extxyz_value(header, "Lattice")
    if not lattice_text:
        return None
    lattice_values = [float(value) for value in lattice_text.split()]
    if len(lattice_values) != 9:
        return None
    species: list[str] = []
    coords: list[list[float]] = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            return None
        species.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    lattice = [lattice_values[0:3], lattice_values[3:6], lattice_values[6:9]]
    return Structure(lattice, species, coords, coords_are_cartesian=True)


def extract_mptrj_candidates(raw_zip: Path, targets: pd.DataFrame, out_path: Path, max_frames: int | None) -> dict[str, Any]:
    if ase is None:
        return {"status": "blocked_ase_unavailable"}
    if out_path.exists() and not max_frames:
        frame = pd.read_parquet(out_path, columns=["composition_key"])
        return {
            "status": "available_cached",
            "path": str(out_path),
            "rows": int(len(frame)),
            "unique_keys": int(frame["composition_key"].nunique()),
        }
    key_to_nsites = {
        key: set(pd.to_numeric(group["structure_nsites"], errors="coerce").dropna().astype(int).tolist())
        for key, group in targets.groupby("composition_key")
    }
    target_keys = set(key_to_nsites)
    rows: list[dict[str, Any]] = []
    scanned = 0
    scanned_members = 0
    matched_key = 0
    matched_key_nsites = 0
    malformed_frames = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw_zip) as archive:
        names = [name for name in archive.namelist() if name.endswith(".extxyz")]
        if not names:
            return {"status": "blocked_no_extxyz_member", "path": str(raw_zip)}
        for name in names:
            scanned_members += 1
            with archive.open(name, "r") as binary:
                text = io.TextIOWrapper(binary, encoding="utf-8")
                while True:
                    count_line = text.readline()
                    if not count_line:
                        break
                    count_line = count_line.strip()
                    if not count_line:
                        continue
                    try:
                        nsites = int(count_line.split()[0])
                    except ValueError:
                        malformed_frames += 1
                        break
                    header = text.readline()
                    if not header:
                        malformed_frames += 1
                        break
                    atom_lines = [text.readline() for _ in range(nsites)]
                    if len(atom_lines) != nsites or any(not line for line in atom_lines):
                        malformed_frames += 1
                        break
                    scanned += 1
                    symbols = [line.split()[0] for line in atom_lines if line.strip()]
                    if len(symbols) != nsites:
                        malformed_frames += 1
                        continue
                    key = composition_key_from_symbols(symbols)
                    if key in target_keys:
                        matched_key += 1
                        if nsites in key_to_nsites[key]:
                            matched_key_nsites += 1
                            structure = structure_from_extxyz_frame(header, atom_lines)
                            if structure is None:
                                malformed_frames += 1
                                continue
                            source_id = quoted_extxyz_value(header, "source_id")
                            material_id = Path(name).stem
                            frame_id = source_id
                            rows.append(
                                {
                                    "training_id": source_id or f"{material_id}:frame_{scanned}",
                                    "material_id": material_id,
                                    "frame_id": frame_id,
                                    "composition_key": key,
                                    "nsites": nsites,
                                    "source_zip_member": name,
                                    "structure_json": json.dumps(
                                        structure.as_dict(),
                                        separators=(",", ":"),
                                        default=json_numpy_default,
                                    ),
                                }
                            )
                    if max_frames and scanned >= max_frames:
                        break
                    if scanned % 100000 == 0:
                        print(
                            f"scanned_frames={scanned} scanned_members={scanned_members}/{len(names)} "
                            f"matched_key={matched_key} matched_key_nsites={matched_key_nsites} rows={len(rows)}",
                            flush=True,
                        )
            if max_frames and scanned >= max_frames:
                break
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return {
        "status": "extracted",
        "path": str(out_path),
        "rows": len(rows),
        "unique_keys": int(pd.Series([row["composition_key"] for row in rows]).nunique()) if rows else 0,
        "scanned_frames": scanned,
        "scanned_members": scanned_members,
        "matched_key_frames": matched_key,
        "matched_key_nsites_frames": matched_key_nsites,
        "malformed_frames": malformed_frames,
    }


def target_structure_path(job_id: str, root: Path) -> tuple[Path | None, str]:
    base = root / job_id
    candidates = [
        (base / "static" / "CONTCAR", "static_CONTCAR"),
        (base / "static" / "POSCAR", "static_POSCAR"),
        (base / "relax" / "CONTCAR", "relax_CONTCAR"),
    ]
    for path, label in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path, label
    return None, "missing"


def run_mptrj_structure_matches(labels: pd.DataFrame, candidates_path: Path, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_exists = candidates_path.exists()
    if candidate_exists:
        candidates = pd.read_parquet(candidates_path)
    else:
        candidates = pd.DataFrame()
    strict_matcher = d3.StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True, attempt_supercell=False)
    loose_matcher = d3.StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10, primitive_cell=True, scale=True, attempt_supercell=False)
    target_cache: dict[str, tuple[Structure | None, str, str | None]] = {}
    candidate_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    sub = labels[labels["model"].isin(MACE_CHGNET)].copy()
    for _, row in sub.iterrows():
        job_id = str(row["pbe_job_id"])
        key = str(row["composition_key"])
        nsites = int(row["structure_nsites"])
        if job_id not in target_cache:
            path, source = target_structure_path(job_id, root)
            target_cache[job_id] = (Structure.from_file(path) if path else None, source, str(path) if path else None)
        target, target_source, target_path = target_cache[job_id]
        cand_key = (key, nsites)
        if cand_key not in candidate_cache:
            if candidate_exists and not candidates.empty:
                cand = candidates[(candidates["composition_key"].astype(str) == key) & (pd.to_numeric(candidates["nsites"], errors="coerce") == nsites)]
                candidate_cache[cand_key] = cand.to_dict("records")
            else:
                candidate_cache[cand_key] = []
        candidate_records = candidate_cache[cand_key]
        strict = False
        loose = False
        matched_ids: list[str] = []
        status = "no_candidate_structure_cache" if not candidate_exists else "no_candidate_structure"
        if target is None:
            status = "target_structure_missing"
            strict_value: bool | float = math.nan
            loose_value: bool | float = math.nan
        elif not candidate_exists:
            strict_value = math.nan
            loose_value = math.nan
        elif not candidate_records:
            strict_value = False
            loose_value = False
        else:
            for cand in candidate_records:
                try:
                    struct = Structure.from_dict(json.loads(cand["structure_json"]))
                except Exception:
                    continue
                cand_id = str(cand.get("training_id", ""))
                if strict_matcher.fit(target, struct):
                    strict = True
                    loose = True
                    matched_ids.append(cand_id)
                    status = "strict_match"
                    break
                if loose_matcher.fit(target, struct):
                    loose = True
                    matched_ids.append(cand_id)
                    status = "loose_match"
            else:
                status = "no_structure_match"
            strict_value = strict
            loose_value = loose
        match_row = {
            "pbe_job_id": job_id,
            "model": row["model"],
            "composition_key": key,
            "nsites": nsites,
            "target_structure_source": target_source,
            "target_structure_path": target_path,
            "candidate_structure_cache_available": candidate_exists,
            "candidate_structure_count": len(candidate_records),
            "in_mptrj_strict": strict_value,
            "in_mptrj_loose": loose_value,
            "held_out_mptrj_strict": (not strict_value) if isinstance(strict_value, bool) else math.nan,
            "held_out_mptrj_loose": (not loose_value) if isinstance(loose_value, bool) else math.nan,
            "match_status": status,
            "matched_training_ids": " ".join(matched_ids),
        }
        rows.append(match_row)
        label_rows.append({**row.to_dict(), **match_row, "coverage_note": "mace_mptrj_controlled_not_full_training" if row["model"] == "mace" else "chgnet_mptrj_training_controlled"})
    return pd.DataFrame(rows), pd.DataFrame(label_rows)


def cross_family_tables(d3_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair = pd.read_csv(d3_dir / "model_pairwise_consistency.csv")
    dft = pd.read_csv(d3_dir / "dft_pairwise_consistency.csv")
    cross = pair[pair["pair"].isin(CROSS_FAMILY_PAIRS)].copy()
    homo = pair[pair["pair"] == HOMOLOGOUS_PAIR].copy()
    pbe3 = dft[dft["database_group"] == "pbe3"].copy()
    all4 = dft[dft["database_group"] == "all4"].copy()
    rows: list[dict[str, Any]] = []
    for name, frame in [
        ("cross_family_excl_orb_sevennet", cross),
        ("homologous_orb_sevennet", homo),
        ("dft_pbe3", pbe3),
        ("dft_all4", all4),
    ]:
        width = frame.groupby("pbe_job_id")["abs_diff_eV_atom"].median()
        rows.append(
            {
                "group": name,
                "n_materials": int(width.notna().sum()),
                "n_pair_values": int(frame["abs_diff_eV_atom"].notna().sum()),
                "pooled_pair_median_eV_atom": float(frame["abs_diff_eV_atom"].median()),
                "pooled_pair_mean_eV_atom": float(frame["abs_diff_eV_atom"].mean()),
                "per_material_width_median_eV_atom": float(width.median()),
                "per_material_width_mean_eV_atom": float(width.mean()),
            }
        )
    pbe_width = pbe3.groupby("pbe_job_id")["abs_diff_eV_atom"].median()
    cross_width = cross.groupby("pbe_job_id")["abs_diff_eV_atom"].median()
    joined = pd.concat([cross_width.rename("cross_family_width"), pbe_width.rename("dft_pbe3_width")], axis=1).dropna()
    if not joined.empty and not np.allclose(joined["cross_family_width"] - joined["dft_pbe3_width"], 0):
        p_value = float(wilcoxon(joined["cross_family_width"], joined["dft_pbe3_width"], alternative="less").pvalue)
    else:
        p_value = math.nan
    rows.append(
        {
            "group": "cross_family_vs_dft_pbe3_test",
            "n_materials": int(len(joined)),
            "n_pair_values": int(len(joined)),
            "pooled_pair_median_eV_atom": math.nan,
            "pooled_pair_mean_eV_atom": math.nan,
            "per_material_width_median_eV_atom": float((joined["cross_family_width"] - joined["dft_pbe3_width"]).median()),
            "per_material_width_mean_eV_atom": float((joined["cross_family_width"] - joined["dft_pbe3_width"]).mean()),
            "cross_width_less_than_dft_rate": float((joined["cross_family_width"] < joined["dft_pbe3_width"]).mean()),
            "wilcoxon_cross_less_than_dft_p": p_value,
            "pass": bool(cross["abs_diff_eV_atom"].median() < pbe3["abs_diff_eV_atom"].median()),
        }
    )
    return cross, pd.DataFrame(rows)


def heldout_gate2(labels: pd.DataFrame, d3_dir: Path) -> pd.DataFrame:
    pair = pd.read_csv(d3_dir / "model_pairwise_consistency.csv")
    dft = pd.read_csv(d3_dir / "dft_pairwise_consistency.csv")
    heldout_jobs = set(labels[(labels["model"].isin(MACE_CHGNET)) & (labels["held_out_mptrj_strict"] == True)]["pbe_job_id"])
    rows: list[dict[str, Any]] = []
    if not heldout_jobs:
        return pd.DataFrame(
            [
                {
                    "subset": "mace_chgnet_mptrj_strict_heldout",
                    "n_materials": 0,
                    "status": "underpowered_or_unavailable",
                }
            ]
        )
    model_pair = pair[(pair["pbe_job_id"].isin(heldout_jobs)) & (pair["pair"] == "chgnet__mace")].copy()
    dft_pbe3 = dft[(dft["pbe_job_id"].isin(heldout_jobs)) & (dft["database_group"] == "pbe3")].copy()
    model_width = model_pair.groupby("pbe_job_id")["abs_diff_eV_atom"].median()
    dft_width = dft_pbe3.groupby("pbe_job_id")["abs_diff_eV_atom"].median()
    joined = pd.concat([model_width.rename("model_width"), dft_width.rename("dft_width")], axis=1).dropna()
    rows.append(
        {
            "subset": "mace_chgnet_mptrj_strict_heldout",
            "n_materials": int(len(joined)),
            "model_pair": "chgnet__mace",
            "model_pooled_pair_median_eV_atom": float(model_pair["abs_diff_eV_atom"].median()) if not model_pair.empty else math.nan,
            "dft_pbe3_pooled_pair_median_eV_atom": float(dft_pbe3["abs_diff_eV_atom"].median()) if not dft_pbe3.empty else math.nan,
            "model_per_material_width_median_eV_atom": float(joined["model_width"].median()) if not joined.empty else math.nan,
            "dft_per_material_width_median_eV_atom": float(joined["dft_width"].median()) if not joined.empty else math.nan,
            "model_width_less_than_dft_rate": float((joined["model_width"] < joined["dft_width"]).mean()) if not joined.empty else math.nan,
            "status": "ok" if len(joined) >= 30 else "underpowered",
        }
    )
    return pd.DataFrame(rows)


def heldout_consensus(labels: pd.DataFrame, d3_dir: Path) -> pd.DataFrame:
    details_path = d3_dir / "consensus_ambiguity_summary.csv"
    pred = pd.read_csv(DATA_ROOT / "processed" / "database_relative_model_checks" / "frontier_model_predictions_clean.csv")
    floor = pd.read_csv(d3_dir / "pairwise_floor_table.csv")
    heldout = set(labels[(labels["model"].isin(MACE_CHGNET)) & (labels["held_out_mptrj_strict"] == True)]["pbe_job_id"])
    if not heldout:
        return pd.DataFrame([{"subset": "mace_chgnet_mptrj_strict_heldout", "status": "underpowered_or_unavailable"}])
    noise = pd.read_csv(DATA_ROOT / "processed" / "dual_noise_floor_checks" / "dual_noise_floors.csv")
    pred = pred[(pred["model"].isin(MACE_CHGNET)) & (pred["pbe_job_id"].isin(heldout))].merge(
        noise[["pbe_job_id", "consensus_pbe3_median", "consensus_all4_median", "Ef_MP", "Ef_OQMD", "Ef_AFLOW", "Ef_JARVIS"]],
        on="pbe_job_id",
        how="left",
    )
    rows: list[dict[str, Any]] = []
    for model, sub in pred.groupby("model"):
        for consensus in ["consensus_pbe3_median", "consensus_all4_median"]:
            dist = (pd.to_numeric(sub[d3.PRED_COL], errors="coerce") - pd.to_numeric(sub[consensus], errors="coerce")).abs()
            rows.append(
                {
                    "subset": "mace_chgnet_mptrj_strict_heldout",
                    "entity_type": "model",
                    "entity": model,
                    "consensus": consensus.replace("consensus_", "").replace("_median", ""),
                    "n": int(dist.notna().sum()),
                    "median_distance_eV_atom": float(dist.median()),
                    "mean_distance_eV_atom": float(dist.mean()),
                    "status": "ok" if dist.notna().sum() >= 30 else "underpowered",
                }
            )
    return pd.DataFrame(rows)


def heldout_element_map(labels: pd.DataFrame, d3_dir: Path) -> pd.DataFrame:
    heldout_jobs = set(labels[(labels["model"].isin(MACE_CHGNET)) & (labels["held_out_mptrj_strict"] == True)]["pbe_job_id"])
    if not heldout_jobs:
        return pd.DataFrame([{"status": "underpowered_or_unavailable", "n_materials": 0}])
    element = pd.read_csv(d3_dir / "element_frontier_map.csv")
    pred = pd.read_csv(DATA_ROOT / "processed" / "database_relative_model_checks" / "frontier_model_predictions_clean.csv")
    floor = pd.read_csv(DATA_ROOT / "processed" / "dual_noise_floor_checks" / "dual_noise_floors.csv")
    # Reuse D3 element map logic on the held-out material subset, but only for CHGNet/MACE.
    pred = pred[(pred["model"].isin(MACE_CHGNET)) & (pred["pbe_job_id"].isin(heldout_jobs))].copy()
    old_frontier = d3.FRONTIER4[:]
    d3.FRONTIER4[:] = MACE_CHGNET
    try:
        frame = d3.element_frontier_map(pred, floor[floor["pbe_job_id"].isin(heldout_jobs)].copy(), np.random.default_rng(20260606), 1000)
        frame["subset"] = "mace_chgnet_mptrj_strict_heldout"
        return frame
    finally:
        d3.FRONTIER4[:] = old_frontier


def write_report(
    args: argparse.Namespace,
    summary: dict[str, Any],
    cross_summary: pd.DataFrame,
    matches_summary: pd.DataFrame,
    heldout_gate: pd.DataFrame,
    consensus: pd.DataFrame,
) -> Path:
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    path = args.docs_dir / "leakage_controlled_consistency_results_20260606.md"
    def md_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No rows._"
        return frame.to_markdown(index=False)
    text = "\n".join(
        [
            "# MatAlign v3 Phase D3-fix Results",
            "",
            "## Summary",
            "",
            f"- Cross-family pooled median: `{summary['cross_family_pooled_median_eV_atom']:.6f}` eV/atom.",
            f"- Cross-family per-material width median: `{summary['cross_family_per_material_width_median_eV_atom']:.6f}` eV/atom.",
            f"- ORB-SevenNet homologous pair median: `{summary['orb_sevennet_pair_median_eV_atom']:.6f}` eV/atom.",
            f"- PBE3 DFT pooled median: `{summary['pbe3_pooled_median_eV_atom']:.6f}` eV/atom.",
            f"- MPtrj status: `{summary['mptrj_status']}`.",
            f"- MPtrj candidate rows: `{summary['mptrj_candidate_rows']}`.",
            f"- MPtrj strict in-sample counts: `{summary['mptrj_strict_in_sample_counts_by_model']}`.",
            f"- MPtrj strict held-out counts: `{summary['mptrj_strict_heldout_counts_by_model']}`.",
            f"- NMI/NCS candidate: `{summary['nmi_ncs_candidate']}`.",
            "",
            "## G2b Cross-Family Consistency",
            "",
            md_table(cross_summary),
            "",
            "## G1 MPtrj Structure Matching",
            "",
            md_table(matches_summary),
            "",
            "## Held-out Gate 2",
            "",
            md_table(heldout_gate),
            "",
            "## Held-out Consensus",
            "",
            md_table(consensus),
            "",
            "## Interpretation",
            "",
            "- The cross-family descriptive result is stronger than D3 because it removes the ORB-SevenNet homologous pair from the main number.",
            "- MPtrj-controlled held-out is only a strong claim when candidate structures are available and strict held-out n is at least 30.",
            "- MACE is MPtrj-controlled only; this does not exclude other possible training-source leakage.",
        ]
    )
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_mptrj_dir.mkdir(parents=True, exist_ok=True)
    args.candidate_dir.mkdir(parents=True, exist_ok=True)

    cross, cross_summary = cross_family_tables(args.d3_dir)
    cross.to_csv(args.out_dir / "cross_family_pairwise_consistency.csv", index=False)
    cross_summary.to_csv(args.out_dir / "cross_family_pairwise_summary.csv", index=False)

    labels_d3 = pd.read_csv(args.d3_dir / "per_model_heldout_labels.csv")
    targets = load_target_keys(labels_d3)
    raw_status = ensure_mptrj(args)
    candidate_path = args.candidate_dir / "mptrj_candidates.parquet"
    local_candidate_export = args.out_dir / "mptrj_candidate_structures.parquet"
    extract_status: dict[str, Any]
    raw_path = Path(raw_status.get("path", ""))
    if args.download_only:
        extract_status = {"status": "not_run_download_only"}
    elif raw_status["status"] in {"available_md5_ok", "downloaded_md5_ok", "available_md5_mismatch", "downloaded_md5_mismatch"} and raw_path.exists():
        if args.force_extract:
            candidate_path.unlink(missing_ok=True)
        extract_status = extract_mptrj_candidates(raw_path, targets, candidate_path, args.max_frames)
    else:
        extract_status = {"status": "blocked_mptrj_raw_unavailable"}
    if candidate_path.exists():
        cand = pd.read_parquet(candidate_path)
        cand.to_parquet(local_candidate_export, index=False)
    else:
        pd.DataFrame(
            columns=["training_id", "material_id", "frame_id", "composition_key", "nsites", "structure_json"]
        ).to_parquet(local_candidate_export, index=False)

    matches, heldout_labels = run_mptrj_structure_matches(labels_d3, candidate_path, args.target_structure_root)
    matches.to_csv(args.out_dir / "mptrj_structure_matches.csv", index=False)
    heldout_labels.to_csv(args.out_dir / "mace_chgnet_mptrj_heldout_labels.csv", index=False)
    heldout_gate = heldout_gate2(heldout_labels, args.d3_dir)
    heldout_gate.to_csv(args.out_dir / "heldout_gate2_consistency_tests.csv", index=False)
    consensus = heldout_consensus(heldout_labels, args.d3_dir)
    consensus.to_csv(args.out_dir / "heldout_consensus_distance_summary.csv", index=False)
    element = heldout_element_map(heldout_labels, args.d3_dir)
    element.to_csv(args.out_dir / "heldout_element_frontier_map.csv", index=False)

    match_summary = (
        heldout_labels.groupby(["model", "coverage_note", "match_status"], dropna=False)
        .agg(
            rows=("pbe_job_id", "count"),
            candidate_structure_rows=("candidate_structure_count", "sum"),
            strict_in_sample=("in_mptrj_strict", lambda s: int((s == True).sum())),
            strict_heldout=("held_out_mptrj_strict", lambda s: int((s == True).sum())),
        )
        .reset_index()
    )

    cross_row = cross_summary[cross_summary["group"] == "cross_family_excl_orb_sevennet"].iloc[0]
    homo_row = cross_summary[cross_summary["group"] == "homologous_orb_sevennet"].iloc[0]
    pbe_row = cross_summary[cross_summary["group"] == "dft_pbe3"].iloc[0]
    strict_in_sample_counts = {
        model: int(heldout_labels[(heldout_labels["model"] == model) & (heldout_labels["in_mptrj_strict"] == True)]["pbe_job_id"].nunique())
        for model in MACE_CHGNET
    }
    heldout_counts = {
        model: int(heldout_labels[(heldout_labels["model"] == model) & (heldout_labels["held_out_mptrj_strict"] == True)]["pbe_job_id"].nunique())
        for model in MACE_CHGNET
    }
    loose_heldout_counts = {
        model: int(heldout_labels[(heldout_labels["model"] == model) & (heldout_labels["held_out_mptrj_loose"] == True)]["pbe_job_id"].nunique())
        for model in MACE_CHGNET
    }
    heldout_ok = all(heldout_counts.get(model, 0) >= 30 for model in MACE_CHGNET)
    heldout_gate_ok = False
    if not heldout_gate.empty and "status" in heldout_gate.columns and heldout_gate.iloc[0].get("status") == "ok":
        heldout_gate_ok = bool(
            heldout_gate.iloc[0].get("model_pooled_pair_median_eV_atom", math.inf)
            < heldout_gate.iloc[0].get("dft_pbe3_pooled_pair_median_eV_atom", -math.inf)
        )
    summary = {
        "cross_family_pooled_median_eV_atom": float(cross_row["pooled_pair_median_eV_atom"]),
        "cross_family_per_material_width_median_eV_atom": float(cross_row["per_material_width_median_eV_atom"]),
        "orb_sevennet_pair_median_eV_atom": float(homo_row["pooled_pair_median_eV_atom"]),
        "pbe3_pooled_median_eV_atom": float(pbe_row["pooled_pair_median_eV_atom"]),
        "g2b_cross_family_pass": bool(cross_row["pooled_pair_median_eV_atom"] < pbe_row["pooled_pair_median_eV_atom"]),
        "target_composition_keys": int(targets["composition_key"].nunique()),
        "mptrj_status": raw_status.get("status"),
        "mptrj_raw": raw_status,
        "mptrj_extract": extract_status,
        "mptrj_candidate_rows": int(pd.read_parquet(candidate_path).shape[0]) if candidate_path.exists() else 0,
        "mptrj_strict_in_sample_counts_by_model": strict_in_sample_counts,
        "mptrj_strict_heldout_counts_by_model": heldout_counts,
        "mptrj_loose_heldout_counts_by_model": loose_heldout_counts,
        "heldout_sufficient_for_mace_chgnet": heldout_ok,
        "heldout_gate2_pass": heldout_gate_ok,
        "nmi_ncs_candidate": bool(cross_row["pooled_pair_median_eV_atom"] < pbe_row["pooled_pair_median_eV_atom"] and heldout_ok and heldout_gate_ok),
        "route_recommendation": "NMI_NCS_candidate" if heldout_ok and heldout_gate_ok else "npj_or_Digital_Discovery_until_MPtrj_heldout_passes",
    }
    (args.out_dir / "leakage_controlled_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = write_report(args, summary, cross_summary, match_summary, heldout_gate, consensus)
    print(json.dumps({**summary, "report_path": str(report), "output_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
