# MatAlign Reference Audit

This repository contains the source code and analysis scripts used to reproduce the
processed MatAlign reference-audit tables from the separately released data package.

## Layout

```text
pyproject.toml
.python-version
configs/project.toml
src/materials_fairness_audit/
scripts/
MANIFEST.csv
CHECKSUMS.sha256
```

## Environment

Python `3.11` is required. The package is managed with `uv`.

```powershell
uv sync
```

If `uv.lock` is absent, `uv` will resolve dependencies from `pyproject.toml`.

## Using Released Data

The processed data are released separately on Figshare or an equivalent data repository.
After downloading and extracting that data package, set the local data path in
`configs/project.toml`.

Most later-stage scripts also accept explicit command-line paths, for example:

```powershell
uv run python scripts/phase14_phase_f_nc_upgrade.py --data-root <path-to-downloaded-data>
```

## Main Script Groups

- `phase10_*`: PBE validation material selection and QC table handling.
- `phase11_*`: clean-subset master table construction and model-vs-reference summaries.
- `phase12_*`: database-relative errors, leakage labels, dual reference-width analysis, pairwise consistency, and held-out checks.
- `phase13_*`: WBM validation on the clean-metal subset.
- `phase14_*`: Phase F distance-response, discovery-list overlap, full-chemistry consistency, and required supplement checks.
- `phase1_build_elements.py`, `phase2_build_matalign.py`, `phase23_compute_metrics.py`, `phase34_recompute_audit.py`, `phase46_analysis.py`: earlier MatAlign table construction and auditing utilities.

## Exclusions

This repository intentionally excludes:

- Remote execution and job-control scripts.
- Raw VASP input/output generation scripts.
- Machine-specific logs and cache files.
- Private configuration files.
- Raw database dumps.

## Basic Verification

Run a syntax check:

```powershell
uv run python -m py_compile (Get-ChildItem -Recurse -Filter *.py | ForEach-Object { $_.FullName })
```

On non-PowerShell shells, run the equivalent `python -m py_compile` over all `.py` files.
