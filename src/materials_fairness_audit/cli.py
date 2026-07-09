from __future__ import annotations

import argparse
from pprint import pprint

from .config import load_settings
from .phase0 import run_phase0


def env_main() -> None:
    settings = load_settings()
    settings.paths.ensure()
    summary = {
        "project_name": settings.project_name,
        "config_path": str(settings.config_path),
        "data_root": str(settings.paths.data_root),
        "raw": str(settings.paths.raw),
        "processed": str(settings.paths.processed),
        "processed_matbench": str(settings.paths.processed_matbench),
        "processed_databases": str(settings.paths.processed_databases),
        "processed_matalign": str(settings.paths.processed_matalign),
        "processed_audit": str(settings.paths.processed_audit),
        "official_repo": str(settings.paths.official_repo),
        "has_mp_api_key": bool(settings.mp_api_key),
    }
    pprint(summary)


def phase0_main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 0 data acquisition and metadata merge.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    settings = load_settings(args.config)
    report = run_phase0(settings, max_files=args.max_files, dry_run=args.dry_run)
    pprint(report)


if __name__ == "__main__":
    phase0_main()
