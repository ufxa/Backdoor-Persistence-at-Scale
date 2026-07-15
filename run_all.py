#!/usr/bin/env python3
"""Run the paper experiment pipeline using an explicit execution profile."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CORE_SCRIPTS = (
    "src/run_crsc_experiment.py",
    "src/run_sensitivity.py",
    "src/run_extra_analyses.py",
    "src/run_v8_sensitivity.py",
    "src/run_v9_sensitivity.py",
)
FULL_ONLY_SCRIPTS = (
    "src/run_transformer_experiment.py",
    "src/run_vision_experiment.py",
    "src/run_vision_blended.py",
)


def run(script: str) -> None:
    print(f"\n==> {script}", flush=True)
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("core", "full"),
        default="core",
        help="core: CPU MLP pipeline; full: core plus transformer and vision",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scripts = CORE_SCRIPTS + (FULL_ONLY_SCRIPTS if args.profile == "full" else ())
    if args.dry_run:
        print("\n".join(scripts))
        return
    for script in scripts:
        run(script)


if __name__ == "__main__":
    main()
