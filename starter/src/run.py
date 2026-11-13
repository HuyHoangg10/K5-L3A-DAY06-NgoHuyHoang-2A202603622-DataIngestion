"""Command-line entry point for the reproducible Day 06 pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_stage(name: str, function: Callable[[], int]) -> None:
    print(f"\n=== Running {name} ===", flush=True)
    status = function()
    if status:
        raise RuntimeError(f"{name} failed with status {status}")


def validate() -> int:
    """Run all lightweight validators without recomputing document embeddings."""
    benchmark_validator = REPO_ROOT / "benchmark" / "validate_benchmark.py"
    completed = subprocess.run([sys.executable, str(benchmark_validator)], check=False)
    if completed.returncode:
        return completed.returncode
    from .validate_cp5 import main as validate_cp5

    return validate_cp5()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or validate the Day 06 ingestion/retrieval experiment."
    )
    parser.add_argument(
        "--stage",
        choices=("cp3", "cp4", "cp5", "validate", "all"),
        default="validate",
        help=(
            "Stage to run. 'all' recomputes local embeddings and can take about "
            "10 minutes on CPU; default 'validate' is lightweight."
        ),
    )
    args = parser.parse_args(argv)

    if args.stage == "validate":
        return validate()

    from .cp3 import main as cp3
    from .cp4 import main as cp4
    from .cp5 import main as cp5

    stages: dict[str, Callable[[], int]] = {"cp3": cp3, "cp4": cp4, "cp5": cp5}
    if args.stage == "all":
        for name in ("cp3", "cp4", "cp5"):
            run_stage(name.upper(), stages[name])
        return validate()

    run_stage(args.stage.upper(), stages[args.stage])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
