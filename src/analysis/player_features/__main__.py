"""CLI entrypoint for player-feature analysis module."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rigorous multi-step player-feature analysis for DAx.")
    parser.add_argument("--repo-root", type=str, default=None, help="Repository root. Defaults to current working directory.")
    parser.add_argument("--input", type=str, default=None, help="Optional input parquet path.")
    parser.add_argument("--output-root", type=str, default=None, help="Optional output directory root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd().resolve()
    input_path = Path(args.input).resolve() if args.input else None
    output_root = Path(args.output_root).resolve() if args.output_root else None
    run_analysis(repo_root=repo_root, input_path=input_path, output_root=output_root, verbose=True)


if __name__ == "__main__":
    main()

