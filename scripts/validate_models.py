"""Validate generated DAx model artifacts."""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate DAx model outputs.")
    parser.add_argument("--task", choices=["logistic", "regression", "all"], default="all")
    parser.add_argument("--models", default="outputs/models")
    parser.add_argument("--metrics", default="outputs/metrics")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(f"Validation request: task={args.task}, models={args.models}, metrics={args.metrics}")
    if args.dry_run:
        return 0
    raise SystemExit("Validation requires regenerated model artifacts; use --dry-run in this cleanup PR.")


if __name__ == "__main__":
    raise SystemExit(main())
