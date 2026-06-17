"""Build DAx feature tables from processed events."""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build DAx player feature tables.")
    parser.add_argument("--input", default="data/processed/events_with_targets.parquet")
    parser.add_argument("--output", default="data/features/player_defensive_actions.parquet")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(f"Feature build: {args.input} -> {args.output}")
    if args.dry_run:
        return 0
    raise SystemExit("Feature execution requires regenerated processed data; use --dry-run in this cleanup PR.")


if __name__ == "__main__":
    raise SystemExit(main())
