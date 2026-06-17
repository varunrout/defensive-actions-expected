"""Train canonical DAx baseline model variants."""
from __future__ import annotations

import argparse

from dax.models.specs import list_model_variants


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DAx model variants.")
    parser.add_argument("--task", choices=["logistic", "regression"], default="logistic")
    parser.add_argument("--variant", default="all")
    parser.add_argument("--input", default="data/features/player_defensive_actions.parquet")
    parser.add_argument("--output", default="outputs/models")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(f"Available {args.task} variants: {', '.join(list_model_variants(args.task))}")
    print(f"Training request: variant={args.variant}, input={args.input}, output={args.output}")
    if args.dry_run:
        return 0
    raise SystemExit("Training requires regenerated features; use --dry-run in this cleanup PR.")


if __name__ == "__main__":
    raise SystemExit(main())
