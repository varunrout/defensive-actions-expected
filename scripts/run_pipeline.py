"""Canonical DAx pipeline entry point."""
from __future__ import annotations

import argparse

WORKFLOW = "download -> process -> event context -> phase proxies -> targets -> player features -> train -> validate -> report"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the canonical DAx workflow or selected stages.")
    parser.add_argument("--stage", choices=["all", "download", "process", "event-context", "phase-proxies", "targets", "player-features", "train", "validate", "report"], default="all")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned workflow without executing stages.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(f"DAx workflow: {WORKFLOW}")
    print(f"Selected stage: {args.stage}; config: {args.config}")
    if args.dry_run:
        return 0
    raise SystemExit("Full execution remains delegated to src/dax modules; use --dry-run or stage-specific scripts in this cleanup PR.")


if __name__ == "__main__":
    raise SystemExit(main())
