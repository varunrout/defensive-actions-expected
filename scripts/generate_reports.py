"""Generate DAx reports from current metrics and figures."""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DAx reports.")
    parser.add_argument("--metrics", default="outputs/metrics")
    parser.add_argument("--figures", default="outputs/figures")
    parser.add_argument("--output", default="outputs/reports")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(f"Report request: metrics={args.metrics}, figures={args.figures}, output={args.output}")
    if args.dry_run:
        return 0
    raise SystemExit("Report generation requires current metrics; use --dry-run in this cleanup PR.")


if __name__ == "__main__":
    raise SystemExit(main())
