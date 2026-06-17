"""Compatibility wrapper for regression baseline training."""

from __future__ import annotations

from pathlib import Path
import sys

from dax.models.training import parse_args as parse_train_args
from dax.models.training import train_regression_models


if __name__ == "__main__":
    args = parse_train_args(["--task", "regression", *sys.argv[1:]])
    if args.dry_run:
        print(f"[dry-run] train task=regression input={args.input}")
        raise SystemExit(0)
    train_regression_models(
        args.input,
        models_dir=Path(args.models_dir) / "regression",
        validation_dir=Path(args.validation_dir) / "regression",
        oof_dir=Path(args.oof_dir) / "regression",
        n_splits=args.n_splits,
        max_rows=args.max_rows,
    )
