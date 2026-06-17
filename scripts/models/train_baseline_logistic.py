"""Compatibility wrapper for logistic baseline training."""

from __future__ import annotations

from pathlib import Path
import sys

from dax.models.training import parse_args as parse_train_args
from dax.models.training import train_logistic_models


if __name__ == "__main__":
    args = parse_train_args(["--task", "logistic", *sys.argv[1:]])
    if args.dry_run:
        print(f"[dry-run] train task=logistic input={args.input}")
        raise SystemExit(0)
    train_logistic_models(
        args.input,
        models_dir=Path(args.models_dir) / "baseline",
        validation_dir=Path(args.validation_dir) / "baseline",
        oof_dir=Path(args.oof_dir) / "baseline",
        n_splits=args.n_splits,
        max_rows=args.max_rows,
    )
