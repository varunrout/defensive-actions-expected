"""Compatibility wrapper for regression baseline training."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
