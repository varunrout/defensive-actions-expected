"""Compatibility wrapper for logistic baseline training."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dax.models.training import train_logistic_models
from dax.models.training import parse_args as parse_train_args


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
