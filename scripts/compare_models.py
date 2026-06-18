from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate model comparison tables.")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--mlflow-enabled", action="store_true")
    parser.add_argument("--tracking-uri")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for task in ["classification", "regression"]:
        path = Path(args.output_dir) / "models" / "comparisons" / f"{task}_model_comparison.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        table["selection_note"] = "transparent multi-metric review required; no single-metric winner"
        table.to_csv(path, index=False)
        print(path)


if __name__ == "__main__":
    main()
