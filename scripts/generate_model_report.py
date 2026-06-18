from __future__ import annotations

import argparse

from dax.models.reporting import write_model_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a model validation report.")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--output", default="outputs/models/reports/model_validation_report.md")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--mlflow-enabled", action="store_true")
    parser.add_argument("--tracking-uri")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(write_model_report(args.config, args.output, args.output_dir))


if __name__ == "__main__":
    main()
