from __future__ import annotations

import argparse

from dax.models.oof_signals import build_player_signals


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build provisional OOF player defensive signals.")
    parser.add_argument("--classification-oof", required=True)
    parser.add_argument("--regression-oof", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--mlflow-enabled", action="store_true")
    parser.add_argument("--tracking-uri")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    signals = build_player_signals(args.classification_oof, args.regression_oof, args.output)
    print(signals.shape)


if __name__ == "__main__":
    main()
