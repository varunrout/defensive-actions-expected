from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dax.models.feature_contracts import get_contracts, load_model_config, resolve_contract
from dax.models.schemas import dataset_fingerprint, normalise_model_schema, validate_model_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a defensive-action modelling dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--output", default="outputs/models/reports/model_data_audit.json")
    return parser.parse_args(argv)


def audit_variants(df: pd.DataFrame, config: dict) -> dict[str, dict[str, dict]]:
    variants: dict[str, dict[str, dict]] = {}
    for task in ["classification", "regression"]:
        variants[task] = {}
        for contract in get_contracts(config, task):
            try:
                variants[task][contract.name] = resolve_contract(df, contract)
            except ValueError as exc:
                variants[task][contract.name] = {"error": str(exc)}
    return variants


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    df = normalise_model_schema(pd.read_parquet(args.input))
    audit = validate_model_dataset(df)
    config = load_model_config(args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audit": audit.__dict__,
        "fingerprint": dataset_fingerprint(args.input, df),
        "variants": audit_variants(df, config),
    }
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(audit)


if __name__ == "__main__":
    main()
