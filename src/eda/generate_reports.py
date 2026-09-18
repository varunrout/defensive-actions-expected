"""CLI entrypoint: regenerate the 6 static EDA HTML reports from feature parquets.

Usage:
    python -m src.eda.generate_reports --dataset all
    python -m src.eda.generate_reports --dataset active
    python -m src.eda.generate_reports --dataset passive
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eda import compute_stats as cs
from src.eda import render
from src.eda.feature_config import DATASETS

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "reports" / "analysis" / "shot_target"


def _check_duplicate_columns(df: pd.DataFrame, dataset_cfg: dict) -> None:
    check = dataset_cfg.get("duplicate_check")
    if not check:
        return
    for col_a, col_b in check.values():
        result = cs.check_duplicate_columns(df, col_a, col_b)
        if not result["present"]:
            continue
        if result["is_duplicate"]:
            print(
                f"  [warning] {col_a} is still 100% identical to {col_b} -- "
                f"known unresolved data issue, excluding both from the report as documented."
            )
        else:
            raise AssertionError(
                f"{col_a} is NO LONGER identical to {col_b}. The known duplicate-column issue "
                f"appears fixed -- update src/eda/feature_config.py to re-include these columns "
                f"and revise the prompt/README that documents this as a known issue."
            )


def generate_dataset_reports(dataset_key: str) -> dict:
    dataset_cfg = DATASETS[dataset_key]
    parquet_path = REPO_ROOT / dataset_cfg["parquet_path"]
    df = pd.read_parquet(parquet_path)

    _check_duplicate_columns(df, dataset_cfg)

    target_col = dataset_cfg["target_col"]
    n_rows = len(df)
    rate = cs.base_rate(df, target_col)

    cat_tables = {col: cs.categorical_rate_table(df, col, target_col) for col in dataset_cfg["categorical"]}
    lifts = [cs.boolean_lift(df, col, target_col) for col in dataset_cfg["boolean"]]
    cont_stats = {col: cs.continuous_distribution(df, col) for col in dataset_cfg["continuous"]}
    disc_stats = {col: cs.discrete_distribution(df, col) for col in dataset_cfg["discrete"]}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        f"{dataset_key}_category_atlas.html": render.render_category_atlas(dataset_cfg, rate, n_rows, cat_tables),
        f"{dataset_key}_flag_ledger.html": render.render_flag_ledger(dataset_cfg, rate, n_rows, lifts),
        f"{dataset_key}_distribution_atlas.html": render.render_distribution_atlas(
            dataset_cfg, rate, n_rows, cont_stats, disc_stats
        ),
    }
    for filename, html_content in outputs.items():
        (OUTPUT_DIR / filename).write_text(html_content, encoding="utf-8")

    return {
        "dataset": dataset_key,
        "n_rows": n_rows,
        "base_rate": rate,
        "n_categorical": len(dataset_cfg["categorical"]),
        "n_boolean": len(dataset_cfg["boolean"]),
        "n_continuous": len(dataset_cfg["continuous"]),
        "n_discrete": len(dataset_cfg["discrete"]),
        "files": list(outputs.keys()),
    }


def _print_summary(summaries: list[dict]) -> None:
    header = f"{'dataset':<10} {'rows':>10} {'base_rate':>10} {'cat':>5} {'bool':>5} {'cont':>5} {'disc':>5}"
    print("\n" + header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['dataset']:<10} {s['n_rows']:>10,} {s['base_rate']:>9.2f}% "
            f"{s['n_categorical']:>5} {s['n_boolean']:>5} {s['n_continuous']:>5} {s['n_discrete']:>5}"
        )
    print(f"\nWrote {sum(len(s['files']) for s in summaries)} HTML files to {OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["active", "passive", "all"], default="all")
    args = parser.parse_args()

    keys = ["active", "passive"] if args.dataset == "all" else [args.dataset]
    summaries = []
    for key in keys:
        print(f"Generating {key} reports...")
        summaries.append(generate_dataset_reports(key))

    _print_summary(summaries)


if __name__ == "__main__":
    main()
