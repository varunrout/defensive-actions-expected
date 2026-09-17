"""CLI entrypoint: verify target_future_xg_10s == 0 in exactly the same
rows as target_future_shot_10s == 0 on both parquets -- the premise Part B
(shot-conditional panels) depends on: xg is a summed-shot-quality value
over a window, structurally 0 when no shot occurred, so an unconditional
mean-xg comparison mostly re-derives the same information as the binary
target. If this premise doesn't hold (a nonzero mismatch count), that
changes the whole rationale for adding shot-conditional panels and should
be investigated before anything else in this prompt runs.

Usage:
    python -m src.eda.generate_structural_zero_check
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eda.feature_config import ACTIVE, PASSIVE

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "eda_xg" / "STRUCTURAL_ZERO_CHECK.json"

SHOT_COL = "target_future_shot_10s"
XG_COL = "target_future_xg_10s"


def check_dataset(dataset_key: str, dataset_cfg: dict) -> dict:
    df = pd.read_parquet(REPO_ROOT / dataset_cfg["parquet_path"], columns=[SHOT_COL, XG_COL])
    n_rows = len(df)

    shot_is_zero = df[SHOT_COL] == 0
    xg_is_zero = df[XG_COL] == 0

    n_mismatches = int((shot_is_zero != xg_is_zero).sum())
    n_shot_zero_xg_nonzero = int((shot_is_zero & ~xg_is_zero).sum())
    n_shot_nonzero_xg_zero = int((~shot_is_zero & xg_is_zero).sum())

    verdict = "confirmed -- xg is structurally zero exactly where shot is zero" if n_mismatches == 0 else "MISMATCH FOUND -- premise does not hold, investigate before trusting Part B/C"

    return {
        "dataset": dataset_key,
        "parquet_path": dataset_cfg["parquet_path"],
        "n_rows": n_rows,
        "n_mismatches": n_mismatches,
        "n_shot_zero_xg_nonzero": n_shot_zero_xg_nonzero,
        "n_shot_nonzero_xg_zero": n_shot_nonzero_xg_zero,
        "verdict": verdict,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    active_result = check_dataset("active", ACTIVE)
    passive_result = check_dataset("passive", PASSIVE)

    output = {
        "method": (
            f"Directly compared ({SHOT_COL} == 0) against ({XG_COL} == 0) row-for-row on both parquets. "
            "Zero mismatches confirms xg is structurally zero exactly where no shot occurred, which is the "
            "premise the shot-conditional panels (Part B/C) depend on."
        ),
        "shot_col": SHOT_COL,
        "xg_col": XG_COL,
        "datasets": {"active": active_result, "passive": passive_result},
        "overall_verdict": (
            "confirmed" if active_result["n_mismatches"] == 0 and passive_result["n_mismatches"] == 0
            else "MISMATCH FOUND"
        ),
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    for r in (active_result, passive_result):
        print(f"{r['dataset']}: n_rows={r['n_rows']:,} n_mismatches={r['n_mismatches']} -- {r['verdict']}")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
