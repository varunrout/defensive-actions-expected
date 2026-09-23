"""Prompt 85, Phase 3: quantify how target_xt_delta_v2 / target_xt_delta_passive
actually changed once rebuilt from the Phase-2-corrected coordinates.

Reads the pre-fix xT prototype parquets (backed up at
.claude_scratch/prefix_backup/ before Phase 2 overwrote anything downstream) and
the post-fix prototype parquets (rebuilt by build_xt_delta_v2_prototype.py /
build_passive_xt_delta_prototype.py, run AFTER Phase 2), and reports, per target:
  - how many rows' target value actually changed (not just "how many rows were
    flagged as exposed" -- an exposed row's value can still come out identical
    if the flip happened to cancel out, e.g. a symmetric grid cell)
  - the full distribution of (new - old), not just a flag count
  - how many rows flipped sign, and by how much the mean/median moved

Usage (run AFTER both build_xt_delta_v2_prototype.py and
build_passive_xt_delta_prototype.py have been re-run against the corrected data):
    .venv/Scripts/python.exe scripts/analysis/phase3_xt_target_delta_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKUP_DIR = REPO_ROOT / ".claude_scratch" / "prefix_backup"
OUT_PATH = REPO_ROOT / "outputs" / "models" / "validation" / "phase3_xt_target_delta_report.json"


def describe(arr: np.ndarray) -> dict:
    s = pd.Series(arr)
    return {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
    }


def compare(old_path: Path, new_path: Path, key: str, target_col: str, label: str) -> dict:
    old = pd.read_parquet(old_path)[[key, target_col]].rename(columns={target_col: "old"})
    new = pd.read_parquet(new_path)[[key, target_col]].rename(columns={target_col: "new"})
    merged = old.merge(new, on=key, how="inner")
    assert len(merged) == len(old) == len(new), f"{label}: row-count mismatch old={len(old)} new={len(new)} merged={len(merged)}"

    both_defined = merged.dropna(subset=["old", "new"])
    delta = (both_defined["new"] - both_defined["old"]).to_numpy()
    n_changed = int((np.abs(delta) > 1e-12).sum())
    n_sign_flip = int(((both_defined["old"] > 0) != (both_defined["new"] > 0)).sum())

    old_nan = int(merged["old"].isna().sum())
    new_nan = int(merged["new"].isna().sum())

    return {
        "label": label,
        "n_rows": int(len(merged)),
        "n_rows_both_defined": int(len(both_defined)),
        "old_nan_count": old_nan,
        "new_nan_count": new_nan,
        "n_rows_target_changed": n_changed,
        "pct_rows_target_changed": 100.0 * n_changed / len(both_defined),
        "n_rows_sign_flipped": n_sign_flip,
        "pct_rows_sign_flipped": 100.0 * n_sign_flip / len(both_defined),
        "old_target_distribution": describe(both_defined["old"].to_numpy()),
        "new_target_distribution": describe(both_defined["new"].to_numpy()),
        "delta_distribution_new_minus_old": describe(delta),
    }


def main() -> None:
    result = {}

    result["target_xt_delta_v2"] = compare(
        BACKUP_DIR / "active_binary_xt_delta_v2.parquet",
        REPO_ROOT / "outputs" / "prototypes" / "active_binary_xt_delta_v2.parquet",
        key="event_id", target_col="target_xt_delta_v2", label="active-xT (target_xt_delta_v2)",
    )
    result["target_xt_delta_passive"] = compare(
        BACKUP_DIR / "passive_xt_delta.parquet",
        REPO_ROOT / "outputs" / "prototypes" / "passive_xt_delta.parquet",
        key="event_id", target_col="target_xt_delta_passive", label="passive-xT (target_xt_delta_passive, unique events)",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for k, r in result.items():
        print(f"\n=== {k} ===")
        print(f"rows changed: {r['n_rows_target_changed']} / {r['n_rows_both_defined']} ({r['pct_rows_target_changed']:.2f}%)")
        print(f"sign flips: {r['n_rows_sign_flipped']} ({r['pct_rows_sign_flipped']:.2f}%)")
        print(f"delta distribution: {r['delta_distribution_new_minus_old']}")
    print(f"\n[write] {OUT_PATH}")


if __name__ == "__main__":
    main()
