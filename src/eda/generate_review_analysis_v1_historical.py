"""CLI entrypoint: resolve REVIEW-tier pairs from
CORRELATION_ANALYSIS_V1_HISTORICAL.json (stage 01's original 51/44-feature
candidate lists), using the exact same Type 1-4 resolution logic as
generate_review_analysis.py. Companion to CORRELATION_ATLAS_V1_HISTORICAL.json
-- keeps the atlas and its methodology page consistent (same pairs).

Reuses generate_review_analysis.resolve_dataset() by temporarily swapping its
module-level DATASETS lookup to the historical config for the call -- the
function only reads dataset_cfg for parquet_path/target_col, both unchanged
between the current and historical configs, so this is a safe substitution,
not a behavioural change to the resolution logic itself.

Usage:
    python -m src.eda.generate_review_analysis_v1_historical
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import generate_review_analysis as rev
from src.eda.feature_config_v1_historical import DATASETS_V1_HISTORICAL

REPO_ROOT = Path(__file__).resolve().parents[2]
CORRELATION_PATH = REPO_ROOT / "reports" / "eda" / "CORRELATION_ANALYSIS_V1_HISTORICAL.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "eda" / "REVIEW_ANALYSIS_V1_HISTORICAL.json"


def main() -> None:
    correlation_data = json.loads(CORRELATION_PATH.read_text(encoding="utf-8"))

    original_datasets = rev.DATASETS
    rev.DATASETS = DATASETS_V1_HISTORICAL
    try:
        results = {key: rev.resolve_dataset(key, correlation_data) for key in DATASETS_V1_HISTORICAL}
    finally:
        rev.DATASETS = original_datasets

    output = {
        "generated_at": correlation_data["generated_at"],
        "source": str(CORRELATION_PATH.relative_to(REPO_ROOT)),
        "datasets": results,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\n=== needs_human_call (V1 historical) ===")
    for key, r in results.items():
        print(f"\n{key}: {len(r['needs_human_call'])} pair(s)")

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
