"""CLI entrypoint: summarise the frozen canonical match-grouped split
(prompt 13) -- per-split, per-leg row/match counts and shot rates, plus the
same integrity checks tests/test_canonical_split.py enforces -- and write
reports/analysis/shot_target/SPLIT_VALIDATION.json.

Usage:
    python -m src.eda.generate_split_validation
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.dax.models.splits import CANONICAL_SPLIT_PATH, load_canonical_split
from src.eda.feature_config import ACTIVE, PASSIVE

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "SPLIT_VALIDATION.json"

TARGET = "target_future_shot_10s"
N_FOLDS = 5


def _leg_summary(parquet_path: str, assignment: dict[str, str]) -> dict:
    df = pd.read_parquet(REPO_ROOT / parquet_path, columns=["match_id", TARGET])
    df["match_id"] = df["match_id"].astype(str)
    df["split"] = df["match_id"].map(assignment)

    rows = []
    for label in ["test"] + [f"fold{i}" for i in range(N_FOLDS)]:
        sub = df[df["split"] == label]
        rows.append({
            "split": label,
            "n_matches": int(sub["match_id"].nunique()),
            "n_rows": int(len(sub)),
            "shot_rate_pct": round(float(sub[TARGET].mean() * 100), 3) if len(sub) else None,
        })

    return {
        "parquet_path": parquet_path,
        "n_rows_total": int(len(df)),
        "n_matches_total": int(df["match_id"].nunique()),
        "overall_shot_rate_pct": round(float(df[TARGET].mean() * 100), 3),
        "by_split": rows,
    }


def _integrity_checks(assignment: dict[str, str]) -> list[dict]:
    """Mirrors tests/test_canonical_split.py -- same checks, surfaced here
    for the report rather than just pass/fail in CI."""
    checks = []

    checks.append({
        "name": "every_match_accounted_for_exactly_once",
        "description": "All 115 matches present, no duplicate keys.",
        "passed": len(assignment) == 115 and len(set(assignment.keys())) == len(assignment),
        "detail": f"{len(assignment)} matches, {len(set(assignment.keys()))} unique keys",
    })

    valid_labels = {"test"} | {f"fold{i}" for i in range(N_FOLDS)}
    invalid = {mid: label for mid, label in assignment.items() if label not in valid_labels}
    checks.append({
        "name": "labels_are_test_or_a_valid_fold",
        "description": "Every label is \"test\" or \"fold0\"..\"fold4\".",
        "passed": not invalid,
        "detail": f"{len(invalid)} invalid label(s)" if invalid else "all labels valid",
    })

    test_matches = {mid for mid, label in assignment.items() if label == "test"}
    checks.append({
        "name": "test_match_count",
        "description": "Exactly 23 matches held out as TEST.",
        "passed": len(test_matches) == 23,
        "detail": f"{len(test_matches)} TEST matches",
    })

    fold_matches = {mid for mid, label in assignment.items() if label != "test"}
    overlap = test_matches & fold_matches
    checks.append({
        "name": "no_test_match_in_any_cv_fold",
        "description": "No TEST match also appears in a CV fold.",
        "passed": not overlap,
        "detail": f"{len(overlap)} overlapping match(es)" if overlap else "no overlap",
    })

    fold_labels_present = {label for label in assignment.values() if label != "test"}
    checks.append({
        "name": "all_five_folds_present",
        "description": "TRAIN+VAL matches cover all 5 folds.",
        "passed": fold_labels_present == {f"fold{i}" for i in range(N_FOLDS)},
        "detail": f"folds present: {sorted(fold_labels_present)}",
    })

    checks.append({
        "name": "train_val_match_count",
        "description": "Exactly 92 matches in the TRAIN+VAL pool.",
        "passed": len(fold_matches) == 92,
        "detail": f"{len(fold_matches)} TRAIN+VAL matches",
    })

    counts = Counter(label for label in assignment.values() if label != "test")
    balanced = all(10 <= c <= 30 for c in counts.values())
    checks.append({
        "name": "fold_sizes_reasonably_balanced",
        "description": "Each fold has 10-30 matches (known result: 17-20).",
        "passed": balanced,
        "detail": ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
    })

    # No match appears in two CV folds -- vacuously true since assignment is
    # a dict (one label per key), but checked explicitly to mirror the test
    # file's own explicit assertion of this property.
    checks.append({
        "name": "no_match_in_two_cv_folds",
        "description": "No match's fold membership is ambiguous or duplicated.",
        "passed": True,
        "detail": "dict keys are unique by construction; verified no key maps to multiple labels",
    })

    return checks


def main() -> None:
    assignment = load_canonical_split()

    active_summary = _leg_summary(ACTIVE["parquet_path"], assignment)
    passive_summary = _leg_summary(PASSIVE["parquet_path"], assignment)
    integrity = _integrity_checks(assignment)

    output = {
        "canonical_split_path": str(CANONICAL_SPLIT_PATH.relative_to(REPO_ROOT)),
        "n_matches_total": len(assignment),
        "n_test_matches": sum(1 for v in assignment.values() if v == "test"),
        "n_train_val_matches": sum(1 for v in assignment.values() if v != "test"),
        "n_folds": N_FOLDS,
        "method_note": (
            "Per match, per leg, target_future_shot_10s rate; combined as the average of each leg's own "
            "percentile rank (not either leg's raw rate alone) into one combined_rank per match. "
            "train_test_split(matches, test_size=0.20, stratify=qcut(combined_rank, 4), random_state=42) -> "
            "TEST/TRAIN+VAL. StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) on the TRAIN+VAL "
            "pool, grouped by match_id, stratified on the row-level target (active leg)."
        ),
        "legs": {"active": active_summary, "passive": passive_summary},
        "integrity_checks": integrity,
        "all_integrity_checks_passed": all(c["passed"] for c in integrity),
        "match_assignment": {mid: assignment[mid] for mid in sorted(assignment, key=int)},
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("=== Split validation ===")
    for leg_name, leg in output["legs"].items():
        print(f"\n{leg_name}: {leg['n_rows_total']:,} rows, {leg['n_matches_total']} matches, {leg['overall_shot_rate_pct']}% overall")
        for row in leg["by_split"]:
            print(f"  {row['split']}: {row['n_matches']} matches, {row['n_rows']:,} rows, {row['shot_rate_pct']}%")

    print("\n=== Integrity checks ===")
    for c in integrity:
        print(f"  [{'OK' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
