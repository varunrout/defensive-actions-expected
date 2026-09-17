"""Add explicit missingness flag columns to already-built feature parquet
files, so downstream analysis can filter/condition on data-quality gaps
instead of silently including or dropping nulls.

These flags document structural absence -- values that were never computed
because there weren't enough freeze-frame attackers/defenders, there was no
next event to compare against, or an action was the first event of its
possession -- not unexplained/random missingness.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_FEATURES = REPO_ROOT / "data" / "features"
PASSIVE_DEFAULT = DATA_FEATURES / "passive_defense.parquet"
ACTIVE_DEFAULT = DATA_FEATURES / "player_defensive_actions.parquet"

# Verified against the real built datasets. If a rebuild of these files ever
# produces different counts, the underlying null semantics have drifted from
# what was checked -- fail loudly rather than silently document the wrong
# thing.
EXPECTED_FALSE_COUNTS = {
    # has_option_2/3 were 279/2181 before the actor-exclusion fix in
    # passive_defense.py (commit d7035be): the on-ball actor was being
    # counted as its own extra attacker candidate, inflating the option
    # pool by one and pushing the "not enough options" boundary out by one
    # rank. With the actor removed, has_option_2's false count now equals
    # the old has_option_3 count (2181) almost exactly, as expected.
    "has_option_2": 2181,
    "has_option_3": 11398,
    "has_screened_outcome": 2860,
    # has_visible_attacker/has_visible_defender were 17/27 before the
    # actor-self-reference fix in player_defense.py's _support_features
    # (nearest_defender_distance & co. no longer count the acting player as
    # their own support): visible_attacker_count/visible_defender_count are
    # built from the same corrected candidate pool, so both counts rose
    # once the actor's own inflated presence was removed.
    "has_visible_attacker": 31,
    "has_visible_defender": 76,
    "has_previous_event": 4534,
}


def _assert_expected_false_count(df: pd.DataFrame, column: str) -> None:
    actual = int((~df[column]).sum())
    expected = EXPECTED_FALSE_COUNTS[column]
    if actual != expected:
        raise ValueError(
            f"{column}: expected {expected} False rows, got {actual}. "
            "Null semantics have drifted from what was verified against the real data -- investigate before writing."
        )


def add_passive_defense_flags(df: pd.DataFrame, *, validate: bool = True) -> pd.DataFrame:
    """Add has_option_2 / has_option_3 / has_screened_outcome.

    All three are structural, not random: has_option_2/3 are False only
    when there weren't that many ranked attacker options in the freeze
    frame (fewer than 2 or 3 visible attackers to rank); has_screened_outcome
    is False only when there was no next event in the data to compare
    against (end of period/match). Existing columns are untouched.
    """
    out = df.copy()
    out["has_option_2"] = out["top_option_2_threat_score"].notnull()
    out["has_option_3"] = out["top_option_3_threat_score"].notnull()
    out["has_screened_outcome"] = out["screened_option_was_avoided"].notnull()
    if validate:
        for column in ("has_option_2", "has_option_3", "has_screened_outcome"):
            _assert_expected_false_count(out, column)
    return out


def add_player_defensive_actions_flags(df: pd.DataFrame, *, validate: bool = True) -> pd.DataFrame:
    """Add has_visible_attacker / has_visible_defender / has_previous_event.

    has_visible_attacker/defender are False when no attacker/defender was
    visible in the freeze frame for that action, including when freeze-frame
    roles are altogether unknown (visible_*_count is then null, and
    `null > 0` is False in pandas -- correctly counted as "no visible
    attacker/defender" rather than raising). has_previous_event is False
    only for the first event of a possession, where there is no prior event
    to look back at. Existing columns are untouched.

    local_10m_region_fully_visible and local_5m_region_fully_visible already
    do this same job for attackers_within_10m / defenders_within_10m /
    local_numerical_balance_10m and attackers_within_5m / defenders_within_5m
    / local_numerical_balance_5m respectively -- deliberately not duplicated
    here.
    """
    out = df.copy()
    out["has_visible_attacker"] = out["visible_attacker_count"] > 0
    out["has_visible_defender"] = out["visible_defender_count"] > 0
    out["has_previous_event"] = out["phase_label_prev_event"].notnull()
    if validate:
        for column in ("has_visible_attacker", "has_visible_defender", "has_previous_event"):
            _assert_expected_false_count(out, column)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add explicit missingness flag columns to existing feature parquet files, in place.")
    parser.add_argument("--passive-input", type=str, default=str(PASSIVE_DEFAULT))
    parser.add_argument("--passive-output", type=str, default=str(PASSIVE_DEFAULT))
    parser.add_argument("--active-input", type=str, default=str(ACTIVE_DEFAULT))
    parser.add_argument("--active-output", type=str, default=str(ACTIVE_DEFAULT))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and validate flag counts without writing output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    passive_input = Path(args.passive_input)
    active_input = Path(args.active_input)
    if not passive_input.exists():
        raise FileNotFoundError(f"Missing input file: {passive_input}")
    if not active_input.exists():
        raise FileNotFoundError(f"Missing input file: {active_input}")

    passive = add_passive_defense_flags(pd.read_parquet(passive_input))
    print(f"passive_defense: {len(passive):,} rows, added has_option_2 / has_option_3 / has_screened_outcome")

    active = add_player_defensive_actions_flags(pd.read_parquet(active_input))
    print(f"player_defensive_actions: {len(active):,} rows, added has_visible_attacker / has_visible_defender / has_previous_event")

    if args.dry_run:
        print("[dry-run] validated flag counts, not writing output")
        return 0

    passive_output = Path(args.passive_output)
    passive_output.parent.mkdir(parents=True, exist_ok=True)
    passive.to_parquet(passive_output, index=False)
    print(f"Saved {passive_output}")

    active_output = Path(args.active_output)
    active_output.parent.mkdir(parents=True, exist_ok=True)
    active.to_parquet(active_output, index=False)
    print(f"Saved {active_output}")

    return 0
