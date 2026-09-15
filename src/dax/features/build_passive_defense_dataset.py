from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dax.features.passive_defense import build_passive_defense_rows

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_FEATURES = REPO_ROOT / "data" / "features"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a passive/off-ball defender-slot dataset from 360 events.")
    parser.add_argument("--max-matches", type=int, default=None, help="Optional limit on matches to load.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DATA_PROCESSED / "events_with_targets.parquet"),
        help="Input events parquet.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_FEATURES / "passive_defense.parquet"),
        help="Output passive-defense parquet.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print planned work without writing output.",
    )
    return parser.parse_args(argv)


def build_passive_defense_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    max_matches: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    events_file = Path(input_path)
    if not events_file.exists():
        raise FileNotFoundError(f"Missing input file: {events_file}")

    df = pd.read_parquet(events_file)
    if "has_360" not in df.columns:
        raise ValueError("Input parquet must contain a 'has_360' column.")

    df = df.sort_values(["match_id", "period", "index"]).reset_index(drop=True)

    if max_matches is not None:
        match_ids = df["match_id"].drop_duplicates().head(max_matches)
        df = df[df["match_id"].isin(match_ids)].copy()

    # Deliberately NOT filtered to has_360 events here: build_passive_defense_rows
    # needs the full per-match event stream (including non-360 events) to compute
    # leakage-safe per-row future-shot/future-xG targets -- a shot recorded on a
    # non-360 event still has to count towards a nearby 360 event's 10s window.
    # The has_360 filter for which events become row-anchors happens inside
    # build_passive_defense_rows itself (only_with_360=True below).
    if verbose:
        print("\n" + "=" * 72)
        print("PASSIVE DEFENSE DATASET BUILD")
        print("=" * 72)
        print(f"Loaded {len(df):,} events ({int(df['has_360'].eq(True).sum()):,} with 360)")

    rows = build_passive_defense_rows(df.to_dict("records"), only_with_360=True, verbose=verbose)
    if not rows:
        raise ValueError("No passive defense rows were built.")

    # 58 columns as of the carrier_x/carrier_y drop (both always identical to
    # ball_x/ball_y in this table -- the on-ball event's location is the ball
    # location; see build_passive_defense_rows' row assembly). Plus the 3
    # missingness flags added by scripts/add_missingness_flags.py afterwards.
    out = pd.DataFrame(rows)
    out = out.sort_values(["match_id", "period", "event_order_in_possession", "defender_slot_index"]).reset_index(drop=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)

    if verbose:
        print(f"Saved {len(out):,} rows to {output}")
        print(f"Matches: {out['match_id'].nunique():,}")
        print("On-ball event types:\n" + out["on_ball_event_type"].value_counts().to_string())
        print("\nDone.")

    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(f"[dry-run] build passive defense dataset from {args.input} -> {args.output}")
        return 0

    build_passive_defense_dataset(
        input_path=args.input,
        output_path=args.output,
        max_matches=args.max_matches,
    )
    return 0
