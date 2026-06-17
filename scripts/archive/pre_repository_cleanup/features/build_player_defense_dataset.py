"""Build the 360-only player defensive action dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_FEATURES = REPO_ROOT / "data" / "features"

sys.path.insert(0, str(REPO_ROOT / "src"))

from dax.features.player_defense import build_player_defensive_actions

REQUIRED_TARGETS = ("target_future_shot_10s", "target_future_xg_10s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a player defensive action dataset from 360 events.")
    parser.add_argument("--max-matches", type=int, default=None, help="Optional limit on the number of matches to load")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional limit on the number of event rows to process")
    parser.add_argument("--input", type=str, default=str(DATA_PROCESSED / "events_with_targets.parquet"))
    parser.add_argument("--output", type=str, default=str(DATA_FEATURES / "player_defensive_actions.parquet"))
    return parser.parse_args()


def _require_targets(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing corrected target columns: {missing}. Rebuild events_with_targets.parquet with pipeline stage 2.")


def main() -> None:
    args = parse_args()
    events_file = Path(args.input)
    if not events_file.exists():
        raise FileNotFoundError(f"Missing input file: {events_file}")

    print("\n" + "=" * 72)
    print("PLAYER DEFENSIVE DATASET BUILD")
    print("=" * 72)

    df = pd.read_parquet(events_file)
    _require_targets(df)
    df = df[df["has_360"] == True].copy()
    df = df.sort_values(["match_id", "period", "index"]).reset_index(drop=True)

    if args.max_matches is not None:
        match_ids = df["match_id"].drop_duplicates().head(args.max_matches)
        df = df[df["match_id"].isin(match_ids)].copy()

    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    print(f"Loaded {len(df):,} 360 events")

    rows = build_player_defensive_actions(df.to_dict("records"), only_with_360=True, defensive_only=True, verbose=True)
    if not rows:
        raise ValueError("No player defensive actions were built.")

    out = pd.DataFrame(rows)
    _require_targets(out)
    if out["target_future_shot_10s"].isna().any() or out["target_future_xg_10s"].isna().any():
        raise ValueError("Corrected targets contain nulls in player defensive dataset.")

    out = out.sort_values(["match_id", "period", "event_index"]).reset_index(drop=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print(f"Saved {len(out):,} rows to {output_path}")
    print(f"Matches: {out['match_id'].nunique():,}")
    print(f"Players: {out['player_id'].nunique():,}")
    print(f"Future-shot-10s rate: {out['target_future_shot_10s'].mean() * 100:.2f}%")
    print(f"Future-xG-10s mean: {out['target_future_xg_10s'].mean():.4f}")
    print("Action families:\n" + out["action_family"].value_counts().to_string())
    print("Position groups:\n" + out["position_group"].value_counts().to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
