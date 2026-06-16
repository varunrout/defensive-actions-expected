"""Build the 360-only player defensive action dataset.

Output:
  data/features/player_defensive_actions.parquet

This is the first player-model table for DAx. It keeps only 360-visible
and defensive-action rows, then adds player, phase, possession, and support
features around each action.
"""

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
from dax.models.attacking_threat import add_xt_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a player defensive action dataset from 360 events.")
    parser.add_argument("--max-matches", type=int, default=None, help="Optional limit on the number of matches to load")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional limit on the number of event rows to process")
    parser.add_argument("--output", type=str, default=str(DATA_FEATURES / "player_defensive_actions.parquet"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events_file = DATA_PROCESSED / "events_with_targets.parquet"
    if not events_file.exists():
        raise FileNotFoundError(f"Missing input file: {events_file}")

    print("\n" + "=" * 72)
    print("PLAYER DEFENSIVE DATASET BUILD")
    print("=" * 72)

    df = pd.read_parquet(events_file)
    df = df[df["has_360"] == True].copy()
    df = df.sort_values(["match_id", "period", "index"]).reset_index(drop=True)

    # Backfill continuous target for older processed files that only have binary targets.
    if "target_xt_10s" not in df.columns:
        df = pd.DataFrame(add_xt_target(df.to_dict("records")))

    if args.max_matches is not None:
        match_ids = df["match_id"].drop_duplicates().head(args.max_matches)
        df = df[df["match_id"].isin(match_ids)].copy()

    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    print(f"Loaded {len(df):,} 360 events")
    print(f"360 coverage: {100.0:.1f}%")

    rows = build_player_defensive_actions(df.to_dict("records"), only_with_360=True, defensive_only=True, verbose=True)
    if not rows:
        print("[ERROR] No player defensive actions were built.")
        return

    out = pd.DataFrame(rows)
    out = out.sort_values(["match_id", "period", "event_index"]).reset_index(drop=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print(f"Saved {len(out):,} rows to {output_path}")
    print(f"Matches: {out['match_id'].nunique():,}")
    print(f"Players: {out['player_id'].nunique():,}")
    print(f"Shot-in-10s rate: {out['target_shot_in_10s'].mean() * 100:.2f}%")
    print(f"xT-10s mean: {out['target_xt_10s'].mean():.4f}")
    print("Action families:\n" + out["action_family"].value_counts().to_string())
    print("Position groups:\n" + out["position_group"].value_counts().to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()


