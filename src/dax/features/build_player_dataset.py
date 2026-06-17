from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dax.features.player_defense import build_player_defensive_actions

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_FEATURES = REPO_ROOT / "data" / "features"
REQUIRED_TARGETS = ("target_future_shot_10s", "target_future_xg_10s")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a player defensive action dataset from 360 events.")
    parser.add_argument("--max-matches", type=int, default=None, help="Optional limit on matches to load.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional limit on rows to process.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DATA_PROCESSED / "events_with_targets.parquet"),
        help="Input events parquet with corrected targets.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_FEATURES / "player_defensive_actions.parquet"),
        help="Output player defensive actions parquet.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print planned work without writing output.",
    )
    return parser.parse_args(argv)


def require_targets(df: pd.DataFrame, required_targets: tuple[str, ...] = REQUIRED_TARGETS) -> None:
    missing = [column for column in required_targets if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing corrected target columns: {missing}. Rebuild events_with_targets.parquet before feature generation."
        )


def build_player_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    max_matches: int | None = None,
    max_rows: int | None = None,
    require_corrected_targets: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    events_file = Path(input_path)
    if not events_file.exists():
        raise FileNotFoundError(f"Missing input file: {events_file}")

    df = pd.read_parquet(events_file)
    if require_corrected_targets:
        require_targets(df)

    if "has_360" not in df.columns:
        raise ValueError("Input parquet must contain a 'has_360' column.")

    df = df[df["has_360"] == True].copy()  # noqa: E712
    df = df.sort_values(["match_id", "period", "index"]).reset_index(drop=True)

    if max_matches is not None:
        match_ids = df["match_id"].drop_duplicates().head(max_matches)
        df = df[df["match_id"].isin(match_ids)].copy()

    if max_rows is not None:
        df = df.head(max_rows).copy()

    if verbose:
        print("\n" + "=" * 72)
        print("PLAYER DEFENSIVE DATASET BUILD")
        print("=" * 72)
        print(f"Loaded {len(df):,} 360 events")

    rows = build_player_defensive_actions(df.to_dict("records"), only_with_360=True, defensive_only=True, verbose=verbose)
    if not rows:
        raise ValueError("No player defensive actions were built.")

    out = pd.DataFrame(rows)
    if require_corrected_targets:
        require_targets(out)
        if out["target_future_shot_10s"].isna().any() or out["target_future_xg_10s"].isna().any():
            raise ValueError("Corrected targets contain nulls in player defensive dataset.")

    out = out.sort_values(["match_id", "period", "event_index"]).reset_index(drop=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)

    if verbose:
        print(f"Saved {len(out):,} rows to {output}")
        print(f"Matches: {out['match_id'].nunique():,}")
        print(f"Players: {out['player_id'].nunique():,}")
        if "target_future_shot_10s" in out.columns:
            print(f"Future-shot-10s rate: {out['target_future_shot_10s'].mean() * 100:.2f}%")
        if "target_future_xg_10s" in out.columns:
            print(f"Future-xG-10s mean: {out['target_future_xg_10s'].mean():.4f}")
        print("Action families:\n" + out["action_family"].value_counts().to_string())
        print("Position groups:\n" + out["position_group"].value_counts().to_string())
        print("\nDone.")

    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(f"[dry-run] build features from {args.input} -> {args.output}")
        return 0

    build_player_dataset(
        input_path=args.input,
        output_path=args.output,
        max_matches=args.max_matches,
        max_rows=args.max_rows,
    )
    return 0
