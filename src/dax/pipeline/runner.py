from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from dax.data.statsbomb_loader import (
    COMPETITIONS_WITH_360,
    TARGET_COMPETITIONS,
    build_all_enriched_events,
    load_360_json,
    load_events,
    load_matches,
    load_target_competitions,
)
from dax.features.build_player_dataset import build_player_dataset
from dax.features.event_context import add_event_context, validate_event_context
from dax.features.phase_segmentation import label_defensive_phases
from dax.targets.short_horizon import add_future_shot_target, add_future_xg_target

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_FEATURES = REPO_ROOT / "data" / "features"
OUTPUTS_DIR = REPO_ROOT / "outputs"
STAGE_CHOICES = ("all", "prepare-data", "build-features", "init-dirs")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the supported DAx pipeline stages.")
    parser.add_argument(
        "--stage",
        choices=STAGE_CHOICES,
        default="all",
        help="Supported stages: prepare-data, build-features, init-dirs, or all.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DATA_PROCESSED / "events_with_targets.parquet"),
        help="Input parquet for the build-features stage.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_FEATURES / "player_defensive_actions.parquet"),
        help="Output parquet for the build-features stage.",
    )
    parser.add_argument("--max-matches", type=int, default=None, help="Optional feature-build match cap.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional feature-build row cap.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned stages without executing them.")
    return parser.parse_args(argv)


def save_json(data: list | dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=str)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def stage_fetch_raw_data() -> None:
    print("\n" + "=" * 70)
    print("STAGE 1: Fetch raw StatsBomb data → data/raw/")
    print("=" * 70)

    print("\n[1a] Competitions")
    competitions = load_target_competitions()
    save_json(competitions.to_dict("records"), DATA_RAW / "competitions.json")
    print(f"  Saved {len(competitions)} competitions")

    print("\n[1b] Matches")
    all_matches: list[pd.DataFrame] = []
    for competition in TARGET_COMPETITIONS:
        competition_id = competition["competition_id"]
        season_id = competition["season_id"]
        label = competition["label"]
        print(f"  {label}...")
        matches = load_matches(competition_id, season_id)
        all_matches.append(matches)
        save_json(matches.to_dict("records"), DATA_RAW / "matches" / f"{competition_id}_{season_id}.json")
        print(f"    {len(matches)} matches")

    all_matches_df = pd.concat(all_matches, ignore_index=True)
    print(f"  Total: {len(all_matches_df)} matches")

    print("\n[1c] Events and 360 freeze-frames")
    match_ids = all_matches_df["match_id"].astype(int).unique()
    for match_id in tqdm(match_ids, desc="Events", unit="match"):
        try:
            events = load_events(match_id)
            save_json(events.to_dict("records"), DATA_RAW / "events" / f"{match_id}.json")
        except Exception as exc:  # pragma: no cover - network/data dependent
            print(f"  [WARN] match {match_id} events: {exc}")
            continue

        match_comp = all_matches_df[all_matches_df["match_id"] == match_id].iloc[0]
        comp_key = (int(match_comp["competition_id"]), int(match_comp["season_id"]))
        if comp_key in COMPETITIONS_WITH_360:
            try:
                freeze_frames = load_360_json(match_id)
                if freeze_frames:
                    save_json(freeze_frames, DATA_RAW / "three-sixty" / f"{match_id}.json")
            except Exception:
                pass


def stage_process_models() -> None:
    print("\n" + "=" * 70)
    print("STAGE 2: Build enriched models → data/processed/")
    print("=" * 70)

    print("\n[2a] Load and enrich events")
    events_enriched = build_all_enriched_events(verbose=True)
    if events_enriched.empty:
        raise ValueError("No events loaded. Stopping.")

    events_enriched = add_event_context(events_enriched)
    context_issues = validate_event_context(events_enriched)
    if context_issues:
        raise ValueError(f"Event context validation failed: {context_issues}")
    save_parquet(events_enriched, DATA_PROCESSED / "events_enriched.parquet")
    print(f"  Saved {len(events_enriched):,} enriched event rows with event context")

    sort_cols = [column for column in ["match_id", "period", "minute", "second", "index"] if column in events_enriched.columns]
    if sort_cols:
        events_enriched = events_enriched.sort_values(sort_cols).reset_index(drop=True)

    print("\n[2b] Add defensive phase labels")
    events_phased = label_defensive_phases(events_enriched.to_dict("records"))
    df_phased = pd.DataFrame(events_phased)
    if sort_cols:
        df_phased = df_phased.sort_values(sort_cols).reset_index(drop=True)
    save_parquet(df_phased, DATA_PROCESSED / "events_with_phases.parquet")
    print(f"  Phase distribution:\n{df_phased['phase_label'].value_counts().to_string()}")

    print("\n[2c] Add future shot and observed future-xG targets")
    df_targets = add_future_shot_target(df_phased)
    df_targets = add_future_xg_target(df_targets)
    if sort_cols:
        df_targets = df_targets.sort_values(sort_cols).reset_index(drop=True)
    save_parquet(df_targets, DATA_PROCESSED / "events_with_targets.parquet")
    shot_rate = df_targets["target_future_shot_10s"].mean() * 100
    future_xg_mean = df_targets["target_future_xg_10s"].mean()
    print(f"  Future-shot-10s rate: {shot_rate:.2f}%")
    print(f"  Future-xG-10s mean: {future_xg_mean:.4f}")

    print("\n[2d] Summary statistics")
    summary = {
        "total_rows": len(df_targets),
        "competitions": df_targets["competition_label"].nunique(),
        "matches": df_targets["match_id"].nunique(),
        "with_360": int(df_targets["has_360"].eq(True).sum()),
        "phases": df_targets["phase_label"].nunique(),
        "future_shot_10s_rate": float(shot_rate),
        "future_xg_10s_mean": float(future_xg_mean),
    }
    save_json(summary, DATA_PROCESSED / "summary.json")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def stage_create_directories() -> None:
    print("\n" + "=" * 70)
    print("STAGE 3: Initialize data folders")
    print("=" * 70)
    directories = [
        DATA_FEATURES / "player_context",
        DATA_FEATURES / "team_context",
        DATA_FEATURES / "threat_models",
        DATA_FEATURES / "attributions",
        OUTPUTS_DIR / "models",
        OUTPUTS_DIR / "validation",
        OUTPUTS_DIR / "oof",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {directory.relative_to(REPO_ROOT)}/")


def run_prepare_data() -> None:
    stage_fetch_raw_data()
    stage_process_models()
    stage_create_directories()


def run_build_features(args: argparse.Namespace) -> None:
    build_player_dataset(
        input_path=args.input,
        output_path=args.output,
        max_matches=args.max_matches,
        max_rows=args.max_rows,
    )


def run_stage(args: argparse.Namespace) -> None:
    if args.stage == "prepare-data":
        run_prepare_data()
    elif args.stage == "build-features":
        run_build_features(args)
    elif args.stage == "init-dirs":
        stage_create_directories()
    elif args.stage == "all":
        run_prepare_data()
        run_build_features(args)
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(f"Unsupported stage: {args.stage}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(f"[dry-run] stage={args.stage}")
        if args.stage in {"build-features", "all"}:
            print(f"[dry-run] build features from {args.input} -> {args.output}")
        return 0

    print("\n*** DAx Pipeline ***")
    print(f"   Repository: {REPO_ROOT}")
    print(f"   Supported stage: {args.stage}")
    run_stage(args)
    print("\nSUCCESS: pipeline stage completed.")
    return 0
