"""
DAx data pipeline: fetch and prepare StatsBomb open data.

This script orchestrates the full data lifecycle:
  1. Fetch all competitions/matches/events/360 from StatsBomb
  2. Save raw JSON to data/raw/
  3. Build enriched event models
  4. Save processed parquets to data/processed/
  5. Prepare feature-engineering inputs

Run:
  python scripts/pipeline/pipeline.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_FEATURES = REPO_ROOT / "data" / "features"
OUTPUTS_DIR = REPO_ROOT / "outputs"

# Create directories
for d in [DATA_RAW, DATA_PROCESSED, DATA_FEATURES]:
    d.mkdir(parents=True, exist_ok=True)

# Add src to path
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from dax.data.statsbomb_loader import (
    build_all_enriched_events,
    load_all_target_matches,
    load_target_competitions,
    load_matches,
    load_events,
    load_360_json,
    TARGET_COMPETITIONS,
    COMPETITIONS_WITH_360,
)
from dax.features.event_context import add_event_context, validate_event_context
from dax.features.phase_segmentation import label_defensive_phases
from dax.targets.short_horizon import add_future_xg_target, add_future_shot_target


def save_json(data: list | dict, path: Path) -> None:
    """Save data to JSON with pretty printing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame as Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def stage_1_fetch_raw_data() -> None:
    """
    Fetch all competitions, matches, events, lineups, and 360 from StatsBomb.
    Save raw JSON to data/raw/.
    """
    print("\n" + "=" * 70)
    print("STAGE 1: Fetch raw StatsBomb data → data/raw/")
    print("=" * 70)

    # Competitions
    print("\n[1a] Competitions")
    comps = load_target_competitions()
    save_json(comps.to_dict("records"), DATA_RAW / "competitions.json")
    print(f"  Saved {len(comps)} competitions")

    # Matches per competition
    print("\n[1b] Matches")
    all_matches: list[dict] = []
    for comp in TARGET_COMPETITIONS:
        cid = comp["competition_id"]
        sid = comp["season_id"]
        label = comp["label"]
        print(f"  {label}...")
        matches = load_matches(cid, sid)
        all_matches.append(matches)
        save_json(
            matches.to_dict("records"),
            DATA_RAW / "matches" / f"{cid}_{sid}.json",
        )
        print(f"    {len(matches)} matches")

    all_matches_df = pd.concat(all_matches, ignore_index=True)
    print(f"  Total: {len(all_matches_df)} matches")

    # Events and 360 per match
    print("\n[1c] Events and 360 freeze-frames")
    match_ids = all_matches_df["match_id"].astype(int).unique()
    for mid in tqdm(match_ids, desc="Events", unit="match"):
        # Events
        try:
            ev = load_events(mid)
            save_json(ev.to_dict("records"), DATA_RAW / "events" / f"{mid}.json")
        except Exception as e:
            print(f"  [WARN] match {mid} events: {e}")
            continue

        # 360 (if available)
        # Check if this match's competition has 360
        match_comp = all_matches_df[all_matches_df["match_id"] == mid].iloc[0]
        comp_key = (int(match_comp["competition_id"]), int(match_comp["season_id"]))
        if comp_key in COMPETITIONS_WITH_360:
            try:
                ff = load_360_json(mid)
                if ff:
                    save_json(ff, DATA_RAW / "three-sixty" / f"{mid}.json")
            except Exception:
                pass  # 360 is optional


def stage_2_process_models() -> None:
    """
    Build enriched event tables with phase labels and threat scores.
    Save processed parquets to data/processed/.
    """
    print("\n" + "=" * 70)
    print("STAGE 2: Build enriched models → data/processed/")
    print("=" * 70)

    print("\n[2a] Load and enrich events")
    events_enriched = build_all_enriched_events(verbose=True)

    if events_enriched.empty:
        print("[ERROR] No events loaded. Stopping.")
        return

    events_enriched = add_event_context(events_enriched)
    context_issues = validate_event_context(events_enriched)
    if context_issues:
        raise ValueError(f"Event context validation failed: {context_issues}")
    save_parquet(events_enriched, DATA_PROCESSED / "events_enriched.parquet")
    print(f"  Saved {len(events_enriched):,} enriched event rows with event context")

    sort_cols = [
        c
        for c in ["match_id", "period", "minute", "second", "index"]
        if c in events_enriched.columns
    ]
    if sort_cols:
        events_enriched = events_enriched.sort_values(sort_cols).reset_index(drop=True)

    # Phase labelling
    print("\n[2b] Add defensive phase labels")
    events_rows = events_enriched.to_dict("records")
    events_phased = label_defensive_phases(events_rows)
    df_phased = pd.DataFrame(events_phased)
    
    # CRITICAL: Re-sort after phase labeling to ensure target labeling works correctly
    sort_cols = [c for c in ["match_id", "period", "minute", "second", "index"] if c in df_phased.columns]
    if sort_cols:
        df_phased = df_phased.sort_values(sort_cols).reset_index(drop=True)
    
    save_parquet(df_phased, DATA_PROCESSED / "events_with_phases.parquet")
    print(f"  Phase distribution:\n{df_phased['phase_label'].value_counts().to_string()}")

    # Short-horizon observed targets
    print("\n[2c] Add future shot and observed future-xG targets")
    df_targets = add_future_shot_target(df_phased)
    df_targets = add_future_xg_target(df_targets)
    
    # Re-sort to guarantee order after target labeling
    sort_cols = [c for c in ["match_id", "period", "minute", "second", "index"] if c in df_targets.columns]
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
        "with_360": df_targets["has_360"].sum(),
        "phases": df_targets["phase_label"].nunique(),
        "future_shot_10s_rate": shot_rate,
        "future_xg_10s_mean": future_xg_mean,
    }
    save_json(summary, DATA_PROCESSED / "summary.json")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def stage_3_create_directories() -> None:
    """Create placeholder directories for downstream stages."""
    print("\n" + "=" * 70)
    print("STAGE 3: Initialize data folders")
    print("=" * 70)

    dirs_to_create = [
        DATA_FEATURES / "player_context",
        DATA_FEATURES / "team_context",
        DATA_FEATURES / "threat_models",
        DATA_FEATURES / "attributions",
        OUTPUTS_DIR / "models",
        OUTPUTS_DIR / "validation",
        OUTPUTS_DIR / "oof",
    ]

    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {d.relative_to(REPO_ROOT)}/")


def main() -> None:
    """Run the full pipeline."""
    print("\n*** DAx Data Pipeline ***")
    print(f"   Repository: {REPO_ROOT}")
    print(f"   Data root: {REPO_ROOT / 'data'}")

    try:
        stage_1_fetch_raw_data()
        stage_2_process_models()
        stage_3_create_directories()

        print("\n" + "=" * 70)
        print("SUCCESS: Pipeline complete!")
        print("=" * 70)
        print("\nData structure:")
        print("  data/raw/              ← JSON files from StatsBomb API")
        print("  data/processed/        ← Enriched Parquet tables")
        print("  data/features/         ← Feature engineering outputs")
        print("  outputs/models/        ← Trained model artifacts")
        print("  outputs/validation/    ← Validation and analysis results")
        print("  outputs/oof/           ← Out-of-fold prediction artifacts")

    except Exception as e:
        print(f"\nERROR: Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()



