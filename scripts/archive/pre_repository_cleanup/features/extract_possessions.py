"""
Extract possession sequences from enriched events with 360 data.
This script:
  1. Loads events_with_targets.parquet
  2. Groups into possessions (same team until turnover)
  3. For each possession with 360 data, extracts:
     - Phase trajectory (how defensive phases evolve)
     - Pressure dynamics (how opponent pressure changes via 360 frames)
     - Zone progression (ball movement through zones)
  4. Saves to data/features/possessions_with_360.parquet
Run:
  python scripts/features/extract_possessions.py
"""
from __future__ import annotations
import warnings
from pathlib import Path
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings("ignore")
# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_FEATURES = REPO_ROOT / "data" / "features"
# Add src to path
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))
from dax.features.possession_sequences import build_possession_sequences
def main() -> None:
    """Extract possession sequences from processed events."""
    print("\n" + "=" * 70)
    print("POSSESSION SEQUENCE EXTRACTION")
    print("=" * 70)
    # Load enriched events with targets (must have phases + 360 data)
    events_file = DATA_PROCESSED / "events_with_targets.parquet"
    if not events_file.exists():
        print(f"[ERROR] File not found: {events_file}")
        print("Please run: python scripts/pipeline/pipeline.py")
        return
    print(f"\n[1] Loading events from: {events_file}")
    df_events = pd.read_parquet(events_file)
    print(f"    Loaded {len(df_events):,} events")
    # Filter for events with 360 data
    print("\n[2] Filtering for events with 360 data")
    df_with_360 = df_events[df_events["has_360"] == True].copy()
    print(f"    {len(df_with_360):,} events have 360 data")
    print(f"    Coverage: {len(df_with_360) / len(df_events) * 100:.1f}%")
    # Convert to list of dicts for possession builder
    print("\n[3] Building possessions")
    events_list = df_with_360.to_dict("records")
    possessions = build_possession_sequences(
        events_list,
        only_with_360=True,
        verbose=True,
    )
    if not possessions:
        print("[ERROR] No possessions extracted. Stopping.")
        return
    # Convert to DataFrame
    print("\n[4] Converting to DataFrame")
    possession_dicts = [p.to_dict() for p in possessions]
    df_possessions = pd.DataFrame(possession_dicts)
    # Save
    output_file = DATA_FEATURES / "possessions_with_360.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_possessions.to_parquet(output_file, index=False)
    print(f"\n[5] Saved {len(df_possessions):,} possession sequences")
    print(f"    Output: {output_file}")
    # Summary statistics
    print("\n[6] Summary Statistics")
    print(f"    Unique matches: {df_possessions['match_id'].nunique()}")
    print(f"    Unique teams attacking: {df_possessions['team_in_possession'].nunique()}")
    print(f"    Avg events per possession: {df_possessions['event_count'].mean():.1f}")
    print(f"    Avg events with 360: {df_possessions['events_with_360'].mean():.1f}")
    print(f"    Avg opponent count: {df_possessions['opponent_count_avg'].mean():.1f}")
    print(f"    Shots in 10s rate: {df_possessions['has_shot_in_10s'].mean() * 100:.2f}%")
    print("\n[7] Phase Transition Examples")
    df_with_transitions = df_possessions[
        df_possessions["phase_transitions"].apply(lambda x: len(x) > 0)
    ]
    if len(df_with_transitions) > 0:
        print(f"    {len(df_with_transitions)} possessions have phase transitions")
    print("\n[8] SUCCESS!")
    print("=" * 70)
if __name__ == "__main__":
    main()
