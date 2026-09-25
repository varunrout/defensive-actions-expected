"""Dashboard data export: locked engineered feature values for the 3 curated matches (prompt 87).

Produces dashboard_data/match_features/{match_id}.json -- one file per curated
match -- for the separate portfolio-website (Next.js) repo to import. Data
only: no frontend code lives in this repo.

This is a READ-ONLY export. It does not train, retune, or recompute anything:
every value here is copied straight out of the already-locked feature tables
that scripts/dashboard/export_match_explorer.py (prompt 83/86) already reads
for its own model-scoring pass:

  Active-phase rows:  data/features/player_defensive_actions.parquet
  Passive-phase rows: data/features/passive_defense.parquet

Both are the corrected post-coordinate-frame-fix parquets (Prompt 85 Phase 2
rebuilt them in place; see reports/modeling/COORDINATE_FRAME_FIX_AND_REPIPELINE.md).
Only columns that survived correlation/VIF/leakage review as LOCKED features
are included -- exactly the ACTIVE (34) and PASSIVE (38) lists declared in
src/eda/feature_config.py (ACTIVE["categorical"|"boolean"|"continuous"|"discrete"]
and PASSIVE[...] respectively). Camera/360-coverage metadata (visibility_limited,
has_360, freeze_frame_count, etc.) and every column feature_config.py already
excludes as "not football signal" are never pulled in here, because this script
only ever reads from feature_config.py's own locked lists -- it does not
re-derive a column list from the parquet's dtypes.

Event-id parity: this script emits exactly the event_id set (dropping none,
adding none) that scripts/dashboard/export_match_explorer.py wrote to
dashboard_data/match_explorer/{match_id}.json for the same match -- verified
by set-equality assertion below, not assumed. Active event_ids come 1:1 from
player_defensive_actions.parquet rows for that match; passive event_ids are
the unique on-ball events in passive_defense.parquet for that match (each
carries 1+ defender-slot rows, exactly as match_explorer nests them under
"defenders").

Passive per-event vs per-defender split. Empirically (checked across the full
passive_defense.parquet, not just one event), 22 of the 38 locked passive
columns are constant within every (event_id) group -- these describe the
on-ball moment itself (on_ball_event_type, phase_label, period, has_option_2/3,
ball_x/y, and all three top_option_N_{threat_score,dx,dy,distance_from_ball,
angle_from_ball} columns, which describe passing options relative to the ball,
not to any one defender). The other 16 vary per defender_slot_index (defender
position, marking_tightness, angle/centrality relative to goal from the
defender's own spot, lane_screening_score_option_N, functional_role,
overload_score, the two in/out-of-box flags, is_wide_lane,
is_goal_side_of_nearest_attacker). This script places the 22 shared columns in
the event-level "features" object and the 16 per-defender columns inside each
entry of a "defenders" list, mirroring how match_explorer/{match_id}.json
already nests multiple defenders under one passive event.

Only LOCKED features that survived correlation review are included (34 active /
38 passive per src/eda/feature_config.py) -- no dropped candidate columns, no
camera-coverage/visibility-quality metadata.

Usage:
    .venv/Scripts/python.exe scripts/dashboard/export_match_features.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eda.feature_config import ACTIVE, PASSIVE  # noqa: E402

EXPLORER_DIR = REPO_ROOT / "dashboard_data" / "match_explorer"
OUT_DIR = REPO_ROOT / "dashboard_data" / "match_features"
ACTIVE_PARQUET = REPO_ROOT / "data" / "features" / "player_defensive_actions.parquet"
PASSIVE_PARQUET = REPO_ROOT / "data" / "features" / "passive_defense.parquet"

CURATED_MATCHES = [3938643, 3857294, 3857298]

ACTIVE_LOCKED = ACTIVE["categorical"] + ACTIVE["boolean"] + ACTIVE["continuous"] + ACTIVE["discrete"]
PASSIVE_LOCKED = PASSIVE["categorical"] + PASSIVE["boolean"] + PASSIVE["continuous"] + PASSIVE["discrete"]

assert len(ACTIVE_LOCKED) == 34, f"expected 34 locked active features, got {len(ACTIVE_LOCKED)}"
assert len(PASSIVE_LOCKED) == 38, f"expected 38 locked passive features, got {len(PASSIVE_LOCKED)}"

# Passive locked columns that are constant within an event_id group (describe
# the on-ball moment, not any one defender) -- empirically verified (see module
# docstring) against the full passive_defense.parquet, not assumed.
PASSIVE_EVENT_LEVEL = [
    "on_ball_event_type", "phase_label", "period", "has_option_2", "has_option_3",
    "ball_x", "ball_y",
    "top_option_1_threat_score", "top_option_2_threat_score", "top_option_3_threat_score",
    "top_option_1_dx", "top_option_1_dy", "top_option_1_distance_from_ball", "top_option_1_angle_from_ball",
    "top_option_2_dx", "top_option_2_dy", "top_option_2_distance_from_ball", "top_option_2_angle_from_ball",
    "top_option_3_dx", "top_option_3_dy", "top_option_3_distance_from_ball", "top_option_3_angle_from_ball",
]
PASSIVE_PER_DEFENDER = [c for c in PASSIVE_LOCKED if c not in PASSIVE_EVENT_LEVEL]
assert len(PASSIVE_EVENT_LEVEL) == 22 and len(PASSIVE_PER_DEFENDER) == 16, (
    len(PASSIVE_EVENT_LEVEL), len(PASSIVE_PER_DEFENDER)
)


def to_native(value):
    """Convert a pandas/numpy scalar to a JSON-safe native Python value."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if not math.isfinite(v) else v
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def row_features(row: pd.Series, cols: list[str]) -> dict:
    return {c: to_native(row[c]) for c in cols}


def build_active_events(df: pd.DataFrame, match_id: int) -> dict[str, dict]:
    sub = df.loc[df["match_id"] == match_id]
    events: dict[str, dict] = {}
    for row in sub.itertuples(index=False):
        row_dict = row._asdict()
        events[row_dict["event_id"]] = {
            "event_id": row_dict["event_id"],
            "location": {"x": to_native(row_dict["ball_x"]), "y": to_native(row_dict["ball_y"])},
            "phase": "active",
            "features": {c: to_native(row_dict[c]) for c in ACTIVE_LOCKED},
        }
    return events


def build_passive_events(df: pd.DataFrame, match_id: int) -> dict[str, dict]:
    sub = df.loc[df["match_id"] == match_id]
    events: dict[str, dict] = {}
    for event_id, grp in sub.groupby("event_id", sort=False):
        first = grp.iloc[0]
        defenders = []
        for d in grp.sort_values("defender_slot_index").itertuples(index=False):
            d_dict = d._asdict()
            defenders.append({
                "defender_slot_index": to_native(d_dict["defender_slot_index"]),
                "location": {"x": to_native(d_dict["defender_x"]), "y": to_native(d_dict["defender_y"])},
                "features": {c: to_native(d_dict[c]) for c in PASSIVE_PER_DEFENDER},
            })
        events[event_id] = {
            "event_id": event_id,
            "location": {"x": to_native(first["ball_x"]), "y": to_native(first["ball_y"])},
            "phase": "passive",
            "features": {c: to_native(first[c]) for c in PASSIVE_EVENT_LEVEL},
            "defenders": defenders,
        }
    return events


def main() -> None:
    print(f"[load] {ACTIVE_PARQUET.relative_to(REPO_ROOT)}")
    active_df = pd.read_parquet(ACTIVE_PARQUET)
    print(f"[load] {PASSIVE_PARQUET.relative_to(REPO_ROOT)}")
    passive_df = pd.read_parquet(PASSIVE_PARQUET)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for match_id in CURATED_MATCHES:
        explorer_path = EXPLORER_DIR / f"{match_id}.json"
        explorer = json.loads(explorer_path.read_text(encoding="utf-8"))
        explorer_ids = {e["event_id"] for e in explorer["events"]}
        explorer_phase = {e["event_id"]: e["phase"] for e in explorer["events"]}

        active_events = build_active_events(active_df, match_id)
        passive_events = build_passive_events(passive_df, match_id)

        our_ids = set(active_events) | set(passive_events)
        missing = explorer_ids - our_ids
        extra = our_ids - explorer_ids
        assert not missing and not extra, (
            f"match {match_id}: event_id set mismatch vs match_explorer -- "
            f"missing={len(missing)} extra={len(extra)}"
        )
        overlap = set(active_events) & set(passive_events)
        assert not overlap, f"match {match_id}: {len(overlap)} event_ids in both active and passive"
        for eid, phase in explorer_phase.items():
            ours_phase = active_events[eid]["phase"] if eid in active_events else passive_events[eid]["phase"]
            assert ours_phase == phase, f"match {match_id} event {eid}: phase mismatch {ours_phase} vs {phase}"

        # Order events the same way match_explorer does (period, timestamp) --
        # reuse the explorer's own event order for a stable, comparable file.
        ordered = [
            active_events[eid] if eid in active_events else passive_events[eid]
            for eid in [e["event_id"] for e in explorer["events"]]
        ]

        payload = {
            "match_id": str(match_id),
            "locked_feature_counts": {"active": len(ACTIVE_LOCKED), "passive": len(PASSIVE_LOCKED)},
            "events": ordered,
        }
        out_path = OUT_DIR / f"{match_id}.json"
        out_path.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
        print(f"  {out_path.relative_to(REPO_ROOT)}: {out_path.stat().st_size / 1e6:.2f} MB, "
              f"{len(ordered)} events (active={len(active_events)}, passive={len(passive_events)}), "
              f"event_id parity OK")

    print("\nDone.")


if __name__ == "__main__":
    main()
