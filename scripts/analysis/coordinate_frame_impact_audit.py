"""Coordinate-frame bug: full impact audit (Prompt 84). Read-only.

Prompt 83 found that `_infer_attack_sign_by_period_team` (src/dax/data/statsbomb_loader.py)
gets the attack-direction sign wrong for a large share of rows: 26,039 of 56,068
(46.4%) of `player_defensive_actions.parquet` rows are stored in a frame
inconsistent with what the rule-based phase labeller assumes.

This script computes, from existing raw data and existing model artifacts only
(no retraining, no locked-file writes):

  1. Event-level frame consistency, generalized from Prompt 83's own audit
     script (scripts/dashboard/audit_frames_and_profile_matches.py) to EVERY
     row of events_with_targets.parquet (not just defensive actions) -- the
     building block every other computation below joins against.
  2. Target-level exposure for all 6 targets (Task 1) -- target_future_shot_10s/
     target_future_xg_10s are shown frame-invariant by direct code inspection
     (src/dax/targets/short_horizon.py references no coordinate column at
     all); target_xt_delta_v2/target_xt_delta_passive are traced through
     their own build scripts' xt_before/xt_after event lookups and the
     resulting row-level exposure share is computed here.
  3. Feature-importance-weighted exposure per reference model (Task 3),
     using each of the 6 models' own already-computed importance/coefficient
     artifact and the frame-dependent feature classification from Task 2
     (hardcoded below, each entry's reasoning documented in the report).

Usage:
    .venv/Scripts/python.exe scripts/analysis/coordinate_frame_impact_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVENTS_PARQUET = REPO_ROOT / "data" / "processed" / "events_with_targets.parquet"
ACTIVE_LOCKED = REPO_ROOT / "data" / "features" / "player_defensive_actions.parquet"
PASSIVE_LOCKED = REPO_ROOT / "data" / "features" / "passive_defense.parquet"
XT_ACTIVE_PROTO = REPO_ROOT / "outputs" / "prototypes" / "active_binary_xt_delta_v2.parquet"
XT_PASSIVE_PROTO = REPO_ROOT / "outputs" / "prototypes" / "passive_xt_delta.parquet"
OUT_PATH = REPO_ROOT / "outputs" / "models" / "validation" / "coordinate_frame_impact_audit.json"


# ---------------------------------------------------------------------------
# Building block: event-level frame consistency (generalizes Prompt 83's own
# audit script from defensive-action rows to every event in the stream).
# ---------------------------------------------------------------------------

def event_frame_consistency() -> pd.DataFrame:
    cols = ["id", "match_id", "team", "home_team", "away_team",
            "attacking_team_before_action", "raw_ball_x", "raw_ball_y", "ball_x", "ball_y"]
    df = pd.read_parquet(EVENTS_PARQUET, columns=cols)

    eq_raw = (np.isclose(df["ball_x"], df["raw_ball_x"], atol=1e-6, equal_nan=False) &
              np.isclose(df["ball_y"], df["raw_ball_y"], atol=1e-6, equal_nan=False))
    eq_flip = (np.isclose(df["ball_x"], 120.0 - df["raw_ball_x"], atol=1e-6, equal_nan=False) &
               np.isclose(df["ball_y"], 80.0 - df["raw_ball_y"], atol=1e-6, equal_nan=False))
    state = np.where(eq_raw, "raw", np.where(eq_flip, "flipped", "unknown"))
    df["state"] = state

    other_team = np.where(df["team"] == df["home_team"], df["away_team"], df["home_team"])
    stored_frame_team = np.where(df["state"] == "raw", df["team"],
                                  np.where(df["state"] == "flipped", other_team, None))
    df["frame_consistent"] = np.where(
        df["state"] == "unknown", np.nan,
        (stored_frame_team == df["attacking_team_before_action"]).astype(float),
    )
    return df[["id", "match_id", "state", "frame_consistent"]]


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result: dict = {}

    print("[1/4] event-level frame consistency (all events_with_targets.parquet rows)...")
    ev = event_frame_consistency()
    n_known = ev["frame_consistent"].notna().sum()
    n_inconsistent = (ev["frame_consistent"] == 0).sum()
    print(f"  {len(ev)} events; {n_known} with known state; "
          f"{n_inconsistent} frame-inconsistent ({n_inconsistent / n_known:.4f})")
    result["event_level_frame_consistency"] = {
        "population": "events_with_targets.parquet, all rows, all 115 matches",
        "rows": int(len(ev)), "rows_known_state": int(n_known),
        "rows_frame_inconsistent": int(n_inconsistent),
        "share_frame_inconsistent_of_known": round(float(n_inconsistent / n_known), 4),
    }

    # Sanity check against Prompt 83's own published figure (defensive-action
    # rows only, joined back via event id == active locked parquet's event_id).
    active_locked = pd.read_parquet(ACTIVE_LOCKED, columns=["event_id"])
    check = active_locked.merge(ev.rename(columns={"id": "event_id"}), on="event_id", how="left")
    n_active_inconsistent = (check["frame_consistent"] == 0).sum()
    print(f"  [sanity check vs Prompt 83] active locked rows inconsistent (this script's general "
          f"event-level method): {n_active_inconsistent} of {len(check)} "
          f"({n_active_inconsistent / len(check):.4f}) -- Prompt 83's own published figure: "
          f"26039 of 56068 (0.4644)")
    result["sanity_check_vs_prompt83"] = {
        "active_locked_rows_inconsistent_this_method": int(n_active_inconsistent),
        "active_locked_rows_total": int(len(check)),
        "share": round(float(n_active_inconsistent / len(check)), 4),
        "prompt83_published_rows_inconsistent": 26039, "prompt83_published_rows_total": 56068,
        "prompt83_published_share": 0.4644,
    }

    # -----------------------------------------------------------------
    # Task 1a -- target_future_shot_10s / target_future_xg_10s (binary +
    # hurdle-continuous legs). Frame-invariance is proven by code inspection
    # (src/dax/targets/short_horizon.py), not computed here -- there is no
    # coordinate column to check exposure against. Recorded for completeness.
    # -----------------------------------------------------------------
    result["task1_short_horizon_targets"] = {
        "targets": ["target_future_shot_10s", "target_future_xg_10s"],
        "used_by_legs": ["active-binary", "passive-binary", "active-continuous (hurdle P(shot) stage + value stage)",
                          "passive-continuous (hurdle P(shot) stage + value stage)"],
        "frame_dependent": False,
        "rows_affected": 0,
        "derivation": (
            "src/dax/targets/short_horizon.py's add_future_shot_target/add_future_xg_target "
            "(the only code that computes these two targets) reference exactly these columns: "
            "match_id, period, possession_column (possession/team_in_possession-derived), index, "
            "event_time_seconds, event_type/type, attacking_team_before_action, and "
            "shot_statsbomb_xg (StatsBomb's own precomputed shot-xG field, not derived from this "
            "project's coordinates). No x/y/location/ball_x/ball_y column is read anywhere in "
            "either function body -- grep-confirmed. The binary target is a pure occurrence flag "
            "(does a Shot-type event happen within 10s / same possession by the same team); the "
            "continuous target is a temporal-window sum of StatsBomb's own xG values gated by that "
            "same occurrence logic. Neither depends on this project's coordinate frame in any way, "
            "so 0 of their own training rows are affected by the frame bug -- proven by inspection, "
            "not sampled."
        ),
    }

    # -----------------------------------------------------------------
    # Task 1b -- target_xt_delta_v2 (active-xT). xt_before = grid lookup at
    # the PREVIOUS event's ball_x/ball_y (match_id+index-1, no period
    # restriction); xt_after = 0 if action_ended_possession, else the NEXT
    # event's shot_statsbomb_xg if that next event is a Shot (frame-
    # invariant), else a grid lookup at the NEXT event's ball_x/ball_y.
    # A row's target is exposed if the previous-event lookup OR the (used)
    # next-event grid lookup drew from a frame-inconsistent event.
    # -----------------------------------------------------------------
    print("\n[2/4] target_xt_delta_v2 exposure (active-xT)...")
    locked_cols = ["event_id", "match_id", "action_x", "action_y", "action_ended_possession"]
    locked = pd.read_parquet(ACTIVE_LOCKED, columns=locked_cols)
    ev_idx = pd.read_parquet(EVENTS_PARQUET, columns=["id", "match_id", "period", "index", "type"])
    ev_idx_by_id = ev_idx.set_index("id")[["match_id", "period", "index"]]
    locked = locked.join(ev_idx_by_id, on="event_id", rsuffix="_ev")

    ev_full = ev_idx.merge(ev.drop(columns=["match_id"]), left_on="id", right_on="id", how="left")
    prev_lookup = ev_full.set_index(["match_id", "index"])[["frame_consistent"]]
    prev_keys = list(zip(locked["match_id"], (locked["index"] - 1).astype("int64")))
    prev_fc = prev_lookup.reindex(prev_keys).reset_index(drop=True)["frame_consistent"]

    ends_here = locked["action_ended_possession"].fillna(False).to_numpy()
    next_lookup = ev_full.set_index(["match_id", "period", "index"])[["type", "frame_consistent"]]
    next_keys = list(zip(locked["match_id"], locked["period"], (locked["index"] + 1).astype("int64")))
    nxt = next_lookup.reindex(next_keys).reset_index(drop=True)
    is_shot_next = (nxt["type"] == "Shot").to_numpy()
    next_used_and_inconsistent = (~ends_here) & (~is_shot_next) & (nxt["frame_consistent"].to_numpy() == 0)
    prev_inconsistent = (prev_fc.to_numpy() == 0)

    exposed = prev_inconsistent | next_used_and_inconsistent
    n_exposed = int(np.nansum(exposed))
    print(f"  xt_before drawn from inconsistent event: {int(np.nansum(prev_inconsistent))} of {len(locked)}")
    print(f"  xt_after drawn from inconsistent event (non-shot, non-ending only): "
          f"{int(np.nansum(next_used_and_inconsistent))} of {len(locked)}")
    print(f"  target_xt_delta_v2 rows exposed (either lookup): {n_exposed} of {len(locked)} "
          f"({n_exposed / len(locked):.4f})")
    result["task1_target_xt_delta_v2"] = {
        "rows_total": int(len(locked)),
        "xt_before_from_inconsistent_event": int(np.nansum(prev_inconsistent)),
        "xt_after_from_inconsistent_event_when_used": int(np.nansum(next_used_and_inconsistent)),
        "rows_exposed_either_lookup": n_exposed,
        "share_exposed": round(float(n_exposed / len(locked)), 4),
        "note": "xt_after only draws a grid lookup (frame-dependent) when action_ended_possession is "
                "False AND the next event is not a Shot (StatsBomb's own xg is frame-invariant); "
                "rows where xt_after is 0.0 or a Shot's own xg are not counted as exposed via that term.",
    }

    # -----------------------------------------------------------------
    # Task 1c -- target_xt_delta_passive (passive-xT). xt_before = grid
    # lookup at the SNAPSHOT'S OWN ball_x/ball_y (i.e. the on-ball event
    # itself); xt_after = same possession-outcome rule as active.
    # -----------------------------------------------------------------
    print("\n[3/4] target_xt_delta_passive exposure (passive-xT)...")
    passive = pd.read_parquet(PASSIVE_LOCKED, columns=["event_id"])
    uniq_event_ids = passive.drop_duplicates(subset="event_id")
    own = ev_full.set_index("id").loc[uniq_event_ids["event_id"], ["match_id", "period", "index", "frame_consistent"]].reset_index()

    own_fc_inconsistent = (own["frame_consistent"].to_numpy() == 0)

    # need action_ended_possession per event -- read fresh (not in `ev`)
    aep = pd.read_parquet(EVENTS_PARQUET, columns=["id", "action_ended_possession"]).set_index("id")
    ends_here_p = aep.loc[own["id"], "action_ended_possession"].fillna(False).to_numpy()

    next_lookup_p = ev_full.set_index(["match_id", "period", "index"])[["type", "frame_consistent"]]
    next_keys_p = list(zip(own["match_id"], own["period"], (own["index"] + 1).astype("int64")))
    nxt_p = next_lookup_p.reindex(next_keys_p).reset_index(drop=True)
    is_shot_next_p = (nxt_p["type"] == "Shot").to_numpy()
    next_used_and_inconsistent_p = (~ends_here_p) & (~is_shot_next_p) & (nxt_p["frame_consistent"].to_numpy() == 0)

    exposed_p_unique = own_fc_inconsistent | next_used_and_inconsistent_p
    n_exposed_unique = int(np.nansum(exposed_p_unique))
    print(f"  xt_before (snapshot's own event) inconsistent: {int(np.nansum(own_fc_inconsistent))} "
          f"of {len(own)} unique events")
    print(f"  xt_after drawn from inconsistent event (non-shot, non-ending only): "
          f"{int(np.nansum(next_used_and_inconsistent_p))} of {len(own)}")
    print(f"  target_xt_delta_passive UNIQUE events exposed: {n_exposed_unique} of {len(own)} "
          f"({n_exposed_unique / len(own):.4f})")

    # Row-level (defender-slot-duplicated) share -- join back onto full
    # passive_defense.parquet population to report in the same row-level
    # style the other legs use (46.4%-style number), not just unique events.
    exposed_by_event = pd.DataFrame({"event_id": own["id"].to_numpy(), "exposed": exposed_p_unique})
    passive_full = pd.read_parquet(PASSIVE_LOCKED, columns=["event_id"])
    row_level = passive_full.merge(exposed_by_event, on="event_id", how="left")
    n_exposed_rows = int(row_level["exposed"].sum())
    print(f"  target_xt_delta_passive DEFENDER-SLOT ROWS exposed: {n_exposed_rows} of {len(row_level)} "
          f"({n_exposed_rows / len(row_level):.4f})")
    result["task1_target_xt_delta_passive"] = {
        "unique_events_total": int(len(own)),
        "unique_events_xt_before_inconsistent": int(np.nansum(own_fc_inconsistent)),
        "unique_events_xt_after_inconsistent_when_used": int(np.nansum(next_used_and_inconsistent_p)),
        "unique_events_exposed_either_lookup": n_exposed_unique,
        "share_exposed_unique_events": round(float(n_exposed_unique / len(own)), 4),
        "defender_slot_rows_total": int(len(row_level)),
        "defender_slot_rows_exposed": n_exposed_rows,
        "share_exposed_defender_slot_rows": round(float(n_exposed_rows / len(row_level)), 4),
        "note": "unique-event share and defender-slot-row share are expected to match (a per-event "
                "property duplicated uniformly across that event's own defender rows); reported both "
                "to confirm.",
    }

    OUT_PATH.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\n[write] {OUT_PATH}")
    print("\nDone (Tasks 1 computed; Task 3 model-importance-weighted exposure computed in a "
          "separate pass -- see coordinate_frame_model_exposure.py).")


if __name__ == "__main__":
    main()
