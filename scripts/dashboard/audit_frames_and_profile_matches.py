"""Coordinate-frame audit + held-out match style profile for the dashboard export (prompt 83).

Read-only. Writes dashboard_data/selection_and_frame_audit.json, the source for
every match-selection and data-quality number in dashboard_data/README.md.

1. Frame audit (all 115 matches, player_defensive_actions.parquet). Raw
   StatsBomb event locations are always in the ACTING team's own frame (own goal
   at x=0). For each locked row, action_x/action_y is compared with the raw event
   location: equal -> "raw" (actor's frame), 180-degree rotation -> "flipped"
   (opponent's frame). The rule-based phase labeller
   (src/dax/features/phase_segmentation.py) assumes the possession team attacks
   toward x=120, so a row is frame-CONSISTENT only when its stored frame is the
   possession (attacking_team) team's frame.

2. Style profile (23 canonical held-out test matches). Computed from RAW
   locations (acting team's own frame), actions by the defending side only,
   so it is not affected by the frame issue in (1). No model output and no
   target value is used -- selection criteria only.

Usage:
    .venv/Scripts/python.exe scripts/dashboard/audit_frames_and_profile_matches.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
ACTIVE_PARQUET = REPO_ROOT / "data" / "features" / "player_defensive_actions.parquet"
SPLIT_PATH = REPO_ROOT / "outputs" / "models" / "splits" / "match_assignment.json"
OUT_PATH = REPO_ROOT / "dashboard_data" / "selection_and_frame_audit.json"


def frame_state(sx: float, sy: float, loc) -> str:
    if not isinstance(loc, list) or len(loc) < 2:
        return "unknown"
    if abs(sx - loc[0]) < 1e-6 and abs(sy - loc[1]) < 1e-6:
        return "raw"
    if abs(sx - (120.0 - loc[0])) < 1e-6 and abs(sy - (80.0 - loc[1])) < 1e-6:
        return "flipped"
    return "unknown"


def main() -> None:
    meta: dict[int, dict] = {}
    for path in glob.glob(str(RAW_DIR / "matches" / "*.json")):
        for m in json.loads(Path(path).read_text(encoding="utf-8")):
            meta[int(m["match_id"])] = m
    assignment = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    test_ids = sorted(int(k) for k, v in assignment.items() if v == "test")

    cols = ["match_id", "event_id", "event_type", "team", "attacking_team", "defending_team",
            "action_x", "action_y", "phase_label", "counterpress"]
    df = pd.read_parquet(ACTIVE_PARQUET, columns=cols)
    recs = []
    for mid, sub in df.groupby("match_id"):
        raw = {e["id"]: e for e in json.loads((RAW_DIR / "events" / f"{mid}.json").read_text(encoding="utf-8"))}
        for r in sub.itertuples(index=False):
            e = raw.get(r.event_id, {})
            loc = e.get("location")
            recs.append({"state": frame_state(r.action_x, r.action_y, loc),
                         "own_x": loc[0] if isinstance(loc, list) else np.nan,
                         "raw_team": e.get("team")})
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    home = {m: meta[m]["home_team"] for m in meta}
    away = {m: meta[m]["away_team"] for m in meta}
    stored_team = np.where(df["state"] == "raw", df["team"],
                           np.where(df["team"] == df["match_id"].map(home), df["match_id"].map(away), df["match_id"].map(home)))
    df["frame_consistent"] = np.where(df["state"] == "unknown", np.nan, (stored_team == df["attacking_team"]).astype(float))

    clear = df[df["event_type"] == "Clearance"]
    frame_audit = {
        "population": "player_defensive_actions.parquet, all 115 matches, all rows",
        "rows": int(len(df)),
        "rows_by_stored_frame_state": {k: int(v) for k, v in df["state"].value_counts().items()},
        "rows_frame_inconsistent_with_phase_labeller": int((df["frame_consistent"] == 0).sum()),
        "share_frame_inconsistent": round(float((df["frame_consistent"] == 0).mean()), 4),
        "clearance_phase_label_mix_by_consistency": {
            ("consistent" if k == 1 else "inconsistent"): {p: round(float(v), 3) for p, v in grp["phase_label"].value_counts(normalize=True).items()}
            for k, grp in clear.groupby("frame_consistent")
        },
        "raw_event_team_equals_parquet_team_share": round(float((df["raw_team"] == df["team"]).mean()), 4),
    }

    t = df[df["match_id"].isin(test_ids) & (df["team"] == df["defending_team"])]
    profile = []
    for (mid, team), g in t.groupby(["match_id", "team"]):
        m = meta[mid]
        profile.append({
            "match_id": int(mid),
            "match": f"{m['home_team']} {m['home_score']}-{m['away_score']} {m['away_team']}",
            "competition": f"{m['competition_name']} {m['season']}", "stage": m.get("competition_stage"),
            "date": m["match_date"], "defending_team": team, "actions": int(len(g)),
            "mean_own_frame_x": round(float(g["own_x"].mean()), 1),
            "share_in_opponent_half": round(float((g["own_x"] > 60).mean()), 2),
            "share_in_final_third": round(float((g["own_x"] > 80).mean()), 2),
            "share_in_own_third": round(float((g["own_x"] < 40).mean()), 2),
            "counterpress_rate": round(float(g["counterpress"].mean()), 2),
        })
    prof = pd.DataFrame(profile)
    by_match = []
    for mid, g in prof.groupby("match_id"):
        by_match.append({
            "match_id": int(mid), "match": g["match"].iloc[0], "competition": g["competition"].iloc[0],
            "stage": g["stage"].iloc[0],
            "height_gap_m": round(float(g["mean_own_frame_x"].max() - g["mean_own_frame_x"].min()), 1),
            "own_third_share_gap": round(float(g["share_in_own_third"].max() - g["share_in_own_third"].min()), 2),
            "mean_counterpress_rate": round(float(g["counterpress_rate"].mean()), 3),
        })
    by_match.sort(key=lambda r: -r["height_gap_m"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "schema_version": 1,
        "frame_audit": frame_audit,
        "held_out_style_profile": {
            "population": "23 canonical held-out test matches; actions by the defending team only; raw StatsBomb "
                          "location in that team's own frame (own goal x=0). No model outputs or targets used.",
            "by_team": profile,
            "by_match_sorted_by_height_gap": by_match,
        },
    }, indent=2), encoding="utf-8")
    print(json.dumps(frame_audit, indent=2))
    for r in by_match:
        print(r)


if __name__ == "__main__":
    main()
