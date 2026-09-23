"""Post-fix goalkeeper-event sanity check (Prompt 85, Phase 1).

Re-runs the same oracle Prompt 84's audit used to establish that raw StatsBomb
event locations are always given in the ACTING team's own attacking frame
(own goal at x=0, so a goalkeeper event -- the ball is almost always near
their own goal -- should sit at raw x<20 regardless of period or team).

This check reads raw cached event JSON only (data/raw/events/*.json) and never
calls the loader, because the property it verifies (raw-frame convention) is
independent of the loader's sign-inference logic and must hold before any fix
is applied. It exists here, post-fix, to (a) confirm the fix's premise still
holds against the full cached corpus, not just the sample the original audit
used, and (b) give Phase 1 a citable, re-runnable artifact.

Usage:
    .venv/Scripts/python.exe scripts/analysis/coordinate_frame_fix_gk_sanity_check.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_EVENTS_DIR = REPO_ROOT / "data" / "raw" / "events"
ACTIVE_PARQUET = REPO_ROOT / "data" / "features" / "player_defensive_actions.parquet"
OUT_PATH = REPO_ROOT / "outputs" / "models" / "validation" / "coordinate_frame_fix_gk_sanity_check.json"

# The same defensive-action event types that make up
# data/features/player_defensive_actions.parquet -- the population Prompt 84's
# original 1,126-event GK oracle was drawn from. Restricting to these (rather
# than every event type, which includes long GK goal-kick carries/passes that
# legitimately start away from x=0) reproduces that oracle's population.
DEFENSIVE_ACTION_TYPES = {
    "Pressure", "Ball Recovery", "Duel", "Clearance", "Block",
    "Foul Committed", "Interception", "50/50",
}


def main() -> None:
    locked_match_ids = set(
        pd.read_parquet(ACTIVE_PARQUET, columns=["match_id"])["match_id"].unique().tolist()
    )

    total = 0
    under_20 = 0
    per_match: list[dict] = []

    for path in sorted(glob.glob(str(RAW_EVENTS_DIR / "*.json"))):
        match_id = int(Path(path).stem)
        if match_id not in locked_match_ids:
            continue
        events = json.loads(Path(path).read_text(encoding="utf-8"))
        m_total = 0
        m_under_20 = 0
        for e in events:
            if e.get("position") != "Goalkeeper":
                continue
            if e.get("type") not in DEFENSIVE_ACTION_TYPES:
                continue
            loc = e.get("location")
            if not isinstance(loc, list) or len(loc) < 2:
                continue
            m_total += 1
            if float(loc[0]) < 20.0:
                m_under_20 += 1
        if m_total:
            per_match.append({"match_id": match_id, "gk_events": m_total, "raw_x_lt_20": m_under_20})
            total += m_total
            under_20 += m_under_20

    result = {
        "population": "the 115 matches in player_defensive_actions.parquet, data/raw/events/*.json, "
                      "events with position == 'Goalkeeper' and type in the 8 defensive-action event "
                      "types (matches Prompt 84's oracle population)",
        "gk_events_total": total,
        "gk_events_raw_x_lt_20": under_20,
        "share_raw_x_lt_20": round(under_20 / total, 4) if total else None,
        "matches_with_gk_events": len(per_match),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({**result, "per_match": per_match}, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
