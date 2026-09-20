"""Prompt 64 prototype: xT-delta target for the active-binary leg only.

Computes xt_before / xt_after / target_xt_delta for every row of
data/features/player_defensive_actions.parquet (read-only -- that file is
NOT modified) and writes them to a separate, row-aligned output,
outputs/prototypes/active_binary_xt_delta.parquet, joined back by
event_id.

  xt_after  = xT value at the defensive action's own resulting location
              (action_x/action_y, already in the locked parquet --
              confirmed identical to data/processed/events_with_targets.parquet's
              own ball_x/ball_y at the same event_id, to floating-point
              exactness).
  xt_before = xT value at the PREVIOUS event's ball location (same
              match_id, index = this event's index - 1, in
              events_with_targets.parquet) -- the attacking state
              immediately before this defensive action, per this
              session's pre-action DAx discussion. events_with_targets.parquet's
              own `index` column is confirmed contiguous per match (no
              gaps), so "index - 1" is a real previous-event lookup, not
              an approximation.
  target_xt_delta = xt_before - xt_after.

A handful of rows (this leg's very first event of a match, index==1) have
no previous event at all -- xt_before is left NaN for those, reported
explicitly in the validation output rather than silently zero-filled.

Usage:
    python scripts/analysis/build_xt_delta_prototype.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dax.targets.expected_threat import get_xt  # noqa: E402

LOCKED_PARQUET = REPO_ROOT / "data" / "features" / "player_defensive_actions.parquet"
EVENTS_PARQUET = REPO_ROOT / "data" / "processed" / "events_with_targets.parquet"
OUT_DIR = REPO_ROOT / "outputs" / "prototypes"
OUT_PATH = OUT_DIR / "active_binary_xt_delta.parquet"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    locked = pd.read_parquet(LOCKED_PARQUET, columns=["event_id", "match_id", "action_x", "action_y"])
    print(f"[data] locked parquet: {len(locked)} rows (read-only, not modified)")

    events = pd.read_parquet(
        EVENTS_PARQUET,
        columns=["id", "match_id", "index", "ball_x", "ball_y"],
    )
    print(f"[data] events_with_targets: {len(events)} rows")

    # events_with_targets.index confirmed contiguous (no gaps) per match_id
    # earlier this session -- reconfirmed here defensively.
    idx_check = events.groupby("match_id")["index"].agg(["min", "max", "count"])
    contiguous = (idx_check["max"] - idx_check["min"] + 1 == idx_check["count"]).all()
    assert contiguous, "events_with_targets.index is not contiguous per match -- previous-event lookup would be wrong"
    print("[check] events_with_targets.index confirmed contiguous per match (no gaps)")

    # Map event_id (== events.id) -> (match_id, index) for the locked rows.
    events_by_id = events.set_index("id")[["match_id", "index"]]
    locked = locked.join(events_by_id, on="event_id", rsuffix="_ev")
    missing_index = locked["index"].isna().sum()
    if missing_index:
        raise ValueError(f"{missing_index} locked rows have no matching event_id in events_with_targets -- stopping")

    # Previous-event lookup: same match_id, index - 1.
    prev_lookup = events.set_index(["match_id", "index"])[["ball_x", "ball_y"]]
    prev_keys = list(zip(locked["match_id"], (locked["index"] - 1).astype("int64")))
    prev = prev_lookup.reindex(prev_keys).reset_index(drop=True)
    n_no_prev = (locked["index"] == 1).sum()
    print(f"[check] {n_no_prev} rows are their match's first event (index==1) -- xt_before will be NaN for these")

    xt_after = get_xt(locked["action_x"].to_numpy(), locked["action_y"].to_numpy())
    xt_before = get_xt(prev["ball_x"].to_numpy(), prev["ball_y"].to_numpy())
    target_xt_delta = xt_before - xt_after

    out = pd.DataFrame({
        "event_id": locked["event_id"].to_numpy(),
        "xt_before": xt_before,
        "xt_after": xt_after,
        "target_xt_delta": target_xt_delta,
    })
    print(f"[check] xt_before NaN count: {out['xt_before'].isna().sum()} (should equal {n_no_prev})")
    print(f"[check] xt_after NaN count: {out['xt_after'].isna().sum()} (should be 0 -- action_x/y always populated)")

    out.to_parquet(OUT_PATH, index=False)
    print(f"[write] {OUT_PATH} ({len(out)} rows, row-aligned to the locked parquet by event_id, "
          "NOT merged into it)")

    print("\n=== target_xt_delta summary (rows with a defined previous event) ===")
    defined = out.dropna(subset=["target_xt_delta"])
    print(defined["target_xt_delta"].describe())


if __name__ == "__main__":
    main()
