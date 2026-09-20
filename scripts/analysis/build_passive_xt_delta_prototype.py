"""Prompt 68 prototype: target_xt_delta_passive, possession-outcome xT-delta
for the passive-binary leg, using the corrected (Prompt 66) definition from
the start -- no v1/v2 detour needed here (see module docstring reasoning in
the companion report).

Computed once per unique event_id (a passive defender snapshot is off-ball;
target_xt_delta_passive measures what happened to the possession the
snapshot sat inside, not anything about any one defender's own position --
same reasoning that already makes target_future_shot_10s/target_future_xg_10s
identical across every defender row sharing an event_id, reconfirmed
directly in this script, not assumed).

  xt_before = xT(ball_x, ball_y) at the snapshot itself. No previous-event
              lookup needed here (unlike active's own xt_before) --
              passive_defense.parquet's ball_x/ball_y IS the ball location
              at the on-ball event the snapshot is anchored to, confirmed
              identical to events_with_targets.parquet's own ball_x/ball_y
              at the same event_id to floating-point exactness.
  xt_after  = 0.0 if this on-ball event ends its own possession
              (action_ended_possession == True -- a general property of
              every event in events_with_targets.parquet, not
              defensive-action-specific, reconfirmed directly: defined
              non-null for all 422,561 rows, computed once by
              add_event_context() over the full event stream).
              Otherwise, the next same-possession event's xT value (same
              match_id/period, index+1, period-boundary-respecting, exactly
              Prompt 66's own convention) -- or that event's own
              shot_statsbomb_xg if it is a Shot, not a grid lookup at the
              shot's location (same reasoning Prompt 66 gave: a shot's
              value is a scoring-probability question, not a
              continue-possession-from-this-zone question).
  target_xt_delta_passive = xt_before - xt_after.

Writes outputs/prototypes/passive_xt_delta.parquet at UNIQUE event_id grain
(one row per event_id, 198,354 rows -- not expanded to the 1,593,181
defender-slot rows of passive_defense.parquet) -- meant to be left-joined
onto passive_defense.parquet by event_id downstream, the same way
target_future_shot_10s/target_future_xg_10s already are one value per event
repeated across every defender row that shares it. passive_defense.parquet
itself, short_horizon.py, and event_context.py are read-only throughout --
none is modified.

Usage:
    python scripts/analysis/build_passive_xt_delta_prototype.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dax.targets.expected_threat import get_xt  # noqa: E402

PASSIVE_PARQUET = REPO_ROOT / "data" / "features" / "passive_defense.parquet"
EVENTS_PARQUET = REPO_ROOT / "data" / "processed" / "events_with_targets.parquet"
OUT_DIR = REPO_ROOT / "outputs" / "prototypes"
OUT_PATH = OUT_DIR / "passive_xt_delta.parquet"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    passive = pd.read_parquet(PASSIVE_PARQUET, columns=["event_id", "ball_x", "ball_y"])
    print(f"[data] passive_defense.parquet: {len(passive)} rows (read-only, not modified)")
    uniq = passive.drop_duplicates(subset="event_id").reset_index(drop=True)
    print(f"[data] unique event_id: {len(uniq)}")

    events = pd.read_parquet(
        EVENTS_PARQUET,
        columns=["id", "match_id", "period", "index", "type", "ball_x", "ball_y", "shot_statsbomb_xg", "action_ended_possession"],
    )
    print(f"[data] events_with_targets: {len(events)} rows")

    ev_by_id = events.set_index("id")
    own = ev_by_id.loc[uniq["event_id"], ["match_id", "period", "index", "action_ended_possession", "ball_x", "ball_y"]].reset_index()
    assert own["index"].notna().all(), "every passive event_id must resolve to a real event"
    ball_mismatch = (own["ball_x"].to_numpy() - uniq["ball_x"].to_numpy())
    assert np.nanmax(np.abs(ball_mismatch)) == 0.0, "passive_defense.parquet's own ball_x must match events_with_targets exactly"

    # --- xt_before: the snapshot's own ball location, no previous-event lookup. ---
    xt_before = get_xt(uniq["ball_x"].to_numpy(), uniq["ball_y"].to_numpy())

    # --- xt_after: possession-outcome, Prompt 66's convention, unchanged. ---
    ends_here = own["action_ended_possession"].fillna(False).to_numpy()
    print(f"[check] action_ended_possession True: {ends_here.sum()} of {len(uniq)} unique events -> xt_after forced to 0.0")

    period_max_index = events.groupby(["match_id", "period"])["index"].max()
    own_period_max = own.set_index(["match_id", "period"]).index.map(period_max_index)
    is_last_of_period = (own["index"].to_numpy() == np.asarray(own_period_max))
    orphan_rows = (~ends_here) & is_last_of_period
    print(f"[check] rows with no next event AND action_ended_possession==False (should be 0): {orphan_rows.sum()}")

    next_lookup = events.set_index(["match_id", "period", "index"])[["type", "ball_x", "ball_y", "shot_statsbomb_xg"]]
    next_keys = list(zip(own["match_id"], own["period"], (own["index"] + 1).astype("int64")))
    nxt = next_lookup.reindex(next_keys).reset_index(drop=True)

    is_shot_next = (nxt["type"] == "Shot").to_numpy()
    print(f"[check] rows whose next event is a Shot: {(is_shot_next & ~ends_here).sum()}")
    n_shot_xg_nan = nxt.loc[is_shot_next, "shot_statsbomb_xg"].isna().sum()
    print(f"[check] of those, shot_statsbomb_xg NaN: {n_shot_xg_nan} (should be 0)")

    xt_after_continue = np.where(
        is_shot_next,
        nxt["shot_statsbomb_xg"].to_numpy(dtype=np.float64),
        get_xt(nxt["ball_x"].to_numpy(), nxt["ball_y"].to_numpy()),
    )
    n_next_ball_nan = nxt.loc[~is_shot_next, "ball_x"].isna().sum()
    print(f"[check] non-shot 'continues' rows with NaN next-event ball location (-> NaN xt_after): {n_next_ball_nan}")

    xt_after = np.where(ends_here | orphan_rows, 0.0, xt_after_continue)
    target_xt_delta_passive = xt_before - xt_after

    out = pd.DataFrame({
        "event_id": uniq["event_id"].to_numpy(),
        "xt_before": xt_before,
        "xt_after": xt_after,
        "target_xt_delta_passive": target_xt_delta_passive,
    })
    print(f"[check] xt_before NaN: {out['xt_before'].isna().sum()}; "
          f"xt_after NaN: {out['xt_after'].isna().sum()}; "
          f"target_xt_delta_passive NaN: {out['target_xt_delta_passive'].isna().sum()}")

    out.to_parquet(OUT_PATH, index=False)
    print(f"[write] {OUT_PATH} ({len(out)} rows, ONE PER UNIQUE event_id -- left-join onto "
          "passive_defense.parquet by event_id downstream, not pre-expanded)")

    print("\n=== target_xt_delta_passive summary (unique events, defined values) ===")
    defined = out.dropna(subset=["target_xt_delta_passive"])
    print(defined["target_xt_delta_passive"].describe())


if __name__ == "__main__":
    main()
