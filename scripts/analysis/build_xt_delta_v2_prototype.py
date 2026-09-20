"""Prompt 66 prototype: corrected xt_after (possession-outcome, not action-location).

Supersedes Prompt 64's xt_after definition for the active-binary leg only.
Prompt 64 used xt_after = xT(action_x, action_y) -- the defensive action's
OWN recorded location (where the defender was standing / where the ball
was when the action happened). Prompt 65's Leakage Audit Part D' and the
Clearance finding both point at the same root cause: for events like a
Clearance, this location is where the ball WAS when cleared, not where it
ENDS UP -- so xt_after was measuring "how dangerous was the spot the
defender had to act from," not "what is the ball's value immediately
after this action," which is what a VAEP/xT-style value-after term is
supposed to mean.

Corrected rule (Varun's explicit decision, this session):
  - If this action ends/wins its own possession (action_ended_possession
    == True, confirmed below as the correct general flag -- see the
    Step-0 writeup in XT_TARGET_PROTOTYPE_V2.md): xt_after = 0.0. A team
    with no ball has no continuing threat from that zone by definition --
    losing the ball ends the value chain, it doesn't relocate it.
  - Otherwise (possession continues): xt_after = value of the NEXT
    same-period event -- the actual next touch, not this action's own
    location. Two sub-cases:
      - If that next event is a Shot: xt_after = the shot's own
        shot_statsbomb_xg (StatsBomb's own expected-goals value for that
        shot), NOT a grid lookup at the shot's location. This matches the
        established literature convention this whole xT approach is
        modelled on -- socceraction's own ExpectedThreat.rate() explicitly
        excludes shots from movement-grid valuation (ratings stay NaN for
        shots there), because a shot's value is better captured by its own
        scoring-probability model than by the coarse "continue possession
        from this zone" grid. shot_statsbomb_xg is populated for 100% of
        the 954 rows this applies to in this dataset (checked directly).
      - Otherwise: xt_after = xT(next event's ball_x, ball_y) via the same
        vendored Karun Singh grid Prompt 64 used (src/dax/targets/expected_threat.py,
        unchanged, not reimplemented here).

xt_before is UNCHANGED from Prompt 64 (same method: previous event's
ball_x/ball_y by match_id+index, no period restriction) -- that side of
the definition was never in question this prompt, per the task brief.

"Next event" is defined as: same match_id AND same period, the row with
index == this row's index + 1 (event_context.py's own canonical event
order is match_id, period, index; its own possession-context flags are
computed with a period-grouped shift(-1), so a next-event lookup that
respects period boundaries is the only choice consistent with how
action_ended_possession was itself derived -- crossing a half-time break
would not be a real "next touch"). Confirmed directly: 0 of 56,068 rows
are the literal last event of their period with action_ended_possession
still False (the edge case the task asked to check) -- action_ended_possession
already covers every such row, so no separate "no next event, no flag"
fallback branch was needed in practice.

Writes outputs/prototypes/active_binary_xt_delta_v2.parquet (event_id,
xt_before, xt_after, target_xt_delta_v2), row-aligned to the locked
parquet by event_id, not merged into it. The v1 file
(active_binary_xt_delta.parquet) is left untouched on disk.

Usage:
    python scripts/analysis/build_xt_delta_v2_prototype.py
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
OUT_PATH = OUT_DIR / "active_binary_xt_delta_v2.parquet"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    locked = pd.read_parquet(
        LOCKED_PARQUET, columns=["event_id", "match_id", "action_x", "action_y", "action_ended_possession"]
    )
    print(f"[data] locked parquet: {len(locked)} rows (read-only, not modified)")

    events = pd.read_parquet(
        EVENTS_PARQUET,
        columns=["id", "match_id", "period", "index", "type", "ball_x", "ball_y", "shot_statsbomb_xg"],
    )
    print(f"[data] events_with_targets: {len(events)} rows")

    # --- xt_before: UNCHANGED from Prompt 64 (previous event, match_id+index, no period restriction). ---
    events_by_id = events.set_index("id")[["match_id", "period", "index"]]
    locked = locked.join(events_by_id, on="event_id", rsuffix="_ev")
    assert locked["index"].notna().all(), "every locked row must resolve to a real event_id"

    prev_lookup_v1 = events.set_index(["match_id", "index"])[["ball_x", "ball_y"]]
    prev_keys_v1 = list(zip(locked["match_id"], (locked["index"] - 1).astype("int64")))
    prev_v1 = prev_lookup_v1.reindex(prev_keys_v1).reset_index(drop=True)
    xt_before = get_xt(prev_v1["ball_x"].to_numpy(), prev_v1["ball_y"].to_numpy())

    # --- xt_after: corrected, possession-outcome definition. ---
    ends_here = locked["action_ended_possession"].fillna(False).to_numpy()
    print(f"[check] action_ended_possession True: {ends_here.sum()} of {len(locked)} rows -> xt_after forced to 0.0")

    # Edge case confirmed empty in Step 0: last-event-of-period rows with
    # action_ended_possession still False. Reconfirmed here defensively.
    period_max_index = events.groupby(["match_id", "period"])["index"].max()
    locked_period_max = locked.set_index(["match_id", "period"]).index.map(period_max_index)
    is_last_of_period = (locked["index"].to_numpy() == np.asarray(locked_period_max))
    orphan_rows = (~ends_here) & is_last_of_period
    print(f"[check] rows with no next event AND action_ended_possession==False (should be 0): {orphan_rows.sum()}")

    next_lookup = events.set_index(["match_id", "period", "index"])[["type", "ball_x", "ball_y", "shot_statsbomb_xg"]]
    next_keys = list(zip(locked["match_id"], locked["period"], (locked["index"] + 1).astype("int64")))
    nxt = next_lookup.reindex(next_keys).reset_index(drop=True)

    is_shot_next = (nxt["type"] == "Shot").to_numpy()
    print(f"[check] rows whose next event is a Shot (valued via shot_statsbomb_xg, not grid lookup): "
          f"{(is_shot_next & ~ends_here).sum()}")
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

    target_xt_delta_v2 = xt_before - xt_after

    out = pd.DataFrame({
        "event_id": locked["event_id"].to_numpy(),
        "xt_before": xt_before,
        "xt_after": xt_after,
        "target_xt_delta_v2": target_xt_delta_v2,
    })
    print(f"[check] xt_before NaN: {out['xt_before'].isna().sum()}; "
          f"xt_after NaN: {out['xt_after'].isna().sum()}; "
          f"target_xt_delta_v2 NaN: {out['target_xt_delta_v2'].isna().sum()}")

    out.to_parquet(OUT_PATH, index=False)
    print(f"[write] {OUT_PATH} ({len(out)} rows, row-aligned to the locked parquet by event_id, "
          "NOT merged into it; v1 file untouched)")

    print("\n=== target_xt_delta_v2 summary (rows with a defined value) ===")
    defined = out.dropna(subset=["target_xt_delta_v2"])
    print(defined["target_xt_delta_v2"].describe())


if __name__ == "__main__":
    main()
