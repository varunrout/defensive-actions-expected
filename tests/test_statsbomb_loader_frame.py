"""Regression tests for the coordinate-frame fix (Prompt 85, Phase 1).

Encodes the invariant established in reports/modeling/COORDINATE_FRAME_IMPACT_AUDIT.md
and reports/modeling/COORDINATE_FRAME_FIX_AND_REPIPELINE.md: raw StatsBomb event
locations are always given in the ACTING team's own attacking frame (own goal at
x=0), so normalizing a row into the possession team's frame depends only on
whether the acting team equals the possession team for that row -- never on a
per-(period, team) inference over ball-progression deltas.
"""

from __future__ import annotations

import pandas as pd

from dax.data.statsbomb_loader import PITCH_X_MAX, PITCH_Y_MAX, _attack_sign_for_row, _normalize_xy


def test_sign_is_positive_when_actor_has_possession():
    assert _attack_sign_for_row("France", "France") == 1


def test_sign_is_negative_when_actor_is_defending_team():
    assert _attack_sign_for_row("England", "France") == -1


def test_sign_defaults_to_positive_when_possession_team_missing():
    assert _attack_sign_for_row("France", None) == 1
    assert _attack_sign_for_row("France", float("nan")) == 1


def test_sign_is_row_local_not_progression_inferred():
    """The old buggy inference aggregated ball-progression deltas per
    (period, possession_team) and could flip an entire team/period's frame
    based on noisy median progression. The fix must be purely row-local:
    the same (team, possession_team) pair always yields the same sign,
    independent of any other row's coordinates."""
    rows = pd.DataFrame(
        [
            {"team": "France", "possession_team": "France"},
            {"team": "England", "possession_team": "France"},
            {"team": "France", "possession_team": "England"},
            {"team": "England", "possession_team": "England"},
        ]
    )
    signs = [_attack_sign_for_row(r["team"], r["possession_team"]) for _, r in rows.iterrows()]
    assert signs == [1, -1, -1, 1]


def test_goalkeeper_event_normalizes_toward_own_goal_when_defending():
    """A goalkeeper event is raw-recorded near their own goal (x close to 0)
    in the acting team's own frame. When that team is NOT the row's possession
    team (e.g. a GK claw-back while the opponent has nominal possession before
    the whistle), normalizing into the possession team's frame must flip it
    toward the far end (x close to 120), not leave it near the possession
    team's own goal."""
    raw_x, raw_y = 5.0, 40.0
    sign = _attack_sign_for_row(team="England", possession_team="France")
    x, y = _normalize_xy(raw_x, raw_y, sign)
    assert x == PITCH_X_MAX - raw_x
    assert y == PITCH_Y_MAX - raw_y
    assert x > 100.0
