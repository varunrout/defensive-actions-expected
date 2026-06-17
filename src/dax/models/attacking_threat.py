"""xT-style attacking threat baseline for DAx MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dax.constants import PITCH_X_MAX, PITCH_Y_MAX


@dataclass
class GridThreatModel:
    n_x: int = 12
    n_y: int = 8
    smoothing: float = 1.0

    def __post_init__(self) -> None:
        self._totals = [[0.0 for _ in range(self.n_y)] for _ in range(self.n_x)]
        self._positives = [[0.0 for _ in range(self.n_y)] for _ in range(self.n_x)]

    def _cell(self, x: float | None, y: float | None) -> tuple[int, int] | None:
        if x is None or y is None:
            return None
        try:
            xf, yf = float(x), float(y)
        except (TypeError, ValueError):
            return None
        import math
        if math.isnan(xf) or math.isnan(yf):
            return None
        if xf < 0 or yf < 0:
            return None

        cx = min(self.n_x - 1, int((xf / PITCH_X_MAX) * self.n_x))
        cy = min(self.n_y - 1, int((yf / PITCH_Y_MAX) * self.n_y))
        return cx, cy

    def fit(self, rows: list[dict[str, Any]]) -> "GridThreatModel":
        # Reset state on every fit; use a future partial_fit for intentional accumulation.
        self._totals = [[0.0 for _ in range(self.n_y)] for _ in range(self.n_x)]
        self._positives = [[0.0 for _ in range(self.n_y)] for _ in range(self.n_x)]
        for row in rows:
            cell = self._cell(row.get("ball_x"), row.get("ball_y"))
            if cell is None:
                continue
            cx, cy = cell
            self._totals[cx][cy] += 1.0
            self._positives[cx][cy] += float(row.get("target_future_shot_10s") or 0.0)
        return self

    def predict_point(self, x: float | None, y: float | None) -> float:
        cell = self._cell(x, y)
        if cell is None:
            return 0.0
        cx, cy = cell
        total = self._totals[cx][cy]
        pos = self._positives[cx][cy]
        return (pos + self.smoothing) / (total + (2 * self.smoothing))

    def predict_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for row in rows:
            out = dict(row)
            out["threat_base_score"] = round(self.predict_point(row.get("ball_x"), row.get("ball_y")), 6)
            scored.append(out)
        return scored


def _event_seconds(row: dict[str, Any]) -> int:
    minute = int(row.get("minute") or 0)
    second = int(row.get("second") or 0)
    return (minute * 60) + second


def _event_type(row: dict[str, Any]) -> Any:
    # Support both internal naming conventions used across the pipeline.
    return row.get("event_type", row.get("type"))


def add_shot_in_10s_target(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add possession-bounded observed future-shot target."""
    import pandas as pd
    from dax.targets.short_horizon import add_future_shot_target
    if not rows:
        return []
    df = pd.DataFrame(rows)
    if "attacking_team_before_action" not in df.columns:
        df["attacking_team_before_action"] = df.get("team_in_possession", df.get("possession_team"))
    return add_future_shot_target(df).to_dict("records")


def add_xt_target(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Add xT-based continuous target: aggregated threat in next 10 seconds.
    
    For each event, collect all attacking events in the next 10 seconds
    (same team, same period, same match) and aggregate their position-based
    expected threat scores.
    
    Args:
        rows: sorted list of event dictionaries
        
    Returns:
        list with added 'target_future_xg_10s' field (continuous threat 0.0-1.0+)
    """
    with_targets: list[dict[str, Any]] = []
    n = len(rows)
    
    # Build a threat model from the current data for position-based scoring
    threat_model = GridThreatModel(n_x=12, n_y=8, smoothing=0.5)
    threat_model.fit(rows)
    
    for i, row in enumerate(rows):
        current_time = _event_seconds(row)
        current_team = row.get("team_in_possession")
        current_match = row.get("match_id")
        current_period = row.get("period")
        
        # Collect threat scores from future events in this possession
        future_threat_scores: list[float] = []
        
        for j in range(i + 1, n):
            next_row = rows[j]
            
            # Stop if we cross match/period boundary
            if next_row.get("match_id") != current_match:
                break
            if next_row.get("period") != current_period:
                break
            
            # Stop if we exceed 10 second window
            if _event_seconds(next_row) - current_time > 10:
                break
            
            # Skip if possession changed (defensive action by other team)
            if next_row.get("team_in_possession") != current_team:
                continue
            
            # Score this event's position for threat
            threat_score = threat_model.predict_point(
                next_row.get("ball_x"),
                next_row.get("ball_y")
            )
            future_threat_scores.append(threat_score)
        
        # Aggregate threat: use sum (captures sustained threat over window)
        # Could also use: max (peak threat), mean (avg threat), etc.
        target_xt = sum(future_threat_scores) if future_threat_scores else 0.0
        
        out = dict(row)
        out["target_future_xg_10s"] = round(target_xt, 6)
        with_targets.append(out)
    
    return with_targets
