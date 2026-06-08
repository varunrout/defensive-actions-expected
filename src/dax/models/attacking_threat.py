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
        for row in rows:
            cell = self._cell(row.get("ball_x"), row.get("ball_y"))
            if cell is None:
                continue
            cx, cy = cell
            self._totals[cx][cy] += 1.0
            self._positives[cx][cy] += float(row.get("target_shot_in_10s") or 0.0)
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


def add_shot_in_10s_target(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    with_targets: list[dict[str, Any]] = []
    n = len(rows)

    for i, row in enumerate(rows):
        current_time = _event_seconds(row)
        current_team = row.get("team_in_possession")
        target = 0

        for j in range(i + 1, n):
            next_row = rows[j]
            if next_row.get("period") != row.get("period"):
                break
            if _event_seconds(next_row) - current_time > 10:
                break
            if next_row.get("team_in_possession") != current_team:
                continue
            if next_row.get("event_type") == "Shot":
                target = 1
                break

        out = dict(row)
        out["target_shot_in_10s"] = target
        with_targets.append(out)

    return with_targets

