"""Expected Threat (xT) grid lookup (prompt 64 prototype).

Standalone module, deliberately separate from `short_horizon.py` -- the
existing shot/xG-in-window target stays exactly as-is, still used by every
already-promoted model on every leg. This module implements only the xT
*lookup* (a value-per-zone grid, looked up by pitch location), not a
from-scratch Markov-chain grid fit -- per Varun's explicit decision this
session, fitting a bespoke grid on this repo's own two-tournament dataset
(WC2022 + Euro2024) risks the same thin-category artifact this project has
hit twice already (`defender_functional_role_unclassified`, prompt 60; thin
`position`/`phase_label` slices, prompts 38/43/46).

Grid source: the publicly published Karun Singh xT grid (the original xT
framework author, https://karun.in/blog/expected-threat.html), fetched from
https://karun.in/blog/data/open_xt_12x8_v1.json and vendored locally at
`src/dax/targets/open_xt_12x8_v1.json` (a static, small (8x12) JSON array,
committed so this module never needs network access at runtime).

Why not import `socceraction.xthreat` directly (its own documented source
for this exact grid, via `socceraction.xthreat.load_model`): confirmed this
session that `socceraction` 1.4.2 (latest PyPI release) cannot actually be
imported in this project's real `.venv` (Python 3.13, numpy 2.4.6, pandas
2.3.3) -- it pins `pandera<0.14.0,>=0.13.4`, and pandera 0.13.4 itself
fails to import under numpy>=2.0 (`AttributeError: np.string_ was removed
in the NumPy 2.0 release`); the newer pandera (0.33.1) that IS numpy-2.x
compatible removes the `pa.SchemaModel` API socceraction's own SPADL schema
classes depend on. A three-way version deadlock, not a typo or a missed
flag -- downgrading numpy/pandas project-wide to satisfy socceraction's
2021-era pins was rejected outright (would risk breaking the other three
legs' already-promoted models and the full 248-test suite). `socceraction`
1.4.2 IS installed in this `.venv` (`pip install socceraction --no-deps`,
confirmed importable at the package level) for provenance/documentation
purposes, but this module does not import it -- the grid-lookup logic
below is a direct, minimal reimplementation of `socceraction.xthreat`'s
own `_get_cell_indexes`/`ExpectedThreat.rate` indexing convention (read
directly from that module's source before writing this), applied to a
grid loaded from the same public JSON that `socceraction.xthreat.load_model`
itself documents as the canonical pretrained source.

Coordinate transform (confirmed, not assumed): this repo's own pitch
convention (confirmed elsewhere in this project, and reconfirmed directly
against `player_defensive_actions.parquet`'s real `action_x`/`action_y`
range, 0-119.9 / 0-80.0) is a 120x80 (yards) pitch, attacking goal at
(120, 40), defending goal at (0, 40). SPADL (socceraction's own coordinate
convention, which the Karun Singh grid is built against) uses a 105x68
(metres) pitch. The transform is a simple per-axis rescale --
`x_spadl = x_project / 120 * 105`, `y_spadl = y_project / 80 * 68` -- no
rotation or flip needed since both conventions already put the attacking
goal at the "high x" end and use the same y-orientation.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

GRID_PATH = Path(__file__).resolve().parent / "open_xt_12x8_v1.json"

# SPADL / socceraction's own pitch convention (metres) -- what the Karun
# Singh grid is defined against.
SPADL_FIELD_LENGTH = 105.0
SPADL_FIELD_WIDTH = 68.0

# This project's own pitch convention (yards) -- confirmed against
# player_defensive_actions.parquet's real action_x/action_y range
# (0.2-119.9 / 0.0-80.0), attacking goal at (120, 40).
PROJECT_FIELD_LENGTH = 120.0
PROJECT_FIELD_WIDTH = 80.0


@lru_cache(maxsize=1)
def _load_grid() -> np.ndarray:
    """Loads the vendored Karun Singh xT grid once per process. Shape
    (8, 12) -- (w=8 width cells, l=12 length cells), matching the grid as
    published (`open_xt_12x8_v1.json`)."""
    if not GRID_PATH.exists():
        raise FileNotFoundError(
            f"xT grid not found at {GRID_PATH}. This module vendors a static copy of the "
            "publicly published Karun Singh grid (https://karun.in/blog/data/open_xt_12x8_v1.json) "
            "rather than fetching it at runtime -- it should not be missing from a checked-out repo."
        )
    data = json.loads(GRID_PATH.read_text(encoding="utf-8"))
    grid = np.array(data, dtype=np.float64)
    if grid.ndim != 2:
        raise ValueError(f"Expected a 2D xT grid, got shape {grid.shape}")
    return grid


def _project_to_spadl(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rescales this project's 120x80-yard coordinates to SPADL's
    105x68-metre convention. Both conventions share the same origin/axis
    orientation (attacking goal at the high-x end, y increasing the same
    direction) -- confirmed directly, not assumed -- so this is a pure
    per-axis rescale, no flip or rotation."""
    x_spadl = np.asarray(x, dtype=np.float64) / PROJECT_FIELD_LENGTH * SPADL_FIELD_LENGTH
    y_spadl = np.asarray(y, dtype=np.float64) / PROJECT_FIELD_WIDTH * SPADL_FIELD_WIDTH
    return x_spadl, y_spadl


def get_xt(x, y) -> np.ndarray:
    """Returns the xT grid value for each (x, y) location, given in this
    project's own 120x80-yard coordinate convention.

    Mirrors `socceraction.xthreat._get_cell_indexes` /
    `ExpectedThreat.rate`'s own indexing convention exactly (read directly
    from that module's installed source, not re-derived from scratch):
    column index from the length axis, row index from the width axis
    counted from the far side (`w - 1 - yj`), clipped to the grid bounds.
    NaN inputs propagate to NaN outputs rather than being silently
    coerced to a boundary cell.

    Parameters
    ----------
    x, y : array-like
        Pitch locations in this project's own 120x80-yard convention.

    Returns
    -------
    np.ndarray
        The xT grid value at each location (float64, NaN where x or y is NaN).
    """
    grid = _load_grid()
    w, l = grid.shape  # noqa: E741 -- l/w match socceraction's own naming

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    nan_mask = np.isnan(x_arr) | np.isnan(y_arr)

    x_spadl, y_spadl = _project_to_spadl(np.nan_to_num(x_arr), np.nan_to_num(y_arr))

    xi = np.clip((x_spadl / SPADL_FIELD_LENGTH * l).astype(np.int64), 0, l - 1)
    yj = np.clip((y_spadl / SPADL_FIELD_WIDTH * w).astype(np.int64), 0, w - 1)
    row = (w - 1) - yj

    values = grid[row, xi]
    values = np.where(nan_mask, np.nan, values)
    return values
