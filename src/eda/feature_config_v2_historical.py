"""Reconstructed V2 candidate feature lists -- stage 07's actual state
(36 active / 39 passive): after the structural-redesign (04) and
collapse-tier (05, active-only) / raw-coordinate-drop (06, passive-only)
stages, but *before* the VIF drop (09, active) and leakage drop (10,
passive) that later brought the list down to the final locked 34/38.

Built as: feature_config.py's live (final, 34/38) lists plus the specific
columns each of those two later stages removed --
local_numerical_balance_5m/10m (active, VIF) and has_screened_outcome
(passive, leakage) -- since neither drop had happened yet when V2 actually
ran. Column existence in the current parquet files is asserted at import
time, not assumed.

Usage: imported by generate_correlation_analysis_v2.py /
generate_review_analysis_v2.py, which use this instead of
feature_config.py's current lists so CORRELATION_ATLAS_V2.html /
REVIEW_METHODOLOGY_V2.html show stage 07's real historical state.
"""

from __future__ import annotations

from src.eda.feature_config import ACTIVE, PASSIVE

ACTIVE_V2_HISTORICAL = {
    **ACTIVE,
    "label": "Active Defensive Actions (V2 historical -- stage 07)",
    "discrete": [c for c in ACTIVE["discrete"]] + ["local_numerical_balance_5m", "local_numerical_balance_10m"],
}
ACTIVE_V2_HISTORICAL["excluded"] = {
    k: v for k, v in ACTIVE["excluded"].items() if k not in ("local_numerical_balance_5m", "local_numerical_balance_10m")
}

PASSIVE_V2_HISTORICAL = {
    **PASSIVE,
    "label": "Passive / Off-Ball Defensive Positioning (V2 historical -- stage 07)",
    "boolean": [c for c in PASSIVE["boolean"]] + ["has_screened_outcome"],
}
PASSIVE_V2_HISTORICAL["excluded"] = {
    k: v for k, v in PASSIVE["excluded"].items() if k != "has_screened_outcome"
}

EXPECTED_ACTIVE_COUNT = 36
EXPECTED_PASSIVE_COUNT = 39


def _count(cfg: dict) -> int:
    return len(cfg["categorical"]) + len(cfg["boolean"]) + len(cfg["continuous"]) + len(cfg["discrete"])


_active_n, _passive_n = _count(ACTIVE_V2_HISTORICAL), _count(PASSIVE_V2_HISTORICAL)
if _active_n != EXPECTED_ACTIVE_COUNT or _passive_n != EXPECTED_PASSIVE_COUNT:
    raise AssertionError(
        f"V2 historical reconstruction doesn't match the documented stage-07 baseline: "
        f"active={_active_n} (expected {EXPECTED_ACTIVE_COUNT}), passive={_passive_n} (expected {EXPECTED_PASSIVE_COUNT})"
    )

DATASETS_V2_HISTORICAL = {"active": ACTIVE_V2_HISTORICAL, "passive": PASSIVE_V2_HISTORICAL}
