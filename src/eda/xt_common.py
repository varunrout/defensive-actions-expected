"""Shared plumbing for the xT-delta (`target_xt_delta`) report suite --
ACTIVE-BINARY LEG ONLY.

This module is the single place where the two methodology adaptations the
xT target forces are decided, computed and recorded, so every `*_xt.py`
report script inherits exactly the same convention rather than each one
re-inventing it.

Read-only inputs, always. `data/features/player_defensive_actions.parquet`
and `outputs/prototypes/active_binary_xt_delta.parquet` are joined in
memory by `event_id` (`how="left"`, row-aligned, 56,068 rows) and NEVER
merged-and-resaved. `src/dax/targets/short_horizon.py` is not imported.

--------------------------------------------------------------------------
ADAPTATION 1 -- the scale-setting statistic
--------------------------------------------------------------------------
Every threshold in the xg_target suite is expressed as a multiple of the
dataset's own OVERALL MEAN target value (flat_margin = 0.5 * mean(xg),
consistency range trigger = 1.0 * mean(xg), review near-identical /
distinct-signal = 0.5x / 1.5x mean(xg), interaction min-marginal-delta =
0.5 * mean(xg)). That works because `target_future_xg_10s` is strictly
non-negative with a mean comfortably above zero.

`target_xt_delta` is roughly symmetric around a mean of about -0.0013
(Prompt 64: skew +0.093, 47.0% negative, 41.7% positive, 11.3% exactly
zero). "0.5 x the overall mean" is therefore NEGATIVE and near-zero for
this target -- multiplying a threshold by it would flip signs and collapse
to nothing. Applying the xg convention literally would be meaningless, not
conservative.

Rather than inventing a fresh constant, the xg convention is TRANSLATED:
each xg threshold is re-expressed as a fraction of xg's own standard
deviation on the very same rows, and that fraction is then applied to
`target_xt_delta`'s standard deviation. Both the source ratio and the
resulting absolute threshold are computed at runtime from the real data
(never hardcoded) and written into every JSON/HTML output, so the
translation is auditable rather than asserted.

    flat_margin_xg   = 0.5 * mean(target_future_xg_10s)
    std_fraction     = flat_margin_xg / std(target_future_xg_10s)   # ~0.0844
    flat_margin_xt   = std_fraction * std(target_xt_delta)

--------------------------------------------------------------------------
ADAPTATION 2 -- the conditional panel ("given a shot" has no xT analogue)
--------------------------------------------------------------------------
Every xg_target report carries a second, conditional panel computed on
`target_future_shot_10s == 1`. Its purpose is to strip out the structural
zero mass (xg is exactly 0 wherever no shot occurred, see
STRUCTURAL_ZERO_CHECK.json) so the remaining variation is about chance
QUALITY rather than chance OCCURRENCE.

`target_xt_delta` has no shot column standing behind it and no
occurrence/quality split. Its structural-zero analogue is the 11.3% of
rows whose delta is EXACTLY zero -- the action did not move the ball
across an xT grid-cell boundary at all (the grid is 8x12 over 105x68m, so
many short actions resolve inside one cell; Prompt 64 section 3.4
spot-checked this directly). The conditional panel in this suite is
therefore `target_xt_delta != 0`, and is labelled "given a non-zero delta"
everywhere, never "given a shot".

--------------------------------------------------------------------------
NOT an adaptation: log-transforms
--------------------------------------------------------------------------
Checked directly before assuming: the xg_target report suite never
log-transforms its target anywhere. Every xg report uses the raw
groupby-mean of `target_future_xg_10s`. The `log1p` framing in this project
lives in the MODELLING legs, not in this report suite, so there is no
log-transform to remove here. What genuinely changed is the scale-setting
statistic (Adaptation 1), the conditional panel (Adaptation 2) and the
explicit two-directional framing of lift (Adaptation 3 below).

--------------------------------------------------------------------------
ADAPTATION 3 -- two-directional framing
--------------------------------------------------------------------------
The xg suite's `lift` is already a signed difference of means, but the
surrounding prose is one-sided throughout ("higher mean xG" = worse for the
defence, monotonically). For `target_xt_delta`, the sign carries the entire
football meaning: POSITIVE = threat fell across the action (good for the
defence), NEGATIVE = threat rose (bad). Every table in this suite therefore
also reports `pct_negative` / `pct_positive` / `pct_zero` alongside the
mean, and rankings state explicitly that they are by |magnitude| across
both directions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

TARGET = "target_xt_delta"
XG_TARGET = "target_future_xg_10s"
SHOT_COL = "target_future_shot_10s"

ACTIVE_PARQUET = REPO_ROOT / "data" / "features" / "player_defensive_actions.parquet"
XT_PARQUET = REPO_ROOT / "outputs" / "prototypes" / "active_binary_xt_delta.parquet"
OUT_DIR = REPO_ROOT / "reports" / "analysis" / "xt_target"

# The xg_target suite's own ratios, restated here so the translation below
# is traceable to a specific source convention rather than a bare number.
XG_FLAT_MARGIN_RATIO = 0.5        # generate_numerical_xg_target_analysis.FLAT_MARGIN_RATIO
XG_RANGE_TRIGGER_RATIO = 1.0      # generate_numerical_xg_target_analysis.RANGE_TRIGGER_RATIO
XG_NEAR_IDENTICAL_RATIO = 0.5     # generate_review_analysis_xg.NEAR_IDENTICAL_RATIO
XG_DISTINCT_SIGNAL_RATIO = 1.5    # generate_review_analysis_xg.DISTINCT_SIGNAL_RATIO

RHO_THRESHOLD = 0.05              # scale-free, carried over unchanged
SMALL_N_THRESHOLD = 500           # feature_config.SMALL_N_THRESHOLD, carried over unchanged
SMALL_N_CONDITIONAL_THRESHOLD = 30  # xg suite's SMALL_N_GIVEN_SHOT_THRESHOLD, carried over


def load_active_xt(columns: list[str] | None = None) -> pd.DataFrame:
    """Active parquet joined to the xT-delta prototype by `event_id`.

    Read-only: neither input file is written back. `how="left"` on the
    56,068-row active parquet; 32 rows carry a NaN `target_xt_delta`
    (Prompt 64 section 2 -- the immediately preceding event is a non-ball
    event with no location), left as NaN rather than zero-filled.
    """
    if columns is not None:
        # `event_id` is the join key and `target_future_xg_10s` is needed by
        # target_scale() for Adaptation 1's runtime threshold translation --
        # both are always loaded, whatever the caller asked for.
        columns = list(dict.fromkeys(list(columns) + ["event_id", XG_TARGET]))
    df = pd.read_parquet(ACTIVE_PARQUET, columns=columns)
    xt = pd.read_parquet(XT_PARQUET)
    return df.merge(xt, on="event_id", how="left")


def target_scale(df: pd.DataFrame) -> dict:
    """Compute Adaptation 1's translated thresholds from the real data.

    Every number here is measured on the rows actually in `df` -- nothing
    is hardcoded from Prompt 64's writeup.
    """
    xt = df[TARGET].dropna()
    xg = df[XG_TARGET]

    xt_mean, xt_std = float(xt.mean()), float(xt.std())
    xg_mean, xg_std = float(xg.mean()), float(xg.std())

    flat_margin_xg = XG_FLAT_MARGIN_RATIO * xg_mean
    std_fraction = flat_margin_xg / xg_std

    nz = xt[xt != 0]
    nz_std = float(nz.std())

    return {
        "target": TARGET,
        "n_rows": int(len(df)),
        "n_defined": int(len(xt)),
        "n_nan": int(df[TARGET].isna().sum()),
        "mean": round(xt_mean, 8),
        "std": round(xt_std, 8),
        "skew": round(float(xt.skew()), 4),
        "excess_kurtosis": round(float(xt.kurtosis()), 4),
        "pct_zero": round(float((xt == 0).mean() * 100), 3),
        "pct_negative": round(float((xt < 0).mean() * 100), 3),
        "pct_positive": round(float((xt > 0).mean() * 100), 3),
        # --- Adaptation 1 audit trail ---
        "xg_mean_same_rows": round(xg_mean, 8),
        "xg_std_same_rows": round(xg_std, 8),
        "xg_flat_margin_absolute": round(flat_margin_xg, 8),
        "xg_flat_margin_as_fraction_of_xg_std": round(std_fraction, 6),
        "flat_margin": round(std_fraction * xt_std, 8),
        "range_trigger": round((XG_RANGE_TRIGGER_RATIO / XG_FLAT_MARGIN_RATIO) * std_fraction * xt_std, 8),
        "near_identical_threshold": round(std_fraction * xt_std, 8),
        "distinct_signal_threshold": round((XG_DISTINCT_SIGNAL_RATIO / XG_FLAT_MARGIN_RATIO) * std_fraction * xt_std, 8),
        # conditional (non-zero-delta) panel scale
        "n_nonzero": int(len(nz)),
        "mean_nonzero": round(float(nz.mean()), 8),
        "std_nonzero": round(nz_std, 8),
        "flat_margin_nonzero": round(std_fraction * nz_std, 8),
        "scale_note": (
            "ADAPTATION 1. target_xt_delta is roughly symmetric with a near-zero, NEGATIVE mean "
            f"({xt_mean:+.6f}), so the xg_target suite's convention of expressing every threshold as a "
            "multiple of the overall mean target value is meaningless here (it would flip signs and collapse "
            "to nothing). Instead the xg convention is TRANSLATED, not replaced: xg's own flat_margin "
            f"(0.5 x mean xg = {flat_margin_xg:.6f} on these same rows) is re-expressed as a fraction of xg's "
            f"own standard deviation ({xg_std:.6f}), giving {std_fraction:.6f}, and that fraction is applied "
            f"to target_xt_delta's standard deviation ({xt_std:.6f}). Every number in this calculation is "
            "computed at runtime from the real parquet, not carried over from a writeup. Scale-free "
            "thresholds (|Spearman rho| >= 0.05, small-n row counts, the substitutive/additive ratio cutoffs) "
            "are carried over completely unchanged."
        ),
        "conditional_panel_note": (
            "ADAPTATION 2. The xg_target suite's second panel conditions on target_future_shot_10s == 1 to "
            "strip the structural zero mass. target_xt_delta has no shot column behind it; its structural-zero "
            f"analogue is the {float((xt == 0).mean() * 100):.1f}% of rows with an EXACTLY zero delta (the "
            "action did not cross an xT grid-cell boundary). The conditional panel throughout this suite is "
            "therefore 'given a non-zero delta' (target_xt_delta != 0), never 'given a shot'."
        ),
        "direction_note": (
            "ADAPTATION 3. Sign carries the football meaning: POSITIVE target_xt_delta = xT fell across the "
            "action (threat reduced, good for the defence), NEGATIVE = xT rose. Rankings are by |magnitude| "
            "across BOTH directions, never one-sided 'higher is worse' lift, and every table also reports the "
            "negative/positive/zero share behind its mean."
        ),
        "log_transform_note": (
            "NOT an adaptation: checked directly rather than assumed -- the xg_target report suite never "
            "log-transforms its target anywhere (every xg report uses the raw groupby mean of "
            "target_future_xg_10s). The log1p framing in this project belongs to the modelling legs, not to "
            "this report suite, so there was no log-transform to remove."
        ),
    }


# --------------------------------------------------------------------------
# Statistics -- the xT counterparts of compute_stats_xg.py
# --------------------------------------------------------------------------

def base_xt(df: pd.DataFrame) -> float:
    return float(df[TARGET].mean())


def _direction_shares(s: pd.Series) -> dict:
    n = len(s)
    if n == 0:
        return {"pct_negative": float("nan"), "pct_positive": float("nan"), "pct_zero": float("nan")}
    return {
        "pct_negative": round(float((s < 0).mean() * 100), 2),
        "pct_positive": round(float((s > 0).mean() * 100), 2),
        "pct_zero": round(float((s == 0).mean() * 100), 2),
    }


def categorical_xt_table(df: pd.DataFrame, col: str) -> list[dict]:
    """Mean target_xt_delta per category, plus the direction shares behind it
    (Adaptation 3). Sorted descending by mean, so the most threat-REDUCING
    categories sit at the top and the most threat-increasing at the bottom --
    both ends are the interesting ends for this target."""
    sub = df[[col, TARGET]].dropna(subset=[TARGET])
    grouped = sub.groupby(sub[col].fillna("(missing)"), dropna=False)[TARGET]
    rows = []
    for idx, s in grouped:
        rows.append({
            "category": str(idx),
            "n": int(len(s)),
            "xt_sum": float(s.sum()),
            "mean_xt": float(s.mean()),
            **_direction_shares(s),
        })
    rows.sort(key=lambda r: r["mean_xt"], reverse=True)
    return rows


def boolean_xt_lift(df: pd.DataFrame, col: str) -> dict:
    sub = df.dropna(subset=[TARGET])
    true_mask = sub[col] == True   # noqa: E712
    false_mask = sub[col] == False  # noqa: E712
    n_true, n_false = int(true_mask.sum()), int(false_mask.sum())
    s_true, s_false = sub.loc[true_mask, TARGET], sub.loc[false_mask, TARGET]
    mean_true = float(s_true.mean()) if n_true else float("nan")
    mean_false = float(s_false.mean()) if n_false else float("nan")
    return {
        "column": col,
        "mean_xt_true": mean_true,
        "mean_xt_false": mean_false,
        "lift": mean_true - mean_false,
        "pct_true": float(sub[col].mean() * 100),
        "n_true": n_true,
        "n_false": n_false,
        "small_n": n_true < SMALL_N_THRESHOLD or n_false < SMALL_N_THRESHOLD,
        "true_direction": _direction_shares(s_true),
        "false_direction": _direction_shares(s_false),
    }


def nonzero_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Adaptation 2's conditional subset -- the xT analogue of the xg suite's
    `df.loc[df[SHOT_COL] == 1]`."""
    return df.loc[df[TARGET].notna() & (df[TARGET] != 0)]


def categorical_xt_table_nonzero(df: pd.DataFrame, col: str) -> list[dict]:
    return categorical_xt_table(nonzero_subset(df), col)


def boolean_xt_lift_nonzero(df: pd.DataFrame, col: str) -> dict:
    return boolean_xt_lift(nonzero_subset(df), col)


# --------------------------------------------------------------------------
# Cross-references to Prompt 64's two carried-forward findings
# --------------------------------------------------------------------------

PROMPT_64_CLEARANCE_FINDING = (
    "Prompt 64 finding carried forward (outputs/prototypes/XT_TARGET_PROTOTYPE.md section 3.1/3.4): for "
    "Clearance events, action_x/action_y -- this target's definition of the post-action location, and "
    "therefore of xt_after -- is the location the clearance was taken FROM, not where the ball ends up. "
    "Last-ditch clearances are very often taken deep in the defending box, a high-xT cell by construction, "
    "so Clearance rows carry a systematically negative delta even though a clearance is a good defensive "
    "outcome. Prompt 64 measured this directly at -0.0315 mean delta for Clearance inside the "
    "action_ended_possession slice, against +0.0009 to +0.0076 for every other event type there. This is a "
    "measurement artefact of the xt_after definition, NOT evidence that clearances are bad defending."
)

PROMPT_64_SHAPE_FINDING = (
    "Prompt 64 finding carried forward (outputs/prototypes/XT_TARGET_PROTOTYPE.md section 3.2): "
    "target_xt_delta is roughly symmetric (skew +0.093), heavy-tailed (excess kurtosis 8.85), 47.0% "
    "negative / 41.7% positive / 11.3% exactly zero. This is a fundamentally different shape from "
    "target_future_xg_10s, which is strictly non-negative, zero-inflated and heavily right-skewed. Any "
    "methodology that assumed non-negativity had to be adapted explicitly -- see this module's Adaptation "
    "1/2/3 notes, restated in every report."
)

CLEARANCE_EVENT_TYPE = "Clearance"
