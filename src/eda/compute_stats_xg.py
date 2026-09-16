"""Dataset-agnostic statistics for the CONTINUOUS xG target
(target_future_xg_10s) -- the xG-side counterpart to compute_stats.py.

compute_stats.py's categorical_rate_table/boolean_lift/base_rate all
hardcode `* 100` to present a 0-1 probability as a percentage, which is the
right unit for the binary target (target_future_shot_10s) but meaningless
for target_future_xg_10s (mean ~0.006-0.008, heavily zero-inflated -- not a
0-100% rate). These are separate functions, not a shared one with a
percentage flag, matching this repo's existing pattern of parallel modules
per methodology/target rather than parameterized branching (e.g.
correlation.py vs correlation_v2.py).

compute_stats.py's continuous_distribution/discrete_distribution are
target-independent (describe a feature's own shape, never reference
target_col) -- there is no xG counterpart needed for those; the existing
Distribution Atlas already covers both target contexts as-is.
"""

from src.eda.feature_config import DISCRETE_TOP_N, SMALL_N_THRESHOLD  # noqa: F401 (kept for parity/future use)


def base_xg_rate(df, target_col: str) -> float:
    return float(df[target_col].mean())


def categorical_xg_table(df, col: str, target_col: str) -> list[dict]:
    """Mean xG per category, always groupby-mean against the full row population."""
    grouped = df.groupby(df[col].fillna("(missing)"), dropna=False)[target_col].agg(
        n="count", xg_sum="sum", mean_xg="mean"
    )
    grouped = grouped.sort_values("mean_xg", ascending=False)
    return [
        {"category": str(idx), "n": int(row["n"]), "xg_sum": float(row["xg_sum"]), "mean_xg": float(row["mean_xg"])}
        for idx, row in grouped.iterrows()
    ]


def boolean_xg_lift(df, col: str, target_col: str) -> dict:
    true_mask = df[col] == True  # noqa: E712
    false_mask = df[col] == False  # noqa: E712
    n_true = int(true_mask.sum())
    n_false = int(false_mask.sum())
    mean_true = float(df.loc[true_mask, target_col].mean()) if n_true else float("nan")
    mean_false = float(df.loc[false_mask, target_col].mean()) if n_false else float("nan")
    lift = mean_true - mean_false
    pct_true = float(df[col].mean() * 100)
    return {
        "column": col,
        "mean_xg_true": mean_true,
        "mean_xg_false": mean_false,
        "lift": lift,
        "pct_true": pct_true,
        "n_true": n_true,
        "n_false": n_false,
        "small_n": n_true < SMALL_N_THRESHOLD or n_false < SMALL_N_THRESHOLD,
    }
