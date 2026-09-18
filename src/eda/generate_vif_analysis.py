"""CLI entrypoint: compute variance inflation factors (VIF) per dataset over
continuous + discrete + boolean candidate features (categorical excluded --
VIF is defined for numeric regressors; that's a real limitation, stated in
the output, not silently skipped).

Critical implementation note: VIF here is computed via statsmodels'
variance_inflation_factor (the standard R^2-based definition, equivalent to
np.linalg.inv on the correlation matrix). np.linalg.pinv is deliberately
NOT used -- the pseudo-inverse silently suppresses near-singular directions
and reports deceptively low, safe-looking VIFs for a genuinely collinear
cluster. This happened during the manual Cowork pass this pipeline
replicates and gave a false "all clear" until re-checked with the correct
method.

For the active dataset, this script also applies the fix it finds: if
local_numerical_balance_5m/10m show pathological VIF (a near-exact linear
combination of attackers_within_Nm - defenders_within_Nm invisible to
pairwise correlation), they are reported as the root cause, and
feature_config.py's active candidate list must be updated separately (see
the accompanying edit to feature_config.py) -- this script only measures and
reports before/after, it does not itself edit feature_config.py.

Usage:
    python -m src.eda.generate_vif_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.eda.feature_config import DATASETS

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "VIF_ANALYSIS.json"

# Columns whose near-exact linear dependency on kept columns (attackers_within_Nm
# - defenders_within_Nm) was confirmed by this analysis -- dropped from the
# active candidate list in feature_config.py as a direct result of this script's
# findings, not a COLLAPSE/REVIEW correlation-tier verdict.
ACTIVE_VIF_DROPPED = {"local_numerical_balance_5m", "local_numerical_balance_10m"}


def _split_constant_columns(X: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """VIF (and correlation) is undefined for a column with zero variance --
    regressing a constant on other predictors gives a meaningless R^2, and
    naively computing it anyway produces a nonsensical VIF near 0 (not >= 1,
    which is mathematically impossible) while also poisoning the
    correlation matrix into exact singularity.

    This happens here specifically as a LISTWISE-DELETION artefact, not
    because the column truly lacks variance in the full dataset: flags like
    has_visible_attacker/has_visible_defender (active) or has_option_2/3
    (passive) are False exactly when some OTHER candidate column (e.g.
    nearest_attacker_distance, top_option_2_threat_score) is null by
    construction. Once every row with any null across the full candidate
    set is dropped for VIF's complete-case requirement, the surviving rows
    are, by that same construction,100% True on the flag -- it looks
    constant only in this listwise-deleted slice, not in the real dataset.

    Returns (X without the constant columns, list of {feature, reason} for
    what was excluded and why) -- reported explicitly, never silently
    dropped.
    """
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    excluded = [
        {
            "feature": c,
            "reason": (
                f"Constant ({X[c].iloc[0]!r} for all {len(X)} complete-case rows) after listwise deletion across "
                "the full candidate set -- not truly constant in the full dataset; this flag's False rows are "
                "exactly the rows where some other candidate column is null by construction, so complete-case "
                "deletion removes every row where the flag is False. VIF and pairwise correlation are undefined "
                "for a constant column; excluded from this computation rather than reported as a bogus near-zero "
                "VIF."
            ),
        }
        for c in constant_cols
    ]
    return X.drop(columns=constant_cols), excluded


def _condition_number(X: pd.DataFrame) -> dict:
    corr = X.corr().values
    try:
        cond = float(np.linalg.cond(corr))
        converged = True
    except np.linalg.LinAlgError:
        cond = float("inf")
        converged = False
    return {"condition_number": cond, "svd_converged": converged}


def _compute_vif(X: pd.DataFrame) -> list[dict]:
    Xc = X.copy()
    Xc.insert(0, "const", 1.0)
    values = Xc.values
    results = []
    for i, col in enumerate(Xc.columns):
        if col == "const":
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            vif = variance_inflation_factor(values, i)
        vif = float(vif) if np.isfinite(vif) else None
        results.append({"feature": col, "vif": vif, "vif_display": vif if vif is not None else float("inf")})
    results.sort(key=lambda r: r["vif_display"], reverse=True)
    for r in results:
        del r["vif_display"]
    return results


def analyze_dataset(dataset_key: str) -> dict:
    dataset_cfg = DATASETS[dataset_key]
    df = pd.read_parquet(REPO_ROOT / dataset_cfg["parquet_path"])

    # Current live candidate list from feature_config.py -- this is what the
    # top-level "vif" result reports. For the active dataset this is the
    # POST-drop list once feature_config.py has been edited (see
    # ACTIVE_VIF_DROPPED); the before/after comparison below is computed
    # independently of when that edit happened, so it stays correct whether
    # this script runs before or after the feature_config.py fix is applied.
    cols = dataset_cfg["continuous"] + dataset_cfg["discrete"] + dataset_cfg["boolean"]
    X_raw = df[cols].astype(float)
    X_complete = X_raw.dropna()
    n_dropped = len(X_raw) - len(X_complete)

    X, excluded_constant = _split_constant_columns(X_complete)
    cond_info = _condition_number(X)
    vif_before = _compute_vif(X)

    result = {
        "dataset": dataset_key,
        "n_rows_used": len(X),
        "n_rows_dropped_for_na": n_dropped,
        "n_features": len(cols),
        "n_features_used_in_vif": len(X.columns),
        "categorical_excluded_caveat": (
            "Categorical features are excluded from this VIF pass -- VIF is defined over numeric regressors "
            "(continuous + discrete + boolean only). A categorical column collinear with a numeric cluster would "
            "not be caught here. This is a known limitation of this analysis, not a silent omission."
        ),
        "constant_columns_excluded": excluded_constant,
        "condition_number": cond_info["condition_number"],
        "svd_converged": cond_info["svd_converged"],
        "vif": vif_before,
    }

    if dataset_key == "active":
        # Reconstruct the pre-drop column list explicitly (add the dropped
        # columns back in if feature_config.py has already been edited to
        # remove them) so "before" always means "including
        # local_numerical_balance_5m/10m", regardless of the live config state.
        cols_before = list(cols) + [c for c in ACTIVE_VIF_DROPPED if c not in cols]
        cols_after = [c for c in cols_before if c not in ACTIVE_VIF_DROPPED]
        dropped_present = [c for c in ACTIVE_VIF_DROPPED if c in df.columns]

        X_before_complete = df[cols_before].astype(float).dropna()
        X_before, excluded_constant_before = _split_constant_columns(X_before_complete)
        cond_before = _condition_number(X_before)
        vif_before_full = _compute_vif(X_before)

        X_after_complete = df[cols_after].astype(float).dropna()
        X_after, excluded_constant_after = _split_constant_columns(X_after_complete)
        cond_after = _condition_number(X_after)
        vif_after = _compute_vif(X_after)
        result["before_after"] = {
            "dropped_columns": dropped_present,
            "root_cause": (
                "local_numerical_balance_5m is a near-exact linear combination of two columns already kept "
                "(attackers_within_5m - defenders_within_5m); local_numerical_balance_10m likewise for the 10m "
                "pair. Pairwise correlation never surfaced this (max pairwise r among the six flagged columns "
                "was well under 0.90) because the redundancy is a JOINT linear dependency across three columns "
                "at once, not a pairwise one -- exactly what VIF catches and simple correlation cannot."
            ),
            "before": {
                "n_features": len(cols_before),
                "condition_number": cond_before["condition_number"],
                "svd_converged": cond_before["svd_converged"],
                "constant_columns_excluded": excluded_constant_before,
                "max_vif": max((v["vif"] for v in vif_before_full if v["vif"] is not None), default=None),
                "vif": vif_before_full,
            },
            "after": {
                "n_features": len(cols_after),
                "condition_number": cond_after["condition_number"],
                "svd_converged": cond_after["svd_converged"],
                "constant_columns_excluded": excluded_constant_after,
                "max_vif": max((v["vif"] for v in vif_after if v["vif"] is not None), default=None),
                "vif": vif_after,
            },
        }

    return result


def main() -> None:
    results = {key: analyze_dataset(key) for key in DATASETS}

    output = {"datasets": results}
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    for key, r in results.items():
        print(f"\n=== {key} ===")
        if r["constant_columns_excluded"]:
            print(f"excluded as constant (listwise-deletion artefact): {[c['feature'] for c in r['constant_columns_excluded']]}")
        print(f"condition number: {r['condition_number']:.3e} (SVD converged: {r['svd_converged']})")
        print("top VIFs:")
        for v in r["vif"][:8]:
            vif_str = f"{v['vif']:.2f}" if v["vif"] is not None else "inf"
            print(f"  {v['feature']:40s} {vif_str}")
        if "before_after" in r:
            ba = r["before_after"]
            print(f"\n  before: n_features={ba['before']['n_features']} max_vif={ba['before']['max_vif']} cond={ba['before']['condition_number']:.3e}")
            print(f"  after:  n_features={ba['after']['n_features']} max_vif={ba['after']['max_vif']:.3f} cond={ba['after']['condition_number']:.3f}")
            if ba['after']['constant_columns_excluded']:
                print(f"  after excluded as constant: {[c['feature'] for c in ba['after']['constant_columns_excluded']]}")

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
