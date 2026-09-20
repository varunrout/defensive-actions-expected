"""Rung 0 of the active-xT leg (prompt 70): architecture decision + dummy/baseline
for target_xt_delta_v2 (active-binary leg, prompts 64/66/67, fully EDA'd and locked
against 34 locked ACTIVE features per reports/analysis/xt_target/FEATURE_LOCK_CONFIRMATION_XT.json).

ARCHITECTURE DECISION (item 0 of the prompt) -- decided here, not assumed
---------------------------------------------------------------------------
target_xt_delta_v2 is signed (35.4% negative, 44.4% positive per XT_TARGET_PROTOTYPE_V2.md,
reconfirmed here: 11,238 of 55,761 defined rows, 20.15%, are EXACTLY zero) and heavy-tailed
but not positive-only (skew -0.306, excess kurtosis 18.33). This rules out the lognormal
hurdle architecture used on active-continuous/passive-continuous (c1d/d1) outright -- a
strictly-positive regression head cannot model a target with real negative mass, and there is
no log transform of a signed quantity.

The remaining question this prompt has to answer with evidence: is the 20.15% exact-zero mass
predictable from the 32 modelling-eligible locked features well enough to be worth a dedicated
classifier stage, or is it close to conditionally random given those features (in which case a
single-stage signed regression is the honest choice)?

Checked directly against reports/analysis/xt_target/active_category_atlas.json and
active_flag_ledger.json (not re-derived from scratch, per the prompt's own instruction):

  - event_type (LOCKED categorical feature) pct-zero by category ranges from 5.70% (Ball
    Recovery) to 27.94% (Foul Committed) -- a ~5x spread across a feature that is directly in
    the modelling feature set.
  - action_retained_defensive_team_control (LOCKED boolean) pct-zero: 24.19% when True vs
    10.63% when False -- more than 2x.
  - is_in_attacking_box / is_in_defending_box (LOCKED booleans) show the same pattern in
    opposite directions (12.56%/13.84% vs 21.04%/20.75%).

This is real, substantial variation from features that are actually available to a model at
prediction time -- not from the three off-pool possession flags (action_ended_possession /
action_changed_possession / action_won_possession) that mechanically TRIGGER the zero (those
remain excluded from feature_config.py, confirmed still excluded, and are not used here either
-- using them would make the classifier stage trivial by construction, not a genuine
prediction). **Decision: two-stage architecture.** Stage A -- LogisticRegression classifying
"this row's delta is nonzero" from the 32 locked features. Stage B -- HuberRegressor (robust to
this leg's heavy tails; explicitly NOT lognormal, per the prompt's constraint) fit only on the
nonzero-delta rows, predicting the signed delta directly, no transform. Combined:
E[delta] = P(nonzero) * E[delta | nonzero], since E[delta | zero] = 0 exactly by construction
(action_ended_possession forces xt_after = 0, so target_xt_delta_v2 = xt_before - 0 = xt_before
is never itself the source of the FALSE label's rows -- the zero here means the true target
value, not a missing observation).

NAMING: v (active-binary shot target), c (active-continuous xg), p (passive-binary),
d (passive-continuous) are all claimed. "x" (for xT) is unclaimed -- checked directly:
no script, report, or output under scripts/models/, reports/modeling/, outputs/models/ uses an
x0/x1-prefixed variant name. Used from Rung 0 onward: x0_dummy, x1_two_stage_huber.

FEATURE SET: same 32 locked ACTIVE features as v1/c1 (34 locked minus
nearest_defender_distance [self-reference data bug, target-agnostic] and defenders_within_5m
[structurally nested inside defenders_within_10m, target-agnostic]) -- reused directly from
train_active_binary_baseline.py's own DesignMatrixBuilder and column lists, not reimplemented.

METRIC-CONVENTION FLAG: dax.models.evaluation.regression_metrics and
dax.models.two_part_xg.compute_hurdle_metrics both hardcode their zero/nonzero split as
`y_true > 0` -- correct for a non-negative xg target, WRONG for this leg's signed target (a
genuine negative row would be misclassified into the "zero" bucket). Neither is reused for the
zero/nonzero breakdown here; this script computes its own split on `y_true == 0` /
`y_true != 0`. Overall mae/rmse/r2/spearman are computed directly (sklearn/scipy), not via
those two helpers, for the same reason -- kept self-contained rather than partially reusing a
function whose other half is wrong for this target.

Split: the same canonical, frozen match-grouped split every other leg uses
(outputs/models/splits/match_assignment.json), loaded via dax.models.splits, never rebuilt.

Rung 0 only -- x0_dummy (grand-mean constant regressor) and x1_two_stage_huber (the architecture
decided above). No gradient boosting, no promotion decision, no full ladder.

Usage:
    .venv/Scripts/python.exe scripts/models/train_active_xt_baseline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import average_precision_score, log_loss

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_active_binary_baseline as vb  # noqa: E402  (DesignMatrixBuilder, feature lists, canonical split reuse)
from dax.models.baselines import ConstantRegressor  # noqa: E402
from dax.models.diagnostics import _ACCENT, _ACCENT_DARK, _MUTED, _new_axes, _save_fixed, ensure_dir  # noqa: E402
from dax.models.evaluation import classification_metrics, safe_spearman  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

LOCKED_PARQUET = REPO_ROOT / "data" / "features" / "player_defensive_actions.parquet"
XT_PARQUET = REPO_ROOT / "outputs" / "prototypes" / "active_binary_xt_delta_v2.parquet"
TARGET_COL = "target_xt_delta_v2"
GROUP_COL = "match_id"
EVENT_COL = "event_id"

REG_DIR = REPO_ROOT / "outputs" / "models" / "regression"
REG_CHARTS_DIR = REG_DIR / "charts"
COMPARISONS_DIR = REPO_ROOT / "outputs" / "models" / "comparisons"
VALIDATION_DIR = REPO_ROOT / "outputs" / "models" / "validation"

VARIANTS = ["x0_dummy", "x1_two_stage_huber"]

FEATURE_COLS = vb.ALL_FEATURE_COLS  # 32 locked ACTIVE features, reused unchanged


def load_data() -> pd.DataFrame:
    """locked player_defensive_actions.parquet left-joined (READ-ONLY, in memory
    only) with the xt-delta v2 prototype by event_id. Neither file is modified."""
    locked = pd.read_parquet(LOCKED_PARQUET)
    xt = pd.read_parquet(XT_PARQUET, columns=[EVENT_COL, TARGET_COL])
    assert not locked[EVENT_COL].duplicated().any(), "locked parquet must be one row per event_id"
    assert not xt[EVENT_COL].duplicated().any(), "xt prototype must be one row per event_id"
    df = locked.merge(xt, on=EVENT_COL, how="left")
    assert len(df) == len(locked), "left-join must not change row count"
    n_nan = int(df[TARGET_COL].isna().sum())
    print(f"[data] locked rows={len(locked)}, xt-delta NaN rows={n_nan} (dropped) -- "
          f"matches Prompt 66/67's own count (307 of 56,068)")
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    n_zero = int((df[TARGET_COL] == 0.0).sum())
    print(f"[data] rows after drop: {len(df)}; exactly-zero: {n_zero} ({100 * n_zero / len(df):.2f}%)")
    return df


# ---------------------------------------------------------------------------
# Fold-safe fit/predict
# ---------------------------------------------------------------------------

def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, variant: str):
    """Returns (combined_pred, extra_dict) -- extra_dict carries whatever
    diagnostic pieces (classifier score, builder, models) the variant produced,
    for reuse in held-out reporting / chart building."""
    if variant == "x0_dummy":
        model = ConstantRegressor(stat="mean")
        model.fit(train_df, train_df[TARGET_COL].to_numpy())
        pred = model.predict(test_df)
        return pred, {"model": model}

    y_nonzero_train = (train_df[TARGET_COL].to_numpy() != 0.0).astype(int)

    clf_builder = vb.DesignMatrixBuilder(add_interactions=False, add_quadratic=False).fit(train_df)
    x_train_clf, clf_feature_names = clf_builder.transform(train_df)
    x_test_clf, _ = clf_builder.transform(test_df)
    clf = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42)
    clf.fit(x_train_clf, y_nonzero_train)
    p_nonzero_test = clf.predict_proba(x_test_clf)[:, 1]

    nonzero_train_df = train_df.loc[train_df[TARGET_COL] != 0.0]
    reg_builder = vb.DesignMatrixBuilder(add_interactions=False, add_quadratic=False).fit(nonzero_train_df)
    x_train_reg, reg_feature_names = reg_builder.transform(nonzero_train_df)
    x_test_reg, _ = reg_builder.transform(test_df)
    reg = HuberRegressor(max_iter=500)
    reg.fit(x_train_reg, nonzero_train_df[TARGET_COL].to_numpy())
    e_delta_given_nonzero_test = reg.predict(x_test_reg)

    combined_pred = p_nonzero_test * e_delta_given_nonzero_test
    extra = {
        "clf": clf, "clf_builder": clf_builder, "clf_feature_names": clf_feature_names,
        "reg": reg, "reg_builder": reg_builder, "reg_feature_names": reg_feature_names,
        "p_nonzero": p_nonzero_test, "e_delta_given_nonzero": e_delta_given_nonzero_test,
    }
    return combined_pred, extra


# ---------------------------------------------------------------------------
# Metrics -- self-contained, correct for a signed target (see module docstring)
# ---------------------------------------------------------------------------

def signed_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    zero = y_true == 0.0
    nonzero = ~zero
    spearman = safe_spearman(y_true, y_pred)
    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "spearman": float(spearman) if not np.isnan(spearman) else float("nan"),
        "mean_prediction": float(y_pred.mean()),
        "mean_observed": float(y_true.mean()),
        "prediction_bias": float((y_pred - y_true).mean()),
        "zero_target_mae": float(mean_absolute_error(y_true[zero], y_pred[zero])) if zero.any() else float("nan"),
        "nonzero_target_mae": float(mean_absolute_error(y_true[nonzero], y_pred[nonzero])) if nonzero.any() else float("nan"),
        "nonzero_target_rmse": float(mean_squared_error(y_true[nonzero], y_pred[nonzero]) ** 0.5) if nonzero.any() else float("nan"),
    }
    return out


# ---------------------------------------------------------------------------
# CV
# ---------------------------------------------------------------------------

def run_cv(df: pd.DataFrame, variant: str) -> dict:
    folds = canonical_grouped_folds(df, group_col=GROUP_COL)
    df = df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()

    fold_rows = []
    oof_pred = np.full(len(df), np.nan)
    y_all = df[TARGET_COL].to_numpy()

    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]

        pred, _ = fit_predict(train_df, test_df, variant)
        oof_pred[test_mask] = pred

        m = signed_regression_metrics(y_all[test_mask], pred)
        m["fold"] = int(fold_id)
        m["n_train"] = int(train_mask.sum())
        m["n_test"] = int(test_mask.sum())
        fold_rows.append(m)

    fold_df = pd.DataFrame(fold_rows)
    oof_metrics = signed_regression_metrics(y_all, oof_pred)
    return {"variant": variant, "fold_metrics": fold_df, "oof_metrics": oof_metrics, "oof_pred": oof_pred, "df": df}


def calibration_bins(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 5) -> list[dict]:
    order = np.argsort(y_pred)
    y_true_sorted, y_pred_sorted = y_true[order], y_pred[order]
    edges = np.array_split(np.arange(len(y_pred_sorted)), n_bins)
    rows = []
    for i, idx in enumerate(edges):
        if len(idx) == 0:
            continue
        rows.append({
            "bin": i, "n": int(len(idx)),
            "mean_predicted": float(y_pred_sorted[idx].mean()),
            "mean_actual": float(y_true_sorted[idx].mean()),
            "gap": float(y_pred_sorted[idx].mean() - y_true_sorted[idx].mean()),
        })
    return rows


def append_to_csv(path: Path, new_rows: list[dict]) -> None:
    """Union-of-columns append (Prompt 54's own fix)."""
    new_df = pd.DataFrame(new_rows)
    if path.exists():
        existing = pd.read_csv(path)
        all_cols = list(existing.columns) + [c for c in new_df.columns if c not in existing.columns]
        existing = existing.reindex(columns=all_cols)
        new_df = new_df.reindex(columns=all_cols)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Charts -- own small function (correct axis labels for a signed xT-delta
# target; dax.models.diagnostics.save_regression_charts hardcodes
# "future xG" axis labels, wrong for this leg -- same reasoning
# train_passive_continuous_baseline.py used for writing its own chart fn).
# ---------------------------------------------------------------------------

def save_xt_regression_charts(chart_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, calib_rows: list[dict], coef_table: pd.DataFrame | None):
    ensure_dir(chart_dir)

    fig, ax = _new_axes()
    ax.scatter(y_pred, y_true, alpha=0.15, s=10, color=_ACCENT, edgecolor="none")
    lo, hi = float(min(y_pred.min(), y_true.min())), float(max(y_pred.max(), y_true.max()))
    ax.plot([lo, hi], [lo, hi], color=_MUTED, linewidth=1.2, linestyle="--", label="y = x")
    ax.legend(loc="upper left")
    ax.set_xlabel("Predicted target_xt_delta_v2")
    ax.set_ylabel("Observed target_xt_delta_v2")
    ax.set_title("Predicted vs observed (signed xT delta)")
    _save_fixed(fig, chart_dir / "predicted_vs_observed.png")

    resid = y_pred - y_true
    fig, ax = _new_axes()
    ax.hist(resid, bins=30, color=_ACCENT, edgecolor=_ACCENT_DARK, linewidth=0.6, alpha=0.85)
    ax.axvline(0, color=_MUTED, linewidth=1.2, linestyle="--")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution")
    _save_fixed(fig, chart_dir / "residual_distribution.png")

    fig, ax = _new_axes()
    bins_x = [r["mean_predicted"] for r in calib_rows]
    ax.plot(bins_x, [r["mean_predicted"] for r in calib_rows], marker="o", color=_ACCENT, linewidth=2, label="mean predicted")
    ax.plot(bins_x, [r["mean_actual"] for r in calib_rows], marker="s", color=_MUTED, linewidth=2, label="mean actual")
    ax.axhline(0, color="#cbd5e1", linewidth=0.8, linestyle=":")
    ax.legend(loc="upper left")
    ax.set_xlabel("Bin (ordered by predicted delta)")
    ax.set_ylabel("target_xt_delta_v2")
    ax.set_title("Calibration: mean predicted vs mean actual, by bin")
    _save_fixed(fig, chart_dir / "calibration_bins.png")

    fig, ax = _new_axes()
    if coef_table is not None and len(coef_table) > 0:
        top = coef_table.reindex(coef_table["abs_coef"].sort_values(ascending=False).index)[:15]
        ax.barh(top["feature"][::-1], top["coef"][::-1], color=_ACCENT)
        ax.set_xlabel("Coefficient (Huber regression, signed-delta scale)")
        ax.set_title("Top 15 |coefficient| -- nonzero-delta regression head")
    else:
        ax.text(0.5, 0.5, "No features (dummy variant)", ha="center", va="center", transform=ax.transAxes, color=_MUTED)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Feature coefficients (n/a)")
    _save_fixed(fig, chart_dir / "feature_coefficients.png")


def main() -> None:
    REG_DIR.mkdir(parents=True, exist_ok=True)
    REG_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] loading data (read-only join, locked files untouched)...")
    df_all = load_data()
    assert len(FEATURE_COLS) == 32, f"Expected 32 locked ACTIVE modelling features, got {len(FEATURE_COLS)}"
    print(f"  32 locked ACTIVE modelling features confirmed (same set as v1/c1)")

    load_canonical_split()
    test_mask_full = canonical_test_mask(df_all, group_col=GROUP_COL)
    trainval = df_all.loc[~test_mask_full].reset_index(drop=True)
    test = df_all.loc[test_mask_full].reset_index(drop=True)
    print(f"  trainval: {len(trainval)} rows / {trainval[GROUP_COL].nunique()} matches; "
          f"held-out test: {len(test)} rows / {test[GROUP_COL].nunique()} matches")

    fold_probe = canonical_grouped_folds(trainval, group_col=GROUP_COL)
    probe_df = trainval.loc[fold_probe["row_index"]].copy()
    probe_df["fold"] = fold_probe["fold"].to_numpy()
    fold_counts = probe_df.groupby("fold").size()
    fold_nonzero_counts = probe_df[probe_df[TARGET_COL] != 0.0].groupby("fold").size()
    print(f"  rows per canonical fold: {fold_counts.to_dict()}")
    print(f"  nonzero-delta rows per canonical fold: {fold_nonzero_counts.to_dict()}")

    print("\n[2/6] cross-validation, x0_dummy and x1_two_stage_huber...")
    comparison_rows = []
    cv_results = {}
    for variant in VARIANTS:
        print(f"  [cv] {variant}")
        result = run_cv(trainval, variant)
        cv_results[variant] = result
        row = {"variant": variant, "n_features": 0 if variant == "x0_dummy" else 32}
        row.update(result["oof_metrics"])
        row["rows"] = len(trainval)
        row["matches"] = trainval[GROUP_COL].nunique()
        comparison_rows.append(row)

    print("\n[3/6] significance test, x1 vs x0 (per-fold RMSE -- no log transform on "
          "either variant here, so no common-scale fix is needed, unlike c0/c1 or d0/d1)...")
    a = cv_results["x1_two_stage_huber"]["fold_metrics"]["rmse"].to_numpy()
    b = cv_results["x0_dummy"]["fold_metrics"]["rmse"].to_numpy()
    diff = a - b
    t_stat, p_val = stats.ttest_rel(a, b)
    if len(set(diff.round(12))) > 1:
        w_stat, w_p = stats.wilcoxon(a, b)
    else:
        w_stat, w_p = float("nan"), float("nan")
    sig = {
        "metric": "rmse (per fold, signed target_xt_delta_v2 scale, lower is better)",
        "fold_rmse_x1": a.tolist(), "fold_rmse_x0": b.tolist(),
        "mean_diff_x1_minus_x0": float(diff.mean()), "std_diff": float(diff.std(ddof=1)),
        "paired_t_stat": float(t_stat), "paired_t_pvalue": float(p_val),
        "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
        "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
        "n_folds": int(len(a)),
    }
    (VALIDATION_DIR / "significance_x0_vs_x1.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")
    print(f"  mean_diff (x1-x0) rmse: {sig['mean_diff_x1_minus_x0']:+.5f}, "
          f"paired_t_p={sig['paired_t_pvalue']:.4f}, wilcoxon_p={sig['wilcoxon_pvalue']}")

    print("\n[4/6] classifier-stage diagnostic (P(nonzero), OOF, x1 only -- not the Rung-0 gate)...")
    oof_df = cv_results["x1_two_stage_huber"]["df"]
    y_nonzero_oof = (oof_df[TARGET_COL].to_numpy() != 0.0).astype(int)
    # Re-run fit_predict per fold already computed p_nonzero internally but did not persist it
    # across folds in run_cv (only the combined prediction is kept there) -- recompute cleanly here.
    p_nonzero_oof = np.full(len(oof_df), np.nan)
    for fold_id in sorted(oof_df["fold"].unique()):
        train_mask = (oof_df["fold"] != fold_id).to_numpy()
        test_mask = (oof_df["fold"] == fold_id).to_numpy()
        train_df, test_df = oof_df.loc[train_mask], oof_df.loc[test_mask]
        y_nonzero_train = (train_df[TARGET_COL].to_numpy() != 0.0).astype(int)
        clf_builder = vb.DesignMatrixBuilder(add_interactions=False, add_quadratic=False).fit(train_df)
        x_train_clf, _ = clf_builder.transform(train_df)
        x_test_clf, _ = clf_builder.transform(test_df)
        clf = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42).fit(x_train_clf, y_nonzero_train)
        p_nonzero_oof[test_mask] = clf.predict_proba(x_test_clf)[:, 1]
    clf_diag = classification_metrics(y_nonzero_oof, p_nonzero_oof)
    clf_diag["note"] = "Diagnostic only (zero-vs-nonzero classifier stage of x1). Rung-0 gate is the combined x1-vs-x0 regression comparison above."
    (VALIDATION_DIR / "x1_classifier_stage_diagnostic.json").write_text(json.dumps(clf_diag, indent=2), encoding="utf-8")
    print(f"  P(nonzero) OOF: roc_auc={clf_diag['roc_auc']:.4f} average_precision={clf_diag['average_precision']:.4f} "
          f"brier={clf_diag['brier_score']:.4f}")

    print("\n[5/6] held-out test readout (fit once on all trainval rows)...")
    held_out_rows = []
    for variant in VARIANTS:
        pred, extra = fit_predict(trainval, test, variant)
        y_true = test[TARGET_COL].to_numpy()
        m = signed_regression_metrics(y_true, pred)
        m["variant"] = variant
        m["readout"] = "FINAL_HELD_OUT_TEST_NOT_FOR_SELECTION"
        m["rows"] = len(test)
        m["matches"] = test[GROUP_COL].nunique()
        held_out_rows.append(m)

        calib_rows = calibration_bins(y_true, pred)
        coef_table = None
        if variant == "x1_two_stage_huber":
            reg, reg_feature_names = extra["reg"], extra["reg_feature_names"]
            coef_table = pd.DataFrame({
                "feature": reg_feature_names, "coef": reg.coef_, "abs_coef": np.abs(reg.coef_),
            }).sort_values("abs_coef", ascending=False).reset_index(drop=True)
            coeff_json = {
                "variant": variant, "intercept": float(reg.intercept_), "n_features": len(reg_feature_names),
                "regression_head_coefficients": coef_table.to_dict(orient="records"),
                "classifier_head_intercept": float(extra["clf"].intercept_[0]),
                "classifier_head_coefficients": {
                    name: float(c) for name, c in zip(extra["clf_feature_names"], extra["clf"].coef_[0])
                },
                "fitted_on": "all trainval rows (canonical split); classifier on all trainval rows, "
                             "regression head on nonzero-delta trainval rows only; held-out test used once for readout only",
            }
            (REG_DIR / f"{variant}.json").write_text(json.dumps(coeff_json, indent=2), encoding="utf-8")
            joblib.dump(extra["clf"], REG_DIR / f"{variant}_classifier.joblib")
            joblib.dump(reg, REG_DIR / f"{variant}_regressor.joblib")
        else:
            joblib.dump(extra["model"], REG_DIR / f"{variant}.joblib")

        save_xt_regression_charts(REG_CHARTS_DIR / variant, y_true, pred, calib_rows, coef_table)
        print(f"  [held-out] {variant}: rmse={m['rmse']:.5f} mae={m['mae']:.5f} r2={m['r2']:.5f} "
              f"spearman={m['spearman']:.4f} zero_mae={m['zero_target_mae']:.5f} nonzero_mae={m['nonzero_target_mae']:.5f}")

    print("\n[6/6] writing comparison CSVs...")
    append_to_csv(COMPARISONS_DIR / "active_xt_baseline_comparison.csv", comparison_rows)
    append_to_csv(COMPARISONS_DIR / "active_xt_baseline_held_out_test_readout.csv", held_out_rows)

    print("\n=== CV comparison (5-fold, OOF metrics) ===")
    print(pd.DataFrame(comparison_rows)[["variant", "rmse", "mae", "r2", "spearman", "zero_target_mae", "nonzero_target_mae"]].to_string(index=False))
    print("\n=== Held-out test readout ===")
    print(pd.DataFrame(held_out_rows)[["variant", "rmse", "mae", "r2", "spearman", "zero_target_mae", "nonzero_target_mae"]].to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
