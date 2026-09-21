"""Rung 0 of the passive-xT leg (prompt 76): architecture decision + dummy/baseline
for target_xt_delta_passive (Prompt 68, fully EDA'd via the extended xt_target
portal, Prompt 69, and locked against 38 locked PASSIVE features per
reports/analysis/xt_target/PASSIVE_FEATURE_LOCK_CONFIRMATION_XT.json).

ARCHITECTURE DECISION (item 0 of the prompt) -- checked on passive's own data,
not copied from active-xT just because the target shapes look superficially similar
---------------------------------------------------------------------------
Both targets are signed and zero-inflated (active: 20.15% exactly zero; passive:
26.29% at the unique-event grain per Prompt 68, 26.37% at this leg's own
defender-slot row grain after the join below) -- similar-looking on the surface,
but Prompt 70's own standard requires checking whether the zero mass is
FEATURE-PREDICTABLE on THIS leg's own data before committing to a two-stage
shape, not assuming it transfers.

**Checked directly against reports/analysis/xt_target/passive_category_atlas.json
(Prompt 69, not re-derived from scratch)**: `on_ball_event_type` (a LOCKED
passive categorical feature) shows a dramatically wider zero-rate range than
anything found on the active leg's own Step-0 check (Prompt 70's event_type
range was 5.70%-27.94%, ~5x) --

    Shot     0.30%  zero (n=24,487)
    Pass     6.73%  zero (n=830,924)
    Carry   49.44%  zero (n=713,760)
    Dribble 52.79%  zero (n=21,429)

a ~176x range across a single locked feature, on large (tens-of-thousands-plus)
per-category samples throughout -- no small-n artefact possible here. Other
categorical features show much weaker but still real variation
(`phase_label`: 23.43%-29.65%, ~1.3x); `defender_functional_role` (a
defender-attribute feature, not an on-ball-event feature) shows almost none
(25.83%-27.76%), a sensible null since the DEFENDER's role says little about
whether the ATTACKING team's possession continues. **The zero mass here is,
if anything, more strongly feature-predictable than on the active leg**, not
less -- there is no basis to skip the two-stage architecture on predictability
grounds.

**Checked whether either of this leg's own existing architectures (`p1e`
passive-binary, `d1` passive-continuous) is a closer structural fit than
active-xT's `x1` before defaulting to copying `x1`**: `p1e_gradient_boosting`
is a single classifier (P(shot)), not two-stage -- not a fit for a signed
continuous target at all. `d1_lognormal_glm`'s hurdle architecture reuses
`p1e_gradient_boosting_calibrated` (an ALREADY-SEPARATELY-PROMOTED classifier
for a DIFFERENT target, P(shot)) as its classifier half, fitting only a new
regression head -- the same "only one surface is this rung's own to build"
shape `c1d_random_forest` had on the active-continuous leg (Prompt 74's own
Step-1 reasoning). That reuse is not available here: there is no existing
"P(nonzero delta)" classifier anywhere in this project to borrow (P(shot) and
"does this action's possession end here" are not the same event -- confirmed
directly, `on_ball_event_type`'s own zero-rate range above is not simply
mirroring the shot-rate distribution: Shot rows are almost NEVER zero, 0.30%,
while Pass rows -- the large majority of on-ball events, and mostly NOT
shots -- are also mostly nonzero, 6.73% zero). Both of this target's stages
are new, exactly the position active-xT's own `x1` was in at its own Rung 0
(Prompt 70) -- `d1`'s cross-target reuse pattern does not apply.

**Checked the regression-stage signal too**, via
reports/analysis/xt_target/passive_numerical_target_atlas.json's own
`pearson_r_nonzero_delta` field: `ball_x` (r=0.290), `defender_x` (r=0.222),
`top_option_3_threat_score` (r=0.218), `top_option_2_threat_score` (r=0.203)
-- real, and stronger than active-xT's own strongest Rung-0/1 correlate
(`distance_to_attacking_box`, r=0.157). `ball_x`'s own strong correlation
here is not a surprise, cross-referenced against Prompt 68's own construction-
coupling finding (`xt_before = xT(ball_x, ball_y)`, `ball_x` r=+0.52 vs
`xt_before`, +0.25 vs the delta) -- a caveat this leg's own EDA already
carries, not new information, but consistent with a real (if partly
construction-driven) numeric signal existing for a regression head to use.

**Decision: two-stage architecture, mirroring `x1`'s own shape, built fresh
(no existing classifier to reuse)** -- `LogisticRegression` for P(nonzero),
`HuberRegressor` for E[delta|nonzero] (robust to this leg's own heavy tails --
Prompt 68 recorded skew -1.779, excess kurtosis 31.02, heavier than active
v2's own -0.306/18.33 -- the same reasoning Prompt 70 used to pick Huber over
plain least-squares). Combined: `E[delta] = P(nonzero) * E[delta|nonzero]`,
since `E[delta|zero] = 0` exactly by construction, the same identity Prompt 70
established for the active leg.

NAMING: `v`/`c`/`p`/`d`/`x` are all claimed (active-binary, active-continuous,
passive-binary, passive-continuous, active-xT). `y` is unclaimed -- checked
directly: no script, report, or output anywhere in scripts/models/,
reports/modeling/, outputs/models/ uses a `y0`/`y1`-prefixed variant name.
Used from Rung 0 onward: `y0_dummy`, `y1_two_stage_huber`.

FEATURE SET: all 38 locked PASSIVE features (`src/eda/feature_config.py`'s
`PASSIVE` pool), reused directly from `train_passive_binary_baseline.py`'s own
`DesignMatrixBuilder`/column lists -- confirmed that script excludes nothing
for this leg (no `EXCLUDED_FOR_THIS_LEG` dict, unlike the active leg's own 2
exclusions), so unlike active-xT's 32-of-34, this is a clean 38-of-38.

SPLIT: the same canonical, frozen match-grouped split every other leg uses
(`outputs/models/splits/match_assignment.json`, shared project-wide, GROUP_COL
= match_id) -- reconfirmed directly against `train_passive_binary_baseline.py`
and `train_passive_continuous_baseline.py`'s own split code before assuming
anything different was needed for "no stable player identity": that
constraint is about PLAYER-level splits/validity checks specifically (the
Player-Level Validity Check / Player-Grouped Split Check are both
active-only, per their own `scope` fields, because passive isn't
player-indexed the same way) -- it has never meant this leg uses a different
match-grouped split. `canonical_grouped_folds`/`canonical_test_mask` are
reused unchanged, same as every other leg.

METRIC-CONVENTION FLAG (carried over from Prompt 70's own finding, reconfirmed
applicable here): `dax.models.evaluation.regression_metrics` and
`dax.models.two_part_xg.compute_hurdle_metrics` both hardcode their
zero/nonzero split as `y_true > 0` -- wrong for this leg's signed target for
the identical reason. This script computes its own `y_true == 0` split.

Usage:
    .venv/Scripts/python.exe scripts/models/train_passive_xt_baseline.py
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_passive_binary_baseline as pb  # noqa: E402  (DesignMatrixBuilder, feature lists, canonical split reuse)
from dax.models.baselines import ConstantRegressor  # noqa: E402
from dax.models.diagnostics import _ACCENT, _ACCENT_DARK, _MUTED, _new_axes, _save_fixed, ensure_dir  # noqa: E402
from dax.models.evaluation import classification_metrics, safe_spearman  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

PASSIVE_PARQUET = REPO_ROOT / "data" / "features" / "passive_defense.parquet"
XT_PARQUET = REPO_ROOT / "outputs" / "prototypes" / "passive_xt_delta.parquet"
TARGET_COL = "target_xt_delta_passive"
GROUP_COL = "match_id"
EVENT_COL = "event_id"

REG_DIR = REPO_ROOT / "outputs" / "models" / "regression"
REG_CHARTS_DIR = REG_DIR / "charts"
COMPARISONS_DIR = REPO_ROOT / "outputs" / "models" / "comparisons"
VALIDATION_DIR = REPO_ROOT / "outputs" / "models" / "validation"

VARIANTS = ["y0_dummy", "y1_two_stage_huber"]

FEATURE_COLS = pb.ALL_FEATURE_COLS  # 38 locked PASSIVE features, all of them, reused unchanged


def load_data() -> pd.DataFrame:
    """passive_defense.parquet (defender-slot grain, 1,593,181 rows) left-joined
    (READ-ONLY, in memory only) with passive_xt_delta.parquet (event grain,
    198,354 rows) by event_id -- neither locked file is modified."""
    passive = pd.read_parquet(PASSIVE_PARQUET)
    xt = pd.read_parquet(XT_PARQUET, columns=[EVENT_COL, TARGET_COL])
    assert not xt[EVENT_COL].duplicated().any(), "xt prototype must be one row per event_id"
    df = passive.merge(xt, on=EVENT_COL, how="left")
    assert len(df) == len(passive), "left-join must not change row count"
    n_nan = int(df[TARGET_COL].isna().sum())
    print(f"[data] passive_defense rows={len(passive)} (unique events={passive[EVENT_COL].nunique()}), "
          f"xt-delta NaN rows={n_nan} (dropped)")
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    n_zero = int((df[TARGET_COL] == 0.0).sum())
    print(f"[data] rows after drop: {len(df)}; exactly-zero: {n_zero} ({100 * n_zero / len(df):.2f}%)")
    return df


# ---------------------------------------------------------------------------
# Fold-safe fit/predict
# ---------------------------------------------------------------------------

def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, variant: str):
    if variant == "y0_dummy":
        model = ConstantRegressor(stat="mean")
        model.fit(train_df, train_df[TARGET_COL].to_numpy())
        pred = model.predict(test_df)
        return pred, {"model": model}

    y_nonzero_train = (train_df[TARGET_COL].to_numpy() != 0.0).astype(int)

    clf_builder = pb.DesignMatrixBuilder(add_interactions=False, add_quadratic=False).fit(train_df)
    x_train_clf, clf_feature_names = clf_builder.transform(train_df)
    x_test_clf, _ = clf_builder.transform(test_df)
    clf = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42)
    clf.fit(x_train_clf, y_nonzero_train)
    p_nonzero_test = clf.predict_proba(x_test_clf)[:, 1]

    nonzero_train_df = train_df.loc[train_df[TARGET_COL] != 0.0]
    reg_builder = pb.DesignMatrixBuilder(add_interactions=False, add_quadratic=False).fit(nonzero_train_df)
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
    return {
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
        m["n_train_events"] = int(train_df[EVENT_COL].nunique())
        m["n_test_events"] = int(test_df[EVENT_COL].nunique())
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
# target; matches the active-xT leg's own save_xt_regression_charts pattern).
# ---------------------------------------------------------------------------

def save_yxt_regression_charts(chart_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, calib_rows: list[dict], coef_table: pd.DataFrame | None):
    ensure_dir(chart_dir)

    fig, ax = _new_axes()
    ax.scatter(y_pred, y_true, alpha=0.1, s=8, color=_ACCENT, edgecolor="none")
    lo, hi = float(min(y_pred.min(), y_true.min())), float(max(y_pred.max(), y_true.max()))
    ax.plot([lo, hi], [lo, hi], color=_MUTED, linewidth=1.2, linestyle="--", label="y = x")
    ax.legend(loc="upper left")
    ax.set_xlabel("Predicted target_xt_delta_passive")
    ax.set_ylabel("Observed target_xt_delta_passive")
    ax.set_title("Predicted vs observed (signed xT delta, passive)")
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
    ax.set_ylabel("target_xt_delta_passive")
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
    assert len(FEATURE_COLS) == 38, f"Expected 38 locked PASSIVE modelling features, got {len(FEATURE_COLS)}"
    print(f"  38 locked PASSIVE modelling features confirmed (same set as p1/d1, no exclusions)")

    load_canonical_split()
    test_mask_full = canonical_test_mask(df_all, group_col=GROUP_COL)
    trainval = df_all.loc[~test_mask_full].reset_index(drop=True)
    test = df_all.loc[test_mask_full].reset_index(drop=True)
    print(f"  trainval: {len(trainval)} rows / {trainval[GROUP_COL].nunique()} matches / "
          f"{trainval[EVENT_COL].nunique()} unique events; "
          f"held-out test: {len(test)} rows / {test[GROUP_COL].nunique()} matches / "
          f"{test[EVENT_COL].nunique()} unique events")

    fold_probe = canonical_grouped_folds(trainval, group_col=GROUP_COL)
    probe_df = trainval.loc[fold_probe["row_index"]].copy()
    probe_df["fold"] = fold_probe["fold"].to_numpy()
    fold_counts = probe_df.groupby("fold").size()
    fold_event_counts = probe_df.groupby("fold")[EVENT_COL].nunique()
    fold_nonzero_counts = probe_df[probe_df[TARGET_COL] != 0.0].groupby("fold").size()
    print(f"  rows per canonical fold: {fold_counts.to_dict()}")
    print(f"  unique events per canonical fold: {fold_event_counts.to_dict()}")
    print(f"  nonzero-delta rows per canonical fold: {fold_nonzero_counts.to_dict()}")

    print("\n[2/6] cross-validation, y0_dummy and y1_two_stage_huber...")
    comparison_rows = []
    cv_results = {}
    for variant in VARIANTS:
        print(f"  [cv] {variant}")
        result = run_cv(trainval, variant)
        cv_results[variant] = result
        row = {"variant": variant, "n_features": 0 if variant == "y0_dummy" else 38}
        row.update(result["oof_metrics"])
        row["rows"] = len(trainval)
        row["matches"] = trainval[GROUP_COL].nunique()
        row["events"] = trainval[EVENT_COL].nunique()
        comparison_rows.append(row)

    print("\n[3/6] significance test, y1 vs y0 (per-fold RMSE -- no log transform on "
          "either variant here, so no common-scale fix is needed)...")
    a = cv_results["y1_two_stage_huber"]["fold_metrics"]["rmse"].to_numpy()
    b = cv_results["y0_dummy"]["fold_metrics"]["rmse"].to_numpy()
    diff = a - b
    t_stat, p_val = stats.ttest_rel(a, b)
    if len(set(diff.round(12))) > 1:
        w_stat, w_p = stats.wilcoxon(a, b)
    else:
        w_stat, w_p = float("nan"), float("nan")
    min_test_events = cv_results["y1_two_stage_huber"]["fold_metrics"]["n_test_events"].min()
    sig = {
        "metric": "rmse (per fold, signed target_xt_delta_passive scale, lower is better)",
        "fold_rmse_y1": a.tolist(), "fold_rmse_y0": b.tolist(),
        "mean_diff_y1_minus_y0": float(diff.mean()), "std_diff": float(diff.std(ddof=1)),
        "paired_t_stat": float(t_stat), "paired_t_pvalue": float(p_val),
        "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
        "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
        "n_folds": int(len(a)),
        "effective_sample_size_note": (
            f"This leg's per-fold ROW count is large, but rows are not independent observations -- "
            f"every defender-slot row sharing an event_id carries the identical target value "
            f"(~8.03 rows/event, confirmed in Prompt 68). The effective sample size for this "
            f"significance test is closer to the per-fold EVENT count (minimum across folds: "
            f"{int(min_test_events)} unique events) than the row count. Read the test's row-count-"
            f"implied power accordingly -- still a real, meaningfully-sized test at the event grain."
        ),
    }
    (VALIDATION_DIR / "significance_y0_vs_y1.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")
    print(f"  mean_diff (y1-y0) rmse: {sig['mean_diff_y1_minus_y0']:+.5f}, "
          f"paired_t_p={sig['paired_t_pvalue']:.4f}, wilcoxon_p={sig['wilcoxon_pvalue']}")

    print("\n[4/6] classifier-stage diagnostic (P(nonzero), OOF, y1 only -- not the Rung-0 gate)...")
    oof_df = cv_results["y1_two_stage_huber"]["df"]
    y_nonzero_oof = (oof_df[TARGET_COL].to_numpy() != 0.0).astype(int)
    p_nonzero_oof = np.full(len(oof_df), np.nan)
    for fold_id in sorted(oof_df["fold"].unique()):
        train_mask = (oof_df["fold"] != fold_id).to_numpy()
        test_mask = (oof_df["fold"] == fold_id).to_numpy()
        train_df, test_df = oof_df.loc[train_mask], oof_df.loc[test_mask]
        y_nonzero_train = (train_df[TARGET_COL].to_numpy() != 0.0).astype(int)
        clf_builder = pb.DesignMatrixBuilder(add_interactions=False, add_quadratic=False).fit(train_df)
        x_train_clf, _ = clf_builder.transform(train_df)
        x_test_clf, _ = clf_builder.transform(test_df)
        clf = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42).fit(x_train_clf, y_nonzero_train)
        p_nonzero_oof[test_mask] = clf.predict_proba(x_test_clf)[:, 1]
    clf_diag = classification_metrics(y_nonzero_oof, p_nonzero_oof)
    clf_diag["note"] = "Diagnostic only (zero-vs-nonzero classifier stage of y1). Rung-0 gate is the combined y1-vs-y0 regression comparison above."
    (VALIDATION_DIR / "y1_classifier_stage_diagnostic.json").write_text(json.dumps(clf_diag, indent=2), encoding="utf-8")
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
        m["events"] = test[EVENT_COL].nunique()
        held_out_rows.append(m)

        calib_rows = calibration_bins(y_true, pred)
        coef_table = None
        if variant == "y1_two_stage_huber":
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

        save_yxt_regression_charts(REG_CHARTS_DIR / variant, y_true, pred, calib_rows, coef_table)
        print(f"  [held-out] {variant}: rmse={m['rmse']:.5f} mae={m['mae']:.5f} r2={m['r2']:.5f} "
              f"spearman={m['spearman']:.4f} zero_mae={m['zero_target_mae']:.5f} nonzero_mae={m['nonzero_target_mae']:.5f}")

    print("\n[6/6] writing comparison CSVs...")
    append_to_csv(COMPARISONS_DIR / "passive_xt_baseline_comparison.csv", comparison_rows)
    append_to_csv(COMPARISONS_DIR / "passive_xt_baseline_held_out_test_readout.csv", held_out_rows)

    print("\n=== CV comparison (5-fold, OOF metrics) ===")
    print(pd.DataFrame(comparison_rows)[["variant", "rmse", "mae", "r2", "spearman", "zero_target_mae", "nonzero_target_mae"]].to_string(index=False))
    print("\n=== Held-out test readout ===")
    print(pd.DataFrame(held_out_rows)[["variant", "rmse", "mae", "r2", "spearman", "zero_target_mae", "nonzero_target_mae"]].to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
