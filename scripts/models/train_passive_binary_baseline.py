"""Train the 4 locked passive-binary baseline variants (p0-p3) on real data.

Target: target_future_shot_10s, computed per-row from that row's own anchor
timestamp (never broadcast across a possession -- this was a caught
leakage bug specific to the passive leg during design, already fixed and
regression-tested; nothing in this script re-derives the target, it is read
as-is from the locked parquet). Dataset: data/features/passive_defense.parquet
(1,593,181 rows / 115 matches -- the real dataset, not a fixture).

Row unit: one row per visible defending-team player-slot per on-ball
attacking event (src/eda/feature_config.py PASSIVE["row_description"]), NOT
one row per independent event. Rows sharing the same event_id describe the
same attacking context through a different defender-slot's geometry, so they
are not independent observations -- the canonical match-grouped split still
prevents cross-split leakage (it groups by match_id), but this within-fold
row correlation is a real property of this leg's data, stated explicitly
in the baseline report rather than silently ignored.

Base rate: 5.97% (vs 7.79% on the active leg) -- NOT directly comparable to
the active leg's base rate, since the two datasets have different row grains
(active: one row per actual defensive action; passive: one row per
defender-slot x on-ball-event pair).

Feature set: all 38 locked PASSIVE features from src/eda/feature_config.py
(4 categorical + 6 boolean + 26 continuous + 2 discrete = 38). No feature is
excluded for this leg (unlike the active leg's 34 -> 32). screened_option_
was_avoided and has_screened_outcome are both leakage exclusions already
enforced upstream in feature_config.py's EXCLUDED_COLUMNS_PASSIVE / never
in PASSIVE's candidate lists -- reasserted defensively below.

No player identity: passive_defense.parquet has no player_id column (Phase 4
of the passive build dropped frame-to-frame player tracking as unreliable).
Any player-disjoint-style leakage recheck for this leg needs a different
mechanism (archetype/event-disjoint, not player-disjoint) -- out of scope
here, deferred to the post-lock validation follow-up prompt.

Split: the canonical, frozen match-grouped split in
outputs/models/splits/match_assignment.json (5-fold, seed 42, computed once
by scripts/pipeline/compute_canonical_split.py, shared with the active leg).
This script never computes a fresh split -- it only loads the existing
assignment via dax.models.splits.

Variants:
  p0_dummy                  -- ConstantClassifier (training-fold positive rate)
  p1_unweighted              -- plain LogisticRegression
  p2_weighted                -- LogisticRegression(class_weight="balanced")
  p3_weighted_interactions   -- p2 + 5 explicit interaction-term pairs, 4 of
                                 which are independently confirmed "interactive"
                                 for a passive-leg target in
                                 reports/analysis/shot_target/FEATURE_LOCK_CONFIRMATION.json's
                                 pattern_analysis_findings.numeric_interaction
                                 section, plus one same-family extension (see
                                 NUMERIC_INTERACTION_PAIRS below).

Unlike the active leg (where Prompts 36/37 split CV from the held-out-test
readout across two runs, and the readout only covered v2/v3), this script
produces one held-out-test readout per variant (including p0/p1) in a single
run, per this leg's explicit instructions.

Usage:
    python scripts/models/train_passive_binary_baseline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dax.models.baselines import ConstantClassifier
from dax.models.diagnostics import save_classification_charts
from dax.models.evaluation import classification_metrics
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split
from eda.feature_config import PASSIVE

REPO_ROOT = Path(__file__).resolve().parents[2]  # scripts/models/ -> repo root
DATA_PATH = REPO_ROOT / "data" / "features" / "passive_defense.parquet"
TARGET_COL = "target_future_shot_10s"
GROUP_COL = "match_id"

MODELS_DIR = REPO_ROOT / "outputs" / "models" / "classification"
CHARTS_DIR = MODELS_DIR / "charts"
COMPARISONS_DIR = REPO_ROOT / "outputs" / "models" / "comparisons"

# -- Feature scope: all 38 locked PASSIVE features, none excluded for this leg. --
CATEGORICAL_COLS = list(PASSIVE["categorical"])  # 4
BOOLEAN_COLS = list(PASSIVE["boolean"])  # 6
CONTINUOUS_COLS = list(PASSIVE["continuous"])  # 26
DISCRETE_COLS = list(PASSIVE["discrete"])  # 2
NUMERIC_COLS = CONTINUOUS_COLS + DISCRETE_COLS  # 28

ALL_FEATURE_COLS = CATEGORICAL_COLS + BOOLEAN_COLS + NUMERIC_COLS
assert len(ALL_FEATURE_COLS) == 38, f"Expected 38 locked passive-binary features, got {len(ALL_FEATURE_COLS)}: {ALL_FEATURE_COLS}"

# Leakage exclusions confirmed in reports/analysis/shot_target/FEATURE_LOCK_CONFIRMATION.json /
# src/eda/feature_config.py EXCLUDED_COLUMNS_PASSIVE -- reasserted here defensively so this
# script fails loudly if either is ever re-added to PASSIVE's candidate lists.
_LEAKAGE_EXCLUDED = {"screened_option_was_avoided", "has_screened_outcome"}
for _col in _LEAKAGE_EXCLUDED:
    assert _col not in ALL_FEATURE_COLS, f"{_col} is a confirmed leakage exclusion and must not appear in the locked feature list"

# -- Interaction pairs (p3 only). --
# The first 4 pairs are each independently classified "interactive" for a passive-leg
# target in FEATURE_LOCK_CONFIRMATION.json's pattern_analysis_findings.numeric_interaction
# section (10 tested pairs total; the 5th tested pair, top_option_2_threat_score x
# top_option_2_distance_from_ball, was classified "additive" and is deliberately excluded).
# The 5th pair below is not itself in that JSON -- it extends the confirmed
# "lane_screening_score_option_2 x engagement_distance_to_carrier -> interactive" pair to
# option 3 of the same lane-screening family, which is present and unchanged in the locked
# 38 (checked directly against PASSIVE["continuous"] above, not assumed from prose).
#
# Note: the task brief's own suggested "zone_defensive_value x engagement_distance_to_carrier"
# pair is NOT used -- zone_defensive_value was dropped from the locked passive feature set
# (REDUNDANCY_DROPPED_PASSIVE in feature_config.py: "CORRELATION_ANALYSIS DROP tier: Spearman
# r=-1.0 vs distance_to_defending_goal"), confirming the brief's own warning that some
# build-plan-era feature names are no longer exact locked columns.
NUMERIC_INTERACTION_PAIRS = [
    ("lane_screening_score_option_1", "marking_tightness"),
    ("lane_screening_score_option_2", "engagement_distance_to_carrier"),
    ("marking_tightness", "engagement_distance_to_carrier"),
    ("overload_score", "attacking_goal_centrality"),
    ("lane_screening_score_option_3", "engagement_distance_to_carrier"),
]
for _a, _b in NUMERIC_INTERACTION_PAIRS:
    assert _a in NUMERIC_COLS, f"{_a} must be one of the locked numeric features"
    assert _b in NUMERIC_COLS, f"{_b} must be one of the locked numeric features"


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    n_matches = df[GROUP_COL].nunique()
    print(f"[data] rows={len(df)} matches={n_matches} base_rate={df[TARGET_COL].mean():.4f}")
    assert len(df) > 100_000 and n_matches > 20, "This does not look like the real dataset."
    return df


# Rung 1 (prompt 51) quadratic features: the locked, continuous features whose
# relationship to target_future_shot_10s is classified strictly "U-shaped" (not the
# weaker "inverse-U" bucket) in reports/analysis/shot_target/passive_numerical_target_atlas.json,
# verified directly against that atlas rather than assumed to mirror the active leg's 4
# U-shaped features (which are different columns on a different dataset). 6 features
# clear this bar, not 4 -- not forced to match the active leg's count:
#   top_option_1/2/3_threat_score (U-shaped, |rho| 0.067-0.085 -- also this leg's 3
#     strongest-correlated features overall per FEATURE_LOCK_CONFIRMATION.json's
#     pattern_analysis_findings.numerical_vs_target_rankings.passive_top)
#   ball_x, defender_x (U-shaped, |rho| 0.036-0.042)
#   angle_to_attacking_goal (U-shaped, |rho| 0.030)
# The weaker "inverse-U" bucket (top_option_3_dx/dy, top_option_2_dy, defender_y,
# |rho| 0.002-0.031) is deliberately excluded -- inverse-U is a distinct, much weaker
# shape classification in that atlas, not the same finding restated.
QUADRATIC_FEATURES = [
    "top_option_1_threat_score",
    "top_option_2_threat_score",
    "top_option_3_threat_score",
    "ball_x",
    "defender_x",
    "angle_to_attacking_goal",
]
for _qf in QUADRATIC_FEATURES:
    assert _qf in CONTINUOUS_COLS, f"{_qf} must be one of the locked continuous passive features"


class DesignMatrixBuilder:
    """Fold-safe design-matrix builder: fit on train rows only, transform both.

    Same preprocessing pattern as the active leg's DesignMatrixBuilder
    (categorical -> OneHotEncoder, numeric -> median-impute + StandardScaler,
    boolean -> passthrough), adapted to the passive feature list. Interaction
    terms (p3 only) are 5 explicit numeric x numeric products built from the
    already-fitted standardized numeric block, so there is no additional
    leakage risk. Quadratic terms (p1b_quadratic only, prompt 51) are the
    square of each QUADRATIC_FEATURES column's already-standardized value,
    named quad__<feature>, same convention as the active leg's Rung 1.
    """

    def __init__(self, add_interactions: bool = False, add_quadratic: bool = False):
        self.add_interactions = add_interactions
        self.add_quadratic = add_quadratic
        self.ohe = OneHotEncoder(handle_unknown="ignore")
        self.num_imputer = SimpleImputer(strategy="median")
        self.num_scaler = StandardScaler()

    def fit(self, train_df: pd.DataFrame) -> "DesignMatrixBuilder":
        self.ohe.fit(train_df[CATEGORICAL_COLS])
        num_imputed = self.num_imputer.fit_transform(train_df[NUMERIC_COLS])
        self.num_scaler.fit(num_imputed)
        self.cat_feature_names_ = list(self.ohe.get_feature_names_out(CATEGORICAL_COLS))
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        cat_block = self.ohe.transform(df[CATEGORICAL_COLS])
        cat_block = np.asarray(cat_block.todense()) if hasattr(cat_block, "todense") else np.asarray(cat_block)

        num_imputed = self.num_imputer.transform(df[NUMERIC_COLS])
        num_block = self.num_scaler.transform(num_imputed)
        num_names = list(NUMERIC_COLS)

        # is_goal_side_of_nearest_attacker is stored as object (True/False/None) --
        # 279/1,593,181 rows (0.018%) are None (no nearest attacker to compare
        # against). astype(float) turns None into NaN; fillna(0.0) treats "unknown"
        # as the majority/negative class, consistent with the other boolean columns'
        # already-binary encoding and the active leg's own boolean-passthrough pattern.
        bool_block = df[BOOLEAN_COLS].astype(float).fillna(0.0).to_numpy()
        bool_names = list(BOOLEAN_COLS)

        blocks = [cat_block, num_block, bool_block]
        names = [*self.cat_feature_names_, *num_names, *bool_names]

        if self.add_interactions:
            inter_block_cols = []
            inter_names = []
            for feat_a, feat_b in NUMERIC_INTERACTION_PAIRS:
                a = num_block[:, num_names.index(feat_a)]
                b = num_block[:, num_names.index(feat_b)]
                inter_block_cols.append(a * b)
                inter_names.append(f"interaction__{feat_a}__x__{feat_b}")

            inter_block = np.column_stack(inter_block_cols)
            blocks.append(inter_block)
            names.extend(inter_names)

        if self.add_quadratic:
            quad_block_cols = []
            quad_names = []
            for feat in QUADRATIC_FEATURES:
                std_col = num_block[:, num_names.index(feat)]
                quad_block_cols.append(std_col ** 2)
                quad_names.append(f"quad__{feat}")

            quad_block = np.column_stack(quad_block_cols)
            blocks.append(quad_block)
            names.extend(quad_names)

        x = np.column_stack(blocks)
        return x, names


def fit_predict_fold(train_df, test_df, y_train, y_test, variant: str):
    if variant == "p0_dummy":
        model = ConstantClassifier()
        model.fit(train_df, y_train)
        score = model.predict_proba(test_df)[:, 1]
        return score, None, None

    add_interactions = variant == "p3_weighted_interactions"
    add_quadratic = variant == "p1b_quadratic"
    builder = DesignMatrixBuilder(add_interactions=add_interactions, add_quadratic=add_quadratic).fit(train_df)
    x_train, feature_names = builder.transform(train_df)
    x_test, _ = builder.transform(test_df)

    class_weight = "balanced" if variant in {"p2_weighted", "p3_weighted_interactions"} else None
    model = LogisticRegression(solver="lbfgs", max_iter=2000, class_weight=class_weight, random_state=42)
    model.fit(x_train, y_train)
    score = model.predict_proba(x_test)[:, 1]
    return score, model, (builder, feature_names)


def run_cv(df: pd.DataFrame, variant: str) -> dict:
    folds = canonical_grouped_folds(df, group_col=GROUP_COL)
    df = df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()

    fold_rows = []
    oof_scores = np.full(len(df), np.nan)
    y_all = df[TARGET_COL].to_numpy()

    for fold_id in sorted(df["fold"].unique()):
        train_mask = df["fold"] != fold_id
        test_mask = df["fold"] == fold_id
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]
        y_train, y_test = y_all[train_mask.to_numpy()], y_all[test_mask.to_numpy()]

        score, _, _ = fit_predict_fold(train_df, test_df, y_train, y_test, variant)
        oof_scores[test_mask.to_numpy()] = score

        metrics = classification_metrics(y_test, score)
        metrics["fold"] = int(fold_id)
        metrics["n_train"] = int(train_mask.sum())
        metrics["n_test"] = int(test_mask.sum())
        fold_rows.append(metrics)
        print(f"    fold {fold_id}: AP={metrics['average_precision']:.4f} roc_auc={metrics['roc_auc']:.4f}")

    fold_df = pd.DataFrame(fold_rows)
    metric_cols = [c for c in fold_df.columns if c not in {"fold", "n_train", "n_test"}]
    fold_mean = fold_df[metric_cols].mean()
    fold_std = fold_df[metric_cols].std(ddof=0)

    overall = classification_metrics(y_all, oof_scores)

    return {
        "variant": variant,
        "fold_metrics": fold_df,
        "fold_mean": fold_mean,
        "fold_std": fold_std,
        "oof_metrics": overall,
        "oof_scores": oof_scores,
        "y_all": y_all,
        "df": df,
    }


def fit_final_and_test(df: pd.DataFrame, test_df: pd.DataFrame, variant: str) -> dict:
    """Fit once on all TRAIN+VAL rows, evaluate once on TEST rows.

    Unlike the active leg (v2/v3 only), every passive-binary variant
    (including p0/p1) gets exactly one such readout in this run, per this
    leg's explicit instructions.
    """
    y_train = df[TARGET_COL].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()
    score, model, extra = fit_predict_fold(df, test_df, y_train, y_test, variant)
    metrics = classification_metrics(y_test, score)
    return {"metrics": metrics, "scores": score, "model": model, "extra": extra}


def coefficient_json(model: LogisticRegression, feature_names: list[str], variant: str) -> dict:
    coefs = model.coef_[0]
    rows = sorted(
        (
            {"feature": name, "coef": float(coef), "abs_coef": float(abs(coef))}
            for name, coef in zip(feature_names, coefs)
        ),
        key=lambda r: r["abs_coef"],
        reverse=True,
    )
    return {
        "variant": variant,
        "target": TARGET_COL,
        "intercept": float(model.intercept_[0]),
        "class_weight": model.get_params().get("class_weight"),
        "n_features": len(feature_names),
        "coefficients": rows,
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)

    df_all = load_data()
    load_canonical_split()  # fails loudly if the canonical split file is missing

    test_mask = canonical_test_mask(df_all, group_col=GROUP_COL)
    trainval_df = df_all.loc[~test_mask].reset_index(drop=True)
    test_df = df_all.loc[test_mask].reset_index(drop=True)
    print(f"[split] train+val rows={len(trainval_df)} matches={trainval_df[GROUP_COL].nunique()}; "
          f"held-out test rows={len(test_df)} matches={test_df[GROUP_COL].nunique()}")

    variants = ["p0_dummy", "p1_unweighted", "p2_weighted", "p3_weighted_interactions"]
    comparison_rows = []
    cv_results = {}

    for variant in variants:
        print(f"[cv] {variant}")
        result = run_cv(trainval_df, variant)
        cv_results[variant] = result
        fold_mean, fold_std, oof = result["fold_mean"], result["fold_std"], result["oof_metrics"]

        row = {"variant": variant, "n_features": 0 if variant == "p0_dummy" else 38}
        for metric in ["log_loss", "brier_score", "average_precision", "roc_auc",
                        "calibration_slope", "calibration_intercept", "expected_calibration_error", "positive_rate"]:
            row[metric] = oof[metric]
            row[f"fold_{metric}_mean"] = fold_mean[metric]
            row[f"fold_{metric}_std"] = fold_std[metric]
        row["rows"] = len(trainval_df)
        row["matches"] = trainval_df[GROUP_COL].nunique()
        comparison_rows.append(row)

        chart_df = pd.DataFrame({"y_true": result["y_all"], "y_score": result["oof_scores"]})
        save_classification_charts(chart_df, CHARTS_DIR / variant)

    # Held-out-test readout: every variant gets exactly one, in this same run.
    final_test_rows = []
    for variant in variants:
        print(f"[final-test-readout] {variant}")
        out = fit_final_and_test(trainval_df, test_df, variant)
        metrics = dict(out["metrics"])
        metrics["variant"] = variant
        metrics["readout"] = "FINAL_HELD_OUT_TEST_NOT_FOR_SELECTION"
        metrics["rows"] = len(test_df)
        metrics["matches"] = test_df[GROUP_COL].nunique()
        final_test_rows.append(metrics)

        if out["model"] is not None:
            builder, feature_names = out["extra"]
            coeff = coefficient_json(out["model"], feature_names, variant)
            coeff["fitted_on"] = "all train+val rows (canonical split); held-out test used once for readout only"
            (MODELS_DIR / f"{variant}.json").write_text(json.dumps(coeff, indent=2), encoding="utf-8")
            joblib.dump(out["model"], MODELS_DIR / f"{variant}.joblib")

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(COMPARISONS_DIR / "passive_binary_baseline_comparison.csv", index=False)

    final_test_df = pd.DataFrame(final_test_rows)
    final_test_df.to_csv(COMPARISONS_DIR / "passive_binary_baseline_held_out_test_readout.csv", index=False)

    for variant, result in cv_results.items():
        result["fold_metrics"].to_csv(MODELS_DIR / f"{variant}_fold_metrics.csv", index=False)

    print("\n=== CV comparison (5-fold, OOF metrics) ===")
    print(comparison_df[["variant", "average_precision", "roc_auc", "log_loss", "brier_score"]].to_string(index=False))
    print("\n=== Final held-out test readout (all variants, NOT for selection) ===")
    print(final_test_df[["variant", "average_precision", "roc_auc", "log_loss", "brier_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
