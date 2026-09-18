"""Train the 4 locked active-binary baseline variants (v0-v3) on real data.

Target: target_future_shot_10s. Dataset: data/features/player_defensive_actions.parquet
(56,068 rows / 115 matches -- the real dataset, not a fixture).

Feature set: the 34 locked ACTIVE features from src/eda/feature_config.py, minus
nearest_defender_distance (known self-reference bug, ~73.6% of rows approx 0m per
the distribution atlas) and defenders_within_5m (confirmed substitutive/nested
with defenders_within_10m for this target -- see reports/analysis/shot_target/FEATURE_INTERACTION_ANALYSIS.json,
"defenders_within_5m x defenders_within_10m -> substitutive"). That leaves 32 features:
6 categorical (one-hot), 9 boolean (passthrough), 11 continuous + 6 discrete = 17
numeric (median-impute + standard-scale).

Split: the canonical, frozen match-grouped split in
outputs/models/splits/match_assignment.json (5-fold, seed 42, computed once by
scripts/pipeline/compute_canonical_split.py). This script never computes a fresh split --
it only loads the existing assignment via dax.models.splits.

Variants:
  v0_dummy                  -- ConstantClassifier (training-fold positive rate)
  v1_unweighted              -- plain LogisticRegression
  v2_weighted                -- LogisticRegression(class_weight="balanced")
  v3_weighted_interactions   -- v2 + 5 explicit interaction-term pairs confirmed
                                 interactive for this target in
                                 reports/analysis/shot_target/FEATURE_INTERACTION_ANALYSIS.json and
                                 reports/analysis/shot_target/MASTER_FINDINGS.md section 6.

Usage:
    python scripts/models/train_active_binary_baseline.py
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
from eda.feature_config import ACTIVE

REPO_ROOT = Path(__file__).resolve().parents[2]  # scripts/<subfolder>/ -> repo root (prompt 48 move: was parents[1] at scripts/ root)
DATA_PATH = REPO_ROOT / "data" / "features" / "player_defensive_actions.parquet"
TARGET_COL = "target_future_shot_10s"
GROUP_COL = "match_id"

MODELS_DIR = REPO_ROOT / "outputs" / "models" / "classification"
CHARTS_DIR = MODELS_DIR / "charts"
COMPARISONS_DIR = REPO_ROOT / "outputs" / "models" / "comparisons"

# -- Feature scope: 34 locked ACTIVE features minus the 2 excluded for this leg. --
EXCLUDED_FOR_THIS_LEG = {
    "nearest_defender_distance": (
        "known self-reference bug: ~73.6% of rows are approx 0m (the action's own "
        "location is being matched to itself as the 'nearest defender') per the "
        "distribution atlas -- not a genuine defensive-distance signal"
    ),
    "defenders_within_5m": (
        "confirmed substitutive/nested with defenders_within_10m for this target "
        "-- reports/analysis/shot_target/FEATURE_INTERACTION_ANALYSIS.json classifies "
        "'defenders_within_5m x defenders_within_10m' as substitutive (5m is a "
        "subset of 10m by construction)"
    ),
}

CATEGORICAL_COLS = list(ACTIVE["categorical"])  # 6
BOOLEAN_COLS = list(ACTIVE["boolean"])  # 9
CONTINUOUS_COLS = [c for c in ACTIVE["continuous"] if c not in EXCLUDED_FOR_THIS_LEG]  # 11
DISCRETE_COLS = [c for c in ACTIVE["discrete"] if c not in EXCLUDED_FOR_THIS_LEG]  # 6
NUMERIC_COLS = CONTINUOUS_COLS + DISCRETE_COLS  # 17

ALL_FEATURE_COLS = CATEGORICAL_COLS + BOOLEAN_COLS + NUMERIC_COLS
assert len(ALL_FEATURE_COLS) == 32, f"Expected 32 locked active-binary features, got {len(ALL_FEATURE_COLS)}: {ALL_FEATURE_COLS}"
for dropped in EXCLUDED_FOR_THIS_LEG:
    assert dropped not in ALL_FEATURE_COLS, f"{dropped} should have been excluded"

# -- Interaction pairs (v3 only), confirmed interactive for target_future_shot_10s. --
NUMERIC_INTERACTION_PAIRS = [
    ("defenders_within_10m", "distance_to_attacking_box"),
    ("visible_defender_count", "attacker_spread"),
    ("possession_elapsed_seconds", "match_time_seconds"),
    ("defenders_between_ball_and_attacking_goal", "attacker_defender_ratio"),
]
CATEGORICAL_INTERACTION = ("defenders_within_10m", "phase_label")

# -- Rung 1 of the model ladder (prompt 39): quadratic terms for the 4 features
# EDA confirmed have a U-shaped (non-monotonic) relationship with the target
# -- a plain logistic coefficient structurally cannot represent a U-shape.
QUADRATIC_FEATURES = [
    "defender_spread",
    "distance_to_attacking_box",
    "visible_defender_count",
    "defenders_between_ball_and_attacking_goal",
]
for _qf in QUADRATIC_FEATURES:
    assert _qf in NUMERIC_COLS, f"{_qf} must be one of the locked numeric features"


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    n_matches = df[GROUP_COL].nunique()
    print(f"[data] rows={len(df)} matches={n_matches} (real dataset, not fixture-scale)")
    assert len(df) > 1000 and n_matches > 20, "This does not look like the real dataset."
    return df


class DesignMatrixBuilder:
    """Fold-safe design-matrix builder: fit on train rows only, transform both.

    Produces a dense numpy matrix plus the matching list of output feature
    names, so coefficients can be reported per named column. Categorical ->
    OneHotEncoder, numeric (continuous+discrete) -> median-impute +
    StandardScaler, boolean -> passthrough (already 0/1, no scaling).
    Optionally appends explicit interaction terms (v3 only), built from the
    already-fitted standardized numeric block and one-hot categorical block
    so there is no additional leakage risk. Optionally appends quadratic
    terms (v1b_quadratic only) for the 4 EDA-confirmed U-shaped features --
    each is the square of that feature's already-imputed-and-scaled column,
    not the raw value, so the added columns stay on a comparable numeric
    scale to the rest of the design matrix (standard practice for
    polynomial terms on standardized inputs).
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
        # Exact "phase_label" one-hot columns only -- CATEGORICAL_COLS also
        # contains "phase_label_prev_event", whose OneHotEncoder output names
        # also start with "phase_label_" (e.g. "phase_label_prev_event_..."),
        # so a naive startswith("phase_label_") match would incorrectly pull
        # in phase_label_prev_event's dummies too. Use the encoder's own
        # per-column category counts to slice out exactly the "phase_label"
        # column's one-hot block (7 categories, per feature_config.py).
        phase_label_pos = CATEGORICAL_COLS.index("phase_label")
        offset = sum(len(cats) for cats in self.ohe.categories_[:phase_label_pos])
        n_phase_label_cats = len(self.ohe.categories_[phase_label_pos])
        self.phase_label_col_indices_ = list(range(offset, offset + n_phase_label_cats))
        assert all(
            self.cat_feature_names_[i].startswith("phase_label_")
            and not self.cat_feature_names_[i].startswith("phase_label_prev_event_")
            for i in self.phase_label_col_indices_
        ), "phase_label one-hot column slicing is wrong"
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        cat_block = self.ohe.transform(df[CATEGORICAL_COLS])
        cat_block = np.asarray(cat_block.todense()) if hasattr(cat_block, "todense") else np.asarray(cat_block)

        num_imputed = self.num_imputer.transform(df[NUMERIC_COLS])
        num_block = self.num_scaler.transform(num_imputed)
        num_names = list(NUMERIC_COLS)

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

            defenders_10m_std = num_block[:, num_names.index(CATEGORICAL_INTERACTION[0])]
            for idx in self.phase_label_col_indices_:
                inter_block_cols.append(defenders_10m_std * cat_block[:, idx])
                inter_names.append(f"interaction__defenders_within_10m__x__{self.cat_feature_names_[idx]}")

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
    if variant == "v0_dummy":
        model = ConstantClassifier()
        model.fit(train_df, y_train)
        score = model.predict_proba(test_df)[:, 1]
        return score, None, None

    add_interactions = variant == "v3_weighted_interactions"
    add_quadratic = variant == "v1b_quadratic"
    builder = DesignMatrixBuilder(add_interactions=add_interactions, add_quadratic=add_quadratic).fit(train_df)
    x_train, feature_names = builder.transform(train_df)
    x_test, _ = builder.transform(test_df)

    class_weight = "balanced" if variant in {"v2_weighted", "v3_weighted_interactions"} else None
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

    Used only for the single, clearly-labelled final readout for v2/v3 --
    never for variant selection.
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

    variants = ["v0_dummy", "v1_unweighted", "v2_weighted", "v3_weighted_interactions"]
    comparison_rows = []
    cv_results = {}

    for variant in variants:
        print(f"[cv] {variant}")
        result = run_cv(trainval_df, variant)
        cv_results[variant] = result
        fold_mean, fold_std, oof = result["fold_mean"], result["fold_std"], result["oof_metrics"]

        row = {"variant": variant, "n_features": 0 if variant == "v0_dummy" else 32}
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

    # Final, single held-out-test readout: v2 and v3 only, clearly labelled.
    final_test_rows = []
    for variant in ["v2_weighted", "v3_weighted_interactions"]:
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

    # v1 also gets a coefficients JSON (fit on all train+val rows for consistency).
    print("[coefficients] v1_unweighted (fit on all train+val rows)")
    y_trainval = trainval_df[TARGET_COL].to_numpy()
    _, model_v1, extra_v1 = fit_predict_fold(trainval_df, trainval_df, y_trainval, y_trainval, "v1_unweighted")
    builder_v1, names_v1 = extra_v1
    coeff_v1 = coefficient_json(model_v1, names_v1, "v1_unweighted")
    coeff_v1["fitted_on"] = "all train+val rows (canonical split); no held-out test readout for v1 per instructions"
    (MODELS_DIR / "v1_unweighted.json").write_text(json.dumps(coeff_v1, indent=2), encoding="utf-8")
    joblib.dump(model_v1, MODELS_DIR / "v1_unweighted.joblib")

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(COMPARISONS_DIR / "active_binary_baseline_comparison.csv", index=False)

    final_test_df = pd.DataFrame(final_test_rows)
    final_test_df.to_csv(COMPARISONS_DIR / "active_binary_baseline_held_out_test_readout.csv", index=False)

    for variant, result in cv_results.items():
        result["fold_metrics"].to_csv(MODELS_DIR / f"{variant}_fold_metrics.csv", index=False)

    print("\n=== CV comparison (5-fold, OOF metrics) ===")
    print(comparison_df[["variant", "average_precision", "roc_auc", "log_loss", "brier_score"]].to_string(index=False))
    print("\n=== Final held-out test readout (v2/v3 only, NOT for selection) ===")
    print(final_test_df[["variant", "average_precision", "roc_auc", "log_loss", "brier_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
