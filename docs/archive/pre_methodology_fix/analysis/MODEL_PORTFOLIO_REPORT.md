# Model Portfolio Report

Generated: 2026-06-15 09:16 UTC

## Executive Summary

- **Dataset / validation context:** 57,637 defensive actions across 115 matches; grouped CV by `match_id`; logistic target base rate 7.95%; regression target mean 0.4199.
- **Best OOF classification model:** `v3_context_enhanced` with ROC-AUC 0.8088 and AP 0.3289.
- **Best OOF regression model:** `v3_context_enhanced` with R2 0.3697, RMSE 0.3116, MAE 0.2262, and Spearman 0.6822.
- **Best direct-score / slice-analysis model:** classification `v4_freeze_geometry` and regression `v4_freeze_geometry`. These are useful for relative segment behavior but are more optimistic than OOF CV because they score saved models on the full dataset.
- **Feature-selection takeaway:** clustered compact variants (`v6` / `v8`) preserve much more signal than the strict interpretable variants (`v5` / `v7`), but they still trail `v3` / `v4` materially.
- **Regularization takeaway:** on filtered feature sets, Ridge had almost no portfolio-level lift (`v5`→`v7` logistic ΔAUC -0.00000; `v6`→`v8` logistic ΔAUC +0.00000; `v5`→`v7` regression ΔR2 +0.00000; `v6`→`v8` regression ΔR2 +0.00000).

## 1. Validation Protocol

- **Classification family:** logistic regression with L2 penalty predicting `target_shot_in_10s`.
- **Regression family:** linear or Ridge regression predicting `target_xt_10s`.
- **Split strategy:** `GroupKFold` by `match_id` to prevent leakage across events from the same match.
- **Primary metrics:** ROC-AUC / Average Precision for classification; R2 / RMSE / MAE / Spearman for regression.
- **Secondary analyses:** slice leaderboards, cross-task alignment, correlation clustering, tactical-vs-proxy ablations, and feature-selection manifests for clustered variants.

## 2. OOF Leaderboard — Classification

| Variant | Feature set | Features | ROC-AUC | AP | ΔAUC vs best | ΔAP vs best |
| --- | --- | --- | --- | --- | --- | --- |
| v0_phase_only | phase-only | 1 | 0.6793 | 0.1289 | -0.1294 | -0.2000 |
| v1_spatial | spatial | 8 | 0.7768 | 0.2903 | -0.0320 | -0.0386 |
| v2_full_baseline | full baseline | 26 | 0.8058 | 0.3181 | -0.0029 | -0.0109 |
| v3_context_enhanced | context enhanced | 44 | 0.8088 | 0.3289 | 0.0000 | 0.0000 |
| v4_freeze_geometry | freeze geometry | 53 | 0.8076 | 0.3281 | -0.0011 | -0.0009 |
| v5_interpretable_clustered | clustered interpretable | 28 | 0.7856 | 0.3143 | -0.0232 | -0.0147 |
| v6_balanced_clustered | clustered balanced | 32 | 0.8006 | 0.3197 | -0.0082 | -0.0092 |
| v7_interpretable_ridge | clustered interpretable ridge | 28 | 0.7856 | 0.3144 | -0.0232 | -0.0145 |
| v8_balanced_ridge | clustered balanced ridge | 32 | 0.8006 | 0.3199 | -0.0082 | -0.0090 |

**Readout:** `v3_context_enhanced` is the strongest OOF classifier. `v4_freeze_geometry` is effectively tied, while `v6`/`v8` are the best reduced-feature alternatives. `v5`/`v7` buy interpretability but give back noticeable discrimination.

## 3. OOF Leaderboard — Regression

| Variant | Feature set | Model | Features | R2 | RMSE | MAE | Spearman | ΔR2 vs best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v0_phase_only | phase-only | ridge | 1 | 0.0348 | 0.3856 | 0.2943 | 0.1454 | -0.3350 |
| v1_spatial | spatial | ridge | 8 | 0.1321 | 0.3656 | 0.2750 | 0.3960 | -0.2376 |
| v2_full_baseline | full baseline | ridge | 26 | 0.3193 | 0.3238 | 0.2360 | 0.6376 | -0.0504 |
| v3_context_enhanced | context enhanced | ridge | 44 | 0.3697 | 0.3116 | 0.2262 | 0.6822 | 0.0000 |
| v4_freeze_geometry | freeze geometry | ridge | 53 | 0.3693 | 0.3117 | 0.2263 | 0.6818 | -0.0004 |
| v5_interpretable_clustered | clustered interpretable | linear | 28 | 0.2440 | 0.3412 | 0.2491 | 0.5726 | -0.1257 |
| v6_balanced_clustered | clustered balanced | linear | 32 | 0.2731 | 0.3346 | 0.2427 | 0.6042 | -0.0966 |
| v7_interpretable_ridge | clustered interpretable ridge | ridge | 28 | 0.2440 | 0.3412 | 0.2491 | 0.5726 | -0.1257 |
| v8_balanced_ridge | clustered balanced ridge | ridge | 32 | 0.2731 | 0.3346 | 0.2427 | 0.6042 | -0.0966 |

**Readout:** `v3_context_enhanced` is the best OOF regressor and `v4_freeze_geometry` is nearly identical. This suggests richer context is valuable for continuous xT, while detailed freeze-frame geometry adds little on top.

## 4. Direct-Score Leaderboard and Slice Caveat

The slice-analysis pipeline scores saved models on the full dataset, so the absolute values below are more optimistic than OOF CV. Use them to compare relative behavior across segments rather than to select the final model alone.

### Classification direct-score leaderboard

| Variant | ROC-AUC | AP |
| --- | --- | --- |
| v4_freeze_geometry | 0.8155 | 0.3379 |
| v3_context_enhanced | 0.8150 | 0.3370 |
| v2_full_baseline | 0.8107 | 0.3241 |
| v6_balanced_clustered | 0.8075 | 0.3282 |
| v8_balanced_ridge | 0.8075 | 0.3284 |
| v5_interpretable_clustered | 0.7920 | 0.3212 |
| v7_interpretable_ridge | 0.7919 | 0.3213 |
| v1_spatial | 0.7790 | 0.2920 |
| v0_phase_only | 0.6815 | 0.1284 |

### Regression direct-score leaderboard

| Variant | R2 | RMSE | MAE | Spearman |
| --- | --- | --- | --- | --- |
| v4_freeze_geometry | 0.3737 | 0.3106 | 0.2256 | 0.6841 |
| v3_context_enhanced | 0.3734 | 0.3107 | 0.2257 | 0.6840 |
| v2_full_baseline | 0.3220 | 0.3231 | 0.2355 | 0.6394 |
| v6_balanced_clustered | 0.2762 | 0.3339 | 0.2422 | 0.6066 |
| v8_balanced_ridge | 0.2762 | 0.3339 | 0.2422 | 0.6066 |
| v5_interpretable_clustered | 0.2467 | 0.3406 | 0.2487 | 0.5750 |
| v7_interpretable_ridge | 0.2467 | 0.3406 | 0.2487 | 0.5750 |
| v1_spatial | 0.1334 | 0.3653 | 0.2748 | 0.3977 |
| v0_phase_only | 0.0353 | 0.3855 | 0.2942 | 0.1501 |

**Readout:** on direct scoring, `v4_freeze_geometry` edges the field in both families. This is directionally useful for segment analysis, but OOF CV still favors `v3` as the safest all-round production candidate.

## 5. Fold Stability

### Classification fold spread

| Variant | Mean AUC | AUC SD | Mean AP | AP SD |
| --- | --- | --- | --- | --- |
| v2_full_baseline | 0.8059 | 0.0070 | 0.3197 | 0.0322 |
| v3_context_enhanced | 0.8088 | 0.0052 | 0.3305 | 0.0296 |
| v4_freeze_geometry | 0.8077 | 0.0048 | 0.3296 | 0.0280 |
| v6_balanced_clustered | 0.8006 | 0.0076 | 0.3209 | 0.0224 |
| v8_balanced_ridge | 0.8006 | 0.0076 | 0.3212 | 0.0224 |

### Regression fold spread

| Variant | Mean R2 | R2 SD | Mean RMSE | RMSE SD |
| --- | --- | --- | --- | --- |
| v2_full_baseline | 0.3192 | 0.0043 | 0.3238 | 0.0050 |
| v3_context_enhanced | 0.3697 | 0.0119 | 0.3115 | 0.0064 |
| v4_freeze_geometry | 0.3693 | 0.0118 | 0.3116 | 0.0064 |
| v6_balanced_clustered | 0.2729 | 0.0067 | 0.3346 | 0.0036 |
| v8_balanced_ridge | 0.2729 | 0.0067 | 0.3346 | 0.0036 |

**Readout:** `v3`/`v4` have the strongest metrics with low fold spread, which is what we want. The clustered variants are stable too, but they plateau lower.

## 6. Cross-Task Alignment (P(shot) vs E[xT))

| Variant | Pearson | Spearman | Agreement | Rows scored |
| --- | --- | --- | --- | --- |
| v0_phase_only | 0.7896 | 0.8592 | 0.6695 | 57637 |
| v1_spatial | 0.7498 | 0.7708 | 0.7555 | 57637 |
| v2_full_baseline | 0.3498 | 0.3439 | 0.6066 | 57593 |
| v3_context_enhanced | 0.3653 | 0.3931 | 0.6333 | 52518 |
| v4_freeze_geometry | 0.3654 | 0.3934 | 0.6336 | 52518 |
| v5_interpretable_clustered | 0.4553 | 0.4399 | 0.6280 | 52518 |
| v6_balanced_clustered | 0.3615 | 0.3455 | 0.6052 | 52518 |
| v7_interpretable_ridge | 0.4551 | 0.4398 | 0.6279 | 52518 |
| v8_balanced_ridge | 0.3612 | 0.3453 | 0.6051 | 52518 |

**Readout:** `v0` and `v1` show very high alignment because both tasks are mostly expressing the same coarse phase/spatial gradient at that simplicity level. From `v2` onward, the correlation drops into the moderate range because the classifier and regressor begin separating shot-trigger risk from continuous threat accumulation. That separation is desirable: the two targets are related, but not redundant.

## 7. Slice Analysis

- **Slice-win leader (classification):** `v4_freeze_geometry` with 30 best-segment wins.
- **Slice-win leader (regression):** `v4_freeze_geometry` with 28 best-segment wins.

### Slice-win counts

| Classification variant | Wins |
| --- | --- |
| v4_freeze_geometry | 30 |
| v3_context_enhanced | 5 |
| v2_full_baseline | 4 |
| v1_spatial | 1 |

| Regression variant | Wins |
| --- | --- |
| v4_freeze_geometry | 28 |
| v3_context_enhanced | 9 |
| v2_full_baseline | 3 |

### Dominant winner by slice dimension

| Dimension | Classification winner | Wins |
| --- | --- | --- |
| action_family | v4_freeze_geometry | 6 |
| action_zone | v4_freeze_geometry | 6 |
| counterpress | v3_context_enhanced | 1 |
| phase_label | v4_freeze_geometry | 4 |
| play_pattern | v4_freeze_geometry | 6 |
| position_group | v4_freeze_geometry | 7 |

| Dimension | Regression winner | Wins |
| --- | --- | --- |
| action_family | v4_freeze_geometry | 4 |
| action_zone | v4_freeze_geometry | 5 |
| counterpress | v3_context_enhanced | 1 |
| phase_label | v4_freeze_geometry | 5 |
| play_pattern | v4_freeze_geometry | 6 |
| position_group | v4_freeze_geometry | 7 |

**Readout:** `v4_freeze_geometry` is the strongest segment specialist. It dominates most slice families, especially action-family, play-pattern, and position-group cuts. `v3` remains competitive in counterpress-heavy and some phase-specific segments.

## 8. Correlation Clusters and Coefficient Stability

### Logistic stability study (`v4_freeze_geometry`) — top correlation clusters

| Cluster | Size | Max abs corr | Features |
| --- | --- | --- | --- |
| 2 | 6 | 1.0000 | action_x, ball_x, distance_to_left_goal, distance_to_right_goal, freeze_opponent_centroid_x, freeze_teammate_centroid_x |
| 3 | 6 | 1.0000 | action_y, ball_y, freeze_opponent_centroid_dy, freeze_opponent_centroid_y, freeze_teammate_centroid_dy, freeze_teammate_centroid_y |
| 5 | 5 | 1.0000 | freeze_frame_count, freeze_opponent_count, freeze_teammate_count, opponent_count, teammate_count |
| 1 | 3 | 0.9597 | event_order_in_possession, phase_transition_count_so_far, seconds_since_possession_start |
| 7 | 2 | 0.9623 | freeze_opponent_nearest_distance, freeze_teammate_nearest_distance |
| 9 | 2 | 0.9595 | possession_duration_total, possession_event_count_total |

### Regression stability study (`v3_context_enhanced`) — top correlation clusters

| Cluster | Size | Max abs corr | Features |
| --- | --- | --- | --- |
| 5 | 5 | 1.0000 | freeze_frame_count, freeze_opponent_count, freeze_teammate_count, opponent_count, teammate_count |
| 2 | 4 | 1.0000 | action_x, ball_x, distance_to_left_goal, distance_to_right_goal |
| 1 | 3 | 0.9597 | event_order_in_possession, phase_transition_count_so_far, seconds_since_possession_start |
| 3 | 2 | 1.0000 | action_y, ball_y |
| 7 | 2 | 0.9623 | freeze_opponent_nearest_distance, freeze_teammate_nearest_distance |
| 8 | 2 | 0.9595 | possession_duration_total, possession_event_count_total |

- Logistic tactical-only ablation drops AUC by -0.0221; proxy-only drops AUC by -0.1114.
- Regression tactical-only ablation drops R2 by -0.1354; proxy-only drops R2 by -0.1702.
- Interpretation: possession-time and progression proxies matter, but they are not the core source of signal. They can inflate coefficients if left unconstrained, especially in linear models with many correlated inputs.

## 9. Feature-Selection Outcomes (Clustering / PCA)

- Logistic `v5` manifest keeps **28** features; `v6` keeps **32**.
- Regression `v5` manifest keeps **28** features; `v6` keeps **32**.
- Stability-selected logistic interpretable set marks **31** raw features as keepers.
- Stability-selected regression interpretable set marks **29** raw features as keepers.

**Readout:**

1. The main redundancy bundles are possession-time proxies, x-axis geometry aliases, y-axis / centroid aliases, freeze-frame counts, and nearest-distance pairs.
2. `v5` is the strict interpretability candidate; it removes more proxies and cluster duplicates, but that costs meaningful predictive power.
3. `v6` is the better compromise for a compact linear model because it retains a few controlled context variables (for example `play_pattern`, `position_group`, and one possession-lifecycle representative).
4. Moving from `v6` to `v8` shows that once the correlated blocks are already pruned, Ridge adds little. The remaining variance is more about missing non-linearity than about coefficient explosion.

## 10. Model-by-Model Interpretation

- **`v0_phase_only`** — sanity baseline. Good for proving the phase taxonomy carries real signal, but far too weak for deployment.
- **`v1_spatial`** — adds the core football geometry: where the action happened and what type of action it was. Big jump in both families confirms location/context is the first major driver.
- **`v2_full_baseline`** — the first production-worthy baseline: freeze-frame support, counts, and possession progression produce a large lift over `v1`.
- **`v3_context_enhanced`** — the best OOF all-rounder. Adds temporal and phase-transition context without overcomplicating the geometry. Strongest recommendation when the target is robust match-held-out generalization.
- **`v4_freeze_geometry`** — the best slice specialist and best direct scorer. Adds fine-grained freeze-frame centroids and player-position detail. Useful when segment performance matters most or when richer geometry is available in production.
- **`v5_interpretable_clustered`** — strict feature-governed linear model. Best choice if stakeholder trust and coefficient clarity matter more than raw performance.
- **`v6_balanced_clustered`** — compact but still competitive. Best reduced-feature candidate if you want a model that is easier to explain and cheaper to maintain without collapsing performance.
- **`v7_interpretable_ridge`** — Ridge version of `v5`. It confirms that regularization alone does not recover the performance lost when aggressively trimming features.
- **`v8_balanced_ridge`** — Ridge version of `v6`. Essentially the same portfolio behavior as `v6`, indicating the clustered balanced set is already stable enough for linear fitting.

## 11. Selection Recommendations

### Recommended default choices

- **Best OOF classification default:** `v3_context_enhanced`.
- **Best OOF regression default:** `v3_context_enhanced`.
- **Best slice-robust specialist:** `v4_freeze_geometry`.
- **Best compact / governed portfolio choice:** `v6_balanced_clustered` (or `v8_balanced_ridge`, effectively tied).
- **Best pure interpretability choice:** `v5_interpretable_clustered` if stakeholder transparency is prioritized over top-end accuracy.

### Recommended deployment framing

1. Use **`v3_context_enhanced` regression** as the main DAx scoring engine because it best fits the continuous threat framing and is strongest on OOF validation.
2. Keep **`v3_context_enhanced` or `v4_freeze_geometry` logistic** as the short-horizon shot-risk companion signal.
3. Use **`v6`/`v8`** for coach-facing interpretability packs, lower-maintenance production baselines, or environments where some features may be unstable / unavailable.
4. Do **not** assume Ridge is the next big gain. The negligible `v6`→`v8` and `v5`→`v7` changes suggest the next improvement path is likely non-linear modeling, interaction design, or better calibration rather than more shrinkage.

## 12. Caveats

- Slice leaderboards are not OOF; they score saved models on the same population used for fit aggregation, so they are best used qualitatively.
- Regression MAPE is not decision-useful here because the target distribution includes near-zero values; prioritize R2, RMSE, MAE, and Spearman instead.
- Coefficients remain associative, not causal. Strong coefficients on possession-lifecycle variables can reflect timing/progression context rather than coachable tactical mechanisms.
- The feature-stability studies were run on `logistic_v4_freeze_geometry` and `regression_v3_context_enhanced`; that is appropriate because they are the main candidates, but it is not a guarantee that every smaller variant behaves identically.

## 13. Artifact Index

### Generated by `build_modeling_report.py`

- `outputs/validation/comparison/model_portfolio/model_leaderboard_all.csv` — model leaderboard all csv
- `outputs/validation/comparison/model_portfolio/cross_task_alignment.csv` — cross task alignment csv
- `outputs/validation/comparison/model_portfolio/fold_stability_summary.csv` — fold stability summary csv
- `outputs/validation/comparison/model_portfolio/slice_win_counts.csv` — slice win counts csv
- `outputs/validation/comparison/model_portfolio/model_portfolio_summary.json` — model portfolio summary json
- `outputs/validation/comparison/slices_latest/summary.json` — slice summary json

### Upstream validation artifacts referenced

- `outputs/oof/baseline/baseline_oof_predictions.parquet`
- `outputs/oof/regression/regression_oof_predictions.parquet`
- `outputs/validation/analysis/feature_stability/logistic_v4_freeze_geometry/tactical_vs_proxy_ablation.csv`
- `outputs/validation/analysis/feature_stability/regression_v3_context_enhanced/tactical_vs_proxy_ablation.csv`
- `outputs/validation/analysis/feature_selection/logistic_v4_freeze_geometry/correlation_heatmap_clustered.png`
- `outputs/validation/analysis/feature_selection/regression_v4_freeze_geometry/correlation_heatmap_clustered.png`
- `outputs/validation/comparison/slices_latest/summary.json`
