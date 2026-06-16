# Defensibility Addendum

This addendum strengthens the evidence package with uncertainty quantification, calibration diagnostics, and slice-level gap stress tests.

## What was added

- Match-bootstrap confidence intervals for each variant (`n_boot=40`).
- Winner-frequency analysis (how often each model is best under resampling).
- Classification calibration checks (ECE + reliability curves).
- Regression decile-bias checks (observed vs predicted by risk bands).
- Slice-level `v4 - v3` gap table across `phase_label`, `action_zone`, and `position_group`.

## Key outcomes

- Classification top mean AUC remains `v3_context_enhanced`: 0.8112 (95% CI 0.8053, 0.8194).
- Regression top mean R2 remains `v3_context_enhanced`: 0.3689 (95% CI 0.3588, 0.3813).
- Classification ranking robustness: `v3_context_enhanced` wins 60.0% of bootstrap runs.
- Regression ranking robustness: `v4_freeze_geometry` wins 50.0% of bootstrap runs.
- Best calibration among tested classifiers: `v4_freeze_geometry` (ECE=0.2951, Brier=0.1765).
- Most stable regression decile bias: `v5_interpretable_clustered` (mean abs decile bias=0.0140).

## Tables

### Classification confidence intervals

| variant | auc | auc_ci_low | auc_ci_high | ap | ap_ci_low | ap_ci_high | brier | brier_ci_low | brier_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v3_context_enhanced | 0.808762 | 0.805333 | 0.819449 | 0.328943 | 0.313829 | 0.349459 | 0.176396 | 0.172663 | 0.178818 |
| v4_freeze_geometry | 0.807641 | 0.798647 | 0.816845 | 0.328059 | 0.314223 | 0.348304 | 0.176463 | 0.173166 | 0.180814 |
| v2_full_baseline | 0.805812 | 0.797542 | 0.813815 | 0.318069 | 0.298338 | 0.333814 | 0.177828 | 0.174458 | 0.182017 |
| v8_balanced_ridge | 0.800572 | 0.790478 | 0.808187 | 0.319926 | 0.303876 | 0.337034 | 0.178983 | 0.175047 | 0.182654 |
| v6_balanced_clustered | 0.800568 | 0.787809 | 0.810177 | 0.319694 | 0.303813 | 0.348001 | 0.178997 | 0.175436 | 0.182670 |
| v5_interpretable_clustered | 0.785610 | 0.775143 | 0.794698 | 0.314292 | 0.299046 | 0.327782 | 0.184869 | 0.181191 | 0.187645 |
| v7_interpretable_ridge | 0.785609 | 0.777564 | 0.794203 | 0.314407 | 0.304118 | 0.332473 | 0.184848 | 0.181099 | 0.188518 |
| v1_spatial | 0.776802 | 0.765657 | 0.784033 | 0.290296 | 0.271911 | 0.306498 | 0.192063 | 0.189474 | 0.194540 |
| v0_phase_only | 0.679346 | 0.672091 | 0.689654 | 0.128944 | 0.119339 | 0.136333 | 0.226034 | 0.224670 | 0.227571 |

### Regression confidence intervals

| variant | r2 | r2_ci_low | r2_ci_high | rmse | rmse_ci_low | rmse_ci_high | mae | mae_ci_low | mae_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v3_context_enhanced | 0.369709 | 0.358846 | 0.381278 | 0.311580 | 0.304711 | 0.316801 | 0.226246 | 0.220488 | 0.228937 |
| v4_freeze_geometry | 0.369303 | 0.358795 | 0.380704 | 0.311680 | 0.306641 | 0.317911 | 0.226344 | 0.223712 | 0.232047 |
| v2_full_baseline | 0.319335 | 0.308463 | 0.325940 | 0.323791 | 0.318354 | 0.331451 | 0.235973 | 0.232984 | 0.240922 |
| v8_balanced_ridge | 0.273116 | 0.265225 | 0.283197 | 0.334604 | 0.328271 | 0.340901 | 0.242676 | 0.238783 | 0.245630 |
| v6_balanced_clustered | 0.273115 | 0.263185 | 0.281432 | 0.334604 | 0.324232 | 0.339522 | 0.242677 | 0.239004 | 0.247010 |
| v7_interpretable_ridge | 0.243978 | 0.235803 | 0.255820 | 0.341245 | 0.334506 | 0.348533 | 0.249099 | 0.245091 | 0.252957 |
| v5_interpretable_clustered | 0.243976 | 0.229737 | 0.252735 | 0.341245 | 0.332479 | 0.348395 | 0.249100 | 0.244609 | 0.253458 |
| v1_spatial | 0.132117 | 0.125278 | 0.135454 | 0.365619 | 0.354866 | 0.373964 | 0.274989 | 0.270803 | 0.277405 |
| v0_phase_only | 0.034750 | 0.032205 | 0.036872 | 0.385583 | 0.381495 | 0.392428 | 0.294325 | 0.290939 | 0.297839 |

### Classification winner frequency

| variant | winner_frequency_auc |
| --- | --- |
| v3_context_enhanced | 0.600000 |
| v4_freeze_geometry | 0.250000 |
| v2_full_baseline | 0.150000 |

### Regression winner frequency

| variant | winner_frequency_r2 |
| --- | --- |
| v4_freeze_geometry | 0.500000 |
| v3_context_enhanced | 0.500000 |

### Largest positive `v4 - v3` slice gaps (classification AUC)

| dimension | slice | v3_metric | v4_metric | delta_v4_minus_v3 | n_rows |
| --- | --- | --- | --- | --- | --- |
| action_zone | attacking_third_left_flank | 0.707189 | 0.708706 | 0.001518 | 3669 |
| position_group | goalkeeper | 0.814669 | 0.815521 | 0.000852 | 2346 |
| position_group | fullback_wingback | 0.817890 | 0.817656 | -0.000234 | 10414 |
| position_group | forward | 0.814618 | 0.814308 | -0.000310 | 7076 |
| position_group | other | 0.809545 | 0.809197 | -0.000348 | 6121 |
| phase_label | box_defence | 0.765058 | 0.764666 | -0.000392 | 9686 |
| position_group | centre_back | 0.808152 | 0.807665 | -0.000487 | 10607 |
| action_zone | attacking_third_center | 0.775736 | 0.775215 | -0.000521 | 12110 |
| action_zone | middle_third_right_flank | 0.680102 | 0.679449 | -0.000653 | 5717 |
| phase_label | second_ball | 0.850728 | 0.849989 | -0.000740 | 6035 |

### Largest positive `v4 - v3` slice gaps (regression R2)

| dimension | slice | v3_metric | v4_metric | delta_v4_minus_v3 | n_rows |
| --- | --- | --- | --- | --- | --- |
| action_zone | attacking_third_left_flank | 0.349081 | 0.349452 | 0.000371 | 3669 |
| position_group | midfielder | 0.335844 | 0.336035 | 0.000190 | 10229 |
| phase_label | counterpress_after_loss | 0.243945 | 0.243981 | 0.000036 | 4070 |
| phase_label | high_press | 0.397689 | 0.397621 | -0.000069 | 9797 |
| action_zone | attacking_third_right_flank | 0.380162 | 0.380043 | -0.000118 | 3794 |
| action_zone | defensive_third_center | 0.378838 | 0.378710 | -0.000127 | 10187 |
| phase_label | settled_low_block | 0.337285 | 0.337013 | -0.000272 | 8915 |
| phase_label | box_defence | 0.390680 | 0.390407 | -0.000272 | 9686 |
| position_group | other | 0.363394 | 0.363120 | -0.000273 | 6121 |
| action_zone | middle_third_right_flank | 0.227362 | 0.227028 | -0.000334 | 5717 |

## New charts

- `outputs/validation/comparison/defensibility/calibration_reliability_curves.png`
- `outputs/validation/comparison/defensibility/regression_decile_bias.png`

## Defensibility checklist (critical gaps)

- Added now: uncertainty CIs, rank robustness, calibration, decile bias, focused slice stress tests.
- Still recommended next: external temporal holdout by competition season, decision-curve utility analysis, and model monitoring thresholds for deployment drift.
