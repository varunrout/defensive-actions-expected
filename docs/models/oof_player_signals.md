# OOF player signals

Player signals are provisional out-of-fold expected-versus-observed summaries. They are not true DAx and are not causal estimates.

The action-level definitions are:

- `shot_suppression_oof = expected_shot_probability - observed_shot_outcome`
- `future_xg_suppression_oof = predicted_future_xg - observed_future_xg`

Positive values mean less threat occurred than expected.

The player-team table includes action and match counts, expected and observed shots, expected and observed future xG, total and mean suppression values, phase/action-family/position summaries, match-bootstrap confidence intervals, standard errors, minimum-action and reliability flags. Sensitivity outputs should compare selected variants, visibility subsets, 360-specific versus all-data models, and low-sample exclusions before any interpretation.
