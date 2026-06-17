# Player Aggregation

## Grain

`data/features/player_defensive_summary.parquet` has one row per player-team. The source is the canonical `player_defensive_actions.parquet` table produced by `src/dax/features/player_defense.py`.

## Metrics and denominators

The summary includes matches, total actions, actions per match, future-shot count/rate, future-xG total/mean, action-family counts/shares, phase counts/shares, pitch-zone counts/shares, action-family-specific target denominators/rates, and phase-specific target denominators/rates. Every rate has an explicit denominator column where appropriate.

## Possession metrics

Possession metrics use canonical fields: `action_won_possession`, `action_ended_possession`, `action_retained_defensive_team_control`, and `action_was_under_opponent_possession`. `has_360` is not treated as reliable visibility.

## Reliability rules

`role_known_actions` uses `freeze_frame_roles_known`. `reliable_visibility_actions` requires both `local_5m_region_fully_visible` and `local_10m_region_fully_visible`. Numerical disadvantage shares count action-level negative `local_numerical_balance_5m` or `local_numerical_balance_10m` rows divided by valid non-null denominators; they are not inferred from player-level means.
