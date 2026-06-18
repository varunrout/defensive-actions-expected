# Descriptive Signal Framework

Signals are provisional, transparent, non-modelled components. They must not be called true DAx and must not be interpreted causally.

| Component | Source | Direction | Denominator | Interpretation | Warning behaviour |
|---|---|---:|---|---|---|
| `activity_index` | `actions_per_match` | Higher is higher activity | `matches` | Action frequency by represented match | Missing if matches/actions unavailable |
| `possession_win_index` | `possession_win_rate` | Higher | possession-win denominator | Share of actions winning possession | Missing below sample threshold |
| `threat_suppression_descriptive_index` | `future_shot_rate` | Lower observed rate is higher index | future-shot denominator | Descriptive post-action shot suppression | Not model-adjusted or causal |
| `phase_versatility_index` | phase-share entropy | Higher | total actions | Diversity across phase proxies | Missing if phase shares unavailable |
| `spatial_aggression_index` | `mean_action_x` | Higher | total actions | Average action height toward attacking goal | No future-xG fallback is allowed |
| `transition_defence_exposure_index` | transition-defence phase share | Higher | total actions | Exposure to transition-defence proxy | Missing if phase absent |
| `box_defence_exposure_index` | configured deep-zone share | Higher | total actions | Approximate box/deep defending exposure | Grid-dependent |
| `local_numerical_difficulty_index` | `mean_local_numerical_balance_10m` | Lower balance is harder | valid 10m denominator | Visible local numerical difficulty | Requires roles and visibility |
| `visibility_reliability_index` | `reliable_visibility_share` | Higher | total actions | Local visibility reliability | Coverage, not player quality |

Missing component inputs remain missing and add warnings; components are not zero-filled.
