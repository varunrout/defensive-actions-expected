# Player defensive actions data dictionary

| Column | Type | Meaning | Calculation | Event-time availability | Missing policy | Model use | Leakage risk |
|---|---|---|---|---|---|---|---|
| match_id | id | Match identifier | Source | Yes | Required | Grouping | Low |
| period | int | Match period | Source | Yes | Required | Feature/grouping | Low |
| possession | id | Possession id | Source | Yes | Required for targets | Boundary | Low |
| event_id | id | Event uuid | Source | Yes | Required | Identifier | Low |
| event_index | int | Canonical event order | Source `index` | Yes | Required | Ordering | Low |
| player_id/player | id/string | Actor | Source | Yes | Drop if missing | Identifier/reporting | Low |
| actor_team | string | Event actor team | Source team | Yes | Validate | Semantics | Medium |
| attacking_team | string | Possession team before action | Context | Yes | Validate | Semantics | Medium |
| defending_team | string | Opponent/defending team | Context | Yes | Validate not equal attacking | Semantics | Medium |
| event_type/action_family | string | Defensive action family | Mapped from event type | Yes | Other/exclude | Categorical | Low |
| possession_elapsed_seconds | float | Seconds since possession start so far | Current time - first event time | Yes | 0 if first | Numeric | Low |
| events_elapsed_in_possession | int | Events observed so far | Running count | Yes | Required | Numeric | Low |
| phase_transitions_observed_so_far | int | Phase changes observed to action | Running count | Yes | 0 | Numeric | Low |
| distance_to_attacking_goal | float | Distance to (120,40) | Geometry | Yes | Missing if no coords | Numeric | Low |
| distance_to_defending_goal | float | Distance to (0,40) | Geometry | Yes | Missing if no coords | Numeric | Low |
| angle_to_attacking_goal | float | Goal angle proxy | Geometry | Yes | Missing if no coords | Numeric | Low |
| target_future_shot_10s | int | Future same-possession shot within 10s | Target builder | No (outcome) | 0 if none | Target only | Target |
| target_future_xg_10s | float | Future same-possession shot xG sum | Target builder | No (outcome) | 0 if none | Target only | Target |

Future-only fields such as `possession_duration_total`, `possession_event_count_total`, and `possession_progress_ratio` are not model features.
