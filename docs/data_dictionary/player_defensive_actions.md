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
| visible_attacker_count / visible_defender_count | int | Stable football-role counts from 360 frame | Teammate flag interpreted via actor attacking/defending context | Yes | 0 if none visible | Numeric | Medium |
| attackers_within_5m / defenders_within_5m | int/null | Local role density within 5m | Euclidean distance if 5m region is visible | Yes | Null when local region unreliable | Numeric | Medium |
| attackers_within_10m / defenders_within_10m | int/null | Local role density within 10m | Euclidean distance if 10m region is visible | Yes | Null when local region unreliable | Numeric | Medium |
| nearest_attacker_distance / nearest_defender_distance | float | Nearest visible role distance | Euclidean distance | Yes | Null when no visible role | Numeric | Medium |
| attacker_centroid_x/y, defender_centroid_x/y | float | Visible role centroids | Mean visible role coordinates | Yes | Null when no visible role | Numeric | Medium |
| attacker_spread / defender_spread | float | Mean distance from role centroid | Visible role geometry | Yes | Null when no visible role | Numeric | Medium |
| defenders_between_ball_and_attacking_goal | int/null | Visible defenders between ball/action and the attacking goal | Count visible defenders with x >= ball_x after attack-direction normalisation | Yes when roles and ball x are known | Null if role context unknown | Numeric | Medium |
| local_numerical_balance_5m / 10m | int/null | Local attackers minus defenders | Role counts in reliable local regions | Yes | Null when local region unreliable | Numeric | Medium |
| visible_area_polygon_area | float | StatsBomb 360 visible polygon area | Shoelace area after coordinate normalisation | Yes | Null if absent/invalid | Visibility control | Low |
| visible_area_fraction_of_pitch | float | Visible polygon share of pitch | Area / 9600 | Yes | Null if absent/invalid | Visibility control | Low |
| ball_inside_visible_area / action_inside_visible_area | bool/null | Whether action point is in visible polygon | Ray-casting point-in-polygon | Yes | Null if absent/invalid | Visibility control | Low |
| local_5m_region_fully_visible / local_10m_region_fully_visible | bool | Conservative local visibility flags | Requires the visible polygon to cover the clipped local buffer around the action | Yes | False if absent/low coverage | Visibility control | Low |
| visibility_quality_band / visibility_limited | string/bool | Coarse 360 coverage quality | Missing/low/medium/high band from visible fraction | Yes | Missing band if no polygon | Visibility control | Low |
| target_future_shot_10s | int | Future same-possession shot within 10s | Target builder | No (outcome) | 0 if none | Target only | Target |
| target_future_xg_10s | float | Future same-possession shot xG sum | Target builder | No (outcome) | 0 if none | Target only | Target |

Future-only fields such as `possession_duration_total`, `possession_event_count_total`, and `possession_progress_ratio` are not model features.
