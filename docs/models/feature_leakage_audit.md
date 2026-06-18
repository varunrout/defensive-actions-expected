# Feature leakage audit

The primary prediction timestamp is the defensive-action timestamp. Default `pre_action_context` models may use action location, coarse event/action labels, phase labels, visibility/geometry available at that timestamp, and possession context known before or at the action. They must not use future outcomes, same-row model predictions, player/team identifiers, match/event IDs, fold IDs, cluster outcome summaries, or descriptive signal outputs.

Feature classifications are enforced by `src/dax/models/leakage.py` and checked when `configs/models.yaml` is loaded. Prohibited examples include `event_id`, `match_id`, `player_id`, `player_name`, `team`, target columns, `future_*`, `*_prediction`, `*_suppression`, and DAx/signal outputs.

`post_action_observed` scopes are diagnostic only and must not be compared as deployment-equivalent candidates.
