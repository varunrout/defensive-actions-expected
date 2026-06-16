# Target definitions

## `target_future_shot_10s`
Observed binary outcome: whether a future shot occurs no more than 10 seconds after the event in the same match, period, relevant possession, and attacking team. The shot event itself never counts for its own row. The search does not cross possession boundaries.

## `target_future_xg_10s`
Observed continuous outcome: sum of StatsBomb shot xG for eligible future shots under the same boundary rules as `target_future_shot_10s`. Multiple rebound shots in the same possession are summed. Missing shot xG is treated as zero.

## Deprecated target
`target_xt_10s` is deprecated as a target because it was created by fitting a grid model on the full dataset and then scoring future events, which is circular across validation folds.
