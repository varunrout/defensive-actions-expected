# Analysis Limitations

## Exposure and minutes

The current framework avoids manufacturing per-90 metrics when reliable minutes are unavailable. Actions per match and action shares are exposure descriptors, not quality ratings.

## 360 visibility

`has_360` indicates 360 data exists, but reliable local context requires local region visibility and role-known checks. Missing or limited visibility can bias role and difficulty features.

## Phase proxies

Phase labels are rule-based tactical proxies. They support stratification and description but are not ground-truth tactical labels.

## Target interpretation

`target_future_shot_10s` and `target_future_xg_10s` are observed future outcomes after actions. Descriptive rates are not causal and are not expected-versus-observed residuals.

## Clustering

Clusters describe style using configured player-level features. They are sensitive to feature selection, sample thresholds, visibility coverage, and dataset composition. They should not be interpreted as player quality tiers.
