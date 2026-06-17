# Chart Catalogue

| Area | Chart | Source table | Default output | Interpretation |
|---|---|---|---|---|
| Data quality | Events by competition | processed events | `outputs/analysis/data_quality/events_by_competition.png` | Competition coverage |
| Data quality | Events by type | event counts | `events_by_type.png` | Event mix |
| Data quality | Rows per match | rows per match | `rows_per_match.png` | Match coverage |
| Data quality | Missingness | missingness table | `missingness.png` | Column completeness |
| Data quality | Phase distribution | phase table | `phase_distribution.png` | Rule-based phase proxy mix |
| Data quality | Target distribution | processed events | `target_future_xg_distribution.png` | Observed future-xG distribution |
| Features | Numeric histograms | feature table | `*_distribution.png` | Feature ranges and skew |
| Features | Correlation heatmap | correlation matrix | `correlation_heatmap.png` | Redundancy diagnostics |
| Features | Missingness by family | diagnostics table | `missingness_by_feature_family.png` | Feature-family completeness |
| Spatial | Pitch grid heatmap | zone summary | `total_action_pitch_heatmap.png` | Spatial density/rates by grid zone |
| Players | Sample/activity charts | player summary | player output directory | Exposure and outcome comparison |
| Clustering | Cluster sizes/centroids/PCA/stability | clustering outputs | clustering output directory | Style-cluster diagnostics |

All chart helpers label axes, handle empty inputs, save PNG, optionally save SVG, and close figures after saving.

## mplsoccer pitch outputs

Football pitch visualisations use `mplsoccer.Pitch` with StatsBomb coordinates: pitch length 120, pitch width 80, defensive goal at x=0 and attacking goal at x=120. The default orientation is horizontal and attacking direction is left to right.

Primary spatial outputs include `outputs/analysis/spatial/all_actions_density.png`, `all_actions_scatter.png`, action-family density maps such as `pressure_density.png`, phase density maps, `possession_win_rate_map.png`, `future_shot_rate_map.png`, and `future_xg_map.png`. Rate maps mask cells below `minimum_spatial_bin_actions` and must be interpreted as descriptive spatial profiles, not tactical ground truth or performance estimates.

The coarse 6x4 pitch grid remains available only as a technical diagnostic table/chart; active football analysis charts should use the mplsoccer pitch functions in `src/dax/analysis/pitch_plotting.py`.
