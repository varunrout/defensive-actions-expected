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
