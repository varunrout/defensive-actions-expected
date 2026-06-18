# Clustering Methodology

## Feature selection

Primary style clustering uses explicit feature groups in `configs/analysis.yaml`: action mix, phase mix, spatial style, possession style, 360 context, and difficulty exposure. Identifiers, matches, total action volume, target totals/rates, denominator columns, and reliability flags are excluded by default.

## Scaling and missing values

Eligible players must meet the configured minimum action threshold. Selected features above the configured missingness threshold are removed, remaining missing values are median-imputed, constant features are removed, and retained features are scaled using the configured scaler.

## Algorithms

K-means, hierarchical agglomerative clustering, and Gaussian mixture models are evaluated across configured cluster-count candidates. GMM outputs include membership probability and low-confidence assignment flags for the selected solution when GMM is selected.

## Evaluation and stability

Solutions are evaluated with silhouette, Calinski-Harabasz, Davies-Bouldin, cluster-size balance, and repeated 80% subsample adjusted Rand index stability. The selection table ranks solutions by average percentile score across those metrics.

## Selection rule

The selected solution is the highest average percentile rank across silhouette, Calinski-Harabasz, inverse Davies-Bouldin, size balance, and subsample ARI stability. Clusters describe defensive style, not player quality.
