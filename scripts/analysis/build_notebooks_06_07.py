"""Build completed analysis notebooks 06 and 07.

These notebooks are generated from code so they can be recreated consistently after
cleanup or git restores. They intentionally depend only on the project data files
and common analysis libraries already used elsewhere in the repository.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf


NOTEBOOKS_DIR = Path(__file__).resolve().parents[2] / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def create_team_clustering_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            """
            # 06 - Team Defensive Profile Clustering

            This notebook turns the player-level defensive-action dataset into **team-tournament defensive profiles** and clusters those profiles into tactical families.

            ## Questions answered
            1. Which teams defend with similar phase mixes and spatial footprints?
            2. Which clusters are front-foot pressers, deep-block absorbers, balanced teams, or risk-exposed teams?
            3. Which team profiles concede the most downstream shot threat (`target_future_shot_10s`) and expected threat (`target_future_xg_10s`)?
            4. Which features most define each cluster?

            **Unit of analysis:** one row per `tournament × team`, filtered to teams with enough matches/actions for stable profiles.
            """
        ),
        code(
            r'''
            from pathlib import Path
            import json
            import warnings

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from IPython.display import display, Markdown
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            from sklearn.metrics import silhouette_score
            from sklearn.preprocessing import StandardScaler

            warnings.filterwarnings('ignore')
            sns.set_theme(style='whitegrid', context='notebook')
            plt.rcParams['figure.figsize'] = (14, 8)
            RANDOM_STATE = 42

            def find_repo_root() -> Path:
                for candidate in [Path.cwd(), Path.cwd().parent, Path.cwd().parents[1] if len(Path.cwd().parents) > 1 else Path.cwd()]:
                    if (candidate / 'data' / 'features' / 'player_defensive_actions.parquet').exists():
                        return candidate
                raise FileNotFoundError('Could not locate data/features/player_defensive_actions.parquet from the current working directory.')

            def mode_or_unknown(s: pd.Series) -> str:
                s = s.dropna()
                return s.mode().iloc[0] if not s.empty else 'Unknown'

            REPO_ROOT = find_repo_root()
            DATA_FEATURES = REPO_ROOT / 'data' / 'features'
            DATA_RAW = REPO_ROOT / 'data' / 'raw'

            df = pd.read_parquet(DATA_FEATURES / 'player_defensive_actions.parquet')

            # Match metadata is optional for the clustering, but it makes the profile unit clearer.
            match_paths = sorted((DATA_RAW / 'matches').glob('*.json'))
            if match_paths:
                matches = pd.concat([pd.read_json(path) for path in match_paths], ignore_index=True)
                meta_cols = [c for c in ['match_id', 'competition_name', 'season', 'competition_stage'] if c in matches.columns]
                meta = matches[meta_cols].drop_duplicates('match_id').copy()
                if {'competition_name', 'season'}.issubset(meta.columns):
                    meta['tournament'] = meta['competition_name'].astype(str) + ' ' + meta['season'].astype(str)
                else:
                    meta['tournament'] = 'Unknown tournament'
                if 'competition_stage' in meta.columns:
                    meta['stage_group'] = np.where(meta['competition_stage'].eq('Group Stage'), 'Group Stage', 'Knockout')
                df = df.merge(meta[['match_id', 'tournament', 'stage_group']], on='match_id', how='left')
            else:
                df['tournament'] = 'All matches'
                df['stage_group'] = 'Unknown'

            df['tournament'] = df['tournament'].fillna('Unknown tournament')
            df['team_profile'] = df['tournament'].astype(str) + ' — ' + df['team'].astype(str)

            print(f'Repository root: {REPO_ROOT}')
            print(f'Defensive actions: {len(df):,}')
            print(f'Teams: {df["team"].nunique()}')
            print(f'Team-tournament profiles: {df["team_profile"].nunique()}')
            display(df[['tournament', 'team', 'phase_label', 'action_family', 'target_future_shot_10s', 'target_future_xg_10s']].head())
            '''
        ),
        md(
            """
            ---
            ## 1. Build team-tournament profile features

            Profiles combine:
            - **Volume / intensity:** defensive actions per match
            - **Outcome risk:** downstream shot rate and mean future xG
            - **Phase identity:** high press, counterpress, low block, box defending, wide defending, central progression defending
            - **Action identity:** pressure, contests, recoveries, interventions
            - **Spatial footprint:** central/wide/deep/high lane shares and mean distance to nearest goal
            - **360 support:** support balance and nearest opponent context
            """
        ),
        code(
            r'''
            profile_key = ['tournament', 'team']

            base = (
                df.groupby(profile_key)
                .agg(
                    matches=('match_id', 'nunique'),
                    actions=('event_id', 'size'),
                    players=('player_id', 'nunique'),
                    shot_rate=('target_future_shot_10s', 'mean'),
                    mean_future_xg=('target_future_xg_10s', 'mean'),
                    has_360_share=('has_360', 'mean'),
                    counterpress_share=('counterpress', 'mean'),
                    central_lane_share=('is_central_lane', 'mean'),
                    wide_lane_share=('is_wide_lane', 'mean'),
                    deep_zone_share=('is_deep_zone', 'mean'),
                    high_zone_share=('is_high_zone', 'mean'),
                    avg_goal_distance=('nearest_goal_distance', 'mean'),
                    avg_support_balance_10m=('freeze_support_balance_10m', 'mean'),
                    avg_support_ratio_10m=('freeze_support_ratio_10m', 'mean'),
                    avg_opponent_nearest_distance=('freeze_opponent_nearest_distance', 'mean'),
                )
                .reset_index()
            )
            base['actions_per_match'] = base['actions'] / base['matches'].clip(lower=1)

            phase_share = pd.crosstab([df['tournament'], df['team']], df['phase_label'], normalize='index').add_prefix('phase__')
            action_share = pd.crosstab([df['tournament'], df['team']], df['action_family'], normalize='index').add_prefix('action__')
            zone_share = pd.crosstab([df['tournament'], df['team']], df['action_zone'], normalize='index').add_prefix('zone__')

            team_profiles = (
                base.set_index(profile_key)
                .join(phase_share, how='left')
                .join(action_share, how='left')
                .join(zone_share, how='left')
                .fillna(0)
                .reset_index()
            )
            team_profiles['profile_name'] = team_profiles['tournament'].astype(str) + ' — ' + team_profiles['team'].astype(str)

            MIN_MATCHES = 3
            MIN_ACTIONS = 250
            profiles = team_profiles[(team_profiles['matches'] >= MIN_MATCHES) & (team_profiles['actions'] >= MIN_ACTIONS)].copy()

            print(f'All profiles: {len(team_profiles)}')
            print(f'Profiles retained (matches >= {MIN_MATCHES}, actions >= {MIN_ACTIONS}): {len(profiles)}')
            display(profiles[['tournament', 'team', 'matches', 'actions', 'actions_per_match', 'shot_rate', 'mean_future_xg']].sort_values('shot_rate', ascending=False).round(4).head(12))
            '''
        ),
        md(
            """
            ---
            ## 2. Choose cluster count and fit K-Means

            We keep risk metrics in the feature set so the clusters represent both style and consequence. The silhouette chart is a sanity check rather than a hard rule; football interpretability still matters.
            """
        ),
        code(
            r'''
            candidate_features = [
                'actions_per_match', 'shot_rate', 'mean_future_xg', 'has_360_share', 'counterpress_share',
                'central_lane_share', 'wide_lane_share', 'deep_zone_share', 'high_zone_share',
                'avg_goal_distance', 'avg_support_balance_10m', 'avg_support_ratio_10m',
                'avg_opponent_nearest_distance',
                'phase__high_press', 'phase__counterpress_after_loss', 'phase__settled_low_block',
                'phase__settled_mid_block', 'phase__box_defence', 'phase__wide_defending',
                'phase__central_progression_defence', 'phase__second_ball',
                'action__pressure', 'action__contest', 'action__recovery', 'action__intervention',
                'action__goalkeeper', 'action__discipline',
            ]
            feature_cols = [c for c in candidate_features if c in profiles.columns]

            X = profiles[feature_cols].replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median(numeric_only=True)).fillna(0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            max_k = min(7, len(profiles) - 1)
            k_results = []
            for k in range(2, max_k + 1):
                labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50).fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else np.nan
                k_results.append({'k': k, 'silhouette': score})
            k_results = pd.DataFrame(k_results)
            display(k_results.round(4))

            if len(k_results):
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.lineplot(data=k_results, x='k', y='silhouette', marker='o', ax=ax)
                ax.set_title('Silhouette score by cluster count')
                ax.set_xlabel('Number of clusters')
                ax.set_ylabel('Silhouette score')
                plt.tight_layout()
                plt.show()

            # Four clusters usually gives interpretable tactical families for this data size.
            N_CLUSTERS = min(4, max(2, len(profiles) // 4))
            kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=100)
            profiles['cluster'] = kmeans.fit_predict(X_scaled)

            centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=feature_cols)
            centers['cluster'] = range(N_CLUSTERS)
            display(centers.set_index('cluster').round(4))
            '''
        ),
        md(
            """
            ---
            ## 3. Give clusters analyst-friendly labels

            Labels are assigned from cluster centroids using clear heuristics. The table keeps the numeric cluster IDs for reproducibility.
            """
        ),
        code(
            r'''
            center_matrix = centers.set_index('cluster').copy()
            center_z = (center_matrix - center_matrix.mean()) / center_matrix.std(ddof=0).replace(0, 1)

            style_components = {
                'Front-foot pressers': ['phase__high_press', 'phase__counterpress_after_loss', 'counterpress_share'],
                'Deep-block absorbers': ['phase__settled_low_block', 'phase__box_defence', 'deep_zone_share'],
                'High-central disruptors': ['central_lane_share', 'high_zone_share', 'phase__central_progression_defence'],
                'Wide-channel protectors': ['wide_lane_share', 'phase__wide_defending'],
                'Risk-exposed profiles': ['shot_rate', 'mean_future_xg'],
            }

            style_scores = pd.DataFrame(index=center_z.index)
            for style_name, style_cols in style_components.items():
                valid_cols = [c for c in style_cols if c in center_z.columns]
                if valid_cols:
                    style_scores[style_name] = center_z[valid_cols].sum(axis=1)
                else:
                    style_scores[style_name] = -999.0

            cluster_labels = {}
            for cluster_id, score_row in style_scores.iterrows():
                ranked = score_row.sort_values(ascending=False)
                primary = ranked.index[0]
                secondary = ranked.index[1] if len(ranked) > 1 else primary
                cluster_labels[cluster_id] = f'{primary} ({secondary}-leaning)'

            profiles['cluster_label'] = profiles['cluster'].map(cluster_labels)

            cluster_summary = (
                profiles.groupby(['cluster', 'cluster_label'])
                .agg(
                    profiles=('profile_name', 'count'),
                    avg_matches=('matches', 'mean'),
                    avg_actions_per_match=('actions_per_match', 'mean'),
                    avg_shot_rate=('shot_rate', 'mean'),
                    avg_xt=('mean_future_xg', 'mean'),
                    high_press=('phase__high_press', 'mean'),
                    counterpress=('counterpress_share', 'mean'),
                    low_block=('phase__settled_low_block', 'mean'),
                    box_defence=('phase__box_defence', 'mean'),
                    central_lane=('central_lane_share', 'mean'),
                    wide_lane=('wide_lane_share', 'mean'),
                )
                .reset_index()
                .sort_values('avg_shot_rate')
            )
            display(cluster_summary.round(4))
            '''
        ),
        md(
            """
            ---
            ## 4. Cluster map and centroid heatmap
            """
        ),
        code(
            r'''
            pca = PCA(n_components=2, random_state=RANDOM_STATE)
            coords = pca.fit_transform(X_scaled)
            profiles['pc1'] = coords[:, 0]
            profiles['pc2'] = coords[:, 1]

            fig, ax = plt.subplots(figsize=(13, 9))
            sns.scatterplot(
                data=profiles,
                x='pc1', y='pc2',
                hue='cluster_label',
                size='shot_rate',
                sizes=(80, 260),
                alpha=0.85,
                ax=ax,
            )
            for _, row in profiles.iterrows():
                label = f"{row['team']} ({str(row['tournament']).split()[-1]})"
                ax.text(row['pc1'] + 0.04, row['pc2'] + 0.04, label, fontsize=8)
            ax.set_title(f'Team defensive profile clusters (PCA explains {pca.explained_variance_ratio_.sum()*100:.1f}% of scaled variance)')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout()
            plt.show()

            heat_cols = [c for c in [
                'actions_per_match', 'shot_rate', 'mean_future_xg', 'counterpress_share', 'central_lane_share', 'deep_zone_share',
                'high_zone_share', 'phase__high_press', 'phase__settled_low_block', 'phase__box_defence',
                'phase__wide_defending', 'action__pressure', 'action__contest', 'action__recovery',
            ] if c in centers.columns]
            center_plot = centers.set_index('cluster')[heat_cols]
            center_plot.index = [f"{idx}: {cluster_labels[idx]}" for idx in center_plot.index]
            center_z = (center_plot - center_plot.mean()) / center_plot.std(ddof=0).replace(0, 1)

            fig, ax = plt.subplots(figsize=(14, 7))
            sns.heatmap(center_z, cmap='vlag', center=0, annot=center_plot.round(3), fmt='', ax=ax)
            ax.set_title('Cluster centroids: raw values annotated, color shows relative z-score')
            plt.tight_layout()
            plt.show()
            '''
        ),
        md(
            """
            ---
            ## 5. Team tables: safest, riskiest, and strongest representatives
            """
        ),
        code(
            r'''
            display_cols = ['tournament', 'team', 'cluster_label', 'matches', 'actions_per_match', 'shot_rate', 'mean_future_xg', 'phase__high_press', 'phase__settled_low_block', 'phase__box_defence', 'counterpress_share']
            display_cols = [c for c in display_cols if c in profiles.columns]

            print('Lowest downstream shot rates among retained team profiles')
            display(profiles.sort_values('shot_rate')[display_cols].head(12).round(4))

            print('Highest downstream shot rates among retained team profiles')
            display(profiles.sort_values('shot_rate', ascending=False)[display_cols].head(12).round(4))

            print('Most representative profiles per cluster (closest to centroid in scaled feature space)')
            distances = kmeans.transform(X_scaled)
            profiles['distance_to_cluster_center'] = [distances[i, c] for i, c in enumerate(profiles['cluster'])]
            representatives = (
                profiles.sort_values('distance_to_cluster_center')
                .groupby('cluster_label', as_index=False)
                .head(5)
                .sort_values(['cluster_label', 'distance_to_cluster_center'])
            )
            display(representatives[['cluster_label', 'tournament', 'team', 'matches', 'shot_rate', 'mean_future_xg', 'distance_to_cluster_center']].round(4))
            '''
        ),
        md(
            """
            ---
            ## 6. Analyst summary
            """
        ),
        code(
            r'''
            print('=' * 90)
            print('TEAM DEFENSIVE CLUSTERING SUMMARY')
            print('=' * 90)
            print(f'Profiles clustered: {len(profiles)}')
            print(f'Clusters: {N_CLUSTERS}')
            print(f'Features used: {len(feature_cols)}')
            print()

            for _, row in cluster_summary.iterrows():
                subset = profiles[profiles['cluster'] == row['cluster']].sort_values('shot_rate')
                examples = ', '.join(subset['team'].head(4).tolist())
                print(f"Cluster {int(row['cluster'])}: {row['cluster_label']}")
                print(f"  Profiles: {int(row['profiles'])}; avg shot rate: {row['avg_shot_rate']*100:.2f}%; avg future xG: {row['avg_xt']:.3f}")
                print(f"  Style: high press {row['high_press']*100:.1f}%, low block {row['low_block']*100:.1f}%, box defence {row['box_defence']*100:.1f}%")
                print(f"  Example lower-risk teams: {examples}")
                print()

            safest = profiles.sort_values('shot_rate').iloc[0]
            riskiest = profiles.sort_values('shot_rate', ascending=False).iloc[0]
            print(f"Safest retained profile by shot rate: {safest['team']} ({safest['tournament']}) — {safest['shot_rate']*100:.2f}%")
            print(f"Riskiest retained profile by shot rate: {riskiest['team']} ({riskiest['tournament']}) — {riskiest['shot_rate']*100:.2f}%")
            '''
        ),
    ]
    return nb


def create_player_archetype_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            """
            # 07 - Player Defensive Archetypes

            This notebook creates **individual player defensive archetypes** from event-level defensive actions.

            ## Questions answered
            1. Which players have similar defensive action profiles?
            2. Which archetypes are pressers, box defenders, wide protectors, ball winners, or support defenders?
            3. Which players are the clearest examples of each archetype?
            4. How do archetypes differ by position group and downstream risk?

            **Unit of analysis:** one row per player, filtered to players with enough defensive actions for a stable profile.
            """
        ),
        code(
            r'''
            from pathlib import Path
            import warnings

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from IPython.display import display, Markdown
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            from sklearn.metrics import silhouette_score
            from sklearn.preprocessing import StandardScaler

            warnings.filterwarnings('ignore')
            sns.set_theme(style='whitegrid', context='notebook')
            plt.rcParams['figure.figsize'] = (14, 8)
            RANDOM_STATE = 42

            def find_repo_root() -> Path:
                for candidate in [Path.cwd(), Path.cwd().parent, Path.cwd().parents[1] if len(Path.cwd().parents) > 1 else Path.cwd()]:
                    if (candidate / 'data' / 'features' / 'player_defensive_actions.parquet').exists():
                        return candidate
                raise FileNotFoundError('Could not locate data/features/player_defensive_actions.parquet from the current working directory.')

            def mode_or_unknown(s: pd.Series) -> str:
                s = s.dropna().astype(str)
                return s.mode().iloc[0] if not s.empty else 'Unknown'

            REPO_ROOT = find_repo_root()
            DATA_FEATURES = REPO_ROOT / 'data' / 'features'
            df = pd.read_parquet(DATA_FEATURES / 'player_defensive_actions.parquet')

            print(f'Repository root: {REPO_ROOT}')
            print(f'Defensive actions: {len(df):,}')
            print(f'Players: {df["player_id"].nunique():,}')
            print(f'Teams: {df["team"].nunique()}')
            display(df[['player', 'team', 'position', 'position_group', 'phase_label', 'action_family', 'target_future_shot_10s']].head())
            '''
        ),
        md(
            """
            ---
            ## 1. Build player profile features

            Each player profile includes volume, risk, phase/action shares, spatial tendency, and 360 support context.
            """
        ),
        code(
            r'''
            profile_key = ['player_id']
            base = (
                df.groupby(profile_key)
                .agg(
                    player=('player', mode_or_unknown),
                    primary_team=('team', mode_or_unknown),
                    primary_position=('position', mode_or_unknown),
                    position_group=('position_group', mode_or_unknown),
                    matches=('match_id', 'nunique'),
                    actions=('event_id', 'size'),
                    shot_rate=('target_future_shot_10s', 'mean'),
                    mean_future_xg=('target_future_xg_10s', 'mean'),
                    has_360_share=('has_360', 'mean'),
                    counterpress_share=('counterpress', 'mean'),
                    central_lane_share=('is_central_lane', 'mean'),
                    wide_lane_share=('is_wide_lane', 'mean'),
                    deep_zone_share=('is_deep_zone', 'mean'),
                    high_zone_share=('is_high_zone', 'mean'),
                    avg_goal_distance=('nearest_goal_distance', 'mean'),
                    avg_support_balance_10m=('freeze_support_balance_10m', 'mean'),
                    avg_support_ratio_10m=('freeze_support_ratio_10m', 'mean'),
                    avg_opponent_nearest_distance=('freeze_opponent_nearest_distance', 'mean'),
                )
                .reset_index()
            )
            base['actions_per_match'] = base['actions'] / base['matches'].clip(lower=1)

            phase_share = pd.crosstab(df['player_id'], df['phase_label'], normalize='index').add_prefix('phase__')
            action_share = pd.crosstab(df['player_id'], df['action_family'], normalize='index').add_prefix('action__')
            zone_share = pd.crosstab(df['player_id'], df['action_zone'], normalize='index').add_prefix('zone__')

            player_profiles = (
                base.set_index('player_id')
                .join(phase_share, how='left')
                .join(action_share, how='left')
                .join(zone_share, how='left')
                .fillna(0)
                .reset_index()
            )

            MIN_ACTIONS = 45
            MIN_MATCHES = 2
            profiles = player_profiles[(player_profiles['actions'] >= MIN_ACTIONS) & (player_profiles['matches'] >= MIN_MATCHES)].copy()

            print(f'All player profiles: {len(player_profiles):,}')
            print(f'Profiles retained (actions >= {MIN_ACTIONS}, matches >= {MIN_MATCHES}): {len(profiles):,}')
            print(f'Action coverage retained: {profiles["actions"].sum() / player_profiles["actions"].sum() * 100:.1f}%')
            display(profiles[['player', 'primary_team', 'primary_position', 'position_group', 'matches', 'actions', 'actions_per_match', 'shot_rate', 'mean_future_xg']].sort_values('actions', ascending=False).head(15).round(4))
            '''
        ),
        md(
            """
            ---
            ## 2. Position-group context

            Position strongly shapes defensive events. Before clustering, inspect the retained sample by position group and risk.
            """
        ),
        code(
            r'''
            position_summary = (
                profiles.groupby('position_group')
                .agg(
                    players=('player_id', 'count'),
                    actions=('actions', 'sum'),
                    median_actions=('actions', 'median'),
                    shot_rate=('shot_rate', 'mean'),
                    mean_future_xg=('mean_future_xg', 'mean'),
                    high_press=('phase__high_press', 'mean'),
                    low_block=('phase__settled_low_block', 'mean'),
                    box_defence=('phase__box_defence', 'mean'),
                    pressure=('action__pressure', 'mean'),
                )
                .sort_values('players', ascending=False)
            )
            display(position_summary.round(4))

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            sns.boxplot(data=profiles, x='position_group', y='shot_rate', ax=axes[0])
            axes[0].set_title('Player downstream shot-rate distribution by position group')
            axes[0].tick_params(axis='x', rotation=35)
            sns.boxplot(data=profiles, x='position_group', y='actions_per_match', ax=axes[1])
            axes[1].set_title('Defensive actions per match by position group')
            axes[1].tick_params(axis='x', rotation=35)
            plt.tight_layout()
            plt.show()
            '''
        ),
        md(
            """
            ---
            ## 3. Fit player archetype clusters

            The clustering uses style and context features. Outcome risk is included so archetypes capture whether a pattern is typically followed by danger.
            """
        ),
        code(
            r'''
            candidate_features = [
                'actions_per_match', 'shot_rate', 'mean_future_xg', 'has_360_share', 'counterpress_share',
                'central_lane_share', 'wide_lane_share', 'deep_zone_share', 'high_zone_share',
                'avg_goal_distance', 'avg_support_balance_10m', 'avg_support_ratio_10m',
                'avg_opponent_nearest_distance',
                'phase__high_press', 'phase__counterpress_after_loss', 'phase__settled_low_block',
                'phase__settled_mid_block', 'phase__box_defence', 'phase__wide_defending',
                'phase__central_progression_defence', 'phase__second_ball',
                'action__pressure', 'action__contest', 'action__recovery', 'action__intervention',
                'action__goalkeeper', 'action__discipline',
            ]
            feature_cols = [c for c in candidate_features if c in profiles.columns]

            X = profiles[feature_cols].replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median(numeric_only=True)).fillna(0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            max_k = min(9, len(profiles) - 1)
            k_results = []
            for k in range(2, max_k + 1):
                labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50).fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else np.nan
                k_results.append({'k': k, 'silhouette': score})
            k_results = pd.DataFrame(k_results)
            display(k_results.round(4))

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.lineplot(data=k_results, x='k', y='silhouette', marker='o', ax=ax)
            ax.set_title('Silhouette score by player archetype count')
            ax.set_xlabel('Number of archetypes')
            ax.set_ylabel('Silhouette score')
            plt.tight_layout()
            plt.show()

            N_CLUSTERS = min(6, max(3, len(profiles) // 35))
            kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=100)
            profiles['cluster'] = kmeans.fit_predict(X_scaled)

            centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=feature_cols)
            centers['cluster'] = range(N_CLUSTERS)
            display(centers.set_index('cluster').round(4))
            '''
        ),
        md(
            """
            ---
            ## 4. Label the archetypes
            """
        ),
        code(
            r'''
            center_matrix = centers.set_index('cluster').copy()
            center_z = (center_matrix - center_matrix.mean()) / center_matrix.std(ddof=0).replace(0, 1)

            style_components = {
                'High-press hunters': ['phase__high_press', 'phase__counterpress_after_loss', 'counterpress_share', 'action__pressure'],
                'Box defenders': ['phase__box_defence', 'phase__settled_low_block', 'deep_zone_share'],
                'Wide-channel protectors': ['phase__wide_defending', 'wide_lane_share'],
                'Ball-winning disruptors': ['action__contest', 'action__recovery', 'action__intervention'],
                'Central screeners': ['phase__central_progression_defence', 'central_lane_share', 'phase__settled_mid_block'],
                'Goalkeeper / last-line actors': ['action__goalkeeper'],
                'Risk-exposed defenders': ['shot_rate', 'mean_future_xg'],
            }

            style_scores = pd.DataFrame(index=center_z.index)
            for style_name, style_cols in style_components.items():
                valid_cols = [c for c in style_cols if c in center_z.columns]
                if valid_cols:
                    style_scores[style_name] = center_z[valid_cols].sum(axis=1)
                else:
                    style_scores[style_name] = -999.0

            archetype_labels = {}
            for cluster_id, score_row in style_scores.iterrows():
                ranked = score_row.sort_values(ascending=False)
                primary = ranked.index[0]
                secondary = ranked.index[1] if len(ranked) > 1 else primary
                archetype_labels[cluster_id] = f'{primary} ({secondary}-leaning, C{int(cluster_id)})'

            profiles['archetype'] = profiles['cluster'].map(archetype_labels)

            archetype_summary = (
                profiles.groupby(['cluster', 'archetype'])
                .agg(
                    players=('player_id', 'count'),
                    actions=('actions', 'sum'),
                    avg_actions_per_match=('actions_per_match', 'mean'),
                    avg_shot_rate=('shot_rate', 'mean'),
                    avg_xt=('mean_future_xg', 'mean'),
                    high_press=('phase__high_press', 'mean'),
                    low_block=('phase__settled_low_block', 'mean'),
                    box_defence=('phase__box_defence', 'mean'),
                    wide_defending=('phase__wide_defending', 'mean'),
                    pressure=('action__pressure', 'mean'),
                    contest=('action__contest', 'mean'),
                    recovery=('action__recovery', 'mean'),
                )
                .reset_index()
                .sort_values('players', ascending=False)
            )
            display(archetype_summary.round(4))

            composition = pd.crosstab(profiles['archetype'], profiles['position_group'], normalize='index')
            fig, ax = plt.subplots(figsize=(13, 7))
            sns.heatmap(composition, cmap='Blues', annot=True, fmt='.0%', ax=ax)
            ax.set_title('Position-group composition by player archetype')
            ax.set_xlabel('Position group')
            ax.set_ylabel('Archetype')
            plt.tight_layout()
            plt.show()
            '''
        ),
        md(
            """
            ---
            ## 5. Archetype map and centroid heatmap
            """
        ),
        code(
            r'''
            pca = PCA(n_components=2, random_state=RANDOM_STATE)
            coords = pca.fit_transform(X_scaled)
            profiles['pc1'] = coords[:, 0]
            profiles['pc2'] = coords[:, 1]

            fig, ax = plt.subplots(figsize=(13, 9))
            sns.scatterplot(
                data=profiles,
                x='pc1', y='pc2',
                hue='archetype',
                style='position_group',
                size='actions',
                sizes=(30, 220),
                alpha=0.75,
                ax=ax,
            )
            ax.set_title(f'Player defensive archetypes (PCA explains {pca.explained_variance_ratio_.sum()*100:.1f}% of scaled variance)')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout()
            plt.show()

            heat_cols = [c for c in [
                'actions_per_match', 'shot_rate', 'mean_future_xg', 'counterpress_share', 'central_lane_share', 'wide_lane_share',
                'deep_zone_share', 'high_zone_share', 'phase__high_press', 'phase__settled_low_block', 'phase__box_defence',
                'phase__wide_defending', 'phase__central_progression_defence', 'action__pressure', 'action__contest',
                'action__recovery', 'action__intervention',
            ] if c in centers.columns]
            center_plot = centers.set_index('cluster')[heat_cols]
            center_plot.index = [f"{idx}: {archetype_labels[idx]}" for idx in center_plot.index]
            center_z = (center_plot - center_plot.mean()) / center_plot.std(ddof=0).replace(0, 1)

            fig, ax = plt.subplots(figsize=(15, 8))
            sns.heatmap(center_z, cmap='vlag', center=0, annot=center_plot.round(3), fmt='', ax=ax)
            ax.set_title('Player archetype centroids: raw values annotated, color shows relative z-score')
            plt.tight_layout()
            plt.show()
            '''
        ),
        md(
            """
            ---
            ## 6. Exemplars and low-risk players by role

            `distance_to_archetype_center` identifies players closest to the K-Means centroid. Low shot rate should be read with context: role, team strength, and action volume all matter.
            """
        ),
        code(
            r'''
            distances = kmeans.transform(X_scaled)
            profiles['distance_to_archetype_center'] = [distances[i, c] for i, c in enumerate(profiles['cluster'])]

            exemplar_cols = ['archetype', 'player', 'primary_team', 'primary_position', 'position_group', 'matches', 'actions', 'shot_rate', 'mean_future_xg', 'distance_to_archetype_center']
            exemplars = (
                profiles.sort_values('distance_to_archetype_center')
                .groupby('archetype', as_index=False)
                .head(8)
                .sort_values(['archetype', 'distance_to_archetype_center'])
            )
            print('Closest archetype exemplars')
            display(exemplars[exemplar_cols].round(4))

            print('Lowest-risk retained players within each position group (minimum volume already applied)')
            low_risk = (
                profiles.sort_values(['position_group', 'shot_rate', 'mean_future_xg'])
                .groupby('position_group', as_index=False)
                .head(8)
            )
            display(low_risk[['position_group', 'player', 'primary_team', 'primary_position', 'archetype', 'matches', 'actions', 'shot_rate', 'mean_future_xg']].round(4))

            print('Highest-volume retained players')
            display(profiles.sort_values('actions', ascending=False)[['player', 'primary_team', 'position_group', 'archetype', 'matches', 'actions', 'actions_per_match', 'shot_rate']].head(20).round(4))
            '''
        ),
        md(
            """
            ---
            ## 7. Analyst summary
            """
        ),
        code(
            r'''
            print('=' * 90)
            print('PLAYER DEFENSIVE ARCHETYPE SUMMARY')
            print('=' * 90)
            print(f'Players clustered: {len(profiles):,}')
            print(f'Archetypes: {N_CLUSTERS}')
            print(f'Features used: {len(feature_cols)}')
            print()

            for _, row in archetype_summary.iterrows():
                subset = profiles[profiles['cluster'] == row['cluster']].sort_values('distance_to_archetype_center')
                examples = ', '.join(subset['player'].head(5).tolist())
                print(f"Cluster {int(row['cluster'])}: {row['archetype']}")
                print(f"  Players: {int(row['players'])}; avg shot rate: {row['avg_shot_rate']*100:.2f}%; avg future xG: {row['avg_xt']:.3f}")
                print(f"  Style: high press {row['high_press']*100:.1f}%, box defence {row['box_defence']*100:.1f}%, pressure {row['pressure']*100:.1f}%")
                print(f"  Exemplars: {examples}")
                print()

            safest = profiles.sort_values(['shot_rate', 'mean_future_xg']).iloc[0]
            print(f"Lowest-risk retained player by shot rate: {safest['player']} ({safest['primary_team']}, {safest['position_group']}) — {safest['shot_rate']*100:.2f}%")
            '''
        ),
    ]
    return nb


def write_notebooks() -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "06_team_defensive_clustering.ipynb": create_team_clustering_notebook(),
        "07_player_defensive_archetypes.ipynb": create_player_archetype_notebook(),
    }
    for filename, nb in notebooks.items():
        path = NOTEBOOKS_DIR / filename
        with path.open("w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Created {path.relative_to(NOTEBOOKS_DIR.parents[0])} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    write_notebooks()

