"""Build a notebook that visualizes all latest slice-comparison outputs.

Input directory:
    outputs/validation/comparison/slices_latest/

Output notebook:
    notebooks/08_slices_latest_visualization.ipynb
"""
from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "08_slices_latest_visualization.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def build_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            """
            # 08 - Slices Latest Model Visualization

            This notebook loads every CSV from `outputs/validation/comparison/slices_latest/` and visualizes the full slice-level model portfolio.

            ## Files loaded
            - `slice_metrics_logistic.csv`
            - `slice_metrics_regression.csv`
            - `best_variant_by_slice_logistic.csv`
            - `best_variant_by_slice_regression.csv`

            ## Coach translation
            - **Logistic models** predict `target_future_shot_10s`: *does this defensive context turn into an opponent shot soon?*
            - **Regression models** predict `target_future_xg_10s`: *how dangerous is the opponent's next attacking window?*

            ## Metrics
            - Logistic: `roc_auc`, `avg_precision` — higher is better.
            - Regression: `r2`, `spearman` — higher is better; `rmse`, `mae` — lower is better.
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

            warnings.filterwarnings('ignore')
            sns.set_theme(style='whitegrid', context='notebook')
            plt.rcParams['figure.figsize'] = (14, 8)
            pd.set_option('display.max_columns', 100)
            pd.set_option('display.max_rows', 200)

            def find_repo_root() -> Path:
                for candidate in [Path.cwd(), Path.cwd().parent, Path.cwd().parents[1] if len(Path.cwd().parents) > 1 else Path.cwd()]:
                    if (candidate / 'outputs' / 'validation' / 'comparison' / 'slices_latest').exists():
                        return candidate
                raise FileNotFoundError('Could not locate outputs/validation/comparison/slices_latest from current working directory.')

            REPO_ROOT = find_repo_root()
            SLICES_DIR = REPO_ROOT / 'outputs' / 'validation' / 'comparison' / 'slices_latest'

            FILES = {
                'logistic': SLICES_DIR / 'slice_metrics_logistic.csv',
                'regression': SLICES_DIR / 'slice_metrics_regression.csv',
                'best_logistic': SLICES_DIR / 'best_variant_by_slice_logistic.csv',
                'best_regression': SLICES_DIR / 'best_variant_by_slice_regression.csv',
            }
            missing = [str(path) for path in FILES.values() if not path.exists()]
            if missing:
                raise FileNotFoundError('Missing required files:\n' + '\n'.join(missing))

            logistic = pd.read_csv(FILES['logistic'])
            regression = pd.read_csv(FILES['regression'])
            best_logistic = pd.read_csv(FILES['best_logistic'])
            best_regression = pd.read_csv(FILES['best_regression'])

            # Numeric safety.
            for df_, cols in [
                (logistic, ['n', 'roc_auc', 'avg_precision']),
                (best_logistic, ['n', 'roc_auc', 'avg_precision']),
                (regression, ['n', 'r2', 'rmse', 'mae', 'spearman']),
                (best_regression, ['n', 'r2', 'rmse', 'mae', 'spearman']),
            ]:
                for col in cols:
                    if col in df_.columns:
                        df_[col] = pd.to_numeric(df_[col], errors='coerce')

            print(f'Repository root: {REPO_ROOT}')
            print(f'Slices latest directory: {SLICES_DIR}')
            print()
            for name, path in FILES.items():
                print(f'{name:16s}: {path.name}')
            print()
            print(f'Logistic rows: {len(logistic):,}')
            print(f'Regression rows: {len(regression):,}')
            print(f'Best logistic rows: {len(best_logistic):,}')
            print(f'Best regression rows: {len(best_regression):,}')
            '''
        ),
        md(
            """
            ---
            ## 1. Data inventory and coverage
            """
        ),
        code(
            r'''
            def inventory(df: pd.DataFrame, name: str) -> None:
                print(f'=== {name} ===')
                print(f'Variants: {df["variant"].nunique()}')
                print(f'Slice dimensions: {df["slice_col"].nunique()}')
                display(pd.DataFrame({'variant': sorted(df['variant'].dropna().unique())}))
                coverage = (
                    df.groupby('slice_col')
                    .agg(
                        rows=('slice_value', 'size'),
                        slice_values=('slice_value', 'nunique'),
                        variants=('variant', 'nunique'),
                        min_n=('n', 'min'),
                        median_n=('n', 'median'),
                        max_n=('n', 'max'),
                    )
                    .reset_index()
                    .sort_values('slice_col')
                )
                display(coverage)

            inventory(logistic, 'Logistic slice metrics')
            inventory(regression, 'Regression slice metrics')
            '''
        ),
        md(
            """
            ---
            ## 2. Metric distributions across all slices and variants

            This answers: *are the model scores generally strong, weak, or variable across football contexts?*
            """
        ),
        code(
            r'''
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))

            sns.histplot(logistic['roc_auc'].dropna(), bins=30, kde=True, ax=axes[0, 0], color='steelblue')
            axes[0, 0].axvline(0.5, color='red', linestyle='--', label='Random = 0.5')
            axes[0, 0].set_title('Logistic ROC-AUC distribution')
            axes[0, 0].legend()

            sns.histplot(logistic['avg_precision'].dropna(), bins=30, kde=True, ax=axes[0, 1], color='darkorange')
            axes[0, 1].set_title('Logistic average precision distribution')

            sns.histplot(regression['r2'].dropna(), bins=30, kde=True, ax=axes[1, 0], color='seagreen')
            axes[1, 0].axvline(0, color='red', linestyle='--', label='No explanatory power = 0')
            axes[1, 0].set_title('Regression R² distribution')
            axes[1, 0].legend()

            sns.histplot(regression['spearman'].dropna(), bins=30, kde=True, ax=axes[1, 1], color='purple')
            axes[1, 1].axvline(0, color='red', linestyle='--', label='No rank signal = 0')
            axes[1, 1].set_title('Regression Spearman distribution')
            axes[1, 1].legend()

            plt.tight_layout()
            plt.show()

            print('Logistic metric summary')
            display(logistic[['roc_auc', 'avg_precision']].describe().round(4))
            print('Regression metric summary')
            display(regression[['r2', 'rmse', 'mae', 'spearman']].describe().round(4))
            '''
        ),
        md(
            """
            ---
            ## 3. Heatmaps: model quality by slice value

            These heatmaps show every variant against each slice value. Rows are football contexts; columns are model variants.
            """
        ),
        code(
            r'''
            def metric_heatmaps(
                df: pd.DataFrame,
                metric: str,
                title_prefix: str,
                cmap: str,
                higher_is_better: bool = True,
                max_values: int = 45,
            ) -> None:
                for slice_col in sorted(df['slice_col'].unique()):
                    sub = df[df['slice_col'] == slice_col].copy()
                    if sub.empty:
                        continue
                    if sub['slice_value'].nunique() > max_values:
                        order = (
                            sub.groupby('slice_value')[metric]
                            .max()
                            .sort_values(ascending=not higher_is_better)
                            .head(max_values)
                            .index
                        )
                        sub = sub[sub['slice_value'].isin(order)]

                    pivot = sub.pivot_table(index='slice_value', columns='variant', values=metric, aggfunc='mean')
                    sort_score = pivot.max(axis=1) if higher_is_better else pivot.min(axis=1)
                    pivot = pivot.loc[sort_score.sort_values(ascending=not higher_is_better).index]

                    fig_height = max(4.5, min(18, 0.35 * len(pivot)))
                    fig, ax = plt.subplots(figsize=(15, fig_height))
                    sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap, linewidths=0.4, ax=ax)
                    ax.set_title(f'{title_prefix}: {metric} by {slice_col}')
                    ax.set_xlabel('Variant')
                    ax.set_ylabel(slice_col)
                    plt.tight_layout()
                    plt.show()

            metric_heatmaps(logistic, 'roc_auc', 'Logistic shot target', 'YlGnBu', higher_is_better=True)
            metric_heatmaps(logistic, 'avg_precision', 'Logistic shot target', 'YlOrRd', higher_is_better=True)
            metric_heatmaps(regression, 'r2', 'Regression future xG target', 'YlGnBu', higher_is_better=True)
            metric_heatmaps(regression, 'spearman', 'Regression future xG target', 'PuBuGn', higher_is_better=True)
            metric_heatmaps(regression, 'rmse', 'Regression future xG target', 'rocket_r', higher_is_better=False)
            metric_heatmaps(regression, 'mae', 'Regression future xG target', 'rocket_r', higher_is_better=False)
            '''
        ),
        md(
            """
            ---
            ## 4. Best variant by slice

            These are the exported winners for each slice value.
            """
        ),
        code(
            r'''
            print('Best logistic variant by slice value')
            display(best_logistic.sort_values(['slice_col', 'roc_auc'], ascending=[True, False]).reset_index(drop=True).round(4))

            print('Best regression variant by slice value')
            display(best_regression.sort_values(['slice_col', 'r2'], ascending=[True, False]).reset_index(drop=True).round(4))
            '''
        ),
        md(
            """
            ---
            ## 5. Variant win counts

            This answers: *which model keeps winning across football contexts?*
            """
        ),
        code(
            r'''
            log_wins = best_logistic['variant'].value_counts().rename_axis('variant').reset_index(name='wins')
            reg_wins = best_regression['variant'].value_counts().rename_axis('variant').reset_index(name='wins')

            fig, axes = plt.subplots(1, 2, figsize=(18, 7))
            sns.barplot(data=log_wins, y='variant', x='wins', ax=axes[0], color='steelblue')
            axes[0].set_title('Logistic slice wins by variant')
            axes[0].set_xlabel('Slice wins')
            axes[0].set_ylabel('Variant')

            sns.barplot(data=reg_wins, y='variant', x='wins', ax=axes[1], color='seagreen')
            axes[1].set_title('Regression slice wins by variant')
            axes[1].set_xlabel('Slice wins')
            axes[1].set_ylabel('Variant')
            plt.tight_layout()
            plt.show()

            display(log_wins)
            display(reg_wins)

            win_by_dimension_log = pd.crosstab(best_logistic['slice_col'], best_logistic['variant'])
            win_by_dimension_reg = pd.crosstab(best_regression['slice_col'], best_regression['variant'])

            fig, axes = plt.subplots(1, 2, figsize=(18, 7))
            sns.heatmap(win_by_dimension_log, annot=True, fmt='d', cmap='Blues', ax=axes[0])
            axes[0].set_title('Logistic wins by slice dimension')
            sns.heatmap(win_by_dimension_reg, annot=True, fmt='d', cmap='Greens', ax=axes[1])
            axes[1].set_title('Regression wins by slice dimension')
            plt.tight_layout()
            plt.show()
            '''
        ),
        md(
            """
            ---
            ## 6. Improvement over baseline model

            Baseline is `v0_phase_only`. Positive deltas mean a variant beats the phase-only model in that slice.
            """
        ),
        code(
            r'''
            def add_baseline_delta(df: pd.DataFrame, metric: str, baseline_variant: str = 'v0_phase_only') -> pd.DataFrame:
                base = (
                    df[df['variant'] == baseline_variant][['slice_col', 'slice_value', metric]]
                    .rename(columns={metric: f'{metric}_baseline'})
                )
                out = df.merge(base, on=['slice_col', 'slice_value'], how='left')
                out[f'{metric}_delta_vs_baseline'] = out[metric] - out[f'{metric}_baseline']
                return out

            logistic_delta = add_baseline_delta(logistic, 'roc_auc')
            regression_delta = add_baseline_delta(regression, 'r2')

            for data, metric, delta_col, title, cmap in [
                (logistic_delta, 'roc_auc', 'roc_auc_delta_vs_baseline', 'Logistic ROC-AUC delta vs v0_phase_only', 'vlag'),
                (regression_delta, 'r2', 'r2_delta_vs_baseline', 'Regression R² delta vs v0_phase_only', 'vlag'),
            ]:
                for slice_col in sorted(data['slice_col'].unique()):
                    sub = data[(data['slice_col'] == slice_col) & (data['variant'] != 'v0_phase_only')].copy()
                    pivot = sub.pivot_table(index='slice_value', columns='variant', values=delta_col, aggfunc='mean')
                    sort_score = pivot.max(axis=1)
                    pivot = pivot.loc[sort_score.sort_values(ascending=False).index]
                    fig_height = max(4.5, min(18, 0.35 * len(pivot)))
                    fig, ax = plt.subplots(figsize=(15, fig_height))
                    sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap, center=0, linewidths=0.4, ax=ax)
                    ax.set_title(f'{title}: {slice_col}')
                    ax.set_xlabel('Variant')
                    ax.set_ylabel(slice_col)
                    plt.tight_layout()
                    plt.show()

            print('Top logistic improvements over baseline')
            display(logistic_delta.sort_values('roc_auc_delta_vs_baseline', ascending=False).head(25).round(4))
            print('Top regression improvements over baseline')
            display(regression_delta.sort_values('r2_delta_vs_baseline', ascending=False).head(25).round(4))
            '''
        ),
        md(
            """
            ---
            ## 7. Bubble charts: sample size vs model quality

            Large slices are more reliable. Small slices can look extreme because there are fewer actions.
            """
        ),
        code(
            r'''
            fig, axes = plt.subplots(1, 2, figsize=(18, 7))

            sns.scatterplot(
                data=logistic,
                x='n', y='roc_auc', hue='variant', style='slice_col',
                size='avg_precision', sizes=(40, 260), alpha=0.75, ax=axes[0]
            )
            axes[0].axhline(0.5, color='red', linestyle='--', linewidth=1)
            axes[0].set_xscale('log')
            axes[0].set_title('Logistic: slice size vs ROC-AUC')
            axes[0].set_xlabel('Slice size n (log scale)')
            axes[0].set_ylabel('ROC-AUC')

            sns.scatterplot(
                data=regression,
                x='n', y='r2', hue='variant', style='slice_col',
                size='spearman', sizes=(40, 260), alpha=0.75, ax=axes[1]
            )
            axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
            axes[1].set_xscale('log')
            axes[1].set_title('Regression: slice size vs R²')
            axes[1].set_xlabel('Slice size n (log scale)')
            axes[1].set_ylabel('R²')

            for ax in axes:
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
            '''
        ),
        md(
            """
            ---
            ## 8. Best and weakest slices
            """
        ),
        code(
            r'''
            print('Top logistic slices by ROC-AUC')
            display(logistic.sort_values('roc_auc', ascending=False)[['variant', 'slice_col', 'slice_value', 'n', 'roc_auc', 'avg_precision']].head(25).round(4))

            print('Weakest logistic slices by ROC-AUC')
            display(logistic.sort_values('roc_auc', ascending=True)[['variant', 'slice_col', 'slice_value', 'n', 'roc_auc', 'avg_precision']].head(25).round(4))

            print('Top regression slices by R²')
            display(regression.sort_values('r2', ascending=False)[['variant', 'slice_col', 'slice_value', 'n', 'r2', 'rmse', 'mae', 'spearman']].head(25).round(4))

            print('Weakest regression slices by R²')
            display(regression.sort_values('r2', ascending=True)[['variant', 'slice_col', 'slice_value', 'n', 'r2', 'rmse', 'mae', 'spearman']].head(25).round(4))
            '''
        ),
        md(
            """
            ---
            ## 9. Slice dimension summaries

            This answers: *which football dimensions are easiest or hardest for the model portfolio?*
            """
        ),
        code(
            r'''
            log_slice_summary = (
                logistic.groupby(['slice_col', 'variant'])
                .agg(
                    slices=('slice_value', 'nunique'),
                    total_n=('n', 'sum'),
                    mean_roc_auc=('roc_auc', 'mean'),
                    median_roc_auc=('roc_auc', 'median'),
                    mean_avg_precision=('avg_precision', 'mean'),
                )
                .reset_index()
            )

            reg_slice_summary = (
                regression.groupby(['slice_col', 'variant'])
                .agg(
                    slices=('slice_value', 'nunique'),
                    total_n=('n', 'sum'),
                    mean_r2=('r2', 'mean'),
                    median_r2=('r2', 'median'),
                    mean_rmse=('rmse', 'mean'),
                    mean_spearman=('spearman', 'mean'),
                )
                .reset_index()
            )

            print('Logistic summary by slice dimension')
            display(log_slice_summary.sort_values(['slice_col', 'mean_roc_auc'], ascending=[True, False]).round(4))
            print('Regression summary by slice dimension')
            display(reg_slice_summary.sort_values(['slice_col', 'mean_r2'], ascending=[True, False]).round(4))

            fig, axes = plt.subplots(1, 2, figsize=(18, 7))
            sns.barplot(data=log_slice_summary, x='slice_col', y='mean_roc_auc', hue='variant', ax=axes[0])
            axes[0].axhline(0.5, color='red', linestyle='--', linewidth=1)
            axes[0].set_title('Mean logistic ROC-AUC by slice dimension')
            axes[0].tick_params(axis='x', rotation=35)

            sns.barplot(data=reg_slice_summary, x='slice_col', y='mean_r2', hue='variant', ax=axes[1])
            axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
            axes[1].set_title('Mean regression R² by slice dimension')
            axes[1].tick_params(axis='x', rotation=35)
            for ax in axes:
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
            '''
        ),
        md(
            """
            ---
            ## 10. Coach-readable interpretation
            """
        ),
        code(
            r'''
            def coach_logistic_band(roc_auc: float, avg_precision: float) -> str:
                if pd.isna(roc_auc):
                    return 'Not enough positives/negatives'
                if roc_auc >= 0.80:
                    return 'Very readable immediate shot-risk pattern'
                if roc_auc >= 0.70:
                    return 'Readable immediate shot-risk pattern'
                if roc_auc >= 0.60:
                    return 'Some useful shot-risk signal'
                if roc_auc >= 0.52:
                    return 'Weak shot-risk signal; use with video'
                return 'No reliable shot-risk separation'

            def coach_regression_band(r2: float, spearman: float) -> str:
                if pd.isna(r2):
                    return 'Not enough threat signal'
                if r2 >= 0.35 and spearman >= 0.60:
                    return 'Strong threat-value and ranking pattern'
                if r2 >= 0.20 and spearman >= 0.45:
                    return 'Useful threat pattern'
                if r2 >= 0.10:
                    return 'Some threat signal'
                if r2 >= 0:
                    return 'Weak but positive threat signal'
                return 'Threat model struggles in this context'

            best_logistic_coach = best_logistic.copy()
            best_logistic_coach['coach_read'] = best_logistic_coach.apply(lambda r: coach_logistic_band(r['roc_auc'], r['avg_precision']), axis=1)

            best_regression_coach = best_regression.copy()
            best_regression_coach['coach_read'] = best_regression_coach.apply(lambda r: coach_regression_band(r['r2'], r['spearman']), axis=1)

            print('Coach read: best logistic model per slice')
            display(best_logistic_coach[['slice_col', 'slice_value', 'n', 'variant', 'roc_auc', 'avg_precision', 'coach_read']].sort_values(['slice_col', 'roc_auc'], ascending=[True, False]).round(4))

            print('Coach read: best regression model per slice')
            display(best_regression_coach[['slice_col', 'slice_value', 'n', 'variant', 'r2', 'spearman', 'rmse', 'coach_read']].sort_values(['slice_col', 'r2'], ascending=[True, False]).round(4))
            '''
        ),
        md(
            """
            ---
            ## 11. Combined shot-risk + future xG dashboard

            This joins the best logistic and best regression result for each slice value.
            """
        ),
        code(
            r'''
            combined = best_logistic.merge(
                best_regression,
                on=['slice_col', 'slice_value', 'n'],
                how='outer',
                suffixes=('_logistic', '_regression'),
            )

            def combined_read(row):
                shot_good = pd.notna(row.get('roc_auc')) and row.get('roc_auc') >= 0.70
                xt_good = pd.notna(row.get('r2')) and row.get('r2') >= 0.20 and (pd.isna(row.get('spearman')) or row.get('spearman') >= 0.45)
                if shot_good and xt_good:
                    return 'Clear tactical context: both shot risk and threat value are readable'
                if xt_good and not shot_good:
                    return 'Threat builds clearly, but shot conversion is less predictable'
                if shot_good and not xt_good:
                    return 'Shot occurrence is readable, but exact threat value is noisy'
                return 'Limited model signal; use video and tactical context before conclusions'

            combined['coach_summary'] = combined.apply(combined_read, axis=1)
            display(
                combined[[
                    'slice_col', 'slice_value', 'n',
                    'variant_logistic', 'roc_auc', 'avg_precision',
                    'variant_regression', 'r2', 'spearman', 'rmse',
                    'coach_summary',
                ]]
                .sort_values(['slice_col', 'n'], ascending=[True, False])
                .round(4)
            )

            fig, ax = plt.subplots(figsize=(12, 8))
            plot_combined = combined.dropna(subset=['roc_auc', 'r2']).copy()
            sns.scatterplot(
                data=plot_combined,
                x='roc_auc', y='r2', hue='slice_col', size='n', sizes=(60, 350), alpha=0.8, ax=ax
            )
            ax.axvline(0.70, color='steelblue', linestyle='--', linewidth=1, label='ROC-AUC 0.70')
            ax.axhline(0.20, color='seagreen', linestyle='--', linewidth=1, label='R² 0.20')
            ax.set_title('Combined best slice performance: shot-risk readability vs future xG readability')
            ax.set_xlabel('Best logistic ROC-AUC')
            ax.set_ylabel('Best regression R²')
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
            '''
        ),
        md(
            """
            ---
            ## 12. Staff-room summary
            """
        ),
        code(
            r'''
            print('=' * 100)
            print('SLICES LATEST MODEL DASHBOARD SUMMARY')
            print('=' * 100)

            top_log = best_logistic.sort_values('roc_auc', ascending=False).iloc[0]
            top_reg = best_regression.sort_values('r2', ascending=False).iloc[0]
            most_log_wins = best_logistic['variant'].value_counts().idxmax()
            most_reg_wins = best_regression['variant'].value_counts().idxmax()

            print(f'Most frequent shot-risk winner: {most_log_wins}')
            print(f'Most frequent future xG-threat winner: {most_reg_wins}')
            print()
            print('Best immediate shot-risk context:')
            print(f"  {top_log['slice_col']} = {top_log['slice_value']} | {top_log['variant']} | ROC-AUC={top_log['roc_auc']:.3f}, AP={top_log['avg_precision']:.3f}, n={int(top_log['n']):,}")
            print()
            print('Best future xG-threat context:')
            print(f"  {top_reg['slice_col']} = {top_reg['slice_value']} | {top_reg['variant']} | R²={top_reg['r2']:.3f}, Spearman={top_reg['spearman']:.3f}, n={int(top_reg['n']):,}")
            print()
            print('Coach rule:')
            print('  Logistic target tells us whether a defensive context becomes a shot soon.')
            print('  Regression target tells us how dangerous the opponent becomes, shot or no shot.')
            print('  Slices where both are strong are the most actionable coaching contexts.')
            '''
        ),
    ]
    return nb


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    with NOTEBOOK_PATH.open('w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Created {NOTEBOOK_PATH}')
    print(f'Cells: {len(nb.cells)}')


if __name__ == '__main__':
    main()

