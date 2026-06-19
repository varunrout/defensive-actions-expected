from __future__ import annotations
import pandas as pd
from .metrics import add_suppression

REVIEW_COLUMNS = [
    'match_id',
    'event_id',
    'team',
    'player',
    'action_family',
    'event_type',
    'coach_expected_shot_probability',
    'coach_observed_future_shot',
    'coach_expected_future_xg_r4',
    'coach_expected_future_xg_two_part',
    'coach_observed_future_xg',
    'coach_shot_suppression',
    'coach_xg_suppression_r4',
    'coach_xg_suppression_two_part',
    'coach_reliable_visibility',
    'reason_selected_for_review',
]


def _pick_with_match_diversity(df: pd.DataFrame, score_col: str, n: int) -> pd.DataFrame:
    if df.empty:
        return df
    if 'match_id' not in df.columns:
        return df.nlargest(n, score_col)
    top = df.sort_values(score_col, ascending=False)
    picks = []
    seen_matches: set[object] = set()
    for _, row in top.iterrows():
        match_id = row['match_id']
        if match_id in seen_matches and len(picks) < max(1, n // 2):
            continue
        picks.append(row)
        seen_matches.add(match_id)
        if len(picks) >= n:
            break
    if len(picks) < n:
        remaining = top[~top['event_id'].isin([pick['event_id'] for pick in picks])].head(n - len(picks))
        picks.extend(list(remaining.to_dict(orient='records')))
    return pd.DataFrame(picks)


def select_representative_events(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    out = add_suppression(df)
    categories = {
        'high expected threat, no shot': out['coach_expected_shot_probability'].ge(out['coach_expected_shot_probability'].quantile(0.9))
        & out['coach_observed_future_shot'].eq(0),
        'low expected threat, shot': out['coach_expected_shot_probability'].le(out['coach_expected_shot_probability'].quantile(0.1))
        & out['coach_observed_future_shot'].eq(1),
        'high expected xg, danger suppressed': out['coach_expected_future_xg_r4'].ge(out['coach_expected_future_xg_r4'].quantile(0.9))
        & out['coach_xg_suppression_r4'].gt(0),
        'observed xg materially above expected': out['coach_xg_suppression_r4'].lt(-0.03),
        'clearance recycled': out.get('coach_clearance_followed_by_opposition_recovery', pd.Series(False, index=out.index)).eq(True),
        'block rebound danger': out.get('coach_block_followed_by_rebound', pd.Series(False, index=out.index)).eq(True),
        'pressure moves danger': out.get('coach_pressure_followed_by_progression', pd.Series(False, index=out.index)).eq(True),
        'transition recovery immediately lost': out.get('coach_recovery_followed_by_immediate_turnover', pd.Series(False, index=out.index)).eq(True),
        'repeated emergency box actions': out.get('coach_is_repeated_defensive_action', pd.Series(False, index=out.index)).eq(True),
    }

    selected = []
    for reason, mask in categories.items():
        subset = out.loc[mask].copy()
        if subset.empty:
            continue
        score_col = 'coach_expected_future_xg_r4'
        picked = _pick_with_match_diversity(subset, score_col=score_col, n=n)
        if picked.empty:
            continue
        picked['reason_selected_for_review'] = reason
        selected.append(picked)
    if not selected:
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    final = pd.concat(selected, ignore_index=True)
    final = final.drop_duplicates(['match_id', 'event_id', 'reason_selected_for_review'])
    for column in REVIEW_COLUMNS:
        if column not in final.columns:
            final[column] = pd.NA
    return final[REVIEW_COLUMNS]
