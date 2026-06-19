from __future__ import annotations
import pandas as pd


def label_box_defence(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    fam = out.get('action_family', pd.Series('', index=out.index)).astype(str).str.lower()
    typ = out.get('event_type', out.get('type', pd.Series('', index=out.index))).astype(str).str.lower()
    possession_secured = out.get('coach_possession_retained', pd.Series(False, index=out.index)).fillna(False).eq(True)
    attack_recycled = out.get('coach_attack_recycled', pd.Series(False, index=out.index)).fillna(False).eq(True)
    rebound_danger = out.get('coach_block_followed_by_rebound', pd.Series(False, index=out.index)).fillna(False).eq(True)
    pressure_progression = out.get('coach_pressure_followed_by_progression', pd.Series(False, index=out.index)).fillna(False).eq(True)
    xg_suppression = out.get('coach_xg_suppression_r4', pd.Series(0.0, index=out.index)).fillna(0.0)
    high_threat_cutoff = float(out.get('coach_expected_shot_probability', pd.Series(0.0, index=out.index)).quantile(0.75))

    out['coach_tactical_label'] = 'video_review_required'
    out.loc[possession_secured & ~attack_recycled, 'coach_tactical_label'] = 'controlled box defence'
    out.loc[typ.str.contains('clearance') & (possession_secured | ~attack_recycled), 'coach_tactical_label'] = 'clearance relief'
    out.loc[typ.str.contains('clearance') & attack_recycled, 'coach_tactical_label'] = 'clearance recycled'
    out.loc[typ.str.contains('block') & rebound_danger, 'coach_tactical_label'] = 'block with rebound danger'
    out.loc[fam.str.contains('pressure') & ~pressure_progression & (xg_suppression > 0.01), 'coach_tactical_label'] = 'pressure suppresses threat'
    out.loc[
        fam.str.contains('pressure')
        & pressure_progression
        & out.get('coach_expected_shot_probability', pd.Series(0.0, index=out.index)).ge(high_threat_cutoff),
        'coach_tactical_label',
    ] = 'pressure moves danger'
    out.loc[(out.get('coach_xg_suppression_r4', pd.Series(0.0, index=out.index)) < -0.04) & out.get('coach_shot_followed', False), 'coach_tactical_label'] = (
        'defensive breakdown (video review category)'
    )
    return out


def label_rules() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'label': [
                'controlled box defence',
                'clearance relief',
                'clearance recycled',
                'block with rebound danger',
                'pressure suppresses threat',
                'pressure moves danger',
                'defensive breakdown (video review category)',
                'video_review_required',
            ],
            'rule': [
                'defensive team secures possession and no immediate recycled danger',
                'clearance followed by secure possession or no opposition recycle inside window',
                'clearance followed by quick opposition continuation inside recycle window',
                'block followed by opposition rebound or shot in immediate window',
                'pressure followed by non-progression and positive xG suppression',
                'pressure followed by progression and high expected threat continuation',
                'observed danger materially above expected and shot continuation',
                'no observable rule met with available event data',
            ],
        }
    )


def label_rule_samples(df: pd.DataFrame) -> pd.DataFrame:
    labeled = label_box_defence(df)
    counts = labeled['coach_tactical_label'].value_counts(dropna=False).rename_axis('label').reset_index(name='sample_size')
    return label_rules().merge(counts, on='label', how='left').fillna({'sample_size': 0})
