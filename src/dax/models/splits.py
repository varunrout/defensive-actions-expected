from __future__ import annotations
import json
import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

# The canonical, frozen match-grouped split (prompt 13) -- computed once by
# scripts/compute_canonical_split.py, never regenerated per training run.
# Both the active and passive legs read match_id -> split label from this
# single file, so a match held out as TEST (or assigned to a given fold) is
# the same match for both legs -- required for comparing active-model vs
# passive-model performance on identical held-out matches.
_REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_SPLIT_PATH = _REPO_ROOT / "outputs" / "models" / "splits" / "match_assignment.json"
_FOLD_LABEL_RE = re.compile(r"^fold(\d+)$")


def load_canonical_split(path: str | Path = CANONICAL_SPLIT_PATH) -> dict[str, str]:
    """Load the frozen match_id -> "test" | "fold0".."fold4" assignment.

    Raises FileNotFoundError with a clear message if the canonical split
    hasn't been computed yet (run scripts/compute_canonical_split.py) --
    never silently falls back to a fresh random split, since that would
    defeat the point of freezing one.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical match split not found at {path}. Run scripts/compute_canonical_split.py once to "
            "generate it -- this loader deliberately does not fall back to computing a fresh split, since "
            "that would silently break comparability between runs and between the active and passive legs."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_test_mask(df: pd.DataFrame, group_col: str = "match_id", assignment: dict[str, str] | None = None) -> pd.Series:
    """Boolean mask: True for rows whose match_id is held out as TEST in the canonical split."""
    assignment = assignment if assignment is not None else load_canonical_split()
    return df[group_col].astype(str).map(assignment).eq("test")


def canonical_grouped_folds(df: pd.DataFrame, group_col: str = "match_id", assignment: dict[str, str] | None = None) -> pd.DataFrame:
    """Canonical-split equivalent of make_grouped_folds' output shape
    (group_col, fold, row_index), restricted to TRAIN+VAL rows -- TEST rows
    are excluded here (they are never a CV fold; evaluate on them
    separately, once, after model selection).

    Raises ValueError if any match_id in df isn't covered by the canonical
    assignment (e.g. a unit-test fixture with synthetic match ids) -- callers
    should catch this and fall back to make_grouped_folds for that case,
    rather than getting a silently partial split.
    """
    assignment = assignment if assignment is not None else load_canonical_split()
    match_ids = df[group_col].astype(str)
    uncovered = set(match_ids.unique()) - set(assignment.keys())
    if uncovered:
        raise ValueError(
            f"{len(uncovered)} match_id(s) in this dataframe are not covered by the canonical split "
            f"({sorted(uncovered)[:5]}{'...' if len(uncovered) > 5 else ''}) -- this looks like a fixture or "
            "a dataset built after the canonical split was frozen. Use make_grouped_folds() for this case."
        )

    labels = match_ids.map(assignment)
    is_test = labels.eq("test")
    fold_labels = labels[~is_test]
    fold_numbers = fold_labels.map(lambda label: int(_FOLD_LABEL_RE.match(label).group(1)))

    out = df.loc[~is_test, [group_col]].copy()
    out["fold"] = fold_numbers.astype(int)
    out["row_index"] = out.index
    return out


def make_grouped_folds(df:pd.DataFrame,target:str,group_col:str='match_id',n_splits:int=5,seed:int=42)->pd.DataFrame:
    groups=df[group_col]; n_groups=groups.nunique()
    if n_groups<2: raise ValueError('Grouped CV requires at least two groups.')
    k=min(n_splits,int(n_groups))
    y=df[target]
    use_strat=target.endswith('shot_10s') and y.nunique()==2 and int(y.sum())>=k
    splitter=StratifiedGroupKFold(k,shuffle=True,random_state=seed) if use_strat else GroupKFold(k)
    folds=pd.Series(index=df.index,dtype='int64')
    X=df[[group_col]]
    for fold,(_,val) in enumerate(splitter.split(X,y if use_strat else None,groups)):
        folds.iloc[val]=fold
    out=df[[group_col]].copy(); out['fold']=folds.astype(int); out['row_index']=df.index
    return out

def fold_summary(df,target,folds,group_col='match_id'):
    x=df.join(folds['fold']) if 'fold' not in df else df
    return x.groupby('fold').agg(rows=(target,'size'),matches=(group_col,'nunique'),target_mean=(target,'mean'),positive_shots=('target_future_shot_10s','sum'),nonzero_xg=('target_future_xg_10s',lambda s:int((s>0).sum()))).reset_index()
