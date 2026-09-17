# Player-Grouped Split Check

**Purpose.** Stress-test the active leg's identity-leakage risk on `position`: build a fold structure where no player appears in more than one fold, and compare its balance against the canonical match-grouped folds over the same rows. Scope: active leg only, TRAIN+VAL rows only (the frozen TEST match set from the canonical split is untouched). Written 2026-09-17.

## Result: clean

Zero players appear in more than one fold, across 5 folds covering 964 distinct players in 45,408 TRAIN+VAL rows over 92 matches. `make_grouped_folds(group_col="player_id")` (the existing, generic fold helper in `src/dax/models/splits.py` -- no new fold logic was written) does this by construction; the script verifies it directly rather than trusting that.

## Fold balance: player-grouped vs match-grouped

| Fold | Player-grouped rows | Player-grouped target mean | Match-grouped rows (same pool) | Match-grouped target mean |
|---|---|---|---|---|
| 0 | 9,531 | 7.65% | 8,450 | 8.97% |
| 1 | 10,064 | 8.02% | 10,053 | 8.26% |
| 2 | 9,452 | 7.69% | 8,753 | 7.21% |
| 3 | 8,142 | 8.43% | 9,342 | 8.06% |
| 4 | 8,219 | 7.92% | 8,810 | 7.13% |

Target-rate spread is narrow either way: 7.65-8.43pp (player-grouped) vs 7.13-8.97pp (match-grouped). Neither structure produces a degenerate or wildly skewed fold. Row counts are comparable (~8.1k-10.1k per fold both ways).

Note that every player-grouped fold still touches ~all 92 matches -- that's expected and not a defect: a fold here is defined by a disjoint *set of players*, and any given match has ~20+ different players drawn from both teams, so almost every match has at least one player landing in each fold. This split isn't meant to test match-to-match generalisation (the canonical split already does that); it's meant to test whether a model could be leaning on player identity, and that's what the zero-overlap check above establishes.

## Position balance

| Fold | Max abs. deviation from overall position-share |
|---|---|
| 0 | 5.14pp |
| 1 | 2.77pp |
| 2 | 4.90pp |
| 3 | 3.99pp |
| 4 | 3.14pp |

`position` is the specific leakage vector named in stage 06 planning (a model could partially memorise a player via `position` acting as an identity proxy). Deviation from the dataset-wide position mix stays under 5.2pp in every fold -- modest, not a red flag.

## Verdict

No material identity-leakage risk from player repetition surfaces in this check. The canonical match-grouped split remains the right default for stage 06 training/evaluation. This player-grouped fold structure is now built, verified, and available (`reports/eda/PLAYER_GROUPED_SPLIT_CHECK.json`, reusable via `make_grouped_folds(df, target, group_col="player_id")`) as a specific stress-test tool if stage 06 wants to re-confirm this once real models are in hand -- but nothing here blocks proceeding with the canonical split as-is.

This closes EDA open item #1 (player-grouped CV for the active legs).
