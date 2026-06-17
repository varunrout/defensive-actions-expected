# Analysis Documentation

In-depth analysis of unidirectional coordinate normalization and data insights.

## Documents

- **ZONE_ANALYSIS.md** - Zone-by-zone shot rate breakdown
  - 12×8 grid analysis with club examples
  - Shot probability by defensive zone (x, y coordinates)
  - Tactical interpretation of zone patterns

- **COUNTER_ATTACK_EXPLANATION.md** - Counter-attack mechanics
  - Why x=1-2 (own box) zones show 17-19% shot rates
  - Transition sequences from defense to attack
  - Unidirectional verification

- **INSIGHT_VALIDATION.md** - Key tactical insights validated with data
  - High pressure → fast counter-attacks (not build-up)
  - Reaction-based decisions in own box vs strategic play in midfield
  - Data-backed tactical principles

- **MODEL_PORTFOLIO_REPORT.md** - Full model comparison and selection document
  - OOF leaderboard across logistic and regression variants (`v0`-`v8`)
  - Slice-analysis summary, cross-task alignment, and fold stability
  - Feature clustering, interpretability, and deployment recommendations

- **DEFENSIBILITY_ADDENDUM.md** - Additional evidence for model defensibility
  - Bootstrap confidence intervals and winner-frequency stability checks
  - Calibration reliability and ECE diagnostics for top classifiers
  - Regression decile-bias diagnostics and `v4-v3` slice stress-test gaps
  - Concrete recommendations for remaining risk-reduction work

## Key Findings

1. **Position > Possession**: Zone 10-11 (attacking) is 5-6× more dangerous than Zone 0-4 (defensive)
2. **Transition Speed Matters**: Counter-attacks from x=1-2 lead to shots 19% of the time
3. **Opponent Pressure Context**: 75% of defensive actions in own box are urgent/reactive
4. **Tactical Response**: Teams counter from defense (fast) vs build from midfield (strategic)
