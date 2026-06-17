# Repository Navigation Guide

## Quick Links

### 📈 Modeling & Strategy
- **`docs/BASELINE_MODELING_STRATEGY.md`** - Complete baseline modeling strategy
  - Problem framing & statistical approach
  - Feature architecture (40+ features across 4 groups)
  - Model selection (logistic regression → tree-based)
  - Evaluation strategy (GroupKFold, ROC-AUC, calibration)
  - Implementation roadmap & success criteria

### 📊 Analysis & Insights
- **`docs/analysis/`** - Tactical insights from unidirectional data
  - `ZONE_ANALYSIS.md` - Shot rate by defensive zone
  - `COUNTER_ATTACK_EXPLANATION.md` - Counter-attack mechanics
  - `INSIGHT_VALIDATION.md` - Data-backed tactical principles

### 🔧 System & Architecture
- **`docs/system/`** - Technical system design
  - `PIPELINE_FIX_SUMMARY.md` - Event ordering bug fix (67× improvement)

### 🚀 Scripts & Automation
- **`scripts/pipeline/`** - Core data pipeline (stages 1-3)
- **`scripts/features/`** - Feature engineering (possessions, player actions)
- **`scripts/analysis/`** - Statistical validation (10-module pipeline)
- **`scripts/visualization/`** - Visualization generation
- **`scripts/utils/`** - Utility functions
- **`scripts/run_downstream_pipeline.ps1`** - One-command orchestrator

### 📖 Getting Started

```bash
# Run full downstream pipeline (features + analysis + visualization)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_downstream_pipeline.ps1

# Or run individual components
python scripts/pipeline/pipeline.py                              # Stage 1-3: Core pipeline
python scripts/features/extract_possessions.py                  # Possession features
python scripts/features/build_player_defense_dataset.py         # Player actions
python scripts/analysis/run_player_feature_analysis.py          # Statistical analysis
python scripts/visualization/visualize_possessions.py           # Visualizations
```

## Key Facts

### Unidirectional Normalization
- All attacks normalized to **left → right** (x=0 → x=120)
- Teams attacking toward **opponent's goal at x=120**
- Both defensive and attacking teams use same coordinate space
- Independent of actual pitch orientation in real matches

### Shot Rate Findings
| Zone | Shot Rate | Mechanism |
|------|-----------|-----------|
| x=1-2 (own box) | 19% | Counter-attacks |
| x=10-11 (opp box center) | 19% | Direct attacks |
| x=4-6 (midfield) | 4% | Build-up plays |

### Data Insights
- **Position matters 6× more than just winning the ball**
- High pressure → fast counter-attacks (not sideways possession)
- 75% of own box actions are urgent/reactive
- Counter-attack success: 1 in 5 clearances lead to shots

## Pipeline Status

✅ **GO_BASELINE** - Ready for model development
- 615,225 enriched events
- 59,491 possession sequences (18.20% shot rate)
- 57,637 player defensive actions (7.95% shot rate)
- 1,012 unique players, 115 matches
- All gates passed (analysis decision: GO_BASELINE)

## Files by Purpose

| File | Purpose |
|------|---------|
| `CLEANUP_SUMMARY.md` | This repository organization |
| `README.md` | Project overview |
| `IMPLEMENTATION.md` | Development phases |
| `PIPELINE_STATUS.md` | Current pipeline status |
| `docs/BASELINE_MODELING_STRATEGY.md` | Complete modeling strategy & roadmap |
| `docs/player_defense_model.md` | Player-level feature design |
| `docs/project_scope.md` | Overall DAx vision and scope |
| `docs/system/PIPELINE_FIX_SUMMARY.md` | Technical bug fixes |
| `docs/analysis/*` | Tactical insights |

---

**Last Updated**: June 10, 2026
**Status**: ✅ Repository organized & ready for development

