# Documentation Index

This folder contains all strategic and technical documentation for the Defensive Actions Expected (DAx) project.

---

## 📈 Modeling Documentation (Start Here)

### Core Strategy Documents
| Document | Purpose | Length |
|----------|---------|--------|
| **[BASELINE_MODELING_STRATEGY.md](BASELINE_MODELING_STRATEGY.md)** | Complete baseline modeling strategy | 10 pages |
| **[MODELING_QUICK_REFERENCE.md](MODELING_QUICK_REFERENCE.md)** | 1-page quick reference | 1 page |
| **[MODELING_PROGRESS.md](MODELING_PROGRESS.md)** | Implementation progress tracker | Checklist |
| **[analysis/MODEL_PORTFOLIO_REPORT.md](analysis/MODEL_PORTFOLIO_REPORT.md)** | Extensive comparison of trained models (`v0`-`v8`) | Detailed report |

### What to Read First
1. **Quick overview?** → Start with `MODELING_QUICK_REFERENCE.md`
2. **Full strategy?** → Read `BASELINE_MODELING_STRATEGY.md`
3. **Ready to implement?** → Follow `MODELING_PROGRESS.md` checklist

---

## 📋 Project Scope & Planning

| Document | Purpose |
|----------|---------|
| [project_scope.md](project_scope.md) | Overall DAx vision, objectives, and phases (1-13) |
| [implementation_plan.md](implementation_plan.md) | Current status and next implementation steps |
| [player_defense_model.md](player_defense_model.md) | Player-level feature design and model rationale |

---

## 📊 Analysis & Insights

| Document | Purpose |
|----------|---------|
| [analysis/ZONE_ANALYSIS.md](analysis/ZONE_ANALYSIS.md) | Shot rates by defensive zone |
| [analysis/COUNTER_ATTACK_EXPLANATION.md](analysis/COUNTER_ATTACK_EXPLANATION.md) | Counter-attack mechanics explained |
| [analysis/INSIGHT_VALIDATION.md](analysis/INSIGHT_VALIDATION.md) | Data-backed tactical principles |
| [analysis/MODEL_PORTFOLIO_REPORT.md](analysis/MODEL_PORTFOLIO_REPORT.md) | Full modeling comparison, slice analysis, and selection guidance |
| [analysis/README.md](analysis/README.md) | Analysis folder index |

---

## 🔧 System & Technical

| Document | Purpose |
|----------|---------|
| [system/REPO_NAVIGATION.md](system/REPO_NAVIGATION.md) | Repository navigation guide |
| [system/PIPELINE_FIX_SUMMARY.md](system/PIPELINE_FIX_SUMMARY.md) | Event ordering bug fix (67× improvement) |
| [system/CLEANUP_SUMMARY.md](system/CLEANUP_SUMMARY.md) | Repository organization history |
| [system/README.md](system/README.md) | System folder index |

---

## 📓 Notebook & Data Quality

| Document | Purpose |
|----------|---------|
| [notebook_findings_summary.md](notebook_findings_summary.md) | Notebook validation results (June 9, 2026) |
| [notebook_remediation_plan.md](notebook_remediation_plan.md) | Data quality remediation plan |

---

## 🏟️ Domain Knowledge

| Document | Purpose |
|----------|---------|
| [possession_sequences.md](possession_sequences.md) | Possession-level features documentation |
| [possession_visualization.md](possession_visualization.md) | Possession visualization guide |

---

## Quick Links by Role

### Data Scientist / ML Engineer
1. [BASELINE_MODELING_STRATEGY.md](BASELINE_MODELING_STRATEGY.md) — Full modeling approach
2. [MODELING_PROGRESS.md](MODELING_PROGRESS.md) — Implementation checklist
3. [system/REPO_NAVIGATION.md](system/REPO_NAVIGATION.md) — Code navigation

### Football Analyst
1. [project_scope.md](project_scope.md) — What is DAx?
2. [player_defense_model.md](player_defense_model.md) — How defensive actions are measured
3. [analysis/INSIGHT_VALIDATION.md](analysis/INSIGHT_VALIDATION.md) — Tactical insights

### Project Manager / Stakeholder
1. [MODELING_QUICK_REFERENCE.md](MODELING_QUICK_REFERENCE.md) — 1-page overview
2. [implementation_plan.md](implementation_plan.md) — Current status
3. [MODELING_PROGRESS.md](MODELING_PROGRESS.md) — Task progress

---

## Document Structure

```
docs/
├── README.md (this file)                      # Documentation index
├── BASELINE_MODELING_STRATEGY.md              # ⭐ Main modeling strategy
├── MODELING_QUICK_REFERENCE.md                # ⭐ 1-page summary
├── MODELING_PROGRESS.md                       # ⭐ Progress tracker
├── project_scope.md                           # Project vision
├── implementation_plan.md                     # Implementation status
├── player_defense_model.md                    # Feature design
├── possession_sequences.md                    # Possession features
├── possession_visualization.md                # Visualization guide
├── notebook_findings_summary.md               # Data quality
├── notebook_remediation_plan.md               # QA plan
├── analysis/                                  # Tactical insights
│   ├── README.md
│   ├── ZONE_ANALYSIS.md
│   ├── COUNTER_ATTACK_EXPLANATION.md
│   ├── INSIGHT_VALIDATION.md
│   └── MODEL_PORTFOLIO_REPORT.md
└── system/                                    # Technical docs
    ├── README.md
    ├── REPO_NAVIGATION.md
    ├── PIPELINE_FIX_SUMMARY.md
    └── CLEANUP_SUMMARY.md
```

---

**Last Updated:** June 10, 2026  
**Status:** Baseline modeling strategy complete, ready for implementation

