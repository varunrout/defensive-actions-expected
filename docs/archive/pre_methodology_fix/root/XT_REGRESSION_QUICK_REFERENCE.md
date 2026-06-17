# xT vs Binary Target: Quick Reference Card

---

## Side-by-Side Comparison

| Feature | `target_shot_in_10s` (Binary) | `xT_from_generic_grid` (Continuous) |
|---------|------|---------|
| **Type** | 0 = No shot, 1 = Shot | 0.0–1.0 = threat score |
| **Sparsity** | 94% negative (5.7% shot rate) | 100% positive (all have signal) |
| **Model Type** | Logistic Regression | Ridge/Lasso Regression |
| **Output** | Probability of shot | Expected threat value |
| **Loss Function** | Binary Cross-Entropy | Mean Squared Error |
| **Primary Metric** | ROC-AUC (≥0.70 target) | R² (≥0.25 target) |
| **Secondary Metrics** | Average Precision, Calibration | RMSE, MAE, Spearman ρ |
| **Interpretation** | "Shot probability reduced by X%" | "Threat reduced by X units" |
| **Training Signal** | Sparse (only shots matter) | Dense (all actions guide learning) |
| **Convergence** | Slower (class imbalance) | Faster (balanced continuous) |
| **Ground Truth** | Yes (actual outcome) | No (proxy model) |
| **Spatial Bias** | Implicit (through features) | Explicit (grid-based) |
| **Circular Dependency** | No | Yes* (xT trained on binary target) |
| **Feature Importance** | Logistic coefs | Ridge coefs |
| **Suitable For** | Direct attribution | Pattern discovery |

*xT is trained on `target_shot_in_10s`, so using xT as target creates dependency

---

## Expected Performance

### Binary Classification (Current)
```
Train & Validation:
├─ v0_phase_only:   AUC ≈ 0.60-0.62
├─ v1_spatial:      AUC ≈ 0.65-0.68
└─ v2_full_base:    AUC ≈ 0.70-0.73

Positive Insights:
├─ Phase ordering validated (press > block)
├─ Location matters (closer to goal = higher threat)
├─ Support balance affects predictions
└─ Position-specific patterns emerge
```

### xT Regression (Proposed)
```
Train & Validation:
├─ v0_phase_only:   R² ≈ 0.15-0.18
├─ v1_spatial:      R² ≈ 0.20-0.25
└─ v2_full_base:    R² ≈ 0.25-0.32

Positive Insights:
├─ Denser learning signal (fewer zeros)
├─ Faster model convergence
├─ Captures threat magnitude
└─ All actions informative
```

---

## When to Use Which

### ✅ Use Binary `target_shot_in_10s` If:
- You want **ground truth** feedback (actual outcomes matter)
- You need **clear interpretability** (probability is intuitive)
- You're computing **player DAx scores** (need actual shot attribution)
- Stakeholders want **"shot probability reduced by X%"** narrative
- You have **domain expertise** in imbalanced learning

### ✅ Use xT Regression If:
- You want **dense learning signal** (all actions matter equally)
- You prefer **faster convergence** (no class imbalance)
- You're building **threat landscape maps** (spatial patterns)
- You want **feature engineering insights** (what really matters?)
- You're okay with **proxy metrics** (xT is model-based, not truth)

### ✅ Use Both (Dual-Track) If:
- You have **time and resources** (2× training)
- You want to **compare approaches** (A/B testing)
- You need **ensemble predictions** (blend both signals)
- You want **robust validation** (cross-check both methods)

---

## Implementation Paths

### 🟢 Path A: Quick Switch to xT (1-2 hours)
```
1. Run: python scripts/models/train_baseline_regression.py
   ↓
2. Check: outputs/validation/regression/regression_model_metrics.json
   ↓
3. Done! Compare R² and RMSE to classify performance
```

### 🟡 Path B: Careful Comparison (4-6 hours)
```
1. Run: python scripts/models/train_baseline_logistic.py
   ↓
2. Run: python scripts/models/train_baseline_regression.py
   ↓
3. Run: python scripts/models/compare_classification_vs_regression.py
   ↓
4. Analyze: outputs/validation/comparison/
   ↓
5. Decide: Which approach is better for your use case?
```

### 🟣 Path C: Ensemble Both (8-12 hours)
```
1. Train both models
   ↓
2. Compare predictions (Spearman ρ > 0.80?)
   ↓
3. Create hybrid score: DAx = α·P(shot) + (1-α)·xT
   ↓
4. Optimize α on holdout set
   ↓
5. Deploy ensemble
```

---

## Decision Matrix

```
┌─ Are predictions well-correlated? (ρ > 0.80)
│  ├─ YES → Approaches agree, either works
│  └─ NO  → Only one captures real patterns
│
├─ Which has better held-out AUC/R²?
│  ├─ Binary (AUC > R²) → Use classification
│  ├─ xT (R² > AUC) → Use regression
│  └─ Similar → Choose based on interpretability
│
├─ Which features rank higher in importance?
│  ├─ Location features → xT may be better (spatial emphasis)
│  ├─ Phase features → Binary may be better (temporal emphasis)
│  └─ Balanced → Could use ensemble
│
└─ Final decision:
   ├─ For player attribution → Use binary
   ├─ For threat mapping → Use regression
   └─ For both → Use ensemble
```

---

## Code Snippet: Switch Your Training

from `train_baseline_logistic.py` → `train_baseline_regression.py`

**What changes:**

```python
# BEFORE (Binary Classification)
from dax.models.baseline_logistic import (
    TARGET_COL,  # "target_shot_in_10s"
    LogisticRegression,
    roc_auc_score,
    average_precision_score,
)

# AFTER (xT Regression)
from dax.models.baseline_regression import (
    TARGET_COL,  # "xT_from_generic_grid"
    Ridge,  # Instead of LogisticRegression
    r2_score,
    mean_squared_error,
)
```

---

## Expected Output Files

### Binary Classification Outputs
```
outputs/
├── models/baseline/
│   ├── logistic_v0_phase_only.joblib
│   ├── logistic_v1_spatial.joblib
│   └── logistic_v2_full_baseline.joblib
├── validation/baseline/
│   ├── baseline_model_metrics.json       (AUC, AP scores)
│   ├── baseline_roc_curves.png
│   ├── baseline_pr_curves.png
│   └── baseline_feature_importance.png
└── oof/baseline/
    └── baseline_oof_predictions.parquet
```

### xT Regression Outputs (New)
```
outputs/
├── models/regression/
│   ├── regression_v0_phase_only.joblib
│   ├── regression_v1_spatial.joblib
│   └── regression_v2_full_baseline.joblib
├── validation/regression/
│   ├── regression_model_metrics.json     (R², RMSE, MAE scores)
│   ├── regression_v0_phase_only_coefficients.csv
│   ├── regression_v1_spatial_fold_metrics.csv
│   └── regression_v2_full_baseline_fold_metrics.csv
└── oof/regression/
    └── regression_oof_predictions.parquet
```

### Comparison Output (New)
```
outputs/
└── validation/comparison/
    └── comparison_v2_full_baseline.json  (correlation analysis)
```

---

## Metrics Interpretation Guide

### Binary Classification Metrics
```
ROC-AUC:
├─ 0.50 = Random (coin flip)
├─ 0.65 = Acceptable 
├─ 0.70 = Good (our target)
└─ 0.80+ = Excellent

Average Precision:
├─ 0.057 = Baseline (5.7% positive rate)
├─ 0.15 = Acceptable (2.6× baseline)
├─ 0.20 = Good (3.5× baseline)
└─ 0.25+ = Excellent
```

### Regression Metrics
```
R² (Coefficient of Determination):
├─ 0.00 = Predicts as well as mean
├─ 0.15 = Explains 15% of variance (acceptable)
├─ 0.25 = Explains 25% of variance (good)
└─ 0.40+ = Explains 40% of variance (excellent)

RMSE (Root Mean Squared Error):
├─ Range: [0, 1] (xT is 0-1)
├─ Typical: 0.10-0.15
├─ Lower is better
└─ Interpretation: ±0.12 typical error

MAE (Mean Absolute Error):
├─ Range: [0, 1]
├─ Typical: 0.06-0.10
├─ Lower is better
└─ Interpretation: ~0.08 average deviation

Spearman ρ (Rank Correlation):
├─ 0.30 = Acceptable
├─ 0.40 = Good
├─ 0.50+ = Strong
└─ Compares ranking, not absolute values
```

---

## Typical Questions

**Q: Should I replace the binary model or run both?**
A: **Run both initially**, then decide. The comparison will show which is better for your use case.

**Q: Will xT regression converge faster?**
A: Yes, typically 20-40% faster due to denser target signal.

**Q: Do the predictions correlate?**
A: Moderately (ρ ≈ 0.35-0.45 expected). They capture different aspects.

**Q: Which is better for player DAx attribution?**
A: Binary (ground truth). Regression is better for threat landscapes.

**Q: Can I use both together?**
A: Yes! Ensemble score: `DAx = 0.6·P(shot) + 0.4·E[xT]`

**Q: What if regression R² seems low (0.15)?**
A: That's normal! Defensive actions have inherent noise. 15-30% explained is solid.

---

## One More Thing

**Why not just use xT directly (without regression)?**

Because xT is already a probabilistic score (grid cell shot rate). Using it:
1. ✅ Gives you direct threat assessment
2. ❌ Throws away information about how features affect it
3. ❌ Doesn't learn phase, location, support interactions

**Regression lets you:**
- Learn which features matter most
- Create interactions (phase × location)
- Build feature importance rankings
- Understand threat drivers

---

## Files to Read First

1. **`SWITCH_TO_XT_SUMMARY.md`** (this file) — Quick reference
2. **`docs/analysis/XT_BASED_TARGET_ANALYSIS.md`** — Detailed analysis
3. **`src/dax/models/baseline_regression.py`** — Code module (if interested)

---

**Ready?** Start with:
```bash
python scripts/models/train_baseline_regression.py --max-rows 5000
```

Then review the metrics JSON to see if results make sense!

