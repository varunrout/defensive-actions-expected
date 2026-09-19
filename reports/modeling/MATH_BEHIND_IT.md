# The Math Behind All Of It

## 0. How to read this document

Every other document under `reports/modeling/` is a **decision record**: what was tried, what won,
why, for people who already know the math and just want the evidence. This document is different.
It is a **teaching document**. It exists so that the math behind every model actually built in this
project -- both completed legs, active-binary and passive-binary -- can be understood from first
principles, not just used.

Two rules govern everything below:

1. **Full derivation, not just formulas.** Every equation is derived from something simpler and
   more obviously true, step by step. Nothing is stated as "here's the formula, trust it."
2. **Explained for a bright 10-11-year-old.** Every derivation starts with a plain-language
   picture of the idea *before* any symbols appear, and every symbol is explained in words the
   first time it shows up. This does not mean skipping steps -- it means no step is assumed
   obvious. Rigor and readability are not in tension here; a step that feels "too small to
   explain" is exactly the step most worth explaining.

**Every worked example below uses a real row from the real held-out test set**, scored by the real
fitted model saved on disk in this repository -- never an invented illustration. Each hand-walked
computation was independently verified by a script
(`scripts/analysis/generate_math_behind_it_examples.py`) that recomputes the same number from the
real model artifact and checks it matches to floating-point precision. The full verification output
is saved at `outputs/models/validation/math_behind_it_worked_examples_verification.json` -- every
number quoted in this document can be traced back to that file, and every one of them matched on
the first run (no example was rounded or adjusted to look right).

**Current promotion status, checked before writing this document, not assumed:**

- **Active-binary**: `v1e_gradient_boosting_calibrated` was promoted to standing reference model in
  Prompt 46 (confirmed against `ACTIVE_BINARY_MODELLING_CLOSEOUT.md`'s own status banner). Section
  1 below covers the full ladder that led there.
- **Passive-binary**: `p1e_gradient_boosting_calibrated` was promoted to standing reference model
  in Prompt 52 (confirmed against `PASSIVE_BINARY_MODEL_LADDER.md` section 7 -- promoted with one
  explicit caveat, the `defender_functional_role="unclassified"` slice). Section 2 below covers
  that leg's full ladder.
- **Active-continuous (xG regression) and passive-continuous**: not yet built. Section 3 is a short
  placeholder, not a preview -- no math is speculated for models that do not exist yet.

---

## 1. Active-binary leg: from `v1_unweighted` to `v1e_gradient_boosting_calibrated`

### 1.1 Logistic regression from first principles (`v1_unweighted`)

**The problem, in plain words.** After a defender does something -- a tackle, an interception, a
block -- we want to ask: "how likely is it that the attacking team gets a shot off in the next 10
seconds?" Not a yes/no guess. A number between 0 and 1, like a weather forecast says "70% chance of
rain" instead of just "it will rain."

**Why not just use a straight line?** The simplest possible model would be: take some numbers that
describe the situation (how many defenders are nearby, how central the ball is, etc.), multiply
each by a weight, add them up, and call that the probability. The trouble is a straight line has no
ceiling or floor -- multiply the weights by big enough numbers and the "probability" comes out as
4.7 or -2.3, which is nonsense. A probability has to stay between 0 and 1, always. We need a
function that takes *any* number (however big or small) and squeezes it into that range, without
ever quite touching 0 or 1.

**Log-odds: the bridge between "any number" and "a probability."** Here's the trick. Instead of
predicting the probability `p` directly, predict something called the **log-odds** of `p`.
"Odds" is a familiar idea from betting: if something has probability `p` of happening, its odds are
`p / (1-p)` -- "how many times more likely it is to happen than not." Odds range from 0 (impossible)
to infinity (certain), which is already better than a raw probability (bounded above at 1), but
still not symmetric or well-behaved for arithmetic. Taking the **logarithm** of the odds fixes that:
`log(odds)` ranges over *all* real numbers, from -infinity (impossible) to +infinity (certain),
with 0 meaning "50-50." That is exactly the unrestricted range a weighted sum of features naturally
produces. So the model computes:

```
z = intercept + coef_1 * feature_1 + coef_2 * feature_2 + ... + coef_n * feature_n
```

and treats `z` as the log-odds: `z = log( p / (1-p) )`.

**Deriving the sigmoid function.** Given `z = log(p / (1-p))`, we want to go the other way: given
`z`, what is `p`? This is pure algebra, one step at a time.

```
z = log( p / (1-p) )
exp(z) = p / (1-p)                          [undo the log by exponentiating both sides]
exp(z) * (1-p) = p                          [multiply both sides by (1-p)]
exp(z) - exp(z)*p = p                       [distribute]
exp(z) = p + exp(z)*p                       [move the p-terms together]
exp(z) = p * (1 + exp(z))                   [factor out p]
p = exp(z) / (1 + exp(z))                   [divide]
```

Divide the top and bottom of that last fraction by `exp(z)`:

```
p = 1 / (1 + exp(-z))
```

This is the **sigmoid function**, usually written `sigmoid(z) = 1 / (1 + exp(-z))`. Two things fall
straight out of this algebra, not just eyeballing a graph:

- **It's always between 0 and 1.** `exp(-z)` is always a positive number (any real number raised as
  a power of `e` is positive), so `1 + exp(-z)` is always bigger than 1, so `1 / (1 + exp(-z))` is
  always between 0 and 1, no matter what `z` is.
- **It's symmetric around `z=0`, `p=0.5`.** When `z=0`, `exp(-0)=1`, so `p = 1/(1+1) = 0.5`. When
  `z` is a large positive number, `exp(-z)` shrinks toward 0, so `p` climbs toward 1. When `z` is a
  large negative number, `exp(-z)` blows up, so `p` shrinks toward 0.

**Deriving log-loss from maximum likelihood.** Now: how does the model *learn* the right
coefficients? Every row in the training data has an actual outcome, `y`, which is either 1 (a shot
happened) or 0 (it didn't). This is called a **Bernoulli trial** -- a single yes/no event with some
probability `p` of "yes." The probability of what actually happened, for one row, can be written as
a single formula that works for both cases:

```
P(y | p) = p^y * (1-p)^(1-y)
```

Check this makes sense: if `y=1`, this becomes `p^1 * (1-p)^0 = p` (the probability of "yes"). If
`y=0`, this becomes `p^0 * (1-p)^1 = 1-p` (the probability of "no"). One formula, both cases.

If we assume every row's outcome is independent of every other row's (a simplifying assumption --
section 2.8 below is exactly about a leg where this assumption needs revisiting), the probability
of the *entire dataset* looking the way it actually looked is the **product** of every row's
individual probability:

```
Likelihood = P(y_1|p_1) * P(y_2|p_2) * ... * P(y_N|p_N)
           = product over all rows of [ p_i^(y_i) * (1-p_i)^(1-y_i) ]
```

"Maximum likelihood estimation" means: choose the coefficients that make this number as large as
possible -- i.e., choose the coefficients that make the data we actually observed look as
*unsurprising* as possible under the model.

**Why take the log.** Multiplying tens of thousands of numbers, each between 0 and 1, produces a
number so close to zero a computer can't represent it accurately (this is a real numerical problem,
not just an inconvenience). Taking the logarithm turns every product into a **sum** --
`log(a*b) = log(a) + log(b)` -- which is both numerically stable to compute and easier to take
derivatives of. Because `log` is a strictly increasing function, whatever coefficients maximize the
likelihood also maximize its log -- nothing about *which* answer is best changes, only how easy it
is to compute.

```
log(Likelihood) = sum over all rows of [ y_i * log(p_i) + (1-y_i) * log(1-p_i) ]
```

**Why negate it.** Every optimization routine in every ML library (including the `lbfgs` solver
this project uses) is written to *minimize* something, by convention. "Maximize the log-likelihood"
and "minimize the negative log-likelihood" are the exact same problem, just phrased with a sign
flip. Negate, and divide by the number of rows `N` to get an average (so the number doesn't grow
just because the dataset is bigger) -- and this is **log-loss**:

```
log-loss = -(1/N) * sum over all rows of [ y_i * log(p_i) + (1-y_i) * log(1-p_i) ]
         = -mean( y*log(p) + (1-y)*log(1-p) )
```

This is exactly the formula every classification metric library uses. It was not handed down from
nowhere -- it is "how surprised was the model by the true outcomes, on average," derived directly
from treating each row as a coin-flip with the model's own predicted probability.

**A brief look at the gradient (why the model can actually be fit at all).** To let a computer find
the coefficients that minimize log-loss, it needs to know, for each coefficient, "if I nudge this
coefficient up slightly, does the loss go up or down, and by how much?" That's the derivative
(gradient) of the loss with respect to each coefficient. This turns out to have an unusually clean
form. First, the derivative of the sigmoid function with respect to its input `z`:

```
d(sigmoid(z))/dz = sigmoid(z) * (1 - sigmoid(z)) = p * (1-p)
```

(This itself is a short derivation from the quotient rule applied to `1/(1+exp(-z))`, omitted here
since the result is what matters for what follows.) Then, using the chain rule (the derivative of
log-loss with respect to `p`, times the derivative of `p` with respect to `z`, times the derivative
of `z` with respect to one coefficient `coef_j`, which is just that coefficient's feature value
`x_j`), the whole chain collapses to a remarkably simple result:

```
d(log-loss)/d(coef_j) = mean( (p_i - y_i) * x_ij )
```

In words: **the gradient for each coefficient is just "how wrong was the prediction, on average,
weighted by how much that feature was present."** If the model is systematically over-predicting on
rows where some feature is large, the gradient pushes that feature's coefficient down; if
under-predicting, it pushes it up. This clean, interpretable gradient is exactly why gradient-based
solvers like `lbfgs` can fit logistic regression efficiently and reliably.

#### Worked example: `v1_unweighted`, one real held-out row

Held-out test row 0 (match `3857257`), real label `y = 0` (no shot followed). `v1_unweighted` uses
32 locked features across 85 design-matrix columns (categorical one-hot + numeric + boolean) -- 6
are shown here for readability; a footnote below accounts for the rest.

| Feature | Real value fed to the model | Real fitted coefficient | Contribution (`coef * value`) |
|---|---|---|---|
| `match_time_seconds` (standardized) | -1.6653 | +0.08009 | -0.13337 |
| `possession_elapsed_seconds` (standardized) | +0.01830 | +0.03972 | +0.00073 |
| `attacking_goal_centrality` (standardized) | +1.60793 | +0.56954 | +0.91579 |
| `is_in_attacking_box` (0/1) | 0 | +0.35938 | 0.0 |
| `counterpress` (0/1) | 0 | -0.04899 | 0.0 |
| `defenders_within_10m` (standardized) | -0.65509 | +0.06306 | -0.04131 |

Sum of these 6 shown contributions: **+0.74183**. The other 79 design-matrix columns (categorical
one-hot dummies for `phase_label`, `position`, `event_type`, etc., plus the remaining numeric
features) contribute a combined **-2.41572** -- not itemized here for readability, but included in
full in the total below (full breakdown: `outputs/models/classification/v1_unweighted.json`).

```
z = intercept + (sum of all 85 contributions)
  = -1.31644 + 0.74183 + (-2.41572)
  = -2.99033

sigmoid(z) = 1 / (1 + exp(2.99033)) = 0.04786
```

**The model's real, actually-stored prediction for this row is `0.047865`** -- matching the
hand-computed `sigmoid(z)` above to 9 decimal places (verified by
`scripts/analysis/generate_math_behind_it_examples.py`, output key `active_1_1_v1_linear`,
`verified_match: true`). The real label was `y=0`, and the model gave this row under a 5% chance of
a shot following -- a genuinely well-calibrated call on this particular row.

### 1.2 Why class weighting breaks calibration (`v2_weighted` / `v3_weighted_interactions` vs `v1`)

**The idea in plain words.** Only about 7.8% of rows in the active-binary dataset are positive (a
shot actually followed). A model that just predicted "no shot, always" would be right 92.2% of the
time -- technically high accuracy, completely useless. `class_weight="balanced"` is a common fix:
make every positive row count more in the loss function, so the model can't get away with ignoring
the minority class.

**Deriving the modified loss.** Ordinary log-loss treats every row equally:

```
log-loss = -mean( y*log(p) + (1-y)*log(1-p) )
```

Balanced weighting multiplies each row's term by a weight `w_i` that depends only on its class:
positive rows get weight `w_pos = N / (2 * N_pos)`, negative rows get `w_neg = N / (2 * N_neg)`
(this specific formula is scikit-learn's convention; the shape of the argument below holds for any
inverse-frequency weighting). The loss becomes:

```
weighted log-loss = -mean( w_i * [ y_i*log(p_i) + (1-y_i)*log(1-p_i) ] )
```

Since positives are rare (`N_pos` small), `w_pos` is large -- getting a positive row wrong now costs
much more than before. Since negatives are common, `w_neg` is small -- getting a negative row wrong
costs less than before.

**Why this breaks calibration, not just in principle but by construction.** The whole derivation in
section 1.1 relied on one specific fact: minimizing *unweighted* log-loss is mathematically
equivalent to maximizing the likelihood of the *true* Bernoulli process that generated the data --
which is exactly what makes the fitted `p` a genuine probability estimate. Once every row's term is
multiplied by a different constant depending on its class, the quantity being minimized is no
longer the likelihood of the true data-generating process. It's the likelihood of a *different,
synthetic* process where positive events are artificially inflated in importance. The model still
converges to *something* -- the coefficients that best satisfy this new, reweighted objective -- but
that something is no longer "the true probability of a shot." It systematically pushes predicted
probabilities for the minority class upward, because getting them right now matters `w_pos / w_neg`
times more than it used to, and the model exploits that by simply predicting higher probabilities
across the board rather than genuinely improving its ranking of which rows are riskier.

#### Worked example: real calibration gap, `v1` vs `v2`

Real, currently-stored out-of-fold expected calibration error (ECE, from
`outputs/models/comparisons/active_binary_baseline_comparison.csv`):

| Variant | OOF ECE | OOF PR-AUC |
|---|---|---|
| `v1_unweighted` | **0.003644** | 0.3521 |
| `v2_weighted` | **0.285480** | 0.3376 |

A roughly **78x** jump in calibration error for a *decrease* in ranking quality -- exactly the
mechanism just derived: the weighted objective is not "get the true probability right," so it
doesn't.

On a real held-out row where the two variants disagree sharply (row 77 of held-out test, match
`3857257`, real label `y=0`): `v1_unweighted` predicts **0.1361**, `v2_weighted` predicts
**0.7060** -- a difference of 0.57 in predicted probability for the exact same real row, same real
features, same real outcome. `v2` is confidently, wrongly, sure a shot was coming; `v1` is
correctly uncertain-but-doubtful. (`active_1_2_calibration_gap` in the verification JSON.)

### 1.3 Why a straight line can't make a U-shape, and why that rung failed (`v1b_quadratic`)

**The idea.** A plain logistic regression's log-odds `z` is a straight-line (linear) function of
each feature: increase `defender_spread` by 1 unit, and `z` always moves by the same fixed amount
`coef * 1`, regardless of where you started. That's a straight line in "feature value vs log-odds"
space. But `defender_spread`'s real relationship with shot risk, per this project's own EDA, is
**U-shaped**: risk is elevated when defenders are bunched very tightly *and* when they're spread
very wide, with a safer middle ground. No single straight-line coefficient can represent "goes up,
then down, then up again" -- a straight line only has one direction.

Adding a squared term, `feature^2`, fixes this specific blind spot: a straight-line combination of
`feature` and `feature^2` can produce a curve, because `feature^2` grows the same way whether
`feature` is very negative or very positive -- exactly the "elevated at both extremes" shape a
U needs. (One coefficient on `feature` for the "which direction" tilt, one coefficient on
`feature^2` for the "curves upward at both ends" bend.)

**Why it didn't survive held-out test on this leg.** Per `ACTIVE_BINARY_MODEL_LADDER.md`'s own
Rung 1 verdict: `v1b_quadratic`'s CV gain over `v1` did not hold up on the held-out test set --
the extra flexibility let the model fit *noise* in the training folds that didn't generalize. This
is a genuinely useful lesson in its own right, not just a footnote: **a model that can represent
more shapes isn't automatically a better model** -- it can also represent shapes that only exist
by chance in one particular sample of data. More flexibility raises the ceiling of what's
representable, but it also raises the risk of representing something that isn't real. The ladder's
job is to check that, not assume it away -- which is exactly what happened here.

### 1.4 Systematic interactions and L1/Lasso regularization (`v1c_systematic_interactions`)

**What an interaction term represents.** `v1`'s coefficients each describe one feature's effect
*in isolation*, holding everything else fixed. But football situations are combinations: maybe
having many defenders nearby only reduces risk when the ball is also far from goal -- close to
goal, extra defenders barely help. A term like `feature_i * feature_j` lets the model respond to
that *combination* directly: its own coefficient describes "how much does the effect of one feature
change depending on the value of the other," something neither feature's own coefficient can say by
itself.

**Deriving the L1 penalty, and why it zeroes coefficients out (not just shrinks them).** Generating
every pairwise product and every squared term from the 17 numeric features gives 153 new candidate
columns (`C(17,2)=136` pairwise + 17 squared) -- far too many to trust blindly; some of these will
be spurious. L1 (Lasso) regularization adds a penalty to the loss:

```
L1 loss = log-loss + lambda * sum( |coef_j| )
```

where `lambda` (`1/C` in scikit-learn's notation) controls how strongly extra coefficients are
punished, and `|coef_j|` is the absolute value of each coefficient (distance from zero, ignoring
sign). Why absolute value rather than squaring (which is L2/Ridge, the "other" common penalty)?
The geometric picture makes this concrete without needing calculus. Picture the space of possible
coefficient values as a 2-D plane (just two coefficients, for the picture -- the real thing has
hundreds of dimensions, but the shape argument doesn't change). The L1 penalty `|c1| + |c2| <= budget`
traces out a **diamond** (a square rotated 45 degrees) around the origin. The L2 penalty
`c1^2 + c2^2 <= budget` traces out a **circle**. Fitting the model means finding where the
loss function's own "best unconstrained answer" contour first touches this constraint shape. A
circle has no corners -- the touching point can land anywhere on its smooth boundary, and will
generally have *both* coefficients nonzero. A diamond has sharp corners sitting exactly on the axes
(where one coefficient is exactly zero) -- and it is disproportionately likely that the best-loss
contour first touches the diamond *at one of those corners*, precisely because the corners are
"points," not spread-out curve, so a comparatively large share of possible loss-contour angles hit
a corner first. That is why L1 regularization tends to zero coefficients out completely (true
feature selection), while L2 only ever shrinks them toward zero without reaching it exactly.

`C` (inverse of `lambda`) was grid-searched on the 5 canonical CV folds only, never against
held-out test.

#### Worked example: a real surviving interaction term

Among `v1c`'s real fitted coefficients (`outputs/models/classification/v1c_systematic_interactions.json`),
100 of the pairwise interaction terms survived L1 with a nonzero coefficient. The single largest
(by `|coefficient|`) is:

```
poly__distance_to_attacking_box__x__defender_attacker_gap_x
```

i.e. `distance_to_attacking_box * defender_attacker_gap_x`, real fitted coefficient
**-0.21064**. On the same real held-out row used throughout this section (row 0, match `3857257`):
`distance_to_attacking_box = 37.40`, `defender_attacker_gap_x = -7.137` (raw values). After the same
standardization `v1`'s numeric features get, then the design matrix's own additional
re-standardization of the polynomial block (so the interaction term itself has comparable scale to
the rest of the design matrix, not just its raw product), this term's real design-matrix value is
**0.43514**. Its real contribution to this row's log-odds:

```
contribution = coefficient * design_matrix_value = -0.21064 * 0.43514 = -0.09166
```

A genuinely small nudge on this particular row -- but a real, physically interpretable one: this
interaction term says "the further the action is from the attacking box, *and* the more the
defenders trail behind the attackers in the x-direction (both features negative or both positive,
here), the lower the risk" -- reducing this row's log-odds by about 0.09 beyond what the two
features' own individual coefficients already captured. (`active_1_4_l1_interaction` in the
verification JSON.)

### 1.5 Random Forest (`v1d_random_forest`)

**Deriving Gini impurity.** A Random Forest builds many decision trees, and each tree decides where
to split by asking: "which yes/no question about the data most cleanly separates positives from
negatives?" To measure "cleanly," first derive a way to measure *messiness*. Take a set of rows
where a fraction `p` are positive (shot followed) and `1-p` are negative. Imagine picking two rows
from this set at random, with replacement, and asking "do they have different labels?" The
probability of picking a positive-then-negative pair is `p * (1-p)`; the probability of
negative-then-positive is `(1-p) * p`. Add these (both orders count as "different"):

```
Gini = p*(1-p) + (1-p)*p = 2*p*(1-p)
```

which is algebraically the same as `1 - p^2 - (1-p)^2` (expand `p^2 + (1-p)^2 = p^2 + 1 - 2p + p^2`,
so `1 - p^2 - (1-p)^2 = 1 - 2p^2 - 1 + 2p = 2p - 2p^2 = 2p(1-p)` -- the two forms are identical).
Gini is 0 when `p=0` or `p=1` (every row the same label, two random draws can never disagree -- a
perfectly "pure," unmessy set), and at its maximum when `p=0.5` (maximum disagreement).

**Deriving information gain with a tiny toy example, before real numbers.** A split's quality is
"how much messiness did we remove." Toy example: 10 rows, 4 positive / 6 negative before any split
(`p=0.4`).

```
Gini_before = 1 - 0.4^2 - 0.6^2 = 1 - 0.16 - 0.36 = 0.48
```

Split into two groups of 5: left group has 1 positive / 4 negative (`p=0.2`), right group has 3
positive / 2 negative (`p=0.6`):

```
Gini_left  = 1 - 0.2^2 - 0.8^2 = 1 - 0.04 - 0.64 = 0.32
Gini_right = 1 - 0.6^2 - 0.4^2 = 1 - 0.36 - 0.16 = 0.48
```

Weighted average impurity after the split (both groups are half the data, so weight 0.5 each):

```
Gini_after = 0.5*0.32 + 0.5*0.48 = 0.16 + 0.24 = 0.40
Information gain = Gini_before - Gini_after = 0.48 - 0.40 = 0.08
```

A tree considers every possible feature and every possible threshold, computes this gain for each,
and picks whichever split produces the largest gain -- the split that does the most to separate the
two classes.

**Deriving why averaging many trees reduces variance (bagging).** Each individual tree, fit on a
bootstrap-resampled subset of the training rows, is a high-variance predictor -- change the sample
slightly and a single tree's predictions can shift a lot. But if you had `n` *fully independent*
estimates of the same underlying quantity, each with variance `Var(single)`, the variance of their
average is a textbook result:

```
Var(mean of n independent estimates) = Var(single) / n
```

In words: averaging cancels out each estimate's own individual noise, because the noise points in
random, unrelated directions and partly cancels out, while the shared true signal (which every tree
is, on average, trying to capture) doesn't cancel -- it adds up. **Honest caveat**: real trees in a
Random Forest are not fully independent (they're all fit on overlapping bootstrap samples of the
same underlying data, and share whichever features are strongest overall), so the real variance
reduction is *less* than the ideal `1/n` this formula predicts -- but the same direction of effect
holds, and it's why averaging many trees is much more stable than trusting any single one.

#### Worked example: real trees, one real row

`v1d_random_forest`'s real fitted forest (`outputs/models/classification/v1d_random_forest.joblib`)
has 600 individual trees, each a full `DecisionTreeClassifier` saved inside the forest object --
real enough to query directly, not a substitute. On the same real held-out row (row 0, match
`3857257`), the first 5 trees' real predicted probabilities:

```
Tree 1: 0.0000
Tree 2: 0.2857
Tree 3: 0.0000
Tree 4: 0.0000
Tree 5: 0.5714
```

Mean of these 5: **0.1714**. Mean of all 600 real trees: **0.074959** -- and the forest's own real
`predict_proba` for this row is **0.074959**, matching the mean of all 600 individual trees exactly
(verified: `active_1_5_random_forest`, `verified_match: true`). This is not a coincidence or an
approximation -- a scikit-learn `RandomForestClassifier`'s `predict_proba` is defined as exactly the
average of its individual trees' `predict_proba` outputs, and this is that definition, checked
against the real object.

The very first tree's real root split (the first, most important question the first tree asks of
every row): **`attacker_spread <= 13.007`** -- in football terms, "are the attackers bunched
relatively close together, or spread wide?" -- the single question that tree found most useful for
separating shot-followed rows from the rest, before any other consideration.

### 1.6 Gradient Boosting (`v1e_gradient_boosting`)

**The boosting idea, derived from "fix what's still wrong."** Random Forest builds many trees
independently and averages them. Gradient boosting builds trees **one after another**, and each new
tree's entire job is to predict what the *current combined model* is still getting wrong. This is a
form of gradient descent -- not on a handful of coefficients like logistic regression, but on the
predicted values themselves (hence "functional" gradient descent: the thing being optimized is a
whole function, approximated tree by tree).

Recall from section 1.1 that log-loss's gradient with respect to the raw score `z` (before the
chain rule reached all the way down to individual coefficients) has the clean form
`p - y` -- the model's predicted probability minus the true label. This is exactly what each new
tree in gradient boosting is trained to predict: not the raw label `y` itself, but the **residual**
`p - y` -- "in which direction, and by how much, is my current combined score wrong for this row?"
If the current model badly overestimates a row's risk (`p` too high, `y=0`), the residual is a large
positive number, and the next tree learns to push its own contribution for rows like that one
downward -- literally correcting the mistake, one small tree at a time.

**The learning rate.** Each new tree's contribution is scaled down by a fixed multiplier before
being added to the running total, `learning_rate` (0.01 for the active leg's tuned `v1e`, per
`outputs/models/classification/v1e_gradient_boosting.json`'s stored hyperparameters). Why damp the
correction instead of applying it in full? Each individual tree is fit on a finite, noisy sample --
its estimate of "what's still wrong" is itself imperfect. Taking one large, full-strength step in
the direction of any single tree's opinion risks over-correcting based on that tree's own noise.
Many small, damped steps, each nudging the combined model a little further in a data-supported
direction, average out individual trees' noise in a similar spirit to Random Forest's averaging --
but here it happens sequentially, with each step informed by everything that came before it, rather
than all at once.

#### Worked example: real boosting rounds, one real row

`v1e_gradient_boosting`'s real tuned model stopped early (via early stopping) at **382 rounds**
(`best_iteration_`), `learning_rate=0.01`. On the same real held-out row, the running raw score
(before the final sigmoid) at several real checkpoints, read directly from the real saved
LightGBM booster:

| Boosting round | Raw score (log-odds) so far |
|---|---|
| 1 | -2.4584 |
| 10 | -2.4621 |
| 50 | -2.3255 |
| 382 (final, `best_iteration_`) | **-2.8272** |

`sigmoid(-2.8272) = 0.055874` -- matching the model's real stored `predict_proba` for this row
exactly (`active_1_6_gradient_boosting`, `verified_match: true`). Notice the raw score does not
move monotonically toward its final value round by round (it actually rises slightly between round
1 and round 50 before falling further) -- each tree is correcting whatever the *current* combined
model gets wrong, and that target itself shifts as the model changes, so the path isn't a straight
line even though the final destination is a single number.

### 1.7 Calibration: Platt scaling and isotonic regression

**Platt scaling, derived as a 1-D version of section 1.1.** Gradient boosting is trained to
*rank* rows well (its loss function rewards putting positives above negatives), which is not
exactly the same objective as "output the true probability." Platt scaling fixes this by taking the
model's raw scores as the *only* input to a brand-new, tiny logistic regression -- literally the
same derivation as section 1.1, but with one input variable (the raw score) instead of 32:

```
calibrated_p = sigmoid( A * raw_score + B )
```

where `A` and `B` are fit, by the exact same maximum-likelihood log-loss minimization derived in
1.1, on a held-out slice of the training data -- learning a simple rescaling that corrects
systematic over- or under-confidence in the raw scores.

**Isotonic regression, derived via a tiny worked example.** Isotonic regression is a different,
non-parametric approach: instead of assuming the correction is a clean S-curve (what Platt/logistic
does), it only assumes the corrected probabilities should be **monotonically non-decreasing** in the
raw score (a higher raw score should never map to a *lower* calibrated probability) -- and finds the
best such mapping directly from the data, with no formula constraining its shape. The algorithm that
does this is called **pool-adjacent-violators (PAV)**: scan the (sorted-by-raw-score) observed
outcomes left to right; wherever a later value is *lower* than an earlier one (a "violation" of
monotonicity), merge the two into a single pooled value (their average), and keep checking backward
in case that pooled value now violates monotonicity with what came before it too.

Tiny worked example, 6 points, raw scores in order, each with an observed value:

```
Points (in raw-score order):  0.1,  0.4,  0.3,  0.5,  0.35,  0.6
                                      ^-- violation: 0.3 < 0.4
```

Pool the violating pair (points 2 and 3): both become their average, `(0.4+0.3)/2 = 0.35`:

```
0.1,  0.35,  0.35,  0.5,  0.35,  0.6
                           ^-- violation: 0.35 < 0.5
```

Pool that violating pair (points 4 and 5): both become `(0.5+0.35)/2 = 0.425`:

```
0.1,  0.35,  0.35,  0.425,  0.425,  0.6
```

Check the whole sequence again: `0.1 < 0.35 <= 0.35 < 0.425 <= 0.425 < 0.6` -- fully non-decreasing,
no more violations, done. The final calibrated mapping is a **step function**: this raw-score range
maps to 0.1, the next to 0.35, the next to 0.425, the last to 0.6 -- exactly what
`CalibratedClassifierCV(method="isotonic")` computes, just at the real dataset's scale (many more
points, real raw scores) instead of 6 toy ones.

#### Worked example: the real chosen calibration method

Both legs' `v1e`/`p1e`-calibrated variants chose **isotonic regression** as the calibration method
(lower OOF ECE than sigmoid at comparable PR-AUC -- active leg: isotonic ECE 0.00274 vs sigmoid's
0.00324, `outputs/models/validation/v1e_calibration_method_comparison.json`). On the same real
held-out row used throughout this section: the raw `v1e_gradient_boosting` score maps to raw
probability **0.055874**; the real fitted isotonic step function maps this to **0.034749** as the
final calibrated probability (`active_1_7_calibration_mapping` in the verification JSON) --
isotonic regression pulled this particular row's probability down, consistent with the calibrated
model tending to be less extreme than the raw model's own confidence on rare-event predictions.

### 1.8 Putting it together: one row, the whole ladder

Same real held-out row (row 0, match `3857257`, real label **y = 0**, no shot followed) scored by
every rung's real, actually-fitted model:

| Rung | Predicted probability |
|---|---|
| `v1_unweighted` | 0.04786 |
| `v1c_systematic_interactions` | 0.04585 |
| `v1d_random_forest` | 0.07496 |
| `v1e_gradient_boosting` (raw) | 0.05587 |
| `v1e_gradient_boosting_calibrated` | **0.03475** |

**Worth stating plainly, not glossed over**: for this specific row, the ladder did *not* improve
monotonically. `v1d_random_forest`'s prediction (0.07496) is actually the *furthest* from the true
answer (the true label is 0, so lower is better here) -- worse than the very first rung,
`v1_unweighted`. `v1c_systematic_interactions` (0.04585) is closer to correct than either `v1d` or
raw `v1e`. Only the final calibrated model ends up closest of all. **This is expected, not a bug**:
every ladder-gate decision in this project (paired significance tests, held-out readouts, cluster
bootstraps) was about *aggregate* metrics across tens of thousands of rows, never about any single
row. Individual rows are noisy; a model that's measurably better on average can still be wrong, or
even worse, on any particular row you pick -- and that's exactly what this one real row shows.

---

## 2. Passive-binary leg: from `p1_unweighted` to `p1e_gradient_boosting_calibrated`

The math in this section is identical in kind to section 1 -- same sigmoid, same log-loss, same L1
diamond, same Gini, same boosting-as-residual-correction, same PAV algorithm. Nothing here is
re-derived; each subsection below points back to where the derivation lives and goes straight to
this leg's own real numbers, which differ because the feature set, the row grain, and (per section
2.8 below) the row-independence assumption itself are all genuinely different from the active leg.

### 2.1 Logistic regression (`p1_unweighted`) -- as derived in section 1.1

#### Worked example: `p1_unweighted`, one real held-out row

Held-out test row 0 (match `3857257`, same match_id as the active leg's example row -- coincidence
of how the canonical split orders held-out matches, not a meaningful link), real label `y = 0`.
`p1_unweighted` uses 38 locked features across 56 design-matrix columns -- 6 shown:

| Feature | Real value fed to the model | Real fitted coefficient | Contribution |
|---|---|---|---|
| `defender_x` (standardized) | -0.04184 | -0.01728 | +0.00072 |
| `ball_x` (standardized) | +0.05241 | -0.01076 | -0.00056 |
| `top_option_1_threat_score` (standardized) | +0.01814 | -0.14689 | -0.00266 |
| `is_in_attacking_box` (0/1) | 0 | +0.78067 | 0.0 |
| `marking_tightness` (standardized) | +0.52346 | -0.02177 | -0.01139 |
| `engagement_distance_to_carrier` (standardized) | -0.99645 | -0.27676 | +0.27578 |

Sum of these 6 shown contributions: **+0.26188**. The other 50 design-matrix columns contribute a
combined **-3.05337** (full breakdown: `outputs/models/classification/p1_unweighted.json`).

```
z = -1.49191 + 0.26188 + (-3.05337) = -4.28340
sigmoid(z) = 1 / (1 + exp(4.28340)) = 0.013608
```

**Matches the model's real stored prediction, 0.013608, exactly** (`passive_2_1_p1_linear`,
`verified_match: true`). Real label `y=0` -- another well-calibrated call.

### 2.2 Why class weighting breaks calibration -- as derived in section 1.2

Real, currently-stored numbers for this leg (`outputs/models/comparisons/passive_binary_baseline_comparison.csv`):

| Variant | OOF ECE | OOF PR-AUC |
|---|---|---|
| `p1_unweighted` | **0.002368** | 0.1928 |
| `p2_weighted` | **0.354227** | 0.1903 |

A roughly **150x** ECE jump here -- an even larger relative blowup than the active leg's ~78x, for
the same reason derived in 1.2: balanced weighting optimizes a synthetic, reweighted objective, not
the true likelihood.

On a real disagreement row (row 183 of held-out test, real label `y=0`): `p1_unweighted` predicts
**0.1462**, `p2_weighted` predicts **0.7523** -- the same qualitative pattern as the active leg's
example, on this leg's own real data (`passive_2_2_calibration_gap`).

### 2.3 Quadratic terms -- and this rung actually survived here (`p1b_quadratic`)

Same math as section 1.3 (a squared standardized feature lets the model bend, not just tilt). The
**real, different outcome on this leg** is worth restating from `PASSIVE_BINARY_MODEL_LADDER.md`
section 2: unlike the active leg's `v1b_quadratic`, this leg's `p1b_quadratic` **did** survive
held-out test (CV PR-AUC 0.1928 to 0.1983, held-out 0.1736 to 0.1805, both real, both gains). Same
underlying math, same kind of check, genuinely different real-world result -- a concrete
demonstration that "does more model flexibility help" isn't answerable from the math alone; it has
to be checked against real held-out data every time, which is exactly what both legs' ladders did.

### 2.4 Systematic interactions and L1 -- as derived in section 1.4

This leg's numeric block has 28 features (not the active leg's 17), so `PolynomialFeatures` produces
`C(28,2)=378` pairwise + 28 squared = 406 candidate columns (vs the active leg's 153).

#### Worked example: a real surviving interaction term, this leg

The largest surviving interaction term by `|coefficient|` (`outputs/models/classification/p1c_systematic_interactions.json`):

```
poly__defender_y__x__ball_y
```

i.e. `defender_y * ball_y`, real fitted coefficient **-0.43264**. On the same real held-out row:
`defender_y = 31.913`, `ball_y = 39.9` (raw values), giving a real (re-standardized) design-matrix
value of **-0.54622**, and a real contribution to this row's log-odds:

```
contribution = -0.43264 * -0.54622 = +0.23632
```

267 pairwise interaction terms survived L1 on this leg (of 378 candidates) -- a smaller *fraction*
pruned than the active leg's terms, but this leg's design matrix is also much larger overall.
(`passive_2_3_l1_interaction` in the verification JSON.)

### 2.5 Random Forest -- as derived in section 1.5

#### Worked example: real trees, one real row, this leg

`p1d_random_forest`'s real forest has 150 trees. On the same real held-out row, the first 5 trees'
real predicted probabilities: `0.01072, 0.01897, 0.03777, 0.06227, 0.03577` -- mean of these 5:
**0.03310**. Mean of all 150 real trees: **0.030218** -- matching the forest's own real
`predict_proba`, **0.030218**, exactly (`passive_2_4_random_forest`, `verified_match: true`).

The first tree's real root split: **`phase_label == "high_press_proxy"`** (a categorical one-hot
split, threshold 0.5 meaning "is this row's phase exactly high-press") -- in football terms, "is
the defending team actively pressing high up the pitch right now" was this leg's first tree's most
useful opening question, a genuinely different kind of question from the active leg's continuous
`attacker_spread` split (this leg's categorical `phase_label` feature turned out more immediately
informative than any single continuous geometry feature, at least for this particular tree's root).

### 2.6 Gradient Boosting -- as derived in section 1.6

#### Worked example: real boosting rounds, one real row, this leg

`p1e_gradient_boosting` stopped early at **164 rounds**, `learning_rate=0.05` (a higher rate than
the active leg's 0.01, and correspondingly a shorter run to convergence). Real raw score at real
checkpoints:

| Boosting round | Raw score (log-odds) so far |
|---|---|
| 1 | -2.7700 |
| 10 | -3.0000 |
| 50 | -3.6677 |
| 164 (final, `best_iteration_`) | **-4.2187** |

`sigmoid(-4.2187) = 0.014504`, matching the model's real stored prediction exactly
(`passive_2_5_gradient_boosting`, `verified_match: true`). Unlike the active leg's example row, this
leg's raw score moves consistently in one direction round over round -- another honest, real
difference between two real rows, not a general rule about how boosting always behaves.

### 2.7 Calibration -- as derived in section 1.7

`p1e_gradient_boosting_calibrated` also chose **isotonic regression** (lower OOF ECE than sigmoid:
0.0018 vs 0.0022, `outputs/models/validation/p1e_calibration_method_comparison.json`). On the real
held-out row: raw score maps to raw probability **0.014504**; the real fitted isotonic step function
maps this to **0.012325** (`passive_2_6_calibration_mapping`) -- again pulled slightly down, same
direction as the active leg's example.

### 2.8 Cluster-aware significance testing -- a check this leg needed that the active leg didn't

**Why treating every row as an independent Bernoulli trial (section 1.1's assumption) overstates
confidence here.** Every derivation in section 1 assumed each row's outcome is statistically
independent of every other row's -- that's what let the dataset's likelihood be written as a
simple product. On the passive leg, this assumption is genuinely wrong in a specific, checkable
way: many rows share the same `event_id` (multiple defender-slots describing the *same* on-ball
attacking event), and those rows are not independent -- they share the same attacking context, the
same ball position, the same outcome.

**The plain-language idea of "effective sample size."** Imagine you wanted to estimate the average
height of people in a room by measuring 100 people -- but 90 of those 100 measurements were actually
just the same 10 people, measured 9 times each (perhaps 9 photos of the same 10 people). You have
100 numbers, but you don't really have 100 *independent pieces of information* about the room's
average height -- you effectively have closer to 10 plus a bit. When rows within a cluster (here, an
`event_id` group) are correlated, each additional row from the same cluster tells you less *new*
information than a row from a brand-new cluster would. The "effective sample size" -- the number of
truly independent pieces of evidence you actually have -- is smaller than the raw row count, and how
much smaller depends on how strongly correlated same-cluster rows are (their intra-class
correlation) and how many rows per cluster there typically are. A significance test (like the
paired t-test used to gate every rung in this ladder) that's computed as if every row were fully
independent will report a **narrower**, over-confident interval than the truth actually supports.

**What a cluster bootstrap does differently.** An ordinary ("naive") bootstrap estimates uncertainty
by resampling individual rows with replacement, many times, and looking at how much the computed
statistic (here, PR-AUC) varies across those resamples. But this inherits the same
independence-that-isn't-real problem: resampling individual rows from a correlated cluster still
treats them as if shuffling any one of them around is as informative as shuffling an independent
row. A **cluster bootstrap** instead resamples *whole event-groups* -- every time an event is drawn,
every row belonging to it goes into that resampled dataset together, preserving whatever correlation
existed inside each cluster instead of averaging it away. Because within-cluster rows now move
together instead of independently, resampling produces more genuine variation from one bootstrap
draw to the next -- which is exactly what a more honest, wider confidence interval requires.

#### Worked example: real naive vs cluster-aware CIs, this leg

Real numbers, `p1e_gradient_boosting` vs `p1_unweighted`, held-out test (`outputs/models/validation/cluster_bootstrap_p1_vs_p1e.json`,
40,147 real `event_id` clusters, average 7.87 rows per cluster, 1,000 bootstrap iterations):

| | Naive (row-level) 95% CI | Cluster-aware 95% CI | Width ratio |
|---|---|---|---|
| `p1e` PR-AUC | [0.2054, 0.2162] | [0.1949, 0.2274] | **3.03x** |
| `p1e - p1` PR-AUC advantage | [0.0340, 0.0403] | [0.0297, 0.0457] | **2.56x** |

The cluster-aware interval for `p1e`'s raw PR-AUC is over 3x wider than the naive one -- concretely
showing how much the naive, independence-assuming calculation understated the true uncertainty.
Both intervals for the advantage over `p1_unweighted` stay entirely above zero, so the *conclusion*
("`p1e` really is better than `p1`") survives this more honest accounting -- but the *precision*
claimed by the naive interval alone would have been overconfident by roughly this same factor, which
is exactly the caveat carried through every promotion decision on this leg (Prompts 50-52).

### 2.9 Putting it together: one row, the whole ladder, this leg

Same real held-out row throughout this section (row 0, real label **y = 0**):

| Rung | Predicted probability |
|---|---|
| `p1_unweighted` | 0.01361 |
| `p1c_systematic_interactions` | 0.00429 |
| `p1d_random_forest` | 0.03022 |
| `p1e_gradient_boosting` (raw) | 0.01450 |
| `p1e_gradient_boosting_calibrated` | **0.01232** |

Same honest observation as section 1.8: the true label is 0, so lower is better here, and
`p1c_systematic_interactions` (0.00429) is actually the *closest* of every rung, closer even than
the final promoted model. `p1d_random_forest` (0.03022) is the furthest -- worse than the very first
rung. The promoted model (`p1e_gradient_boosting_calibrated`) is a modest improvement over the raw
Rung 0 baseline on this one row (0.01232 vs 0.01361), but not the best possible answer among all
five rungs *for this specific row*. Exactly the same lesson as the active leg's section 1.8: the
promotion decisions in this project were built on aggregate evidence across the whole held-out test
set (tournament slices, error slices, cluster-aware bootstraps), never on any single row -- and a
real row, picked before looking at any of these numbers, demonstrates that plainly.

---

## 3. Active-continuous (xG regression) and passive-continuous -- not yet built

**Active-continuous (xG regression).** This leg of the project -- predicting a continuous expected-
goals-style value rather than a binary shot/no-shot outcome, for the active (on-action) leg -- has
not been started as of this document's writing. This section will be filled in once that modeling
work happens.

**Passive-continuous.** Likewise not yet started for the passive (off-ball positioning) leg. This
section will be filled in once that modeling work happens.

No math is speculated for either of these in advance -- the derivations in sections 1 and 2 above
cover only models that actually exist in this repository today.
