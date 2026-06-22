# Modelling techniques

## Classification for future shot probability

The classification task estimates whether a shot occurs in the short horizon after a defensive action. This produces an expected future-shot probability for each action. It is useful for identifying actions after which the opponent still creates a shot despite the defensive intervention.

## Regression for future xG

The regression task estimates the amount of future xG after the action. This captures shot quality as well as shot occurrence, but it is harder because many defensive actions are followed by zero xG.

## Two-part future xG signal

A two-part future-xG signal separates the chance of any future shot from the expected xG conditional on that threat. This is a practical way to handle sparse xG outcomes while keeping the football interpretation readable.

## Sensitivity models

Sensitivity models compare the primary interpretation with alternative model specifications. They help answer whether a conclusion depends heavily on one feature set, modelling family or 360-data availability. Sensitivity comparisons should be treated cautiously when the sensitivity file is incomplete, smoke-sized or identical to the primary file.

## OOF predictions

Out-of-fold predictions are central to the project. Each prediction should come from a model that did not train on that row. This avoids presenting in-sample fitted values as if they were honest generalisation estimates.

## Key model variants

### `b7_full_with_360`

Primary classification-style future-shot variant with the full available feature set including 360 context when eligible. This is used as a canonical shot-probability reference where coverage is complete for the selected population.

### `b6_full_without_360`

Classification sensitivity variant without 360 features. It is useful for checking whether conclusions are dependent on 360-derived context. Full OOF coverage is required before strong sensitivity claims can be made.

### `r4_full_with_360`

Primary regression-style future-xG variant with the full available feature set including 360 context where eligible. It provides a canonical expected future-xG reference for analysis populations with complete coverage.

### `r6_nonlinear_candidate`

Regression sensitivity candidate intended to test whether nonlinear modelling choices materially change expected future-xG conclusions. If the local file is missing, smoke-sized or a placeholder copy of the primary output, the comparison should be labelled as a limitation rather than interpreted as model agreement.

### `conditional_tweedie`

Two-part or conditional-style future-xG variant used to represent sparse positive xG outcomes in a way that is easier to interpret than a single unconditional regression alone. It supports future-xG analysis alongside classification and regression variants.

## Interpretation discipline

Model outputs estimate future threat after recorded defensive actions. They do not, by themselves, prove that an individual defender caused the threat level to change. Football conclusions should combine model output, population definition, sample size, metadata coverage, sensitivity checks and video or event review.
