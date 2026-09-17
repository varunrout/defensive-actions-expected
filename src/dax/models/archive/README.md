# Archived (dead) modeling code

Soft-deleted on 2026-09-16 -- confirmed zero imports from any current entry
point (`scripts/*.py`) or from the test suite. Kept here instead of hard-deleted
in case something surfaces that still needed it.

- `specs.py` -- never imported by anything, including its own tests.
  Its function-local imports of `baseline_logistic.py`/`baseline_regression.py`
  never actually run, since nothing calls into `specs.py` itself.
