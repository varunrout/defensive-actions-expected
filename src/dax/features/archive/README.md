# Archived (dead) feature-engineering code

Soft-deleted on 2026-09-16 -- confirmed zero imports from any current entry
point (`scripts/run_pipeline.py`, `scripts/build_features.py`) or from the
test suite. Kept here instead of hard-deleted in case something surfaces
that still needed it.

- `possession_sequences.py` -- only referenced by an already-archived
  script (`scripts/archive/pre_repository_cleanup/features/extract_possessions.py`);
  not used by the current pipeline (`src/dax/pipeline/runner.py` -- the module
  `run_pipeline.py` actually calls -- uses `event_context.py`/`phase_segmentation.py`
  instead, both still live).
