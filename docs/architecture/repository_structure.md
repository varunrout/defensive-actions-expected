# Repository structure

The canonical workflow is:

`download -> process -> event context -> phase proxies -> targets -> player features -> train -> validate -> report`

## Active structure

- `configs/`: pipeline, competition and model-spec references.
- `src/dax/`: reusable package code.
- `scripts/`: five thin active CLI entry points.
- `docs/methodology/`: current methodology notes.
- `docs/architecture/`: repository and pipeline architecture.
- `docs/validation/`: validation protocol and corrected-result status.
- `docs/data_dictionary/`: current data dictionary.
- `notebooks/`: sequential notebook index; historical notebooks are archived.
- `outputs/`: generated artifact landing zones; historical generated artifacts are archived.

## Deviations from the issue target tree

- The existing `src/dax/features/event_context.py` remains under `features/` rather than being moved to `context/` to avoid behaviour changes in this cleanup-only PR.
- Historical notebooks are archived instead of mechanically renamed into the new sequence, because they have not been rerun against current pipeline outputs.
- Current output folders are kept empty or with `.gitkeep`; pre-fix generated artifacts are preserved under `outputs/archive/pre_methodology_fix/`.
