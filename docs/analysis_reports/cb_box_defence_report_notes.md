# CB box-defence report notes

## Current report scope

The current centre-back own-box analysis focuses on recorded defensive actions by centre-backs in the defending team's own penalty-box area. It is one tactical population, not a full defensive evaluation system.

## Verified current result numbers

The current local report numbers are:

| Metric | Count |
| --- | ---: |
| Defensive actions | 56,068 |
| Centre-back actions | 10,588 |
| Own-box actions | 4,842 |
| Centre-back own-box actions | 1,442 |
| Matches | 115 |
| Complete canonical model coverage for selected actions | 1,442 |
| Spatial phase overlap | 0 |

The spatial phase overlap is currently `0` and should be treated as a diagnostic, not as proof that the football concept is absent.

## Interpretation notes

Complete canonical model coverage for the selected 1,442 actions means the current selected population can be joined to the canonical model outputs used by the report. That supports report completeness for this population, but it does not remove the need for football review or sensitivity checks.

Competition metadata is currently unknown in the local artifacts. Upstream metadata enrichment is therefore a future task before competition, season or stage comparisons are interpreted strongly.

Model sensitivity warnings are expected when primary files are used as sensitivity placeholders or when sensitivity outputs are smoke-sized. In those cases, the report should label the issue as a validation-mode limitation rather than claiming that models agree.

## What not to overclaim

Do not treat this report as a player ranking system, a causal defensive value estimate or a complete team defensive model. It is a validated first coach-analysis population that demonstrates the script-based workflow and identifies the next data and modelling gaps.
