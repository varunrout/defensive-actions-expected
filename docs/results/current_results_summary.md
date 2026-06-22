# Current results summary

## Validated current results

The current verified centre-back own-box analysis contains:

| Result | Value |
| --- | ---: |
| Defensive actions | 56,068 |
| Centre-back actions | 10,588 |
| Own-box actions | 4,842 |
| Centre-back own-box actions | 1,442 |
| Matches | 115 |
| Complete canonical model coverage for selected actions | 1,442 |
| Spatial phase overlap | 0 |

The selected centre-back own-box population has complete canonical model coverage for the 1,442 selected actions. This supports the current report's completeness checks for that tactical population.

The spatial phase overlap is currently 0. This is a diagnostic result for phase-label and spatial-definition alignment, not a football conclusion that box defending is absent.

## Known limitations

- Competition metadata is currently unknown in local artifacts, so competition, season and stage comparisons should not be over-interpreted.
- Full `b6_full_without_360` and `r6_nonlinear_candidate` sensitivity OOF files are not yet available in the current validated state.
- Sensitivity warnings are expected when primary OOF files are reused as placeholders or when sensitivity files are smoke-sized.
- The current CB box-defence report covers one tactical population only.
- The project does not yet claim causal defensive value, invisible defending or final player DAx attribution.

## Interpretation boundary

The current results demonstrate a reproducible analysis population and complete canonical model joins for that population. They should be read as a validated workflow milestone, not as final player rankings or definitive tactical conclusions.
