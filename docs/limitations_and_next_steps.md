# Limitations and next steps

## Current limitations

- **Competition metadata is unknown in current local artifacts.** Upstream metadata enrichment is needed before competition, season or stage breakdowns can be interpreted strongly.
- **Full sensitivity OOF files are not yet available for all variants.** In particular, complete `b6_full_without_360` and `r6_nonlinear_candidate` outputs are needed for stronger sensitivity comparisons.
- **Phase-label and spatial own-box populations do not overlap yet.** The current spatial phase overlap is 0 and should be treated as a diagnostic for definitions, coordinates and phase labelling.
- **The current CB box-defence analysis is one tactical population.** It does not cover wide defending, pressing, transition defending, deep-block crossing pressure, team profiles or player archetypes yet.
- **No causal defensive value is claimed.** Current outputs estimate and analyse future threat after recorded actions; they do not prove counterfactual suppression.

## Next steps

1. **Upstream metadata enrichment**
   - Add reliable competition, season and stage context to local artifacts.
   - Re-run readiness and CB analysis once metadata is populated.

2. **Full sensitivity OOF generation**
   - Generate complete sensitivity files for the required classification and regression variants.
   - Replace placeholder or smoke-sized comparisons with real model disagreement checks.

3. **Tactical population expansion**
   - Add scripted analyses for wide 1v1s, crossing control, transition/recovery defending, pressing/counterpressing and deep-block situations.

4. **Player and team profiling**
   - Build player role archetypes and team/system defensive style summaries after validated tactical populations are available.

5. **Portfolio report and dashboard layer**
   - Package the methodology, validation, results and limitations into a polished reviewer-facing story and optional dashboard/reporting workflow.
