# Evaluation and validation

## Evaluation logic

Evaluation asks whether model outputs are complete, honest and stable enough to support the intended football interpretation. The project uses short-horizon observed outcomes, grouped validation and out-of-fold predictions to reduce leakage and overfitting risk.

## OOF integrity

Out-of-fold integrity means every analysed row is scored only by a model that did not train on that row. OOF files should have stable identifiers, no duplicate predictions for the same action/model variant and complete coverage for the selected analysis population.

## Train/test separation and fold-level predictions

Train/test separation matters because defensive action context can repeat within matches and teams. Fold-level prediction files help verify that predictions are not in-sample fitted values. Match-aware or grouped splitting is important when evaluating generalisation across football contexts.

## Completeness checks

Completeness checks confirm that required inputs exist and cover the selected population. Examples include:

- defensive action row counts;
- selected tactical-population counts;
- required model variant coverage;
- duplicate OOF row checks;
- fold coverage checks;
- match coverage checks;
- processed-event timeline columns;
- competition, season and stage metadata availability;
- boolean visibility coverage where used.

## Model sensitivity comparisons

Sensitivity comparisons check whether conclusions are robust to alternative model variants, such as with-360 versus without-360 classification outputs or primary versus nonlinear regression candidates. These comparisons are useful only when both sides are complete enough to support the selected population.

## Placeholder and smoke-sized sensitivity limitations

If a sensitivity file is smoke-sized, incomplete or identical to the primary output, the analysis should not claim model agreement. It should report a validation-mode warning and treat the section as a limitation until full sensitivity OOF files are generated.

## Football sanity checks

Technical validation is necessary but not sufficient. Results should also be checked against football expectations, tactical definitions, coordinate conventions and representative event or video-review candidates. Unexpected diagnostics, such as a phase-label population not overlapping with a spatial population, should be labelled clearly and reviewed before being used as conclusions.
