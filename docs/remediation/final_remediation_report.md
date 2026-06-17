# Final remediation report

## 1. Executive summary
This remediation corrected core methodological defects in the repository: possession-bounded targets, event-time feature leakage, attacking-goal geometry, defensive action inclusion semantics, stateful phase proxy labelling, and documentation claims. The repository now describes the implemented system as post-action short-horizon attacking threat modelling, not causal DAx.

## 2. Original critical problems
Targets could cross possessions; continuous xT targets were circular; feature specs used future-only possession totals; nearest-goal geometry contradicted the normalised attack direction; defensive action types included unsupported `Shield` and generic goalkeeper events; phase labels were overclaimed.

## 3. Files changed
Core package files under `src/dax/features/`, `src/dax/models/`, and `src/dax/targets/`; tests under `tests/`; documentation under `README.md` and `docs/`; CI under `.github/workflows/ci.yml`.

## 4. Logic corrections
Added canonical event context utilities and possession-bounded target builders.

## 5. Leakage corrections
Removed `possession_duration_total`, `possession_event_count_total`, and `possession_progress_ratio` from baseline model specifications and added a denylist guard.

## 6. Football semantic corrections
Separated actor/attacking/defending team fields in player defensive output where context exists. Excluded `Shield` and generic `Goal Keeper` from defensive action types.

## 7. Pipeline architecture changes
Reusable target and event context logic now lives under `src/dax/`. Full raw/processed manifest architecture remains documented future work.

## 8. Target definition changes
`target_future_shot_10s` and `target_future_xg_10s` are observed same-possession outcomes. `target_xt_10s` is deprecated as a target.

## 9. Feature definition changes
Attacking goal is explicitly `(120, 40)`; defending goal is `(0, 40)`. Time-safe possession elapsed/count-so-far fields replace future totals.

## 10. Validation changes
Added deterministic unit tests for target boundaries, leakage, geometry and grid fit reset. Full tournament holdouts were not regenerated.

## 11. Old versus corrected metrics
Old metrics are invalid for current claims because they used stale targets/features. Corrected full-data metrics were not regenerated in this environment, so no replacement performance numbers are claimed.

## 12. Tests added
Short-horizon target tests, methodology invariant tests, fixture data.

## 13. Tests run
- Pre-change: `python -m pytest -q` → 8 passed, 3 failed.
- Post-change: `python -m pytest -q` → 29 passed.

## 14. Commands used
`find .. -name AGENTS.md -print`, `rg --files`, `git checkout -b fix/methodology-pipeline-rebuild`, `python -m pytest -q`, file inspection with `sed`, and local edit commands.

## 15. Notebook updates
Notebook outputs were not rerun. Existing notebook outputs should be considered historical until regenerated from corrected pipeline outputs.

## 16. Documentation updates
README and methodology/data dictionary/remediation documents were rewritten or added to remove unsupported causal and production claims.

## 17. Archived artifacts
No large output artifacts were moved in this environment. Historical outputs should be placed under `outputs/archive/pre_methodology_fix/` before full regeneration.

## 18. Remaining limitations
Full raw-to-processed offline pipeline rebuild, manifests/checksums, structured failure manifests, removal of sys.path manipulation, stale output archiving, tournament holdout metrics, calibration/subgroup validation plots, and notebook reruns remain to be completed on full data.

## 19. Work still required for true DAx
Pre-action threat, post-action threat, counterfactual no-intervention threat, option availability/removal, multi-defender attribution, role/team/opportunity adjustment, human validation, and uncertainty estimates.

## 20. Exact reproduction commands
```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```
