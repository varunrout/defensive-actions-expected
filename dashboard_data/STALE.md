# STALE -- do not use until regenerated

This folder (`dashboard_data/`, Prompt 83's export for the separate `portfolio-website` repo) was
generated **before** the coordinate-frame loader fix (Phase 1, commit `8b7d0a8`, prompt 85) and the
full re-pipeline that followed it (Phases 2-5, prompt 85; see
[`reports/modeling/COORDINATE_FRAME_FIX_AND_REPIPELINE.md`](../reports/modeling/COORDINATE_FRAME_FIX_AND_REPIPELINE.md)).

Every file in this folder embeds predictions from the **pre-fix** models across all 6 legs:

- `legs_summary.json`, `methodology_steps.json`, `selection_and_frame_audit.json`, and every file
  under `match_explorer/` were produced by re-scoring the pre-fix `v1e_gradient_boosting_calibrated`,
  `p1e_gradient_boosting_calibrated`, `c1d_random_forest`, `d1_lognormal_glm`,
  `x1c_random_forest`, and `y1c_random_forest` artifacts on pre-fix features.
- For the 4 non-xT legs, the underlying targets (`target_future_shot_10s`, `target_future_xg_10s`)
  are proven frame-invariant, but roughly a third to over four-fifths of each model's own feature
  importance sits on features that were computed from the wrong coordinate frame for up to ~46% of
  rows (`COORDINATE_FRAME_IMPACT_AUDIT.md`, section 3) -- so per-row predictions in this export can
  differ from what the corrected models now produce.
- For the 2 xT legs (`x1c_random_forest`, `y1c_random_forest`), the problem is more severe: the
  **target itself** (`target_xt_delta_v2` / `target_xt_delta_passive`) was directly frame-dependent
  for 69.13% / 67.19% of its own training rows, and the corrected retraining in this prompt (see the
  linked report's Phase 5) produced a fresh promotion decision on genuinely different target values.
  Any xT-delta number in this export (`legs_summary.json`'s xT rows, `match_explorer/*.json`'s
  xT-delta fields) reflects the pre-fix models and pre-fix target.

**Do not use anything in this folder** (or ship it to the `portfolio-website` repo) until it is
regenerated from the corrected pipeline. Regenerating `dashboard_data/` is explicitly **out of
scope for this prompt** (prompt 85's own non-goals) -- this file only marks the folder stale; it
does not touch, delete, or regenerate any file inside it.
