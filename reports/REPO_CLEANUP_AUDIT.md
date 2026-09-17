# Repo Cleanup Audit (Part B)

Scoped back in prompt 18, never delivered. This is an audit only -- nothing
listed here has been deleted or moved. Every item needs your explicit
go-ahead (per your standing "never delete without asking first" rule).
Written 2026-09-17.

## Delete candidates (high confidence)

1. **`local_archive/`** -- 3.4 GB. Already gitignored (never committed). Contains
   one dated snapshot, `data_pre_cleanup_20260617_164826/` (features, models,
   processed, raw, validation), plus a stray `interrupted_script_conversion.patch`.
   This is a "just in case" backup from a data cleanup done three months ago
   (17 June). By far the single biggest thing on disk that isn't source code
   or a tracked artifact. If the 17 June cleanup has been fine since, this is
   the strongest single candidate for deletion -- it's not backing up anything
   current, and it's gitignored either way so deleting it changes nothing in
   git history.

2. **`mlflow.db.bak_20260916182833`** -- 2.5 MB. A one-off timestamped backup
   of `mlflow.db`, one day old (16 Sep). If `mlflow.db` itself is intact and
   working, this backup has served its purpose. Low risk, trivial size --
   included mainly for completeness.

3. **`Claude outputs/`** (note the space in the folder name, capital C) -- 552 KB,
   untracked. This is a stale, superseded copy: every file in it duplicates
   something that now lives properly under `reports/eda/`, `reports/eda_xg/`,
   or `claude/eda-prompts/` (e.g. `Claude outputs/correlation_atlas.html` ==
   `reports/eda/correlation-atlas.html`; `Claude outputs/01-correlation-analysis-prompt.md`
   == `claude/eda-prompts/01-correlation-analysis-prompt.md`). Looks like an
   early, ad hoc output location from before the `reports/`/`claude/` structure
   was settled. Safe to delete once you've confirmed nothing in it is newer
   than its `reports/`/`claude/` counterpart (spot-checked a few -- they're
   not).

## Worth a look, not clear-cut

4. **`ARCHITECTURE.md`, `FIXES.md`, `FUNCTIONAL.md`** (repo root) -- 12-13 KB
   each, dated 17-22 July, never committed to git (no history under these
   paths at all -- they were never tracked). `docs/` has its own
   `project_overview.md`, `project_roadmap.md`, `limitations_and_next_steps.md`,
   `portfolio_story.md` and `README.md` (updated 16 Sep, current), and
   `docs/README.md` doesn't reference these three root files. They read like
   an earlier engineering-hygiene planning pass (FIXES.md pairs with
   `create-issues.sh`, FUNCTIONAL.md pairs with `create-functional-issues.sh`
   -- both scripts turn these into GitHub issues). If those issues have
   already been filed, the source docs may be safe to archive or delete; if
   you still plan to run either script against current state, keep them and
   refresh the content first, since they're two months stale against a repo
   that's moved on substantially since July.

5. **`create-issues.sh` / `create-functional-issues.sh`** -- these are NOT
   duplicates of each other (diffed them) -- one targets engineering hygiene
   (CI, tests, packaging), the other functional completeness (does the
   analysis do what it claims). Genuinely different scripts with similar
   names. No action needed unless #4 above is resolved, since they consume
   FIXES.md / FUNCTIONAL.md respectively.

## Not cleanup candidates (checked, ruled out)

- `outputs/coach_analysis/` (`cb_box_defence/` -- execution_summary.json,
  figures, tables, video_review) -- untracked but recently generated
  (current), reads as active analysis output, not stray. Left alone.
- `reports/eda_xg/active_category_atlas.json` / `active_flag_ledger.json` /
  `passive_category_atlas.json` / `passive_flag_ledger.json` -- untracked, but
  these are the live data files for the `.html` reports sitting right next to
  them (matching timestamps, matching names) -- not orphaned duplicates, just
  not yet `git add`ed.
- `.pytest_cache/`, `.ruff_cache/`, `.idea/`, `.venv/`, `mlruns/` -- all
  already gitignored, standard tool caches/environments. Normal to have on
  disk, not a cleanup item.

## What I'd do, if you want a recommendation

Delete `local_archive/` (biggest win, lowest risk -- gitignored, three months
stale) and `mlflow.db.bak_20260916182833` first. Hold off on `Claude outputs/`
and the three root docs until you've had a quick look -- say the word on any
of these five and I'll action it directly.
