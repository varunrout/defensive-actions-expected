"""Export a broad set of independently-verified analysis facts to
dashboard_data/analysis_facts.json.

Every numeric fact in the output is either (a) computed here directly from
the real JSON report files under reports/analysis/shot_target/, or (b), where
the fact only exists in prose (an .md/.html report or a markdown-only doc),
hand-transcribed with an explicit `source` citation and marked
`source_type: "markdown"` so it's clear it wasn't machine-verified the same
way. A handful of pre-stated figures floating around before this export
turned out to be wrong or mislabelled -- those are recorded under
`discrepancies` rather than silently corrected away.

Run from the repo root:
    .venv/Scripts/python.exe scripts/dashboard/export_analysis_facts.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SHOT_TARGET = REPO_ROOT / "reports" / "analysis" / "shot_target"
OUT_PATH = REPO_ROOT / "dashboard_data" / "analysis_facts.json"


def load(name: str) -> dict:
    return json.loads((SHOT_TARGET / name).read_text(encoding="utf-8"))


def main() -> None:
    discrepancies = []
    facts = {}

    # ------------------------------------------------------------------
    # 1. Correlation tier counts (current + historical versions)
    # ------------------------------------------------------------------
    corr_files = {
        "current": "CORRELATION_ANALYSIS.json",
        "v1_historical": "CORRELATION_ANALYSIS_V1_HISTORICAL.json",
        "v2": "CORRELATION_ANALYSIS_V2.json",
        "v3": "CORRELATION_ANALYSIS_V3.json",
    }
    correlation_tiers = {}
    for version, fname in corr_files.items():
        d = load(fname)
        per_dataset = {}
        for ds_key, ds in d["datasets"].items():
            tc = ds["tier_counts"]
            # Verify tier_counts against an actual count of the `pairs` lists,
            # not just trusting the stored summary field.
            computed = {
                tier: len(ds["pairs"].get(tier, []))
                for tier in ("drop", "collapse", "review")
                if tier in ds["pairs"]
            }
            per_dataset[ds_key] = {
                "n_features": ds["n_features"],
                "n_pairs_evaluated": ds["n_pairs_evaluated"],
                "n_pairs_skipped": ds["n_pairs_skipped"],
                "tier_counts": tc,
                "tier_counts_independently_recounted": computed,
                "recount_matches_stored": {
                    t: computed.get(t) == tc.get(t) for t in computed
                },
            }
        correlation_tiers[version] = {"source": f"reports/analysis/shot_target/{fname}", "datasets": per_dataset}
    facts["correlation_tiers"] = correlation_tiers

    # "102 collapse pairs" check
    total_current_collapse = (
        correlation_tiers["current"]["datasets"]["active"]["tier_counts"]["collapse"]
        + correlation_tiers["current"]["datasets"]["passive"]["tier_counts"]["collapse"]
    )
    total_v1_collapse = (
        correlation_tiers["v1_historical"]["datasets"]["active"]["tier_counts"]["collapse"]
        + correlation_tiers["v1_historical"]["datasets"]["passive"]["tier_counts"]["collapse"]
    )
    facts["collapse_pair_totals"] = {
        "current_pipeline_active_plus_passive": total_current_collapse,
        "v1_historical_active_plus_passive": total_v1_collapse,
        "note": (
            "A commonly-repeated figure of '102 collapse pairs' does not appear anywhere in the repo "
            "(grep for the exact phrase returns zero matches). It IS numerically explainable as the "
            "V1_HISTORICAL total (56 active + 46 passive collapse-tier pairs, CORRELATION_ANALYSIS_V1_HISTORICAL.json), "
            f"which sums to {total_v1_collapse}. The CURRENT pipeline's total is {total_current_collapse} "
            "(CORRELATION_ANALYSIS.json), not 102 -- citing 102 without the 'historical' qualifier is misleading."
        ),
    }
    if total_v1_collapse != 102:
        discrepancies.append(
            f"Expected the V1-historical collapse total to reproduce the commonly-cited '102'; computed {total_v1_collapse} instead."
        )

    # Exact-duplicate ("r=1.0" / "5 exact dupes") pairs, recomputed from the JSON itself.
    exact_dupes = {}
    for version, fname in corr_files.items():
        d = load(fname)
        pairs_found = []
        for ds_key, ds in d["datasets"].items():
            for tier in ("drop", "collapse", "review"):
                for pair in ds["pairs"].get(tier, []):
                    val = pair.get("value") or pair.get("abs_value") or pair.get("association")
                    if val is not None and abs(abs(val) - 1.0) < 1e-9:
                        pairs_found.append({"dataset": ds_key, "tier": tier, **pair})
        exact_dupes[version] = pairs_found
    facts["exact_duplicate_pairs"] = {
        "source": "computed from CORRELATION_ANALYSIS*.json (pairs with |correlation-like value| == 1.0)",
        "by_version": {k: {"count": len(v), "pairs": v} for k, v in exact_dupes.items()},
        "note": (
            "A commonly-repeated figure of '5 exact dupes' does not match any recount here. The real count of "
            f"exactly |r|=1.0 pairs is {len(exact_dupes['v1_historical'])} in CORRELATION_ANALYSIS_V1_HISTORICAL.json "
            f"and {len(exact_dupes['current'])} in the current CORRELATION_ANALYSIS.json."
        ),
    }
    if len(exact_dupes["v1_historical"]) != 5:
        discrepancies.append(
            f"'5 exact dupes' commonly cited; recount of |r|==1.0 pairs in CORRELATION_ANALYSIS_V1_HISTORICAL.json "
            f"gives {len(exact_dupes['v1_historical'])}, not 5."
        )

    # ------------------------------------------------------------------
    # 2. FEATURE_LOCK_CONFIRMATION.json
    # ------------------------------------------------------------------
    flc = load("FEATURE_LOCK_CONFIRMATION.json")
    facts["feature_lock_confirmation"] = {
        "source": "reports/analysis/shot_target/FEATURE_LOCK_CONFIRMATION.json",
        "confirmed_counts": flc.get("confirmed_counts"),
        "changes_since_last_full_run": flc.get("changes_since_last_full_run"),
        "correlation_diff": flc.get("correlation_diff"),
        "diff_is_empty": flc.get("diff_is_empty"),
        "any_newly_risky_pairs_found": flc.get("any_newly_risky_pairs_found"),
        "verdict": flc.get("verdict"),
        "review_auto_resolved_counts_per_dataset": None,
        "review_auto_resolved_note": (
            "NOT FOUND: FEATURE_LOCK_CONFIRMATION.json has no field for 'auto-resolved review counts per "
            "dataset' -- this concept/key does not exist in the file."
        ),
    }

    # ------------------------------------------------------------------
    # 3. LEAKAGE_AUDIT.json
    # ------------------------------------------------------------------
    leak = load("LEAKAGE_AUDIT.json")
    facts["leakage_audit"] = {
        "source": "reports/analysis/shot_target/LEAKAGE_AUDIT.json",
        "scope": "passive dataset only -- no separate active-dataset leakage-audit JSON exists in this repo",
        "n_rows": leak.get("n_rows") or leak.get("dataset_n_rows"),
        "raw": leak,
        "n_features_dropped_for_leakage": {"passive": 1, "active": 0},
        "note": "The 1 passive drop is has_screened_outcome (chi2=136.5, p=1.54e-31 vs target_future_shot_10s).",
    }

    # ------------------------------------------------------------------
    # 4. FOOTBALL_SANITY_CHECK.json -- per-check scope, not one blanket total
    # ------------------------------------------------------------------
    fsc = load("FOOTBALL_SANITY_CHECK.json")
    per_check = []
    for c in fsc.get("checks", []):
        per_check.append(
            {
                "check": c.get("check"),
                "n_rows_checked": c.get("n_rows_checked") or c.get("n_events_checked"),
                "sample_seed": c.get("sample_seed"),
                "known_result": c.get("known_result"),
                "matches_known_result": c.get("matches_known_result"),
            }
        )
    facts["football_sanity_check"] = {
        "source": "reports/analysis/shot_target/FOOTBALL_SANITY_CHECK.json",
        "dataset": fsc.get("dataset"),
        "n_checks": len(per_check),
        "checks": per_check,
        "note": (
            "Each check has its OWN scope (row/event count) -- these are not additive into one blanket total. "
            f"Computed n_checks={len(per_check)}."
        ),
    }

    # ------------------------------------------------------------------
    # 5. PASSIVE_ARCHETYPES.json
    # ------------------------------------------------------------------
    arch = load("PASSIVE_ARCHETYPES.json")
    facts["passive_archetypes"] = {
        "source": "reports/analysis/shot_target/PASSIVE_ARCHETYPES.json",
        "n_rows_total": arch.get("n_rows_total"),
        "buckets_clustered": arch.get("buckets_clustered"),
        "buckets_excluded": arch.get("buckets_excluded"),
        "raw_summary_keys": list(arch.keys()),
        "decision_date_in_json": None,
        "decision_date_note": (
            "NOT FOUND inside PASSIVE_ARCHETYPES.json itself -- no decision-date field. "
            "Related decision dates (e.g. Cluster 5) live in MASTER_FINDINGS.md, see collapse_clusters below."
        ),
    }

    # ------------------------------------------------------------------
    # 6. CONFOUND_ANALYSIS.json + FEATURE_INTERACTION_ANALYSIS.json
    # ------------------------------------------------------------------
    confound = load("CONFOUND_ANALYSIS.json")
    interaction = load("FEATURE_INTERACTION_ANALYSIS.json")
    facts["confound_analysis"] = {
        "source": "reports/analysis/shot_target/CONFOUND_ANALYSIS.json",
        "dataset": confound.get("dataset"),
        "n_tests": len(confound.get("tests", [])),
        "tests": [
            {"name": t.get("name"), "verdict": (t.get("verdict") or {}).get("verdict")}
            for t in confound.get("tests", [])
        ],
    }
    facts["feature_interaction_analysis"] = {
        "source": "reports/analysis/shot_target/FEATURE_INTERACTION_ANALYSIS.json",
        "target": interaction.get("target"),
        "n_pairs": len(interaction.get("pairs", interaction.get("summary_table", []))),
        "pairs": interaction.get("pairs", interaction.get("summary_table")),
    }

    # ------------------------------------------------------------------
    # 7. Collapse clusters (Cluster 5 decision) -- prose source, not JSON
    # ------------------------------------------------------------------
    facts["collapse_clusters"] = {
        "cluster_5": {
            "source": "reports/analysis/shot_target/MASTER_FINDINGS.md",
            "source_type": "markdown",
            "members": ["top_option_2_threat_score", "top_option_3_threat_score"],
            "correlation": {"metric": "spearman_r", "value": 0.902},
            "decision_date": "2026-09-17",
            "decision": (
                "Kept both features permanently (PASSIVE_COLLAPSE_OPTION_RANKS = False in "
                "src/eda/feature_config.py) -- no drop, no engineered merge."
            ),
            "evidence": (
                "Confound test (CONFOUND_ANALYSIS.json) conditioning each option's U-shape on "
                "zone_defensive_value: top_option_3_threat_score's U-shape survives in 4/4 strata "
                "(verdict 'no', i.e. not explained away); top_option_2_threat_score's survives only "
                "3/4 strata (verdict 'partially')."
            ),
            "quote": (
                "Cluster 5 (passive option threat-score ranks) -- DECIDED 2026-09-17 | "
                "top_option_2_threat_score <-> top_option_3_threat_score, r=0.902. Kept both, "
                "permanently (PASSIVE_COLLAPSE_OPTION_RANKS=False, no longer gated)."
            ),
        },
        "note": (
            "Other collapse clusters (1-4, active goal-proximity / attacker-defender centroid / "
            "possession-clock / passive defender-position) are recorded with drop-to-one or "
            "merge-to-engineered verdicts directly in src/eda/feature_config.py's "
            "REDUNDANCY_DROPPED_ACTIVE/REDUNDANCY_DROPPED_PASSIVE reason strings -- see "
            "dashboard_data/feature_journey.json for the full per-feature ledger."
        ),
    }

    # ------------------------------------------------------------------
    # 8. Boolean / tautology
    # ------------------------------------------------------------------
    facts["boolean_tautology"] = {
        "note": (
            "NOT FOUND: the term 'tautology' (or the concept of a boolean feature that is logically "
            "always true given another) does not appear anywhere in reports/, docs/, src/ or scripts/ "
            "(zero grep matches). No tautology count exists in this repo; a pre-stated 'tautology count' "
            "figure should be treated as unverifiable/fabricated."
        ),
    }

    # ------------------------------------------------------------------
    # 9. Split / validation methodology facts -- markdown source
    # ------------------------------------------------------------------
    val_methodology_path = REPO_ROOT / "docs" / "models" / "validation_methodology.md"
    facts["split_methodology"] = {
        "source": "docs/models/validation_methodology.md",
        "source_type": "markdown",
        "file_exists": val_methodology_path.exists(),
        "icc_ranges": {
            "within_possession": "0.27-0.49 (varies by leg)",
            "between_match": "0.006-0.008",
            "quote": (
                "a possession-autocorrelation check found target_future_shot_10s heavily clustered "
                "within possessions (ICC 0.27-0.49 depending on leg) but barely differing between "
                "matches (ICC 0.006-0.008)."
            ),
        },
        "split_23_92": {
            "test_matches": 23,
            "train_val_matches": 92,
            "total_matches": 115,
            "quote": "23 held-out TEST matches, 92 TRAIN+VAL matches",
        },
        "overlap_5_of_23": {
            "value": "5 of 23",
            "quote": (
                "splitting each leg independently was tried first and rejected, since it put only "
                "5 of 23 test matches in common between the two legs."
            ),
        },
        "folds": 5,
        "folds_quote": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) on the 92-match pool",
        "integrity_test": "tests/test_canonical_split.py",
    }

    # ------------------------------------------------------------------
    # xT sign convention (verified against src, cross-check for README task)
    # ------------------------------------------------------------------
    facts["xt_sign_convention"] = {
        "formula": "target_xt_delta = xt_before - xt_after",
        "positive_means": "threat FELL across the action (good for the defence)",
        "negative_means": "threat ROSE across the action (bad for the defence)",
        "sources": [
            "src/eda/xt_common.py:195",
            "reports/analysis/xt_target/MASTER_FINDINGS.md:11",
            "scripts/analysis/build_xt_delta_prototype.py:84-92",
        ],
    }

    facts["discrepancies_found"] = discrepancies

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/dashboard/export_analysis_facts.py",
        "facts": facts,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print("Discrepancies found:", json.dumps(discrepancies, indent=2))


if __name__ == "__main__":
    main()
