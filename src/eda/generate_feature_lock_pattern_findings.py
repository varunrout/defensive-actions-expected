"""CLI entrypoint: prompt 34 -- fold prompts 21-31's pattern-analysis
findings into both lock-confirmation reports (extends them in place, does
not create a third file).

Premise correction, verified before writing anything: this prompt asks to
pull from reports/eda/PATTERN_ANALYSIS_CLOSEOUT.md (prompt 32) -- that file
does not exist (prompt 32 was never run in this session). The prompt's own
"Method" section already requires re-reading every figure from its live
source JSON rather than trusting the closeout doc, so this script does
exactly that directly, without needing the closeout file to exist.

Second correction, verified empirically rather than assumed: the prompt
claims tournament stability (Sec 4) and numeric interaction analysis
(Sec 7) are "target-agnostic in practice." They are not -- both are
computed independently per target and produce DIFFERENT verdicts on some
features/pairs:
  - Tournament stability: 3/10 features differ between
    reports/eda/TOURNAMENT_STABILITY_CHECK.json and the xg version
    (defenders_within_5m, local_numerical_balance_5m, attackers_within_5m).
  - Numeric interaction: 6/10 pairs differ between
    reports/eda/FEATURE_INTERACTION_ANALYSIS.json and the xg version's
    unconditional classification.
Both are therefore reported IN FULL in both lock-confirmation files (not
cross-referenced), each scoped to its own target's actual numbers.

Only slicer redundancy (SLICER_REDUNDANCY.json) and player-level validity
(PLAYER_LEVEL_VALIDITY_CHECK.json) are genuinely target-agnostic --
confirmed by construction (neither references a target column) and by the
xg versions being literal mirrors (generate_correlation_vif_xg_mirror.py).
These ARE cross-referenced from the xg file to the binary file rather than
duplicated.

Third correction: the prompt says reports/eda_xg/CONFOUND_ANALYSIS.json has
"2 tests" needing reproduction-confirmation. Reading it directly finds 5
(prompt 29 already extended it with 3 has_option_2/3 tests) -- all 5 are
reported, not just the original 2.

Usage:
    python -m src.eda.generate_feature_lock_pattern_findings
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EDA_DIR = REPO_ROOT / "reports" / "eda"
EDA_XG_DIR = REPO_ROOT / "reports" / "eda_xg"

BINARY_LOCK_PATH = EDA_DIR / "FEATURE_LOCK_CONFIRMATION.json"
XG_LOCK_PATH = EDA_XG_DIR / "FEATURE_LOCK_CONFIRMATION_XG.json"

CLOSEOUT_PATH = EDA_DIR / "PATTERN_ANALYSIS_CLOSEOUT.md"

PREMISE_CORRECTIONS = [
    (
        "reports/eda/PATTERN_ANALYSIS_CLOSEOUT.md (prompt 32) does not exist in this session -- prompt 32 was "
        "never run. Every figure below is read directly from its live source JSON instead, which the prompt's "
        "own Method section required anyway."
    ),
    (
        "Tournament stability is NOT target-agnostic, despite the prompt's claim -- verified by diffing "
        "reports/eda/TOURNAMENT_STABILITY_CHECK.json against reports/eda_xg/TOURNAMENT_STABILITY_CHECK.json "
        "directly: 3 of 10 features (defenders_within_5m, local_numerical_balance_5m, attackers_within_5m) get a "
        "DIFFERENT verdict depending on target. Reported in full in both lock-confirmation files, not "
        "cross-referenced."
    ),
    (
        "Numeric interaction analysis is NOT target-agnostic either -- verified by diffing "
        "reports/eda/FEATURE_INTERACTION_ANALYSIS.json against the xg version's unconditional classification: "
        "6 of 10 pairs differ. Reported in full in both lock-confirmation files, not cross-referenced."
    ),
    (
        "reports/eda_xg/CONFOUND_ANALYSIS.json has 5 tests, not the 2 the prompt assumed -- prompt 29 already "
        "extended it with 3 has_option_2/3 tests. All 5 are reported here."
    ),
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def numerical_rankings(active_atlas: dict, passive_atlas: dict, n: int = 5) -> dict:
    return {
        "active_top": [
            {"feature": f["feature"], "spearman_rho": f["spearman_rho"], "shape": f["shape"]}
            for f in active_atlas["features"][:n]
        ],
        "passive_top": [
            {"feature": f["feature"], "spearman_rho": f["spearman_rho"], "shape": f["shape"]}
            for f in passive_atlas["features"][:n]
        ],
        "note": f"Top {n} of each dataset's locked pool by |Spearman rho|, read from the atlas's own sort order.",
    }


def confound_summary(confound: dict) -> dict:
    return {
        "n_tests": len(confound["tests"]),
        "tests": [{"name": t["name"], "title": t["title"], "verdict": t["verdict"]["verdict"]} for t in confound["tests"]],
    }


def tournament_stability_summary(data: dict) -> dict:
    return {
        "n_features": len(data["features"]),
        "features": [
            {"feature": f["feature"], "dataset": f["dataset"], "verdict": f["verdict"]}
            for f in data["features"]
        ],
        "excluded": data["excluded_from_investigation"],
    }


def slice_stratification_binary_summary(v1: dict, v2: dict) -> dict:
    v1_summary = v1["cross_dataset_summary"]
    v2_summary = v2["cross_dataset_summary"]
    ex1 = v1["results"]["active"]["defenders_within_10m"]["phase_label"]
    ex2 = v1["results"]["passive"]["marking_tightness"]["phase_label"]

    return {
        "v1_categorical_slicers": {
            "n_cells": sum(len(sl) for f in v1["results"].values() for sl in f.values()),
            "n_genuine_divergences": v1_summary["n_genuine_divergences"],
            "n_features_with_at_least_one_diverging_slicer": v1_summary["n_features_with_at_least_one_diverging_slicer"],
        },
        "v2_archetype_and_boolean_slicers": {
            "n_cells": sum(len(sl) for f in v2["results"].values() for sl in f.values()),
            "n_genuine_divergences": v2_summary["n_genuine_divergences"],
        },
        "worked_examples": [
            {
                "feature": "defenders_within_10m", "slicer": "phase_label", "dataset": "active",
                "overall_shape": ex1["overall_shape"],
                "diverging": f"{ex1['n_categories_diverging']}/{ex1['n_categories']}",
                "takeaway": "Every phase category diverges from the overall shape -- the strongest interaction-term candidate found in the whole slice-stratification corpus.",
            },
            {
                "feature": "marking_tightness", "slicer": "phase_label", "dataset": "passive",
                "overall_shape": ex2["overall_shape"],
                "diverging": f"{ex2['n_categories_diverging']}/{ex2['n_categories']}",
                "takeaway": "Only 3 of 7 phases reproduce the overall monotonic-decreasing pattern -- real category-dependence, but not concentrated in a single outlier category.",
            },
        ],
    }


def slicer_redundancy_summary(data: dict) -> dict:
    out = {}
    for ds in ("active", "passive"):
        c = data["datasets"][ds]["clusters"]["consensus_both_metrics_agree"]
        out[ds] = {"redundant_clusters": c["clusters"], "independent_slicers": c["independent_slicers"]}
    out["note"] = "Consensus view (both Cramer's V and NMI agree) -- the one to trust, per prompt 27."
    return out


def numeric_interaction_summary(data: dict, given_shot: bool = False) -> dict:
    key = "classification_given_shot" if given_shot else "classification"
    counts = Counter(p[key] for p in data["pairs"])
    return {
        "classification_counts": dict(counts),
        "pairs": [
            {"feature_a": p["feature_a"], "feature_b": p["feature_b"], "dataset": p["dataset"], "classification": p[key]}
            for p in data["pairs"]
        ],
    }


def player_validity_summary(data: dict) -> dict:
    return {
        "n_distinct_players": data["row_concentration"]["n_distinct_players"],
        "gini_coefficient": data["row_concentration"]["gini_coefficient"],
        "train_test_player_overlap_pct": data["train_test_player_overlap"]["overlap_pct_of_test_players"],
        "risk_pairs": data["risk_pairs"],
        "scope_limitation": "Active dataset only -- passive has no player_id column.",
    }


def build_binary_findings() -> dict:
    active_atlas = _load(EDA_DIR / "active_numerical_target_atlas.json")
    passive_atlas = _load(EDA_DIR / "passive_numerical_target_atlas.json")
    confound = _load(EDA_DIR / "CONFOUND_ANALYSIS.json")
    tournament = _load(EDA_DIR / "TOURNAMENT_STABILITY_CHECK.json")
    slice_v1 = _load(EDA_DIR / "SLICE_STRATIFICATION.json")
    slice_v2 = _load(EDA_DIR / "SLICE_STRATIFICATION_V2.json")
    slicer_redundancy = _load(EDA_DIR / "SLICER_REDUNDANCY.json")
    interaction = _load(EDA_DIR / "FEATURE_INTERACTION_ANALYSIS.json")
    player_validity = _load(EDA_DIR / "PLAYER_LEVEL_VALIDITY_CHECK.json")

    return {
        "source_note": (
            "Every number below re-read live from its source JSON at the time this section was generated -- see "
            "generate_feature_lock_pattern_findings.py for the exact reads. reports/eda/PATTERN_ANALYSIS_CLOSEOUT.md "
            "does not exist in this session; not used as a source."
        ),
        "numerical_vs_target_rankings": numerical_rankings(active_atlas, passive_atlas),
        "confound_tests": confound_summary(confound),
        "tournament_stability": {
            **tournament_stability_summary(tournament),
            "target_agnosticism_note": (
                "NOT target-agnostic despite how it might look (it's about tournament, not target) -- verified "
                "against the xg version directly: 3/10 features get a different verdict. This section reports "
                "the binary-target numbers specifically; see leakage_confirmation_xg's sibling section in "
                "FEATURE_LOCK_CONFIRMATION_XG.json for the xg numbers, not assumed identical."
            ),
        },
        "slice_stratification": slice_stratification_binary_summary(slice_v1, slice_v2),
        "slicer_redundancy": {
            **slicer_redundancy_summary(slicer_redundancy),
            "target_agnosticism_note": "Genuinely target-agnostic (confirmed: never references a target column, computed once, mirrored unchanged into reports/eda_xg/).",
        },
        "numeric_interaction": {
            **numeric_interaction_summary(interaction),
            "target_agnosticism_note": (
                "NOT target-agnostic despite the prompt's claim -- verified against the xg version's unconditional "
                "classification directly: 6/10 pairs differ. This section reports the binary-target numbers "
                "specifically; see FEATURE_LOCK_CONFIRMATION_XG.json's sibling section for the xg numbers."
            ),
        },
        "player_level_validity": {
            **player_validity_summary(player_validity),
            "target_agnosticism_note": "Genuinely target-agnostic (confirmed: never references a target column, computed once, mirrored unchanged into reports/eda_xg/).",
        },
        "corrections_to_prompt_34s_premise": PREMISE_CORRECTIONS,
    }


def build_xg_findings() -> dict:
    active_atlas = _load(EDA_XG_DIR / "active_numerical_target_atlas.json")
    passive_atlas = _load(EDA_XG_DIR / "passive_numerical_target_atlas.json")
    structural_zero = _load(EDA_XG_DIR / "STRUCTURAL_ZERO_CHECK.json")
    confound = _load(EDA_XG_DIR / "CONFOUND_ANALYSIS.json")
    slice_v1 = _load(EDA_XG_DIR / "SLICE_STRATIFICATION.json")
    tournament = _load(EDA_XG_DIR / "TOURNAMENT_STABILITY_CHECK.json")
    interaction = _load(EDA_XG_DIR / "FEATURE_INTERACTION_ANALYSIS.json")

    slice_summary = slice_v1["cross_dataset_summary"]
    disagreement_example = next(r for r in slice_summary["conditioning_agreement"] if not r["agrees"] and r["diverges_unconditional"] and not r["diverges_given_shot"])

    return {
        "source_note": (
            "Every number below re-read live from its source JSON. reports/eda/PATTERN_ANALYSIS_CLOSEOUT.md does "
            "not exist in this session; not used as a source."
        ),
        "numerical_vs_target_rankings_xg": numerical_rankings(active_atlas, passive_atlas),
        "structural_zero_check": {
            "overall_verdict": structural_zero["overall_verdict"],
            "method": structural_zero["method"],
            "note": "target_future_xg_10s == 0 exactly wherever target_future_shot_10s == 0 -- this is why every shot-conditional panel in this pipeline exists: the unconditional xg curve mostly re-derives occurrence in different units.",
        },
        "confound_tests_xg": {
            **confound_summary(confound),
            "note": "5 tests, not 2 -- prompt 29 already extended this file with 3 has_option_2/3 tests; all 5 reported here.",
        },
        "slice_stratification_xg": {
            "n_genuine_divergences_unconditional": slice_summary["n_genuine_divergences"],
            "n_genuine_divergences_given_shot": slice_summary["n_genuine_divergences_given_shot"],
            "n_conditioning_disagreements": slice_summary["n_conditioning_disagreements"],
            "occurrence_vs_quality_worked_example": {
                **disagreement_example,
                "takeaway": "Diverges unconditionally but that divergence is occurrence-only -- flat once conditioned on target_future_shot_10s==1. The unconditional divergence was about whether a shot happens, not about its quality.",
            },
        },
        "tournament_stability_xg": {
            **tournament_stability_summary(tournament),
            "target_agnosticism_note": (
                "NOT target-agnostic -- 3/10 features (defenders_within_5m, local_numerical_balance_5m, "
                "attackers_within_5m) get a different verdict than reports/eda/FEATURE_LOCK_CONFIRMATION.json's "
                "tournament_stability section. Reported here in full, xg-specific, not cross-referenced."
            ),
        },
        "numeric_interaction_xg": {
            "unconditional": numeric_interaction_summary(interaction, given_shot=False),
            "given_shot": numeric_interaction_summary(interaction, given_shot=True),
            "n_pairs_where_unconditional_and_given_shot_agree": interaction["n_pairs_where_unconditional_and_given_shot_agree"],
            "target_agnosticism_note": (
                "NOT target-agnostic -- 6/10 pairs' unconditional classification differs from "
                "reports/eda/FEATURE_LOCK_CONFIRMATION.json's numeric_interaction section. Reported here in full, "
                "xg-specific, not cross-referenced."
            ),
        },
        "cross_referenced_target_agnostic_sections": {
            "slicer_redundancy": "See reports/eda/FEATURE_LOCK_CONFIRMATION.json's pattern_analysis_findings.slicer_redundancy -- genuinely target-agnostic (confirmed), this analysis doesn't depend on target type.",
            "player_level_validity": "See reports/eda/FEATURE_LOCK_CONFIRMATION.json's pattern_analysis_findings.player_level_validity -- genuinely target-agnostic (confirmed), active dataset only.",
        },
        "corrections_to_prompt_34s_premise": PREMISE_CORRECTIONS,
    }


def main() -> None:
    binary_lock = _load(BINARY_LOCK_PATH)
    xg_lock = _load(XG_LOCK_PATH)

    binary_lock["pattern_analysis_findings"] = build_binary_findings()
    xg_lock["pattern_analysis_findings"] = build_xg_findings()

    BINARY_LOCK_PATH.write_text(json.dumps(binary_lock, indent=2, default=str), encoding="utf-8")
    XG_LOCK_PATH.write_text(json.dumps(xg_lock, indent=2, default=str), encoding="utf-8")

    print(f"PATTERN_ANALYSIS_CLOSEOUT.md exists: {CLOSEOUT_PATH.exists()}")
    print(f"Wrote pattern_analysis_findings into {BINARY_LOCK_PATH}")
    print(f"Wrote pattern_analysis_findings into {XG_LOCK_PATH}")


if __name__ == "__main__":
    main()
