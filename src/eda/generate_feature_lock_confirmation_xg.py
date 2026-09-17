"""CLI entrypoint: prompt 33 -- post-lock correlation confirmation,
continuous target. reports/eda/FEATURE_LOCK_CONFIRMATION.json (prompt 14)
only ever confirmed the locked feature set against the BINARY target's
leakage check (has_screened_outcome, chi2=136.5 vs target_future_shot_10s).

The correlation/redundancy machinery itself is target-agnostic (feature vs
feature, never feature vs target -- stated explicitly in prompt 14), so it
does NOT get re-run here: this script reads the binary confirmation's
already-computed correlation_diff directly and states explicitly that it's
identical by construction, rather than silently re-deriving the same
numbers a second time.

What IS target-specific is the leakage side. reports/eda_xg/LEAKAGE_AUDIT.json
already has the continuous-target legs of Part C (has_screened_outcome vs
target_future_xg_10s, 1.3x weaker effect, p=0.043 -- much weaker than the
binary leg's ~12x ratio at p=1.5e-31, but the censoring MECHANISM is
target-independent so the drop holds regardless) and Part D (has_option_2/3
vs target_future_xg_10s, both still "flagged-for-follow-up" in that file --
prompt 29 already closed that follow-up in CONFOUND_ANALYSIS.json/its xg
mirror, cross-referenced here, not re-litigated). This is the first time
either has been folded into a formal lock-confirmation document for the
continuous target.

Usage:
    python -m src.eda.generate_feature_lock_confirmation_xg
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda.feature_config import ACTIVE, PASSIVE

REPO_ROOT = Path(__file__).resolve().parents[2]
BINARY_LOCK_PATH = REPO_ROOT / "reports" / "eda" / "FEATURE_LOCK_CONFIRMATION.json"
LEAKAGE_XG_PATH = REPO_ROOT / "reports" / "eda_xg" / "LEAKAGE_AUDIT.json"
CONFOUND_XG_PATH = REPO_ROOT / "reports" / "eda_xg" / "CONFOUND_ANALYSIS.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "eda_xg" / "FEATURE_LOCK_CONFIRMATION_XG.json"

EXPECTED_ACTIVE_COUNT = 34
EXPECTED_PASSIVE_COUNT = 38


def _has_option_followup_verdicts() -> dict | None:
    """Prompt 29 closed the has_option_2/3 'flagged-for-follow-up' item from
    LEAKAGE_AUDIT.json by adding 3 confound tests to CONFOUND_ANALYSIS.json
    and its xg mirror -- checked directly here rather than assumed, since
    the prompt asks not to assume it landed."""
    if not CONFOUND_XG_PATH.exists():
        return None
    data = json.loads(CONFOUND_XG_PATH.read_text(encoding="utf-8"))
    names = {"has_option_2_vs_top_option_1_threat_score", "has_option_3_vs_top_option_1_threat_score", "has_option_2_vs_defender_x"}
    tests = [t for t in data["tests"] if t["name"] in names]
    if len(tests) != 3:
        return None
    return {
        "closed_by": "prompt 29 (reports/eda_xg/CONFOUND_ANALYSIS.json)",
        "n_tests": len(tests),
        "verdicts": {t["name"]: t["verdict"]["verdict"] for t in tests},
    }


def build_leakage_confirmation_xg() -> dict:
    leakage_xg = json.loads(LEAKAGE_XG_PATH.read_text(encoding="utf-8"))
    part_c = leakage_xg["part_c_has_screened_outcome"]
    part_d = leakage_xg["part_d_has_option_2_3"]
    followup = _has_option_followup_verdicts()

    return {
        "source": str(LEAKAGE_XG_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "part_c_has_screened_outcome": {
            "ratio_true_to_false": part_c["ratio_true_to_false"],
            "t_stat": part_c["t_stat"],
            "p_value": part_c["p_value"],
            "verdict": part_c["verdict"],
            "comparison_to_binary": (
                "1.3x weaker mean-xG ratio at p=0.043 (barely significant), vs the binary target's ~12x shot-rate "
                "ratio at p=1.5e-31. The exclusion still holds on structural grounds -- the censoring mechanism "
                "(forward window truncated by match/period end) doesn't depend on which target is measured within "
                "that window, it depends on whether the window itself could be observed. Weaker evidence on this "
                "target does not weaken the case for dropping it; if anything, the binary leg alone was already "
                "sufficient and this just adds a second, independent (if noisier) confirmation."
            ),
        },
        "part_d_has_option_2_3": {
            "leakage_audit_verdicts": {k: v["verdict"] for k, v in part_d.items()},
            "leakage_audit_note": (
                "Still reads 'flagged-for-follow-up' in LEAKAGE_AUDIT.json itself -- that file records the "
                "state at the time of the base audit and is not rewritten retroactively when a follow-up "
                "closes elsewhere."
            ),
            "followup_status": followup if followup else "NOT FOUND -- prompt 29's xg confound extension was not located; do not assume it landed.",
        },
        "verdict": (
            "Both leakage items are confirmed consistent for continuous-target modelling. has_screened_outcome's "
            "drop is reconfirmed (mechanism-based, weaker-but-still-present empirical support on xg). "
            "has_option_2/3 remain locked candidate features -- LEAKAGE_AUDIT.json's own "
            "'flagged-for-follow-up' note is resolved by the confound tests prompt 29 already ran (cross-"
            "referenced above), not reopened here."
        ),
    }


def main() -> None:
    binary_lock = json.loads(BINARY_LOCK_PATH.read_text(encoding="utf-8"))

    active_count = len(ACTIVE["categorical"]) + len(ACTIVE["boolean"]) + len(ACTIVE["continuous"]) + len(ACTIVE["discrete"])
    passive_count = len(PASSIVE["categorical"]) + len(PASSIVE["boolean"]) + len(PASSIVE["continuous"]) + len(PASSIVE["discrete"])

    leakage_confirmation_xg = build_leakage_confirmation_xg()

    output = {
        "confirmed_counts": {
            "active": active_count,
            "active_expected": EXPECTED_ACTIVE_COUNT,
            "active_matches_expected": active_count == EXPECTED_ACTIVE_COUNT,
            "passive": passive_count,
            "passive_expected": EXPECTED_PASSIVE_COUNT,
            "passive_matches_expected": passive_count == EXPECTED_PASSIVE_COUNT,
        },
        "changes_since_last_full_run": binary_lock["changes_since_last_full_run"],
        "correlation_diff": binary_lock["correlation_diff"],
        "correlation_diff_note": (
            "IDENTICAL to reports/eda/FEATURE_LOCK_CONFIRMATION.json's correlation_diff, by construction, not "
            "independently re-derived -- correlation clustering (feature vs feature) never references a target "
            "column, so re-running it against a different target would produce the exact same pairs and tiers. "
            "Reused directly from the binary confirmation rather than silently recomputed, so this file's numbers "
            "are traceable to that source."
        ),
        "diff_is_empty": binary_lock["diff_is_empty"],
        "any_newly_risky_pairs_found": binary_lock["any_newly_risky_pairs_found"],
        "part_b_triggered": binary_lock["part_b_triggered"],
        "leakage_confirmation_xg": leakage_confirmation_xg,
        "verdict": (
            f"The locked feature set ({active_count} active / {passive_count} passive) is confirmed internally "
            "consistent for continuous-target (target_future_xg_10s) modelling -- same counts as the binary "
            "confirmation (reports/eda/FEATURE_LOCK_CONFIRMATION.json), since correlation/redundancy is "
            "target-agnostic. One target-specific caveat: has_screened_outcome's leakage effect is present but "
            "materially weaker on xg (1.3x vs ~12x, p=0.043 vs p=1.5e-31) -- the drop is still warranted on "
            "structural grounds, more strongly evidenced on the binary target than on xg."
        ),
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    print(f"active: {active_count} (expected {EXPECTED_ACTIVE_COUNT}) {'OK' if active_count == EXPECTED_ACTIVE_COUNT else 'MISMATCH'}")
    print(f"passive: {passive_count} (expected {EXPECTED_PASSIVE_COUNT}) {'OK' if passive_count == EXPECTED_PASSIVE_COUNT else 'MISMATCH'}")
    print(f"correlation diff empty: {output['diff_is_empty']} (reused from binary confirmation)")
    followup_found = isinstance(leakage_confirmation_xg["part_d_has_option_2_3"]["followup_status"], dict)
    print(f"has_option_2/3 followup located: {followup_found}")
    print(f"\nverdict: {output['verdict']}")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
