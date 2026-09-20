"""CLI entrypoint: leakage audit for `target_xt_delta`, ACTIVE dataset.

STEP-0 SCOPE NOTE, stated plainly rather than papered over.
reports/analysis/xg_target/LEAKAGE_AUDIT.json/.html were produced by
generate_leakage_audit_xg.py / generate_leakage_report_xg.py (confirmed by
reading them), and BOTH ARE PASSIVE-DATASET-ONLY. Their Parts B, C and D
quantify three passive-specific columns -- `screened_option_was_avoided`,
`has_screened_outcome`, `has_option_2`/`has_option_3` -- none of which
exist in data/features/player_defensive_actions.parquet. No active-side
leakage audit exists anywhere in this repo for any target (searched
reports/analysis/shot_target/ and reports/analysis/xg_target/; both hold a
single passive-only LEAKAGE_AUDIT).

Since this prompt is scoped to the active-binary leg only, a literal
mirror is impossible. What is mirrored is the METHOD, part by part, and
where a part has no active counterpart that is said so in the report
itself rather than substituted for silently:

  Part A -- IDENTICAL METHOD, reused by import. `generate_leakage_audit.part_a`
    is target- and dataset-independent (a schema/naming scan), so it is
    imported unchanged and pointed at the ACTIVE parquet and the ACTIVE
    candidate list.

  Part B/C/D -- NO ACTIVE COUNTERPART. The columns those parts test do not
    exist here. Stated as such in the report.

  Part B' (substituting for B/C) -- the same method (groupby mean target +
    Welch's t-test, exactly as generate_leakage_audit_xg.py uses) applied
    to the ACTIVE dataset's own censoring/computability proxy,
    `has_previous_event`. This is the closest structural analogue of
    passive's `has_screened_outcome`: both flag rows where an adjacent
    event needed to compute something was unavailable.

  Part C' (substituting for D) -- the same method applied to the three
    `action_*_possession` flags, which feature_config.py excludes with the
    reason "structural zero (0.00% shot rate by target-window definition)".
    That justification is target-specific; whether it survives on this
    target is exactly a leakage-audit question.

  Part D' -- NEW, no xg counterpart, and the most important finding for
    this target. `target_xt_delta = xt_before - xt_after`, and `xt_after`
    is computed directly from `action_x`/`action_y`. Several LOCKED active
    features are deterministic functions of that same action location, so
    part of the target is definitionally recoverable from them. This is a
    construction-coupling risk that simply does not arise for
    `target_future_xg_10s` (a forward-looking outcome), so there is no
    established method to mirror -- the method used is the plain one
    (Pearson r of each locked feature against `xt_after`, `xt_before` and
    the delta), and it is labelled as new rather than presented as the
    established audit.

Usage:
    python -m src.eda.generate_leakage_audit_xt
"""

from __future__ import annotations

import json

import pandas as pd
from scipy.stats import ttest_ind

from src.eda import xt_common as xc
from src.eda.feature_config import ACTIVE
from src.eda.generate_leakage_audit import _candidate_columns, part_a

OUTPUT_PATH = xc.OUT_DIR / "LEAKAGE_AUDIT.json"
TARGET = xc.TARGET

# Columns that feed target_xt_delta's own construction (Prompt 64 section 2).
TARGET_CONSTRUCTION_COLUMNS = ["action_x", "action_y"]


def _split_stats(df: pd.DataFrame, col: str) -> dict:
    """groupby mean target + Welch's t-test -- the exact method
    generate_leakage_audit_xg.py uses for its Parts B/C, with the
    negative/zero/positive shares added (Adaptation 3)."""
    sub = df.dropna(subset=[TARGET])
    rows = {}
    for label, mask in (("True", sub[col] == True), ("False", sub[col] == False)):  # noqa: E712
        s = sub.loc[mask, TARGET]
        if len(s) == 0:
            continue
        rows[label] = {
            "n": int(len(s)),
            "mean_xt": round(float(s.mean()), 8),
            "std_xt": round(float(s.std()), 8),
            "pct_zero": round(float((s == 0).mean() * 100), 2),
            "pct_negative": round(float((s < 0).mean() * 100), 2),
            "pct_positive": round(float((s > 0).mean() * 100), 2),
        }
    s_t = sub.loc[sub[col] == True, TARGET]   # noqa: E712
    s_f = sub.loc[sub[col] == False, TARGET]  # noqa: E712
    t_stat, p_value = ttest_ind(s_t, s_f, equal_var=False)
    mt, mf = rows.get("True", {}).get("mean_xt"), rows.get("False", {}).get("mean_xt")
    return {
        "column": col,
        "target_col": TARGET,
        "by_value": rows,
        "delta": round(mt - mf, 8) if mt is not None and mf is not None else None,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "zero_share_gap_pp": (
            round(rows["False"]["pct_zero"] - rows["True"]["pct_zero"], 2)
            if "True" in rows and "False" in rows else None
        ),
    }


def part_b_prime(df: pd.DataFrame) -> dict:
    """has_previous_event -- the ACTIVE analogue of passive's
    has_screened_outcome censoring-mechanism proxy."""
    r = _split_stats(df, "has_previous_event")
    tr, fa = r["by_value"]["True"], r["by_value"]["False"]
    r.update({
        "substitutes_for": "Part C of the passive xg audit (has_screened_outcome)",
        "method": "groupby mean target + Welch's t-test -- identical to generate_leakage_audit_xg.part_c_xg",
        "verdict": "flagged -- real censoring-shaped effect, but on the ZERO SHARE rather than the mean",
        "explanation": (
            f"has_previous_event=False rows (n={fa['n']:,}) are the first event of a possession, so there is no "
            f"in-possession predecessor. Their mean delta ({fa['mean_xt']:+.6f}) is only modestly different from "
            f"True's ({tr['mean_xt']:+.6f}, Welch p={r['p_value']:.3e}) -- but the EXACTLY-ZERO share is "
            f"{fa['pct_zero']:.1f}% versus {tr['pct_zero']:.1f}%, a {r['zero_share_gap_pp']:+.1f}pp gap. Reading "
            "the mean alone, as the xg audit's method does, would have understated this almost entirely: the "
            "effect lives in the target's zero mass, not its centre. That is a direct consequence of this "
            "target's shape (Prompt 64: symmetric, 11.3% exact zeros) and is why the zero/negative/positive "
            "shares are reported alongside every mean in this suite. Reported as a finding about the target's "
            "measurement, NOT as a proposal to drop has_previous_event -- it is a locked feature and this audit "
            "does not change feature_config.py."
        ),
    })
    return r


def part_c_prime(df: pd.DataFrame) -> dict:
    """The three action_*_possession flags, excluded from the ACTIVE
    candidate list on a target-specific 'structural zero' justification."""
    results = {}
    for col in ("action_changed_possession", "action_ended_possession", "action_won_possession"):
        r = _split_stats(df, col)
        tr = r["by_value"].get("True", {})
        r.update({
            "substitutes_for": "Part D of the passive xg audit (has_option_2 / has_option_3)",
            "method": "groupby mean target + Welch's t-test -- identical to generate_leakage_audit_xg's Parts B-D",
            "feature_config_exclusion_reason": ACTIVE["excluded"].get(col),
            "verdict": "exclusion justification does NOT transfer to this target -- flagged, not reopened",
            "explanation": (
                f"feature_config.py excludes {col} because it is a structural zero against "
                f"target_future_shot_10s (0.00% shot rate by that target's own window definition). Against "
                f"{TARGET} the True group is not degenerate at all: n={tr.get('n', 0):,}, mean "
                f"{tr.get('mean_xt', float('nan')):+.6f}, only {tr.get('pct_zero', float('nan')):.1f}% exactly "
                f"zero, {tr.get('pct_negative', float('nan')):.1f}% negative and "
                f"{tr.get('pct_positive', float('nan')):.1f}% positive -- real row-level variance. This "
                "reproduces Prompt 64's central result (the degenerate-zero collapse is what the xT target was "
                "built to fix) on the full active dataset rather than only inside the two slices Prompt 64 "
                "examined. Reported as an observation, NOT a proposal to unlock the column: changing "
                "feature_config.py is outside this report suite's remit, and the column would still need its own "
                "leakage review before any such change."
            ),
        })
        results[col] = r
    return results


def part_d_prime(df: pd.DataFrame) -> dict:
    """NEW -- no xg counterpart. Construction coupling between locked
    location-derived features and the target's own xt_after term."""
    locked = sorted(set(ACTIVE["continuous"]) | set(ACTIVE["discrete"]))
    sub = df.dropna(subset=[TARGET])
    rows = []
    for col in locked:
        if not pd.api.types.is_numeric_dtype(sub[col]):
            continue
        rows.append({
            "feature": col,
            "r_vs_xt_after": round(float(sub[col].corr(sub["xt_after"])), 4),
            "r_vs_xt_before": round(float(sub[col].corr(sub["xt_before"])), 4),
            "r_vs_target": round(float(sub[col].corr(sub[TARGET])), 4),
        })
    rows.sort(key=lambda r: abs(r["r_vs_target"]), reverse=True)

    # Confirm directly (not assumed) that the construction columns themselves
    # are already out of the candidate list.
    candidate_cols = _candidate_columns(ACTIVE)
    construction_status = {
        c: {
            "in_candidate_list": c in candidate_cols,
            "exclusion_reason": ACTIVE["excluded"].get(c),
            "r_vs_xt_after": round(float(sub[c].corr(sub["xt_after"])), 4) if c in sub.columns else None,
        }
        for c in TARGET_CONSTRUCTION_COLUMNS
    }

    strongest = rows[0] if rows else None
    return {
        "method": (
            "NEW for this target -- no xg counterpart exists, and this is labelled as new rather than presented "
            "as the established audit method. Pearson r of every locked numerical ACTIVE feature against "
            "xt_after, xt_before and the delta itself."
        ),
        "why_this_part_exists": (
            f"{TARGET} = xt_before - xt_after, and xt_after is looked up directly from action_x/action_y "
            "(Prompt 64 section 2). Any feature that is a deterministic function of the action's own pitch "
            "location therefore shares a definitional term with half the target. target_future_xg_10s, being a "
            "forward-looking outcome measured after the action, has no such coupling -- so this risk is specific "
            "to the xT target and had to be checked rather than inherited."
        ),
        "target_construction_columns": construction_status,
        "locked_feature_correlations": rows,
        "strongest": strongest,
        "verdict": "flagged -- construction coupling is real and must be stated when interpreting this atlas",
        "explanation": (
            f"The construction columns themselves (action_x/action_y) are already OUT of the ACTIVE candidate "
            f"list, excluded as verified duplicates of ball_x/ball_y -- confirmed directly here, not assumed. "
            f"But several LOCKED features are still deterministic functions of the same location: the strongest "
            f"is {strongest['feature']} at r={strongest['r_vs_target']:+.4f} against the target and "
            f"r={strongest['r_vs_xt_after']:+.4f} against xt_after specifically. This is NOT leakage in the "
            "forward-window sense the passive audit deals with -- nothing here is knowable only because of what "
            "happens after the action. It is construction coupling: the feature and one term of the target are "
            "computed from the same coordinates. A model fed these features will partly recover xt_after by "
            "definition, and a strong correlation in the Numerical Target Atlas should be read with that in "
            "mind rather than as discovered defensive signal. No feature is dropped on this evidence; it is "
            "recorded so the atlas is read correctly."
        ),
        "prompt_64_cross_reference": xc.PROMPT_64_CLEARANCE_FINDING,
        "clearance_link": (
            "This is the same root cause as Prompt 64's Clearance artefact, generalised: because xt_after scores "
            "the action's OWN location rather than the ball's resulting location, every location-derived feature "
            "inherits that definitional choice. The Clearance case is the most visible symptom (see the Category "
            "Atlas, where Clearance ranks last of 8 event types); this part is the underlying mechanism."
        ),
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = xc.load_active_xt()
    candidate_cols = _candidate_columns(ACTIVE)

    a = part_a(df, candidate_cols)
    b = part_b_prime(df)
    c = part_c_prime(df)
    d = part_d_prime(df)

    output = {
        "dataset": "active",
        "leg": "active-binary only (passive leg out of scope for this target)",
        "parquet_path": ACTIVE["parquet_path"],
        "n_rows": len(df),
        "target_col": TARGET,
        "target_scale": xc.target_scale(df),
        "step_0_scope_note": (
            "CONFIRMED BY READING THE SOURCE, not inferred from filenames: the xg portal's LEAKAGE_AUDIT.json/"
            ".html are produced by generate_leakage_audit_xg.py / generate_leakage_report_xg.py, and both are "
            "PASSIVE-DATASET-ONLY. Their Parts B/C/D test screened_option_was_avoided, has_screened_outcome and "
            "has_option_2/3 -- none of which exist in the active parquet. No active-side leakage audit exists "
            "in this repo for any target. This prompt is scoped to the active leg, so a literal mirror is "
            "impossible. Part A's method is reused by direct import (it is dataset- and target-independent). "
            "Parts B/C/D are reported as HAVING NO ACTIVE COUNTERPART, and the same method is applied instead "
            "to the active dataset's own analogous columns (Parts B' and C'). Part D' is new to this target and "
            "is labelled as new rather than presented as the established method."
        ),
        "methodology": (
            "Part A (target- and dataset-independent schema/naming scan) imported unchanged from "
            "generate_leakage_audit.part_a. Parts B'/C' use the same groupby-mean-target + Welch's t-test the xg "
            "audit uses, with negative/zero/positive shares added because this target's effects can live in the "
            "zero mass rather than the mean. Part D' is a new construction-coupling check specific to this "
            "target."
        ),
        "parts_with_no_active_counterpart": {
            "part_b_screened_option_was_avoided": "passive-only column; does not exist in the active parquet",
            "part_c_has_screened_outcome": "passive-only column; does not exist in the active parquet",
            "part_d_has_option_2_3": "passive-only columns; do not exist in the active parquet",
        },
        "part_a_known_leaky_scan": a,
        "part_b_prime_has_previous_event": b,
        "part_c_prime_action_possession_flags": c,
        "part_d_prime_construction_coupling": d,
        "prompt_64_cross_references": {
            "clearance_action_x_artefact": xc.PROMPT_64_CLEARANCE_FINDING,
            "distribution_shape": xc.PROMPT_64_SHAPE_FINDING,
        },
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    print("=== Part A ===", "clean" if a["verdict"] == "clean" else "ISSUES FOUND")
    print(f"Part B' has_previous_event: zero-share gap {b['zero_share_gap_pp']:+.1f}pp, Welch p={b['p_value']:.3e}")
    for col, r in c.items():
        tr = r["by_value"].get("True", {})
        print(f"Part C' {col}: True n={tr.get('n'):,} mean={tr.get('mean_xt'):+.6f} zero={tr.get('pct_zero'):.1f}%")
    print(f"Part D' strongest construction coupling: {d['strongest']['feature']} "
          f"r_vs_target={d['strongest']['r_vs_target']:+.4f} r_vs_xt_after={d['strongest']['r_vs_xt_after']:+.4f}")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
