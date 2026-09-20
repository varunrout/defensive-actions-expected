"""CLI entrypoint: confound (reversal) testing against `target_xt_delta`,
ACTIVE dataset.

STEP-0 SCOPE NOTE, stated plainly.
reports/analysis/xg_target/CONFOUND_ANALYSIS.json holds 5 tests, produced by
generate_confound_analysis_xg.py (2 tests) and
generate_confound_analysis_part_d_xg.py (3 more) -- confirmed by reading
both. ALL FIVE ARE PASSIVE-DATASET tests, on passive-only columns
(marking_tightness, lane_screening_score_option_*, top_option_*_threat_score,
has_option_2/3, defender_x). None of those columns exist in the active
parquet.

Searching the whole repo, exactly ONE active-dataset confound test exists,
anywhere, for any target: `defenders_within_10m_vs_box_proximity`, added to
reports/analysis/shot_target/CONFOUND_ANALYSIS.json by
generate_confound_analysis_part_a.py (its "test_7"). The xg mirror never
received it -- generate_confound_analysis_part_d_xg.py's own docstring says
so explicitly ("reports/analysis/xg_target/CONFOUND_ANALYSIS.json currently
only has the original 2 tests mirrored (never received prompt 24 Part A's 5
additions)").

So this file:
  Test 1 -- MIRRORS that single existing active test exactly (same marginal
    column, same quartile labels, same confound-selection rule recomputed
    live, same reliability caveat carried forward), against target_xt_delta.
  Test 2 -- NEW, labelled as new. Motivated directly by Prompt 64's
    Clearance/action_x finding, which the Category Atlas confirmed
    resurfaces in this portal. It asks whether Clearance's negative mean
    delta survives conditioning on whether the action happened in the
    defending box -- i.e. whether the artefact is purely a deep-box
    location effect or something broader.

METHOD carried over unchanged: `_verdict()` is imported from
generate_confound_analysis and reused byte-for-byte. It compares only the
SIGN of first-to-last deltas across strata, never a magnitude threshold, so
it is already target-scale-agnostic -- the same reason
generate_confound_analysis_xg.py reused it. QCUT_N = 4 likewise unchanged.
Only the table builders needed an xT variant (mean delta, plus the
negative/zero/positive shares).

ADAPTATION 2: each test carries a `given_nonzero_delta` sub-object in place
of the xg version's `given_shot` sub-object.

Usage:
    python -m src.eda.generate_confound_analysis_xt
"""

from __future__ import annotations

import json

import pandas as pd

from src.eda import xt_common as xc
from src.eda.generate_confound_analysis import QCUT_N, _verdict

OUTPUT_PATH = xc.OUT_DIR / "CONFOUND_ANALYSIS.json"
TARGET = xc.TARGET


def _agg(s: pd.Series) -> dict:
    return {
        "n": int(len(s)),
        # "rate" key name kept because the reused _verdict() hardcodes it --
        # it holds a mean xT delta here, not a shot-rate percentage.
        "rate": round(float(s.mean()), 8) if len(s) else None,
        "pct_zero": round(float((s == 0).mean() * 100), 2) if len(s) else None,
        "pct_negative": round(float((s < 0).mean() * 100), 2) if len(s) else None,
        "pct_positive": round(float((s > 0).mean() * 100), 2) if len(s) else None,
    }


def _marginal_table_xt(df: pd.DataFrame, bins: pd.Series, labels: list[str]) -> list[dict]:
    g = df.groupby(bins, observed=True)[TARGET]
    got = {str(idx): _agg(s) for idx, s in g}
    return [{"bin": lbl, **got.get(lbl, {"n": 0, "rate": None, "pct_zero": None, "pct_negative": None, "pct_positive": None})}
            for lbl in labels]


def _stratified_table_xt(df, marginal_bins, confound_bins, marginal_labels, confound_labels) -> list[dict]:
    g = df.groupby([confound_bins, marginal_bins], observed=True)[TARGET]
    got = {(str(a), str(b)): _agg(s) for (a, b), s in g}
    strata = []
    for stratum in confound_labels:
        row_bins, rates = [], []
        for lbl in marginal_labels:
            a = got.get((stratum, lbl), {"n": 0, "rate": None, "pct_zero": None, "pct_negative": None, "pct_positive": None})
            row_bins.append({"bin": lbl, **a})
            if a["rate"] is not None:
                rates.append(a["rate"])
        strata.append({
            "stratum": stratum,
            "bins": row_bins,
            "first_to_last_delta_pp": round(rates[-1] - rates[0], 8) if len(rates) >= 2 else None,
        })
    return strata


def _bin_series(df: pd.DataFrame, spec: dict) -> pd.Series:
    col, kind = spec["marginal_col"], spec.get("marginal_type", "numeric")
    if kind == "numeric":
        return pd.qcut(df[col], QCUT_N, labels=spec["marginal_labels"])
    if kind == "boolean":
        return df[col].astype(bool).map({False: spec["marginal_labels"][0], True: spec["marginal_labels"][1]})
    if kind == "category_indicator":
        eq = df[col].astype(str) == spec["marginal_category"]
        return eq.map({False: spec["marginal_labels"][0], True: spec["marginal_labels"][1]})
    raise ValueError(kind)


def _confound_series(df: pd.DataFrame, spec: dict) -> pd.Series:
    if spec.get("confound_type", "numeric") == "boolean":
        return df[spec["confound_col"]].astype(bool).map({True: "True", False: "False"})
    return pd.qcut(df[spec["confound_col"]], QCUT_N, labels=spec["confound_labels"])


def run_test_xt(df: pd.DataFrame, spec: dict, conditional: bool = False) -> dict:
    sub = df.dropna(subset=[TARGET])
    m_bins = _bin_series(sub, spec)
    c_bins = _confound_series(sub, spec)
    marginal = _marginal_table_xt(sub, m_bins, spec["marginal_labels"])
    strata = _stratified_table_xt(sub, m_bins, c_bins, spec["marginal_labels"], spec["confound_labels"])

    result = {
        "name": spec["name"],
        "title": spec["title"],
        "marginal_column": spec["marginal_col"],
        "confound_column": spec["confound_col"],
        "marginal_type": spec.get("marginal_type", "numeric"),
        "confound_type": spec.get("confound_type", "numeric"),
        "proposed_confound_reason": spec["proposed_confound_reason"],
        "provenance": spec["provenance"],
        "n_rows_used": int(len(sub)),
        "marginal_table": marginal,
        "stratified_table": strata,
        "verdict": _verdict(marginal, strata),
    }
    for k in ("confound_selection_note", "reliability_note", "prompt_64_link", "is_new_test"):
        if k in spec:
            result[k] = spec[k]

    if not conditional:
        result["given_nonzero_delta"] = run_test_xt(xc.nonzero_subset(sub), spec, conditional=True)
    return result


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = xc.load_active_xt()

    # --- Test 1: mirror of the single existing active confound test ---
    # Confound selection recomputed live, exactly as
    # generate_confound_analysis_part_a.py does it, rather than hardcoding
    # which column it picked.
    corr_def = float(df["defenders_within_10m"].corr(df["is_in_defending_box"].astype(int)))
    corr_att = float(df["defenders_within_10m"].corr(df["is_in_attacking_box"].astype(int)))
    stronger = "is_in_attacking_box" if abs(corr_att) >= abs(corr_def) else "is_in_defending_box"

    test_1 = {
        "name": "defenders_within_10m_vs_box_proximity",
        "title": "defenders_within_10m vs box proximity",
        "marginal_col": "defenders_within_10m",
        "marginal_type": "numeric",
        "marginal_labels": ["Q1 fewest", "Q2", "Q3", "Q4 most"],
        "confound_col": stronger,
        "confound_type": "boolean",
        "confound_labels": ["False", "True"],
        "proposed_confound_reason": (
            "more nearby defenders correlating with more danger (not less) is opposite naive intuition -- "
            "plausibly because defenders converge precisely when the situation is already dangerous (box proximity)"
        ),
        "provenance": (
            "MIRROR of the only active-dataset confound test that exists anywhere in this repo: "
            "reports/analysis/shot_target/CONFOUND_ANALYSIS.json's 'defenders_within_10m_vs_box_proximity', "
            "added by generate_confound_analysis_part_a.py. The xg portal never received it -- its own "
            "CONFOUND_ANALYSIS.json holds 5 passive-only tests. Same marginal column, same quartile labels, "
            "same live confound-selection rule, same reliability caveat, run against target_xt_delta."
        ),
        "confound_selection_note": (
            f"Checked both candidates directly on this run: corr(defenders_within_10m, is_in_defending_box)="
            f"{corr_def:.4f}, corr(defenders_within_10m, is_in_attacking_box)={corr_att:.4f} -- used "
            f"{stronger}, the stronger correlation. Same rule the binary-target version applies."
        ),
        "reliability_note": (
            "UNRELIABLE, not a clean finding -- caveat carried forward from the binary-target version and "
            "re-checked against this portal's own Tournament Stability Check: defenders_within_10m is a "
            "GENUINE TOURNAMENT-LEVEL DIFFERENCE, not arbitrary train/test noise. This test runs on the pooled "
            "115-match population, which mixes two populations. Read the numbers below as a description of the "
            "pooled data, not as evidence about a single stable phenomenon."
        ),
    }

    # --- Test 2: NEW, motivated by Prompt 64's Clearance/action_x finding ---
    test_2 = {
        "name": "clearance_vs_defending_box_proximity",
        "title": "Clearance's negative mean delta vs defending-box proximity",
        "marginal_col": "event_type",
        "marginal_type": "category_indicator",
        "marginal_category": xc.CLEARANCE_EVENT_TYPE,
        "marginal_labels": ["not Clearance", "Clearance"],
        "confound_col": "is_in_defending_box",
        "confound_type": "boolean",
        "confound_labels": ["False", "True"],
        "proposed_confound_reason": (
            "Clearance's strongly negative mean delta is an artefact of WHERE clearances happen (deep in the "
            "defending box, a high-xT cell that xt_after scores as the post-action location), not of clearances "
            "themselves -- so conditioning on defending-box proximity should flatten it"
        ),
        "provenance": (
            "NEW TEST -- it has no counterpart in the xg or binary portals and is labelled as new rather than "
            "presented as an established one. It exists because Prompt 64's Clearance/action_x finding "
            "resurfaced in this portal's Category Atlas exactly as predicted (Clearance ranks last of 8 "
            "event_type categories), and the confound-testing method is the right established instrument in "
            "this suite for asking whether a marginal pattern survives conditioning."
        ),
        "is_new_test": True,
        "prompt_64_link": xc.PROMPT_64_CLEARANCE_FINDING,
    }

    tests = [run_test_xt(df, test_1), run_test_xt(df, test_2)]

    output = {
        "dataset": "active",
        "leg": "active-binary only (passive leg out of scope for this target)",
        "parquet_path": "data/features/player_defensive_actions.parquet",
        "n_rows": len(df),
        "target_col": TARGET,
        "target_scale": xc.target_scale(df),
        "step_0_scope_note": (
            "CONFIRMED BY READING THE SOURCE: reports/analysis/xg_target/CONFOUND_ANALYSIS.json's 5 tests are "
            "ALL passive-dataset tests on passive-only columns, produced by generate_confound_analysis_xg.py "
            "(2) and generate_confound_analysis_part_d_xg.py (3). Exactly one active-dataset confound test "
            "exists anywhere in this repo for any target -- 'defenders_within_10m_vs_box_proximity' in the "
            "binary portal, from generate_confound_analysis_part_a.py; the xg portal never received it. Test 1 "
            "below mirrors that test exactly. Test 2 is new and is labelled as new."
        ),
        "methodology": (
            "Same quartile/boolean-stratification method as the binary and xg confound analyses. _verdict() is "
            "imported from generate_confound_analysis and reused BYTE-FOR-BYTE -- it compares only the SIGN of "
            "first-to-last deltas across strata, never a magnitude threshold, so it is target-scale-agnostic "
            "and needed no adaptation (the same reason generate_confound_analysis_xg.py reused it). QCUT_N=4 "
            "unchanged. Only the table builders differ: mean target_xt_delta instead of a shot-rate percentage, "
            "with the negative/zero/positive shares added. Each test carries a 'given_nonzero_delta' sub-object "
            "in place of the xg version's 'given_shot' -- this target's structural-zero analogue (Adaptation 2)."
        ),
        "prompt_64_cross_references": {
            "clearance_action_x_artefact": xc.PROMPT_64_CLEARANCE_FINDING,
            "distribution_shape": xc.PROMPT_64_SHAPE_FINDING,
        },
        "tests": tests,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    for t in tests:
        print(f"\n=== {t['title']} (xT delta) ===")
        print("Marginal:", [(b["bin"], b["rate"]) for b in t["marginal_table"]])
        print(f"Verdict: {t['verdict']['verdict']} "
              f"({t['verdict']['n_strata_reversal_survives']}/{t['verdict']['n_strata_total']} strata)")
        g = t["given_nonzero_delta"]
        print(f"Given non-zero delta (n={g['n_rows_used']:,}): {g['verdict']['verdict']} "
              f"({g['verdict']['n_strata_reversal_survives']}/{g['verdict']['n_strata_total']})")

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
