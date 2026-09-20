"""CLI entrypoint: tournament stability check against
`target_xt_delta_passive` -- PASSIVE features only. Computes the JSON and
renders its HTML in one pass.

STEP-0 DISPOSITION: genuine passive build. `reports/analysis/xg_target/
TOURNAMENT_STABILITY_CHECK.json` covers BOTH legs inside one file (its
`flat_margins_by_dataset` block carries an `active` and a `passive` entry,
and its feature list mixes the two) -- confirmed by reading it. This
portal's existing `TOURNAMENT_STABILITY_CHECK.json` is Prompt 67's
ACTIVE-ONLY output, because Prompt 65 scoped the whole suite to one leg and
filtered `INVESTIGATE` down to its 6 active entries.

The xg portal's one-file-per-report convention therefore CANNOT be mirrored
literally here without editing Prompt 67's active output, which this pass
does not do. The passive half is a separate `PASSIVE_*` file and both the
portal index and MASTER_FINDINGS say so rather than leaving a reader to
wonder why the two portals are shaped differently.

Carried over UNCHANGED (target-independent or scale-free): the tournament
split, the imported `INVESTIGATE` list filtered to its 4 PASSIVE entries,
bin edges computed once on pooled data and reused for both tournaments, and
`classify_shape()`'s heuristic.

ADAPTED: `flat_margin`, recomputed from THIS leg's own standard deviation
(Adaptation 1). A naive reuse of the active leg's absolute margin would
have been wrong in the direction that matters -- see the threshold block in
the output.

Usage:
    python -m src.eda.generate_tournament_stability_check_xt_passive
"""

from __future__ import annotations

import json

import pandas as pd

from src.eda import xt_passive_common as pc  # re-points xt_common; MUST precede the generator imports
from src.eda import xt_common as xc
from src.eda import generate_tournament_stability_check_xt as t1
from src.eda import generate_tournament_stability_report_xt as tr1
from src.eda.generate_numerical_target_analysis import N_QUANTILE_BINS, classify_shape
from src.eda.generate_tournament_stability_check import INVESTIGATE
from src.eda.tournament_mapping import load_match_tournament_map

JSON_PATH = xc.OUT_DIR / "PASSIVE_TOURNAMENT_STABILITY_CHECK.json"
HTML_PATH = xc.OUT_DIR / "PASSIVE_TOURNAMENT_STABILITY_CHECK.html"
ATLAS_PATH = xc.OUT_DIR / "passive_numerical_target_atlas.json"
TARGET = xc.TARGET

PASSIVE_INVESTIGATE = [f for f, ds in INVESTIGATE.items() if ds == "passive"]


def _load_train_test_finding_passive(feature: str) -> dict:
    """Restate THIS report's own atlas finding for context -- read live from
    the passive atlas JSON this portal just wrote, never re-typed by hand.
    Same approach as the active and xg versions; only the file differs."""
    data = json.loads(ATLAS_PATH.read_text(encoding="utf-8"))
    entry = next((f for f in data["features"] if f["feature"] == feature), None)
    if entry is None:
        return {"in_atlas": False,
                "note": f"{feature} is not in the reconstructed PASSIVE numerical pool this portal's atlas covers."}
    cc = entry["consistency_check"]
    return {
        "in_atlas": True,
        "overall_shape": entry["shape"],
        "spearman_rho": entry["spearman_rho"],
        "train_val_shape": cc.get("train_val_shape"),
        "test_shape": cc.get("test_shape"),
        "train_test_consistent": cc.get("consistent"),
        "shape_under_robust_margin": entry.get("shape_under_robust_margin"),
        "shape_differs_under_robust_margin": entry.get("shape_differs_under_robust_margin"),
    }


def main() -> None:
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    assert ATLAS_PATH.exists(), (
        f"{ATLAS_PATH} must exist before this report runs -- it restates the passive Numerical Target Atlas's "
        "own train/test finding for each feature rather than re-deriving it. Run "
        "`python -m src.eda.generate_numerical_xt_target_passive` first."
    )
    # Declared substitution: the active module's loader points at
    # active_numerical_target_atlas.json. Rebinding it is what makes
    # investigate_feature reusable byte-for-byte. Asserted above by the
    # existence check, and the active module is not edited.
    t1._load_train_test_finding = _load_train_test_finding_passive

    tournament_map = load_match_tournament_map()
    df = xc.load_active_xt()
    scale = xc.target_scale(df)
    flat_margin = scale["flat_margin"]
    robust_fm = scale["robust_scale"]["robust_flat_margin"]

    results = []
    for f in PASSIVE_INVESTIGATE:
        r = t1.investigate_feature(f, df, tournament_map, flat_margin)
        r["dataset"] = "passive"
        # Prompt 67's robust-threshold recomputation, repeated per feature on
        # THIS leg's own numbers: would either tournament's shape label change
        # under the MAD-derived margin? If the verdict itself flips, the
        # genuine-difference call rests on the threshold, not on the data.
        alt_shapes = {}
        for label in ("WC2022", "Euro2024"):
            bins = r["per_tournament"][label]["bins"]
            values = [b["mean_xt"] for b in bins]
            alt_shapes[label] = (
                classify_shape([b["bin"] for b in bins], values, flat_margin=robust_fm)
                if values else "insufficient data"
            )
        alt_verdict = ("arbitrary train/test noise" if alt_shapes["WC2022"] == alt_shapes["Euro2024"]
                       else "genuine tournament-level difference")
        r["robust_margin_recheck"] = {
            "shapes_under_robust_margin": alt_shapes,
            "verdict_under_robust_margin": alt_verdict,
            "verdict_flips_under_robust_margin": bool(alt_verdict != r["verdict"]),
            "note": (
                "Recomputed on this run with the MAD-derived robust flat margin "
                f"({robust_fm:.6f}) in place of the std-derived one ({flat_margin:.6f}). This check does not "
                "exist in the xg or active-leg versions of this report; it is added because this leg's "
                "std-derived margin spans "
                f"{scale['robust_scale']['pct_rows_inside_flat_margin']:.1f}% of all rows, so a shape-based "
                "verdict could rest on the threshold rather than on the tournaments. Where the verdict flips, "
                "that is reported and the std-derived verdict is still the headline -- swapping the threshold "
                "silently would make this half of the portal incomparable with its active half."
            ),
        }
        results.append(r)

    n_flip = sum(1 for r in results if r["robust_margin_recheck"]["verdict_flips_under_robust_margin"])

    output = {
        "dataset": "passive",
        "leg": "passive (this portal also covers the active-binary leg -- see TOURNAMENT_STABILITY_CHECK.json)",
        "target": TARGET,
        "target_scale": scale,
        "tournaments": ["WC2022", "Euro2024"],
        "n_quantile_bins": N_QUANTILE_BINS,
        "flat_margin": flat_margin,
        "n_rows": int(len(df)),
        "n_unique_events": int(df["event_id"].nunique()),
        "step_0_scope_note": (
            "The xg portal's TOURNAMENT_STABILITY_CHECK covers BOTH legs in ONE file -- confirmed by reading "
            "it: its flat_margins_by_dataset block carries an 'active' and a 'passive' entry and its feature "
            "list mixes the two. That one-file convention is NOT mirrored here, and the reason is stated "
            "rather than left implicit: this portal's TOURNAMENT_STABILITY_CHECK.json is Prompt 67's "
            "ACTIVE-ONLY output, and merging a passive section into it would mean editing active-leg report "
            "content that this pass deliberately leaves frozen. The passive half is this separate file. "
            "Feature selection is the imported INVESTIGATE list filtered to its 4 PASSIVE entries -- the same "
            "list the xg version uses for its own passive half, not re-derived from this portal's own "
            "flagged-inconsistent features, so the same features are compared across all three targets."
        ),
        "excluded_from_investigation": {
            "nearest_defender_distance": (
                "already-known self-reference bug -- an ACTIVE-dataset column and not part of the passive "
                "INVESTIGATE set at all, so nothing is excluded from this half beyond what the imported list "
                "already omits. Recorded for symmetry with the active and xg versions."
            ),
        },
        "methodology": (
            "xT-delta counterpart to the xg and binary TOURNAMENT_STABILITY_CHECKs, passive half -- same "
            "tournament split, same imported feature list (passive entries), same bin edges computed once on "
            "the pooled data and reused for both tournaments, same classify_shape() heuristic. The ONLY "
            "threshold change is flat_margin, translated from the xg suite's 0.5x-mean-xg convention via xg's "
            "own standard-deviation fraction applied to THIS leg's std. One field has no xg counterpart: "
            "zero_crossing_differs_between_tournaments, which asks whether the binned curve changes DIRECTION "
            "between tournaments -- impossible to ask of a non-negative target. One field has no counterpart "
            "in the ACTIVE xT version either: robust_margin_recheck, added because this leg's std-derived "
            "margin is wide enough that a shape verdict could rest on it."
        ),
        "threshold_recomputation": {
            "std_derived_flat_margin": flat_margin,
            "mad_derived_robust_flat_margin": robust_fm,
            "active_v2_std_derived_flat_margin": 0.004929,
            "n_features_whose_verdict_flips_under_robust_margin": n_flip,
            "n_features_total": len(results),
            "note": (
                "flat_margin was RECOMPUTED from this leg's own standard deviation, not transferred from the "
                "active leg. That matters concretely: a naive transfer of active's own absolute margin "
                "(0.004929) onto a target whose std is 0.0367 rather than 0.0584 would have made the 'flat' "
                "band nearly twice as wide as it should be, and this report's verdicts are shape comparisons, "
                "so a wider flat band makes tournaments agree more often and would have understated genuine "
                "differences. The robust MAD-derived alternative is computed per feature alongside, and the "
                "count of verdicts that flip under it is reported above rather than hidden."
            ),
        },
        "row_grain_note": pc.PROMPT_68_SHARED_TARGET_FINDING,
        "prompt_68_cross_references": {
            "distribution_shape": pc.PROMPT_68_SHAPE_FINDING,
            "shared_target_row_grain": pc.PROMPT_68_SHARED_TARGET_FINDING,
        },
        "features": results,
    }

    JSON_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    binary = json.loads((xc.REPO_ROOT / "reports" / "analysis" / "shot_target"
                         / "TOURNAMENT_STABILITY_CHECK.json").read_text(encoding="utf-8"))
    xg = json.loads((xc.REPO_ROOT / "reports" / "analysis" / "xg_target"
                     / "TOURNAMENT_STABILITY_CHECK.json").read_text(encoding="utf-8"))
    html = tr1.build_report(output, binary, xg)
    for needle, replacement in [
        ("TOURNAMENT STABILITY CHECK -- XT DELTA &amp;middot; ACTIVE-BINARY LEG ONLY",
         "TOURNAMENT STABILITY CHECK -- XT DELTA &amp;middot; PASSIVE LEG"),
        ("Train/Test-Inconsistent Features, Split by Tournament (xT delta)",
         "Train/Test-Inconsistent Features, Split by Tournament (xT delta, passive)"),
        ("The same 6 ACTIVE features, the same tournament split (WC2022 vs Euro2024) and the same method as "
         "the binary and xG checks, against mean xT delta. Only flat_margin was adapted. Active-binary leg "
         "only -- the 4 passive features in the shared INVESTIGATE list are out of scope.",
         "The 4 PASSIVE features of the shared INVESTIGATE list, the same tournament split (WC2022 vs "
         "Euro2024) and the same method as the binary and xG checks, against mean xT delta. flat_margin was "
         "recomputed from this leg's own std, not transferred from the active leg, and each verdict is "
         "re-checked against a MAD-derived robust margin -- a check neither the xG nor the active xT version "
         "carries."),
    ]:
        assert needle in html, (
            f"Expected {needle!r} in generate_tournament_stability_report_xt's output so it could be "
            "relabelled for the passive leg. It is not there -- that module must have changed."
        )
        html = html.replace(needle, replacement)
    HTML_PATH.write_text(html, encoding="utf-8")

    for r in results:
        print(f"{r['feature']}: {r['verdict']}")
        print(f"  WC2022 shape={r['per_tournament']['WC2022']['shape']}  "
              f"Euro2024 shape={r['per_tournament']['Euro2024']['shape']}  "
              f"zero-crossing differs={r['zero_crossing_differs_between_tournaments']}  "
              f"robust-margin verdict flips={r['robust_margin_recheck']['verdict_flips_under_robust_margin']}")
    print(f"\n{n_flip}/{len(results)} verdicts flip under the robust margin")
    print(f"Wrote {JSON_PATH} and {HTML_PATH}")


if __name__ == "__main__":
    main()
