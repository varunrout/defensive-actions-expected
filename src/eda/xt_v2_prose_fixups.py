"""Post-pass prose corrections for three re-pointed Prompt-65 generators.

STEP-0 DISPOSITION: these three scripts are RE-POINTS in every computation
they perform -- not one number, threshold, bin edge or verdict in them is
v1-specific, and all three recompute their thresholds at runtime from the
real parquet. Each, however, carries exactly ONE hardcoded sentence that
asserts v1's distribution shape as a fact, and one of them repeats it in
two more places:

  generate_review_analysis_xt.py (JSON `note`, and the HTML rendered from it)
    "...a multiple of target_xt_delta's own near-zero, NEGATIVE mean would
     be meaningless."

  generate_tournament_stability_check_xt.py (JSON `methodology` + its HTML)
    "...because a multiple of this target's near-zero NEGATIVE mean would
     be meaningless."

  generate_numerical_xt_target_reports.py (rendered HTML)
    "<target> can be negative and is roughly symmetric."
    "roughly symmetric, NN% negative -- NOT a rate and NOT non-negative."
    "...because a multiple of this target's own near-zero, negative mean
     would be meaningless."

v2's mean is +0.00338, i.e. POSITIVE, and its skew is -0.306, i.e. not
symmetric. Publishing those sentences beside v2's own tables would have
contradicted the numbers immediately above them.

Duplicating three large generators to change one sentence each would have
been worse than the problem -- it triples the surface area that has to stay
in sync. Instead each correction is applied here at the narrowest possible
granularity, and every one ASSERTS that it matched. If any Prompt-65 script
is ever changed, this fails loudly rather than silently publishing a stale
claim. Corrections are applied to the JSON and to the HTML rendered from it
alike, so the pass is order-independent within the driver.

The replacement preserves the ORIGINAL ARGUMENT where it still holds. The
translation is still used and still justified -- see
`xt_v2_common.target_scale_v2`'s `scale_note` -- but for the reason that
survives (the mean is only ~6% of a std, so a mean-multiple threshold is an
arbitrary sliver, and keeping the translation is what makes v1 and v2
comparable) rather than the reason that does not (a negative mean would
flip every threshold's sign).

NOT corrected, deliberately: several reports contain the phrase "near-zero
and NEGATIVE (-0.001316)". That is `xt_v2_common`'s own v2 text describing
what was true of v1, in the past tense, as part of explaining why the
convention was re-examined. It is accurate and stays.

Usage:
    python -m src.eda.xt_v2_prose_fixups
"""

from __future__ import annotations

from src.eda import xt_v2_common as v2  # noqa: F401  (re-points xt_common)
from src.eda import xt_common as xc

_WHY = (
    "because this target's mean, at only about 6% of one standard deviation, makes a mean-multiple threshold "
    "an arbitrary sliver -- and because keeping the same translation is what makes this portal numerically "
    "comparable with the v1 portal it supersedes. (v1 justified this translation by its mean being NEGATIVE, "
    "which would have flipped every threshold's sign. That no longer applies -- v2's mean is positive -- so "
    "the justification is restated rather than repeated.)"
)

# (filename, old fragment, new fragment, required?). Applied as raw text to
# both JSON and HTML; the replacement contains no quote or backslash
# characters, so it stays valid inside a JSON string literal.
FIXUPS: list[tuple[str, str, str, bool]] = [
    # --- Review Methodology -------------------------------------------------
    ("REVIEW_ANALYSIS.json",
     "standard-deviation fraction -- a multiple of target_xt_delta's own near-zero, NEGATIVE mean would be meaningless.",
     f"standard-deviation fraction, {_WHY}", True),

    # --- Tournament Stability Check ----------------------------------------
    ("TOURNAMENT_STABILITY_CHECK.json",
     "fraction, because a multiple of this target's near-zero NEGATIVE mean would be meaningless.",
     f"fraction, {_WHY}", True),

    # --- Numerical Target Atlas (rendered HTML only) ------------------------
    # Longer/more specific fragment first: the second would otherwise match
    # inside the first.
    ("active_numerical_target_atlas.html",
     "can be negative and is roughly symmetric.",
     "can be negative, is left-skewed and is heavy-tailed.", True),
    ("active_numerical_target_atlas.html",
     "roughly symmetric, ",
     "left-skewed and heavy-tailed, ", True),
    ("active_numerical_target_atlas.html",
     "because a multiple of this target's own near-zero, negative\nmean would be meaningless.",
     _WHY, True),
]


def apply() -> None:
    applied = skipped = already = 0
    for fname, old, new, required in FIXUPS:
        path = xc.OUT_DIR / fname
        text = path.read_text(encoding="utf-8")
        if old not in text:
            # Idempotent: a re-run over already-corrected output is a no-op,
            # not a failure. Only a file that has NEITHER the old fragment nor
            # the new one indicates the upstream generator really did change.
            if new in text:
                already += 1
                continue
            if required:
                raise AssertionError(
                    f"{fname}: expected the v1 fragment {old[:70]!r}... so it could be corrected for v2. "
                    "Not found -- the Prompt-65 generator must have changed. Refusing to publish prose that "
                    "has not been verified against the target it now describes."
                )
            skipped += 1
            continue
        path.write_text(text.replace(old, new), encoding="utf-8")
        applied += 1

    # The Review Methodology and Tournament Stability HTML are rendered FROM
    # the JSON just corrected (their renderers escape the text, so patching
    # the HTML by the same literal would be brittle). Re-render them instead,
    # so the two can never disagree.
    from src.eda import generate_review_report_xt, generate_tournament_stability_report_xt
    generate_review_report_xt.main()
    generate_tournament_stability_report_xt.main()

    print(f"Applied {applied} v2 prose corrections to re-pointed Prompt-65 outputs"
          + (f" ({already} already applied)" if already else "")
          + (f" ({skipped} optional fragment(s) not present)" if skipped else "")
          + "; re-rendered REVIEW_METHODOLOGY.html and TOURNAMENT_STABILITY_CHECK.html from the corrected JSON")


def main() -> None:
    apply()


if __name__ == "__main__":
    main()
