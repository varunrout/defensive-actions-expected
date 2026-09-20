"""CLI entrypoint: render the v2 FEATURE_LOCK_CONFIRMATION_XT.json as HTML.

STEP-0 DISPOSITION: RE-POINT + three label corrections.

`generate_feature_lock_report_xt.py` renders the v2 JSON correctly -- every
number, table and body string it prints comes from the JSON, which the v2
confirmation script already wrote with v2-correct prose. Three CARD LABELS,
however, are literals in the renderer itself and state v1 conclusions:

  1. "carried forward from Prompt 64 -- CONFIRMED" / "symmetric,
     negative-capable target". v2's skew is -0.306; it is not symmetric,
     and the finding carried forward is Prompt 66's, not Prompt 64's.
  2. "carried forward from Prompt 64 -- RESURFACED" on the Clearance card.
     The artefact did not resurface under v2 -- it reversed.
  3. "carried forward from Prompt 64 -- the motivating problem, confirmed
     fixed (off-pool)" on the possession-flag card, plus its lead-in
     sentence. Under v2 those flags are construction inputs, which is a
     different statement from "confirmed fixed".

Rather than duplicate a 220-line renderer to change three labels, they are
corrected by explicit, declared substitution on the rendered HTML. Each
substitution is ASSERTED to have matched, so if Prompt 65's renderer ever
changes this script fails loudly instead of silently publishing the stale
label. That trade -- a tiny declared patch over a large duplicate -- is the
same reasoning used for the flag-ledger masthead in
`generate_reports_xt_v2.py`.

Usage:
    python -m src.eda.generate_feature_lock_report_xt_v2
"""

from __future__ import annotations

import json

from src.eda import xt_v2_common as v2  # re-points xt_common; MUST precede the generator import
from src.eda import xt_common as xc
from src.eda import generate_feature_lock_report_xt as fr1

INPUT_PATH = xc.OUT_DIR / "FEATURE_LOCK_CONFIRMATION_XT.json"
OUTPUT_PATH = xc.OUT_DIR / "FEATURE_LOCK_CONFIRMATION_XT.html"

# (what to find, what to replace it with, why). Every one is asserted below.
LABEL_CORRECTIONS: list[tuple[str, str, str]] = [
    (
        "carried forward from Prompt 64 -- CONFIRMED",
        "carried forward from Prompt 66 -- SHAPE CHANGED",
        "v2 is left-skewed (-0.306) and far heavier-tailed (kurtosis 18.33), not the near-symmetric "
        "target v1 confirmed.",
    ),
    (
        "symmetric, negative-capable target",
        "left-skewed, heavy-tailed, negative-capable target",
        "v1's descriptor is factually wrong for v2.",
    ),
    (
        "carried forward from Prompt 64 -- RESURFACED",
        "carried forward from Prompt 66 -- REVERSED",
        "the Clearance artefact does not resurface under v2; it fully reverses (8th of 8 -> 1st of 8).",
    ),
    (
        "carried forward from Prompt 64 -- the motivating problem, confirmed fixed (off-pool)",
        "carried forward from Prompt 66 -- these flags are now CONSTRUCTION INPUTS (measured off-pool)",
        "under v2 action_ended_possession triggers xt_after = 0, so its True group equals xt_before by "
        "construction -- a stronger statement than 'no longer degenerate'.",
    ),
    (
        "explained in Prompt 64 section 2",
        "explained in Prompt 66 section 1",
        "the NaN accounting changed: v2 carries 307 NaN (32 from xt_before, unchanged from v1, plus 277 "
        "from xt_after's next-event lookup), not v1's 32.",
    ),
    (
        "Pattern-analysis findings (xT delta)",
        "Pattern-analysis findings (xT delta v2)",
        "title should name the target version.",
    ),
    (
        "carried forward from Prompt 64",
        "carried forward from Prompt 66",
        "catch-all for any remaining provenance label; runs last.",
    ),
]


def build_report_v2(data: dict) -> str:
    html = fr1.build_report(data)
    for needle, replacement, why in LABEL_CORRECTIONS:
        if needle == "carried forward from Prompt 64":
            html = html.replace(needle, replacement)  # catch-all, may legitimately match nothing
            continue
        assert needle in html, (
            f"Expected to find {needle!r} in the v1 feature-lock renderer's output so it could be corrected "
            f"for v2 ({why}). It is not there -- generate_feature_lock_report_xt.py must have changed. "
            "Refusing to publish a report whose provenance labels have not been verified."
        )
        html = html.replace(needle, replacement)
    return html


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    OUTPUT_PATH.write_text(build_report_v2(data), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} ({len(LABEL_CORRECTIONS)} provenance labels corrected for v2)")


if __name__ == "__main__":
    main()
