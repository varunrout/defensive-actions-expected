"""Prompt 85, Phase 2: Clearance / phase_label sanity re-check, pre-fix vs post-fix.

Re-runs the Clearance/phase_label sanity check referenced in the Phase-1 loader-fix
handoff, comparing the pre-fix `events_with_phases.parquet` (backed up at
`.claude_scratch/prefix_backup/events_with_phases.parquet` before Phase 2 regenerated
the pipeline) against the post-fix `events_with_phases.parquet` produced by the
corrected `_attack_sign_for_row` loader logic (commit 8b7d0a8).

`label_defensive_phases` (src/dax/features/phase_segmentation.py) labels a
Clearance/Block event 'box_defence' when its own ball_x >= 95 (its own attacking-frame
convention: the acting player -- a defender here -- attacks toward x=120 in their own
frame, so ball_x >= 95 means the clearance happened deep in the ACTOR'S OWN defensive
third, near their own goal), and 'high_press_proxy' only via the generic `ball_x <= 30`
fallback rule (not Clearance-specific), which fires for a Clearance/Block only when its
own ball_x is low (i.e. NOT near the actor's own goal in their own frame). This script
reports, for both pre-fix and post-fix events_with_phases.parquet, the full phase_label
distribution restricted to Clearance events and specifically how many/what share of
Clearance events sitting deep near the actor's own goal (ball_x >= 95 -- the intended
"clearance near own goal" population) are mislabeled 'high_press_proxy' (should be
near-zero by construction of the rule, since that ball_x range routes to 'box_defence'
before the high_press_proxy branch is ever reached) versus any other phase.

Usage:
    .venv/Scripts/python.exe scripts/analysis/coordinate_frame_fix_clearance_phase_sanity_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PRE_PATH = REPO_ROOT / ".claude_scratch" / "prefix_backup" / "events_with_phases.parquet"
POST_PATH = REPO_ROOT / "data" / "processed" / "events_with_phases.parquet"
OUT_PATH = REPO_ROOT / "outputs" / "models" / "validation" / "coordinate_frame_fix_clearance_phase_sanity_check.json"


def summarize(df: pd.DataFrame, label: str) -> dict:
    clearances = df[df["type"] == "Clearance"].copy() if "type" in df.columns else df[df.get("event_type") == "Clearance"].copy()
    n = len(clearances)
    phase_counts = clearances["phase_label"].value_counts().to_dict()
    own_goal_zone = clearances[clearances["ball_x"] >= 95]
    own_goal_zone_phase_counts = own_goal_zone["phase_label"].value_counts().to_dict()
    high_press_in_own_goal_zone = int((own_goal_zone["phase_label"] == "high_press_proxy").sum())
    return {
        "source": label,
        "n_clearance_events": int(n),
        "phase_label_distribution": {k: int(v) for k, v in phase_counts.items()},
        "n_clearance_events_ball_x_ge_95_own_goal_zone": int(len(own_goal_zone)),
        "phase_label_distribution_in_own_goal_zone": {k: int(v) for k, v in own_goal_zone_phase_counts.items()},
        "high_press_proxy_mislabels_in_own_goal_zone": high_press_in_own_goal_zone,
        "high_press_proxy_mislabel_rate_in_own_goal_zone": (
            high_press_in_own_goal_zone / len(own_goal_zone) if len(own_goal_zone) else 0.0
        ),
    }


def main() -> None:
    pre = pd.read_parquet(PRE_PATH, columns=["type", "ball_x", "ball_y", "phase_label"])
    post = pd.read_parquet(POST_PATH, columns=["type", "ball_x", "ball_y", "phase_label"])

    pre_summary = summarize(pre, "pre_fix (backed up before Phase 2 rebuild)")
    post_summary = summarize(post, "post_fix (Phase 2 rebuild, commit 8b7d0a8 loader)")

    result = {
        "pre_fix": pre_summary,
        "post_fix": post_summary,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=== PRE-FIX Clearance phase_label distribution ===")
    print(json.dumps(pre_summary, indent=2))
    print("\n=== POST-FIX Clearance phase_label distribution ===")
    print(json.dumps(post_summary, indent=2))
    print(f"\n[write] {OUT_PATH}")


if __name__ == "__main__":
    main()
