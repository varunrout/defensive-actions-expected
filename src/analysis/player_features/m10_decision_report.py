"""Step 10: Consolidated go/no-go decision report."""

from __future__ import annotations

from typing import Any

import json
from pathlib import Path

from .config import AnalysisConfig


MANDATORY_STEPS = [
    "01_schema_quality",
    "02_target_label_audit",
    "03_univariate_signal",
    "06_leakage_and_confounding",
    "07_stability_and_shift",
    "08_redundancy_and_selection",
    "09_sanity_negative_controls",
]


def run(step_results: dict[str, dict[str, Any]], cfg: AnalysisConfig) -> dict[str, Any]:
    """Build final markdown + json decision artifact."""
    failed = [name for name in MANDATORY_STEPS if not bool(step_results.get(name, {}).get("pass", False))]
    decision = "GO_BASELINE" if not failed else "HOLD"

    decision_obj = {
        "decision": decision,
        "failed_mandatory_steps": failed,
        "steps": step_results,
    }

    json_path = cfg.report_dir / "decision.json"
    json_path.write_text(json.dumps(decision_obj, indent=2), encoding="utf-8")

    lines = [
        "# Player Feature Analysis Decision",
        "",
        f"Decision: **{decision}**",
        "",
        "## Mandatory step results",
    ]
    for name in MANDATORY_STEPS:
        status = "PASS" if bool(step_results.get(name, {}).get("pass", False)) else "FAIL"
        lines.append(f"- {name}: {status}")

    lines += ["", "## Failed gates", ""]
    if failed:
        lines.extend([f"- {name}" for name in failed])
    else:
        lines.append("- None")

    md_path = cfg.report_dir / "player_feature_analysis_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "decision": decision,
        "failed_mandatory_steps": failed,
        "decision_json": str(json_path),
        "decision_markdown": str(md_path),
        "pass": decision == "GO_BASELINE",
    }

