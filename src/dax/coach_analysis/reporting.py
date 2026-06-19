from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

MIN_SAMPLE_ACTIONS = 30
MIN_SAMPLE_MATCHES = 5


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows available._"
    data = frame.head(max_rows).fillna("")
    cols = [str(c) for c in data.columns]
    rows = [[str(v) for v in row] for row in data.to_numpy()]
    widths = [len(c) for c in cols]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row, strict=False)]
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths, strict=False)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = ["| " + " | ".join(value.ljust(w) for value, w in zip(row, widths, strict=False)) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def sample_warning(actions: int, matches: int, min_actions: int = MIN_SAMPLE_ACTIONS, min_matches: int = MIN_SAMPLE_MATCHES) -> str:
    warnings = []
    if actions < min_actions:
        warnings.append(f"low action sample: {actions} < {min_actions}")
    if matches < min_matches:
        warnings.append(f"low match sample: {matches} < {min_matches}")
    return "; ".join(warnings) if warnings else ""


def data_derived_conclusions(
    table: pd.DataFrame,
    subgroup_col: str,
    metric_col: str,
    comparison_col: str | None = None,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Generate conclusion records from calculated rows only.

    Each record contains the subgroup, metric, value, comparison value,
    difference, actions, matches, CI and a minimum-sample warning.
    """
    required = {subgroup_col, metric_col, "actions", "matches"}
    if table.empty or not required.issubset(table.columns):
        return []
    ranked = table.dropna(subset=[metric_col]).sort_values(metric_col, ascending=False).head(top_n)
    if ranked.empty:
        return []
    comparison_value = None
    if comparison_col and comparison_col in table.columns:
        comparison_value = pd.to_numeric(table[comparison_col], errors="coerce").mean()
    elif metric_col in table.columns:
        comparison_value = pd.to_numeric(table[metric_col], errors="coerce").mean()
    records = []
    for _, row in ranked.iterrows():
        value = float(row[metric_col])
        comp = float(comparison_value) if comparison_value is not None and pd.notna(comparison_value) else None
        records.append(
            {
                "subgroup": str(row[subgroup_col]),
                "metric": metric_col,
                "value": value,
                "comparison_value": comp,
                "difference": None if comp is None else value - comp,
                "actions": int(row.get("actions", 0)),
                "matches": int(row.get("matches", 0)),
                "ci_low": None if pd.isna(row.get("ci_low", pd.NA)) else float(row.get("ci_low")),
                "ci_high": None if pd.isna(row.get("ci_high", pd.NA)) else float(row.get("ci_high")),
                "minimum_sample_warning": sample_warning(int(row.get("actions", 0)), int(row.get("matches", 0))),
            }
        )
    return records


def render_conclusions(conclusions: list[dict[str, Any]]) -> str:
    if not conclusions:
        return "_No data-derived conclusions available for this population._"
    lines = []
    for item in conclusions:
        ci = "not available"
        if item["ci_low"] is not None and item["ci_high"] is not None:
            ci = f"[{item['ci_low']:.4f}, {item['ci_high']:.4f}]"
        warning = f" Warning: {item['minimum_sample_warning']}." if item["minimum_sample_warning"] else ""
        comp = "n/a" if item["comparison_value"] is None else f"{item['comparison_value']:.4f}"
        diff = "n/a" if item["difference"] is None else f"{item['difference']:+.4f}"
        lines.append(
            f"- **{item['subgroup']}** — {item['metric']}={item['value']:.4f}; "
            f"comparison={comp}; difference={diff}; actions={item['actions']}; "
            f"matches={item['matches']}; 95% match-bootstrap CI={ci}.{warning}"
        )
    return "\n".join(lines)


def write_markdown_report(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# {title}", ""]
    for heading, content in sections:
        body.extend([f"## {heading}", "", content, ""])
    path.write_text("\n".join(body))
