from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dax.coach_analysis.execution import add_model_selection_args, build_common_parser, execution_summary, resolve_paths
from dax.coach_analysis.loaders import oof_coverage, read_optional_table, select_variant, validate_inputs, validate_schema
from dax.coach_analysis.reporting import markdown_table, write_json, write_markdown_report


def parse_args(argv: list[str] | None = None):
    parser = build_common_parser("Validate coach-analysis data and OOF readiness.")
    add_model_selection_args(parser)
    return parser.parse_args(argv)


def _variant_status(df, variant):
    selected = select_variant(df, variant)
    return {
        "variant": variant,
        "rows": int(len(selected)),
        "selection": selected.attrs.get("variant_selection", ""),
        "available": bool(len(selected)) if any(c in df.columns for c in ["variant", "model_variant", "candidate", "model_name", "run_name"]) else not df.empty,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolve_paths(args, "readiness")
    report_path = paths.output_root / "report.md"
    summary_path = paths.output_root / "execution_summary.json"

    status = validate_inputs(paths.root)
    actions = read_optional_table("data/features/player_defensive_actions.parquet", paths.root)
    classification = read_optional_table("outputs/oof/classification_oof.parquet", paths.root)
    regression = read_optional_table("outputs/oof/regression_oof.parquet", paths.root)
    two_part = read_optional_table("outputs/oof/two_part_future_xg_oof_exploratory.parquet", paths.root)

    schemas = {
        "actions": validate_schema(actions, [["match_id", "event_id"]], ["competition", "player", "team"]),
        "classification_oof": validate_schema(classification, [["match_id", "event_id"]], ["fold"]),
        "regression_oof": validate_schema(regression, [["match_id", "event_id"]], ["fold"]),
        "two_part_exploratory_oof": validate_schema(two_part, [["match_id", "event_id"]], ["fold"]),
    }
    variants = {
        "classification_candidate": _variant_status(classification, args.classification_variant),
        "classification_sensitivity": _variant_status(classification, args.classification_sensitivity),
        "regression_candidate": _variant_status(regression, args.regression_variant),
        "regression_sensitivity": _variant_status(regression, args.regression_sensitivity),
        "two_part_exploratory": _variant_status(two_part, args.two_part_variant),
    }
    coverages = {
        "classification": oof_coverage(actions, select_variant(classification, args.classification_variant)),
        "regression": oof_coverage(actions, select_variant(regression, args.regression_variant)),
        "two_part_exploratory": oof_coverage(actions, select_variant(two_part, args.two_part_variant)),
    }
    timeline_cols = [c for c in ["period", "minute", "second", "event_index", "timestamp"] if c in actions.columns]
    competition_cols = [c for c in ["competition", "competition_name", "competition_id"] if c in actions.columns]
    visibility_cols = [c for c in actions.columns if "visible" in c.lower() or "360" in c.lower()]
    coverage_context = {
        "processed_event_timeline_available": bool(timeline_cols),
        "timeline_columns": timeline_cols,
        "competition_coverage_available": bool(competition_cols),
        "competition_columns": competition_cols,
        "competitions": sorted(actions[competition_cols[0]].dropna().astype(str).unique().tolist()) if competition_cols else [],
        "visibility_coverage_available": bool(visibility_cols),
        "visibility_columns": visibility_cols,
        "visibility_non_null_rows": int(actions[visibility_cols].notna().any(axis=1).sum()) if visibility_cols else 0,
    }
    duplicate_failures = {name: cov.get("duplicate_predictions", 0) for name, cov in coverages.items() if cov.get("duplicate_predictions", 0)}
    missing_inputs = status.loc[~status["exists"], "path"].tolist()
    summary = execution_summary(
        "completed_with_missing_inputs" if missing_inputs else "completed",
        missing_inputs=missing_inputs,
        schemas=schemas,
        variants=variants,
        oof_coverage=coverages,
        coverage_context=coverage_context,
        duplicate_prediction_failures=duplicate_failures,
    )
    sections = [
        ("Input existence", markdown_table(status)),
        ("Schema validation", f"```json\n{schemas}\n```"),
        ("Explicit model selections", f"```json\n{variants}\n```"),
        ("OOF row, match, fold and duplicate coverage", f"```json\n{coverages}\n```"),
        ("Timeline, competition and visibility coverage", f"```json\n{coverage_context}\n```"),
        ("Readiness conclusion", "Missing inputs prevent substantive coach findings." if missing_inputs else "All checked inputs are present; review duplicate and coverage details above."),
    ]
    write_markdown_report(report_path, "Coach analysis readiness", sections)
    write_json(summary_path, summary)
    return 2 if duplicate_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
