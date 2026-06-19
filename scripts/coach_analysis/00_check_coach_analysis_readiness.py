from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dax.coach_analysis.execution import add_model_selection_args, build_common_parser, execution_summary, resolve_paths
from dax.coach_analysis.loaders import CoachAnalysisInputError, oof_coverage, require_table, select_required_two_part, select_required_variant, validate_inputs, validate_schema, validate_unique_predictions
from dax.coach_analysis.reporting import markdown_table, write_json, write_markdown_report
from dax.coach_analysis.timeline import validate_processed_timeline
from dax.coach_analysis.visibility import visibility_report


def parse_args(argv: list[str] | None = None):
    parser = build_common_parser("Validate coach-analysis data and OOF readiness.")
    add_model_selection_args(parser)
    return parser.parse_args(argv)


def _read(path: Path, root: Path, errors: list[str], required: bool = True):
    try:
        return require_table(path, root, required=required)
    except CoachAnalysisInputError as exc:
        errors.append(str(exc))
        return __import__("pandas").DataFrame()


def _check_variant(label, frame, selector, errors):
    try:
        selected = selector(frame)
        validate_unique_predictions(selected, label=label)
        return selected, {"rows": int(len(selected)), "selection": selected.attrs.get("variant_selection", "")}
    except CoachAnalysisInputError as exc:
        errors.append(str(exc))
        return frame.iloc[0:0].copy(), {"rows": 0, "error": str(exc)}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolve_paths(args, "readiness")
    errors: list[str] = []

    status = validate_inputs(paths.root)
    actions = _read(args.actions_input, paths.root, errors)
    processed_events = _read(args.processed_events_input, paths.root, errors)
    classification = _read(args.classification_oof, paths.root, errors)
    regression = _read(args.regression_oof, paths.root, errors)
    two_part = _read(args.two_part_oof, paths.root, errors)

    schemas = {
        "actions": validate_schema(actions, [["match_id", "event_id"]], ["competition_label", "competition", "player", "team"]),
        "processed_events": validate_schema(processed_events, [["match_id", "event_id"]], ["period"]),
        "classification_oof": validate_schema(classification, [["match_id", "event_id"]], ["fold"]),
        "regression_oof": validate_schema(regression, [["match_id", "event_id"]], ["fold"]),
        "two_part_oof": validate_schema(two_part, [["match_id", "event_id"]], ["fold"]),
    }
    for name, schema in schemas.items():
        if not schema["valid"]:
            errors.append(f"Invalid schema for {name}: {schema['missing_required_options']}")

    b7, b7_status = _check_variant("classification primary", classification, lambda df: select_required_variant(df, args.classification_variant, label="classification primary"), errors)
    b6, b6_status = _check_variant("classification sensitivity", classification, lambda df: select_required_variant(df, args.classification_sensitivity, label="classification sensitivity"), errors)
    r4, r4_status = _check_variant("regression primary", regression, lambda df: select_required_variant(df, args.regression_variant, label="regression primary"), errors)
    r6, r6_status = _check_variant("regression sensitivity", regression, lambda df: select_required_variant(df, args.regression_sensitivity, label="regression sensitivity"), errors)
    tp, tp_status = _check_variant("two-part exploratory", two_part, lambda df: select_required_two_part(df, args.two_part_classification_variant, args.two_part_conditional_variant), errors)

    coverages = {
        "classification_primary": oof_coverage(actions, b7),
        "classification_sensitivity": oof_coverage(actions, b6),
        "regression_primary": oof_coverage(actions, r4),
        "regression_sensitivity": oof_coverage(actions, r6),
        "two_part": oof_coverage(actions, tp),
    }
    for name, cov in coverages.items():
        if cov.get("coverage_rate") is not None and cov["coverage_rate"] < 1.0:
            errors.append(f"Prediction coverage below eligible population for {name}: {cov['coverage_rate']:.3f}")
    timeline = validate_processed_timeline(processed_events)
    if not timeline["valid"]:
        errors.append("Processed event timeline is missing or invalid for next-event sequence construction.")
    visibility = visibility_report(actions)
    variants = {
        "classification_primary": b7_status,
        "classification_sensitivity": b6_status,
        "regression_primary": r4_status,
        "regression_sensitivity": r6_status,
        "two_part": tp_status,
    }
    summary = execution_summary(
        "completed_with_errors" if errors else "completed",
        errors=errors,
        selected_variants=variants,
        schemas=schemas,
        oof_coverage=coverages,
        processed_event_timeline=timeline,
        visibility=visibility,
        allow_partial=bool(args.allow_partial),
    )
    sections = [
        ("Input existence", markdown_table(status)),
        ("Schema validation", f"```json\n{schemas}\n```"),
        ("Selected variants", f"```json\n{variants}\n```"),
        ("OOF coverage", f"```json\n{coverages}\n```"),
        ("Processed event timeline", f"```json\n{timeline}\n```"),
        ("Visibility coverage", f"```json\n{visibility}\n```"),
        ("Errors", "\n".join(f"- {e}" for e in errors) if errors else "No readiness errors."),
    ]
    write_markdown_report(paths.output_root / "report.md", "Coach analysis readiness", sections)
    write_json(paths.output_root / "execution_summary.json", summary)
    return 0 if (not errors or args.allow_partial) else 2


if __name__ == "__main__":
    raise SystemExit(main())
