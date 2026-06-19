from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from dax.coach_analysis.bootstrap import add_match_bootstrap_by_group
from dax.coach_analysis.execution import add_model_selection_args, build_common_parser, execution_summary, resolve_paths
from dax.coach_analysis.labels import label_rules
from dax.coach_analysis.loaders import CoachAnalysisInputError, join_oof_strict, require_table, select_required_two_part, select_required_variant, validate_unique_predictions
from dax.coach_analysis.metrics import add_suppression, first_existing, summary_table
from dax.coach_analysis.plotting import action_pitch_map, horizontal_metric_chart
from dax.coach_analysis.populations import apply_visibility_filter, box_defence_population
from dax.coach_analysis.reporting import data_derived_conclusions, markdown_table, render_conclusions, write_json, write_markdown_report
from dax.coach_analysis.representative_events import select_representative_events
from dax.coach_analysis.timeline import add_next_events, validate_processed_timeline

KEYS = ["match_id", "event_id"]


def parse_args(argv: list[str] | None = None):
    parser = build_common_parser("Analyse centre-back box-defence risk.")
    add_model_selection_args(parser)
    return parser.parse_args(argv)


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _prefix_predictions(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep = set(KEYS) | {"fold"}
    rename = {c: f"{prefix}_{c}" for c in df.columns if c not in keep and not c.endswith("variant")}
    return df.rename(columns=rename)


def _select_prediction_sets(args, root: Path) -> list[tuple[str, pd.DataFrame]]:
    classification = require_table(args.classification_oof, root)
    regression = require_table(args.regression_oof, root)
    two_part = require_table(args.two_part_oof, root)
    selected = [
        ("classification_primary", _prefix_predictions(select_required_variant(classification, args.classification_variant, label="classification primary"), "b7")),
        ("classification_sensitivity", _prefix_predictions(select_required_variant(classification, args.classification_sensitivity, label="classification sensitivity"), "b6")),
        ("regression_primary", _prefix_predictions(select_required_variant(regression, args.regression_variant, label="regression primary"), "r4")),
        ("regression_sensitivity", _prefix_predictions(select_required_variant(regression, args.regression_sensitivity, label="regression sensitivity"), "r6")),
        ("two_part", _prefix_predictions(select_required_two_part(two_part, args.two_part_classification_variant, args.two_part_conditional_variant), "two_part")),
    ]
    for label, frame in selected:
        validate_unique_predictions(frame, KEYS, label)
    return selected


def _canonical_competition(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    source = first_existing(out, ["competition_label", "competition", "competition_name"])
    out["coach_competition"] = out[source].astype(str) if source else "unknown"
    return out


def _prepare_data(args, root: Path) -> tuple[pd.DataFrame, dict]:
    actions = require_table(args.actions_input, root)
    events = require_table(args.processed_events_input, root)
    timeline = validate_processed_timeline(events)
    if not timeline["next_event_sequence_possible"]:
        raise CoachAnalysisInputError("Processed events cannot support next-event sequence construction.")
    joined = join_oof_strict(actions, _select_prediction_sets(args, root), KEYS)
    with_next = add_next_events(joined, events)
    return add_suppression(_canonical_competition(with_next)), timeline


def _event_text(df: pd.DataFrame, prefix: str = "") -> pd.Series:
    col = first_existing(df, [f"{prefix}event_type", f"{prefix}type", f"{prefix}action_family"])
    return df[col].astype(str).str.lower() if col else pd.Series("", index=df.index)


def _augment_sequences(population: pd.DataFrame) -> pd.DataFrame:
    out = population.copy()
    action = _event_text(out)
    next_text = _event_text(out, "next_")
    second_text = _event_text(out, "second_next_")
    same_possession = out.get("next_possession", pd.Series(pd.NA, index=out.index)).eq(out.get("possession", pd.Series(pd.NA, index=out.index)))
    same_team_next = out.get("next_team", pd.Series(pd.NA, index=out.index)).eq(out.get("team", pd.Series(pd.NA, index=out.index)))
    out["coach_possession_secured"] = same_team_next.fillna(False) & ~same_possession.fillna(True)
    out["coach_clearance_outcome"] = "not clearance"
    out.loc[action.str.contains("clearance") & out["coach_possession_secured"], "coach_clearance_outcome"] = "clearance relief"
    out.loc[action.str.contains("clearance") & ~out["coach_possession_secured"], "coach_clearance_outcome"] = "clearance recycled"
    out["coach_block_outcome"] = "not block"
    out.loc[action.str.contains("block") & (next_text.str.contains("shot|rebound") | second_text.str.contains("shot|rebound")), "coach_block_outcome"] = "block followed by rebound"
    out.loc[action.str.contains("block") & out["coach_block_outcome"].eq("not block"), "coach_block_outcome"] = "block without immediate rebound"
    next_x = pd.to_numeric(out.get("next_x", out.get("next_location_x", pd.Series(pd.NA, index=out.index))), errors="coerce")
    x = pd.to_numeric(out.get("x", out.get("location_x", pd.Series(pd.NA, index=out.index))), errors="coerce")
    progressed = next_x.gt(x + 5)
    out["coach_pressure_outcome"] = "not pressure"
    out.loc[action.str.contains("pressure") & progressed.fillna(False), "coach_pressure_outcome"] = "pressure followed by progression"
    out.loc[action.str.contains("pressure") & ~progressed.fillna(False), "coach_pressure_outcome"] = "pressure delayed/no progression"
    out["coach_immediate_re_turnover"] = same_team_next.fillna(False) & out.get("second_next_team", pd.Series(pd.NA, index=out.index)).ne(out.get("team", pd.Series(pd.NA, index=out.index)))
    out["coach_duel_outcome"] = "not duel"
    out.loc[action.str.contains("duel") & out["coach_possession_secured"], "coach_duel_outcome"] = "duel won and secured"
    out.loc[action.str.contains("duel") & ~out["coach_possession_secured"], "coach_duel_outcome"] = "duel unresolved/lost"
    if {"match_id", "possession"}.issubset(out.columns):
        out["coach_repeated_box_action"] = out.groupby(["match_id", "possession"]).cumcount().gt(0)
    else:
        out["coach_repeated_box_action"] = False
    return out


def _metric_table(population: pd.DataFrame, group_col: str, n_boot: int, seed: int) -> pd.DataFrame:
    if population.empty or group_col not in population.columns:
        return pd.DataFrame()
    table = summary_table(population, [group_col])
    metric = first_existing(population, ["observed_future_xg", "future_xg", "xg_within_10s", "y_true_xg"])
    if metric:
        ci = add_match_bootstrap_by_group(population, [group_col], metric, n_boot=n_boot, seed=seed)
        table = table.merge(ci[[group_col, "ci_low", "ci_high"]], on=group_col, how="left")
    return table


def _sensitivity_tables(population: pd.DataFrame) -> dict[str, pd.DataFrame]:
    specs = {
        "b7_vs_b6_expected_shot": ["b7_expected_shot_probability", "b6_expected_shot_probability", "b7_y_pred_proba", "b6_y_pred_proba"],
        "r4_vs_r6_expected_xg": ["r4_expected_future_xg", "r6_expected_future_xg", "r4_y_pred", "r6_y_pred"],
        "r4_vs_two_part_expected_xg": ["r4_expected_future_xg", "two_part_expected_future_xg", "r4_y_pred", "two_part_y_pred"],
    }
    out = {}
    for name, cols in specs.items():
        pairs = [(cols[i], cols[i + 1]) for i in range(0, len(cols), 2) if cols[i] in population.columns and cols[i + 1] in population.columns]
        if pairs:
            a, b = pairs[0]
            diff = pd.to_numeric(population[a], errors="coerce") - pd.to_numeric(population[b], errors="coerce")
            out[name] = pd.DataFrame({"comparison": [name], "rows": [int(diff.notna().sum())], "mean_difference": [float(diff.mean())]})
        else:
            out[name] = pd.DataFrame({"comparison": [name], "rows": [0], "mean_difference": [pd.NA]})
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolve_paths(args, "cb_box_defence")
    tables = paths.child("tables")
    figures = paths.child("figures")
    video_dir = paths.child("video_review")
    try:
        actions, timeline = _prepare_data(args, paths.root)
        population = _augment_sequences(box_defence_population(actions, centre_backs_only=True))
    except CoachAnalysisInputError as exc:
        if not args.allow_partial:
            write_json(paths.output_root / "execution_summary.json", execution_summary("failed", error=str(exc)))
            write_markdown_report(paths.output_root / "report.md", "Centre-back box-defence analysis", [("Error", str(exc))])
            return 2
        population = pd.DataFrame()
        timeline = {}
    reliable = apply_visibility_filter(population, reliable_only=True)
    wc_euros = population[population.get("coach_competition", pd.Series(dtype=str)).astype(str).str.contains("World Cup|Euros|Euro", case=False, na=False)].copy() if not population.empty else population
    group_cols = ["coach_box_zone", "action_family", "event_type", "coach_clearance_outcome", "coach_block_outcome", "coach_pressure_outcome", "coach_duel_outcome", "coach_repeated_box_action", "coach_competition"]
    generated = {}
    conclusion_records = []
    for col in group_cols:
        if col in population.columns:
            table = _metric_table(population, col, args.bootstrap_samples, args.seed)
            _save_table(table, tables / f"by_{col}.csv")
            generated[col] = len(table)
            metric = first_existing(table, ["future_xg_per_action", "future_shot_rate", "expected_future_xg"])
            if metric:
                conclusion_records.extend(data_derived_conclusions(table.rename(columns={col: "subgroup"}), "subgroup", metric, top_n=3, min_actions=args.min_actions, min_matches=args.min_matches))
    if not reliable.empty:
        table = _metric_table(reliable, "coach_box_zone", args.bootstrap_samples, args.seed)
        _save_table(table, tables / "reliable_visibility_by_box_zone.csv")
    for name, table in _sensitivity_tables(population).items():
        _save_table(table, tables / f"{name}.csv")
        if not table.empty and table["rows"].iloc[0]:
            conclusion_records.append({"subgroup": name, "metric": "mean_difference", "value": float(table["mean_difference"].iloc[0]), "comparison_value": 0.0, "difference": float(table["mean_difference"].iloc[0]), "actions": int(table["rows"].iloc[0]), "matches": int(population["match_id"].nunique()) if "match_id" in population.columns else 0, "ci_low": None, "ci_high": None, "minimum_sample_warning": "" if table["rows"].iloc[0] >= args.min_actions else f"low action sample: {table['rows'].iloc[0]} < {args.min_actions}"})
    video = select_representative_events(population, n=20, reason="CB box-defence candidate for video review")
    _save_table(video, video_dir / "candidate_events.csv")
    _save_table(label_rules(), tables / "tactical_label_rules.csv")
    if "coach_box_zone" in population.columns:
        zone_table = _metric_table(population, "coach_box_zone", args.bootstrap_samples, args.seed)
        metric = first_existing(zone_table, ["future_xg_per_action", "future_shot_rate", "expected_future_xg"])
        if metric:
            horizontal_metric_chart(zone_table, "coach_box_zone", metric, "CB box-defence threat by mutually exclusive box zone", figures / "zone_threat_horizontal.png")
    action_pitch_map(population, "Centre-back box-defence actions", figures / "cb_box_defence_pitch_map.png")
    competition_counts = population["coach_competition"].value_counts().to_dict() if "coach_competition" in population.columns else {}
    sections = [
        ("Population", f"Centre-back box-defence actions: {len(population)}; matches: {population['match_id'].nunique() if 'match_id' in population.columns else 0}; World Cup/Euros rows: {len(wc_euros)}."),
        ("Processed event timeline", f"```json\n{timeline}\n```"),
        ("Competition counts", f"```json\n{competition_counts}\n```"),
        ("Generated tables", markdown_table(pd.DataFrame([{"table": k, "rows": v} for k, v in generated.items()]))),
        ("Data-derived conclusions", render_conclusions(conclusion_records)),
        ("Reliable-visibility sensitivity", f"Reliable-visibility subset actions: {len(reliable)}."),
        ("Video review", f"Candidate event table written with {len(video)} rows."),
    ]
    write_markdown_report(paths.output_root / "report.md", "Centre-back box-defence analysis", sections)
    write_json(paths.output_root / "execution_summary.json", execution_summary("completed", actions=int(len(population)), matches=int(population["match_id"].nunique()) if "match_id" in population.columns else 0, competition_counts=competition_counts, generated_tables=generated, data_derived_conclusions=conclusion_records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
