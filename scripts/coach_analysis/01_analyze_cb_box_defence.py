from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from dax.coach_analysis.bootstrap import add_match_bootstrap_by_group
from dax.coach_analysis.execution import add_model_selection_args, build_common_parser, execution_summary, resolve_paths
from dax.coach_analysis.labels import label_rules
from dax.coach_analysis.loaders import CoachAnalysisInputError, add_canonical_model_columns, join_oof_strict, normalise_event_id, require_table, select_required_two_part, select_required_variant, validate_unique_predictions
from dax.coach_analysis.metrics import add_suppression, first_existing, summary_table
from dax.coach_analysis.plotting import action_pitch_map, horizontal_metric_chart
from dax.coach_analysis.populations import apply_visibility_filter, box_defence_population
from dax.coach_analysis.reporting import data_derived_conclusions, markdown_table, render_conclusions, write_json, write_markdown_report
from dax.coach_analysis.representative_events import select_representative_events
from dax.coach_analysis.timeline import add_next_events, normalise_processed_events, validate_processed_timeline
from dax.coach_analysis.zones import CoordinateError, add_pitch_zones

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


def _clean_metadata_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned.isna() | cleaned.eq("") | cleaned.str.lower().eq("nan"))


def _coalesce_metadata_aliases(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    values = pd.Series(pd.NA, index=df.index, dtype="string")
    for col in candidates:
        if col not in df.columns:
            continue
        candidate = _clean_metadata_series(df[col])
        values = values.where(values.notna(), candidate)
    return values.fillna("unknown")


def _canonical_competition(df: pd.DataFrame) -> pd.DataFrame:
    """Expose competition metadata under stable coach-analysis column names.

    The processed StatsBomb pipeline carries competition metadata as
    competition_label, competition_id and season_id on action rows. Some
    historical extracts use competition/competition_name and season/season_name
    instead, so the coach-analysis layer accepts all known aliases and falls
    back to unknown only when metadata is genuinely absent or null.
    """
    out = df.copy()
    aliases = {
        "coach_competition": ["competition_label", "competition", "competition_name"],
        "coach_competition_stage": ["competition_stage", "stage", "stage_name"],
        "coach_season": ["season_name", "season", "season_id"],
        "coach_competition_id": ["competition_id"],
        "coach_season_id": ["season_id"],
    }
    for target, candidates in aliases.items():
        out[target] = _coalesce_metadata_aliases(out, candidates)
    return out


def _prepare_data(args, root: Path) -> tuple[pd.DataFrame, dict]:
    actions = normalise_event_id(require_table(args.actions_input, root))
    events = normalise_processed_events(require_table(args.processed_events_input, root))
    timeline = validate_processed_timeline(events)
    if not timeline["next_event_sequence_possible"]:
        raise CoachAnalysisInputError("Processed events cannot support next-event sequence construction.")
    joined = join_oof_strict(actions, _select_prediction_sets(args, root), KEYS)
    with_next = add_next_events(joined, events, window_seconds=args.sequence_window_seconds)
    return add_suppression(add_canonical_model_columns(_canonical_competition(with_next))), timeline


def _event_text(df: pd.DataFrame, prefix: str = "") -> pd.Series:
    col = first_existing(df, [f"{prefix}event_type", f"{prefix}type", f"{prefix}action_family"])
    return df[col].astype(str).str.lower() if col else pd.Series("", index=df.index)


def _augment_sequences(population: pd.DataFrame, window_seconds: float = 10.0) -> pd.DataFrame:
    out = population.copy()
    action = _event_text(out)
    next_text = _event_text(out, "next_")
    second_text = _event_text(out, "second_next_")
    next_team = out.get("next_team", pd.Series(pd.NA, index=out.index))
    action_team = out.get("team", pd.Series(pd.NA, index=out.index))
    same_team_next = next_team.eq(action_team)
    won_flag = out.get("action_won_possession", out.get("possession_won", pd.Series(False, index=out.index))).fillna(False).astype(bool)
    changed_flag = out.get("action_changed_possession", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    out["coach_possession_secured"] = same_team_next.fillna(False) | won_flag | changed_flag
    out["coach_clearance_outcome"] = "not clearance"
    out.loc[action.str.contains("clearance") & out["coach_possession_secured"], "coach_clearance_outcome"] = "clearance relief"
    out.loc[action.str.contains("clearance") & next_text.str.contains("out|stoppage|throw|corner|goal kick"), "coach_clearance_outcome"] = "out of play"
    out.loc[action.str.contains("clearance") & out["coach_clearance_outcome"].eq("not clearance") & next_team.ne(action_team) & out.get("next_event_within_window", pd.Series(False, index=out.index)).fillna(False), "coach_clearance_outcome"] = "opposition recycle"
    out.loc[action.str.contains("clearance") & out["coach_clearance_outcome"].eq("not clearance"), "coach_clearance_outcome"] = "contested continuation"
    out.loc[action.str.contains("clearance") & next_team.isna(), "coach_clearance_outcome"] = "unknown"
    out["coach_block_outcome"] = "not block"
    next_within = out.get("next_event_within_window", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    second_within = out.get("second_next_event_within_window", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    out.loc[action.str.contains("block") & next_team.ne(action_team) & ((next_text.str.contains("shot|rebound") & next_within) | (second_text.str.contains("shot|rebound") & second_within)), "coach_block_outcome"] = "block followed by opposition rebound or shot"
    out.loc[action.str.contains("block") & out["coach_possession_secured"], "coach_block_outcome"] = "block followed by defensive recovery"
    out.loc[action.str.contains("block") & next_text.str.contains("out|stoppage|whistle"), "coach_block_outcome"] = "block followed by stoppage"
    out.loc[action.str.contains("block") & out["coach_block_outcome"].eq("not block"), "coach_block_outcome"] = "unresolved"
    next_x = pd.to_numeric(out.get("next_x", out.get("next_location_x", pd.Series(pd.NA, index=out.index))), errors="coerce")
    x = pd.to_numeric(out.get("x", out.get("location_x", pd.Series(pd.NA, index=out.index))), errors="coerce")
    direction = out.get("attack_direction", pd.Series(1, index=out.index)).fillna(1).replace({"left_to_right": 1, "right_to_left": -1})
    signed_dx = (next_x - x) * pd.to_numeric(direction, errors="coerce").fillna(1)
    opposition_next = next_team.ne(action_team)
    out["coach_pressure_outcome"] = "not pressure"
    out.loc[action.str.contains("pressure") & out["coach_possession_secured"], "coach_pressure_outcome"] = "defensive turnover"
    out.loc[action.str.contains("pressure") & opposition_next & next_within & signed_dx.gt(5), "coach_pressure_outcome"] = "opposition progresses"
    out.loc[action.str.contains("pressure") & opposition_next & next_within & signed_dx.between(-5, 5), "coach_pressure_outcome"] = "opposition recycles sideways"
    out.loc[action.str.contains("pressure") & opposition_next & next_within & signed_dx.lt(-5), "coach_pressure_outcome"] = "opposition forced backward"
    out.loc[action.str.contains("pressure") & opposition_next & ~next_within, "coach_pressure_outcome"] = "no immediate continuation observed"
    out.loc[action.str.contains("pressure") & next_team.isna(), "coach_pressure_outcome"] = "unknown"
    second_next_team = (
        out["second_next_team"].astype("string")
        if "second_next_team" in out.columns
        else pd.Series("", index=out.index, dtype="string")
    )
    second_next_within_window = (
        out["second_next_event_within_window"].fillna(False).astype(bool)
        if "second_next_event_within_window" in out.columns
        else pd.Series(False, index=out.index)
    )
    out["coach_immediate_re_turnover"] = (
        out["coach_possession_secured"].fillna(False).astype(bool)
        & second_next_within_window
        & second_next_team.ne(action_team.astype("string")).fillna(False)
    )
    out["coach_duel_outcome"] = "not duel"
    out.loc[action.str.contains("duel") & out["coach_possession_secured"], "coach_duel_outcome"] = "duel won and secured"
    out.loc[action.str.contains("duel") & ~out["coach_possession_secured"], "coach_duel_outcome"] = "duel unresolved/lost"
    if {"match_id", "possession"}.issubset(out.columns):
        original_index = out.index
        sort_cols = [c for c in ["match_id", "period", "possession", "coach_event_seconds", "event_index"] if c in out.columns]
        ordered = out.sort_values(sort_cols).copy() if sort_cols else out.copy()
        previous_time = ordered.groupby(["match_id", "period", "possession"], dropna=False)["coach_event_seconds"].shift(1) if "coach_event_seconds" in ordered.columns and "period" in ordered.columns else pd.Series(pd.NA, index=ordered.index)
        repeated = (ordered["coach_event_seconds"] - previous_time).le(window_seconds) if "coach_event_seconds" in ordered.columns else pd.Series(False, index=ordered.index)
        out["coach_repeated_box_action"] = repeated.reindex(original_index).fillna(False).astype(bool)
    else:
        out["coach_repeated_box_action"] = False
    return out


def _metric_table(population: pd.DataFrame, group_col: str, n_boot: int, seed: int) -> pd.DataFrame:
    if population.empty or group_col not in population.columns:
        return pd.DataFrame()
    table = summary_table(population, [group_col])
    metric = first_existing(population, ["coach_observed_xg", "target_future_xg_10s", "observed_future_xg", "future_xg", "xg_within_10s", "y_true_xg"])
    if metric:
        ci = add_match_bootstrap_by_group(population, [group_col], metric, n_boot=n_boot, seed=seed)
        table = table.merge(ci[[group_col, "ci_low", "ci_high"]], on=group_col, how="left")
    return table


def _phase_spatial_diagnostic(stage_counts: dict[str, int | str | None]) -> dict[str, int | str | None]:
    spatial = int(stage_counts.get("centre_back_own_box_actions", 0) or 0)
    phase = int(stage_counts.get("centre_back_phase_labelled_box_defence_actions", 0) or 0)
    overlap = int(stage_counts.get("spatial_phase_overlap", 0) or 0)
    return {
        "primary_population": "spatial_own_box_centre_back_actions",
        "spatial_own_box_centre_back_actions": spatial,
        "phase_labelled_box_defence_centre_back_actions": phase,
        "overlap_count": overlap,
        "coordinate_x_column": stage_counts.get("coordinate_x_column"),
        "coordinate_y_column": stage_counts.get("coordinate_y_column"),
        "interpretation": "Spatial own-box centre-back analysis is the primary population. Phase-labelled box_defence is retained as a diagnostic because it can use a different upstream tactical definition or coordinate normalisation; overlap is not forced to match.",
    }


def _sensitivity_warning(
    population: pd.DataFrame,
    cols: list[str],
    comparison_mask: pd.Series,
    min_actions: int,
    min_matches: int,
) -> str:
    rows = int(comparison_mask.sum())
    comparison_matches = int(population.loc[comparison_mask, "match_id"].nunique()) if "match_id" in population.columns else 0
    warnings = []
    if rows and len(cols) == 2 and all(c in population.columns for c in cols):
        compared = population.loc[comparison_mask, cols]
        a = pd.to_numeric(compared[cols[0]], errors="coerce")
        b = pd.to_numeric(compared[cols[1]], errors="coerce")
        if not compared.empty and a.reset_index(drop=True).equals(b.reset_index(drop=True)):
            warnings.append("validation-mode limitation: sensitivity variant is identical to the primary variant; do not interpret zero disagreement as model robustness")
    if rows < min_actions:
        warnings.append(f"validation-mode limitation: smoke-sized sensitivity action sample: {rows} < {min_actions}")
    if comparison_matches < min_matches:
        warnings.append(f"validation-mode limitation: smoke-sized sensitivity match sample: {comparison_matches} < {min_matches}")
    if len(population) and rows < len(population):
        warnings.append(f"validation-mode limitation: sensitivity comparison covers {rows} of {len(population)} selected actions")
    return "; ".join(warnings)


def _sensitivity_tables(population: pd.DataFrame, min_actions: int = 30, min_matches: int = 5) -> dict[str, pd.DataFrame]:
    specs = {
        "b7_vs_b6_expected_shot": ["coach_expected_shot_b7", "coach_expected_shot_b6"],
        "r4_vs_r6_expected_xg": ["coach_expected_xg_r4", "coach_expected_xg_r6"],
        "r4_vs_two_part_expected_xg": ["coach_expected_xg_r4", "coach_expected_xg_two_part"],
    }
    out = {}
    for name, cols in specs.items():
        pairs = [(cols[i], cols[i + 1]) for i in range(0, len(cols), 2) if cols[i] in population.columns and cols[i + 1] in population.columns]
        if pairs:
            a, b = pairs[0]
            diff = pd.to_numeric(population[a], errors="coerce") - pd.to_numeric(population[b], errors="coerce")
            comparison_mask = diff.notna()
            rows = int(comparison_mask.sum())
            matches = int(population.loc[comparison_mask, "match_id"].nunique()) if "match_id" in population.columns else 0
            out[name] = pd.DataFrame({"comparison": [name], "rows": [rows], "matches": [matches], "mean_difference": [float(diff.mean())], "warning": [_sensitivity_warning(population, [a, b], comparison_mask, min_actions, min_matches)]})
        else:
            out[name] = pd.DataFrame({"comparison": [name], "rows": [0], "matches": [0], "mean_difference": [pd.NA], "warning": ["validation-mode limitation: required sensitivity columns are unavailable"]})
    return out




def _population_stage_counts(actions: pd.DataFrame, population: pd.DataFrame) -> dict[str, int | str | None]:
    if actions.empty:
        return {"all_defensive_actions": 0, "all_centre_back_actions": 0, "own_box_actions": 0, "centre_back_own_box_actions": 0, "phase_labelled_box_defence_actions": 0, "centre_back_phase_labelled_box_defence_actions": 0, "spatial_phase_overlap": 0, "coordinate_x_column": None, "coordinate_y_column": None}
    pos = actions.get("position_group", pd.Series("", index=actions.index)).astype(str).str.lower()
    cb_mask = pos.eq("centre_back") | pos.isin(["cb", "centre back", "center back"])
    own_box_mask = actions.get("coach_defensive_box_zone", pd.Series("outside defensive box", index=actions.index)).ne("outside defensive box")
    phase = actions.get("phase_label", pd.Series("", index=actions.index)).astype(str).str.lower().eq("box_defence")
    xcol = actions["coach_coordinate_x_column"].iloc[0] if "coach_coordinate_x_column" in actions.columns and len(actions) else None
    ycol = actions["coach_coordinate_y_column"].iloc[0] if "coach_coordinate_y_column" in actions.columns and len(actions) else None
    return {
        "all_defensive_actions": int(len(actions)),
        "all_centre_back_actions": int(cb_mask.sum()),
        "own_box_actions": int(own_box_mask.sum()),
        "centre_back_own_box_actions": int((cb_mask & own_box_mask).sum()),
        "phase_labelled_box_defence_actions": int(phase.sum()),
        "centre_back_phase_labelled_box_defence_actions": int((cb_mask & phase).sum()),
        "spatial_phase_overlap": int((cb_mask & own_box_mask & phase).sum()),
        "coordinate_x_column": None if pd.isna(xcol) else str(xcol),
        "coordinate_y_column": None if pd.isna(ycol) else str(ycol),
    }

def _write_category_video_files(population: pd.DataFrame, video_dir: Path) -> dict[str, int]:
    categories = {
        "clearance_relief.csv": population.get("coach_clearance_outcome", pd.Series(dtype=str)).eq("clearance relief"),
        "clearance_recycled.csv": population.get("coach_clearance_outcome", pd.Series(dtype=str)).eq("opposition recycle"),
        "block_rebound.csv": population.get("coach_block_outcome", pd.Series(dtype=str)).eq("block followed by opposition rebound or shot"),
        "pressure_progression.csv": population.get("coach_pressure_outcome", pd.Series(dtype=str)).eq("opposition progresses"),
        "immediate_re_turnover.csv": population.get("coach_immediate_re_turnover", pd.Series(dtype=bool)).fillna(False).astype(bool),
        "high_expected_threat_no_shot.csv": population.get("coach_expected_shot_b7", pd.Series(dtype=float)).gt(population.get("coach_expected_shot_b7", pd.Series(dtype=float)).quantile(0.75)) & population.get("coach_observed_shot", pd.Series(dtype=float)).eq(0),
        "observed_threat_above_expected.csv": population.get("coach_observed_xg", pd.Series(dtype=float)).gt(population.get("coach_expected_xg_r4", pd.Series(dtype=float))),
        "repeated_box_actions.csv": population.get("coach_repeated_box_action", pd.Series(dtype=bool)).fillna(False).astype(bool),
    }
    counts = {}
    for filename, mask in categories.items():
        subset = population[mask.reindex(population.index, fill_value=False)] if not population.empty else population
        if "match_id" in subset.columns:
            subset = subset.sort_values("match_id").groupby("match_id", as_index=False, group_keys=False).head(2)
        table = select_representative_events(subset, n=20, reason=filename.replace(".csv", "").replace("_", " "))
        _save_table(table, video_dir / filename)
        counts[filename] = len(table)
    return counts

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolve_paths(args, "cb_box_defence")
    tables = paths.child("tables")
    figures = paths.child("figures")
    video_dir = paths.child("video_review")
    try:
        actions, timeline = _prepare_data(args, paths.root)
        zoned_actions = add_pitch_zones(actions, strict=True)
        population = _augment_sequences(box_defence_population(zoned_actions, centre_backs_only=True), args.sequence_window_seconds)
        stage_counts = _population_stage_counts(zoned_actions, population)
    except (CoachAnalysisInputError, CoordinateError) as exc:
        if not args.allow_partial:
            write_json(paths.output_root / "execution_summary.json", execution_summary("failed", error=str(exc)))
            write_markdown_report(paths.output_root / "report.md", "Centre-back box-defence analysis", [("Error", str(exc))])
            return 2
        population = pd.DataFrame()
        timeline = {}
        stage_counts = {"error": str(exc)}
    if not args.allow_partial and (stage_counts.get("all_centre_back_actions", 0) == 0 or stage_counts.get("own_box_actions", 0) == 0 or stage_counts.get("centre_back_own_box_actions", 0) == 0):
        write_json(paths.output_root / "execution_summary.json", execution_summary("failed", filter_stage_counts=stage_counts))
        write_markdown_report(paths.output_root / "report.md", "Centre-back box-defence analysis", [("Filter-stage counts", f"```json\n{stage_counts}\n```")])
        return 2
    phase_spatial_diagnostic = _phase_spatial_diagnostic(stage_counts)
    reliable = apply_visibility_filter(population, reliable_only=True)
    wc_euros = population[population.get("coach_competition", pd.Series(dtype=str)).astype(str).str.contains("World Cup|Euros|Euro", case=False, na=False)].copy() if not population.empty else population
    group_cols = ["coach_defensive_box_zone", "action_family", "event_type", "coach_clearance_outcome", "coach_block_outcome", "coach_pressure_outcome", "coach_duel_outcome", "coach_repeated_box_action", "coach_competition", "coach_competition_stage", "coach_season"]
    generated = {}
    conclusion_records = []
    conclusion_groups = {"dangerous contexts": [], "suppression contexts": [], "possession-control contexts": [], "competition comparison": [], "visibility sensitivity": [], "model disagreement": []}
    for col in group_cols:
        if col in population.columns:
            table = _metric_table(population, col, args.bootstrap_samples, args.seed)
            _save_table(table, tables / f"by_{col}.csv")
            generated[col] = len(table)
            metric = first_existing(table, ["future_xg_per_action", "future_shot_rate", "expected_future_xg"])
            if metric:
                records = data_derived_conclusions(table.rename(columns={col: "subgroup"}), "subgroup", metric, top_n=3, min_actions=args.min_actions, min_matches=args.min_matches)
                conclusion_records.extend(records)
                group_name = "competition comparison" if col == "coach_competition" else "possession-control contexts" if col in {"coach_clearance_outcome", "coach_duel_outcome"} else "dangerous contexts"
                conclusion_groups[group_name].extend(records)
                if "xg_suppression" in table.columns:
                    sup_records = data_derived_conclusions(table.rename(columns={col: "subgroup"}), "subgroup", "xg_suppression", top_n=3, min_actions=args.min_actions, min_matches=args.min_matches)
                    conclusion_groups["suppression contexts"].extend(sup_records)
                    conclusion_records.extend(sup_records)
    if not reliable.empty:
        table = _metric_table(reliable, "coach_defensive_box_zone", args.bootstrap_samples, args.seed)
        _save_table(table, tables / "reliable_visibility_by_defensive_box_zone.csv")
        metric = first_existing(table, ["future_xg_per_action", "future_shot_rate", "expected_future_xg"])
        if metric:
            vis_records = data_derived_conclusions(table.rename(columns={"coach_defensive_box_zone": "subgroup"}), "subgroup", metric, top_n=3, min_actions=args.min_actions, min_matches=args.min_matches)
            conclusion_groups["visibility sensitivity"].extend(vis_records)
            conclusion_records.extend(vis_records)
    sensitivity_warnings = {}
    for name, table in _sensitivity_tables(population, min_actions=args.min_actions, min_matches=args.min_matches).items():
        if not table.empty and table.get("warning") is not None:
            warning = str(table["warning"].iloc[0])
            if warning:
                sensitivity_warnings[name] = warning
        _save_table(table, tables / f"{name}.csv")
        if not table.empty and table["rows"].iloc[0]:
            record = {"subgroup": name, "metric": "mean_difference", "value": float(table["mean_difference"].iloc[0]), "comparison_value": 0.0, "difference": float(table["mean_difference"].iloc[0]), "actions": int(table["rows"].iloc[0]), "matches": int(population["match_id"].nunique()) if "match_id" in population.columns else 0, "ci_low": None, "ci_high": None, "minimum_sample_warning": "" if table["rows"].iloc[0] >= args.min_actions else f"low action sample: {table['rows'].iloc[0]} < {args.min_actions}"}
            if name in sensitivity_warnings:
                record["minimum_sample_warning"] = "; ".join(x for x in [record["minimum_sample_warning"], sensitivity_warnings[name]] if x)
            conclusion_records.append(record)
            conclusion_groups["model disagreement"].append(record)
    video = select_representative_events(population, n=20, reason="CB box-defence candidate for video review")
    _save_table(video, video_dir / "candidate_events.csv")
    category_video_counts = _write_category_video_files(population, video_dir)
    _save_table(label_rules(), tables / "tactical_label_rules.csv")
    if "coach_defensive_box_zone" in population.columns:
        zone_table = _metric_table(population, "coach_defensive_box_zone", args.bootstrap_samples, args.seed)
        metric = first_existing(zone_table, ["future_xg_per_action", "future_shot_rate", "expected_future_xg"])
        if metric:
            horizontal_metric_chart(zone_table, "coach_defensive_box_zone", metric, "CB box-defence threat by mutually exclusive defensive box zone", figures / "zone_threat_horizontal.png")
    action_pitch_map(population, "Centre-back box-defence actions", figures / "cb_box_defence_pitch_map.png")
    competition_counts = population["coach_competition"].value_counts().to_dict() if "coach_competition" in population.columns else {}
    competition_metadata_counts = {
        col: population[col].value_counts(dropna=False).to_dict()
        for col in ["coach_competition", "coach_competition_stage", "coach_season"]
        if col in population.columns
    }
    sections = [
        ("Filter-stage counts", f"```json\n{stage_counts}\n```"),
        ("Phase vs spatial box-defence diagnostic", f"```json\n{phase_spatial_diagnostic}\n```"),
        ("Population", f"Primary population is spatial own-box centre-back actions: {len(population)}; matches: {population['match_id'].nunique() if 'match_id' in population.columns else 0}; World Cup/Euros rows: {len(wc_euros)}. Phase-label overlap is diagnostic only, not ground truth."),
        ("Processed event timeline", f"```json\n{timeline}\n```"),
        ("Competition metadata counts", f"```json\n{competition_metadata_counts}\n```"),
        ("Generated tables", markdown_table(pd.DataFrame([{"table": k, "rows": v} for k, v in generated.items()]))),
        ("Dangerous contexts", render_conclusions(conclusion_groups["dangerous contexts"])),
        ("Suppression contexts", render_conclusions(conclusion_groups["suppression contexts"])),
        ("Possession-control contexts", render_conclusions(conclusion_groups["possession-control contexts"])),
        ("Competition comparison", render_conclusions(conclusion_groups["competition comparison"])),
        ("Visibility sensitivity", render_conclusions(conclusion_groups["visibility sensitivity"])),
        ("Model disagreement", render_conclusions(conclusion_groups["model disagreement"])),
        ("Model-sensitivity validation warnings", f"```json\n{sensitivity_warnings}\n```"),
        ("Reliable-visibility sensitivity", f"Reliable-visibility subset actions: {len(reliable)}."),
        ("Video review", f"Candidate event table written with {len(video)} rows. Category files: {category_video_counts}."),
    ]
    write_markdown_report(paths.output_root / "report.md", "Centre-back box-defence analysis", sections)
    write_json(paths.output_root / "execution_summary.json", execution_summary("completed", filter_stage_counts=stage_counts, phase_spatial_diagnostic=phase_spatial_diagnostic, actions=int(len(population)), matches=int(population["match_id"].nunique()) if "match_id" in population.columns else 0, competition_counts=competition_counts, competition_metadata_counts=competition_metadata_counts, model_sensitivity_warnings=sensitivity_warnings, generated_tables=generated, data_derived_conclusions=conclusion_records, conclusion_groups=conclusion_groups, category_video_counts=category_video_counts, canonical_model_completeness={c: int(population[c].notna().sum()) for c in ["coach_expected_shot_b7", "coach_expected_shot_b6", "coach_expected_xg_r4", "coach_expected_xg_r6", "coach_expected_xg_two_part", "coach_observed_shot", "coach_observed_xg"] if c in population.columns}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
