from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from dax.coach_analysis.bootstrap import add_match_bootstrap_by_group
from dax.coach_analysis.execution import add_model_selection_args, build_common_parser, execution_summary, resolve_paths
from dax.coach_analysis.labels import label_box_defence, label_rules
from dax.coach_analysis.loaders import join_oof, read_optional_table, select_variant
from dax.coach_analysis.metrics import add_suppression, first_existing, summary_table
from dax.coach_analysis.plotting import action_pitch_map, horizontal_metric_chart
from dax.coach_analysis.populations import apply_visibility_filter, box_defence_population
from dax.coach_analysis.reporting import data_derived_conclusions, markdown_table, render_conclusions, write_json, write_markdown_report
from dax.coach_analysis.representative_events import select_representative_events
from dax.coach_analysis.sequences import construct_sequences


def parse_args(argv: list[str] | None = None):
    parser = build_common_parser("Analyse centre-back box-defence risk.")
    add_model_selection_args(parser)
    return parser.parse_args(argv)


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _prepare_data(args, root: Path) -> pd.DataFrame:
    actions = read_optional_table("data/features/player_defensive_actions.parquet", root)
    if actions.empty:
        return actions
    classification = select_variant(read_optional_table("outputs/oof/classification_oof.parquet", root), args.classification_variant)
    regression = select_variant(read_optional_table("outputs/oof/regression_oof.parquet", root), args.regression_variant)
    two_part = select_variant(read_optional_table("outputs/oof/two_part_future_xg_oof_exploratory.parquet", root), args.two_part_variant)
    joined = join_oof(actions, classification, regression, two_part)
    return construct_sequences(add_suppression(joined))


def _augment_cb_box(population: pd.DataFrame) -> pd.DataFrame:
    if population.empty:
        return population
    out = label_box_defence(population)
    action = first_existing(out, ["action_family", "event_type", "type"])
    poss = first_existing(out, ["possession_won", "won_possession", "defensive_action_won"])
    if poss:
        out["coach_possession_secured"] = out[poss].fillna(False).astype(bool)
    else:
        out["coach_possession_secured"] = False
    action_text = out[action].astype(str).str.lower() if action else pd.Series("", index=out.index)
    out["coach_clearance_outcome"] = "not clearance"
    out.loc[action_text.str.contains("clearance") & out["coach_possession_secured"], "coach_clearance_outcome"] = "clearance relief"
    out.loc[action_text.str.contains("clearance") & ~out["coach_possession_secured"], "coach_clearance_outcome"] = "clearance recycled"
    out["coach_block_outcome"] = "not block"
    out.loc[action_text.str.contains("block") & ~out["coach_possession_secured"], "coach_block_outcome"] = "block followed by rebound/continued attack"
    out["coach_pressure_outcome"] = "not pressure"
    out.loc[action_text.str.contains("pressure") & ~out["coach_possession_secured"], "coach_pressure_outcome"] = "pressure followed by progression risk"
    out["coach_duel_outcome"] = "not duel"
    out.loc[action_text.str.contains("duel") & out["coach_possession_secured"], "coach_duel_outcome"] = "duel won"
    out.loc[action_text.str.contains("duel") & ~out["coach_possession_secured"], "coach_duel_outcome"] = "duel lost/unsecured"
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolve_paths(args, "cb_box_defence")
    tables = paths.child("tables")
    figures = paths.child("figures")
    video_dir = paths.child("video_review")
    actions = _prepare_data(args, paths.root)
    if actions.empty:
        summary = execution_summary("completed_with_missing_inputs", missing_inputs=["data/features/player_defensive_actions.parquet"])
        write_markdown_report(paths.output_root / "report.md", "Centre-back box-defence analysis", [("Data availability", "No defensive-action feature table was available; no football conclusions were generated.")])
        write_json(paths.output_root / "execution_summary.json", summary)
        return 0

    population = _augment_cb_box(box_defence_population(actions, centre_backs_only=True))
    reliable = apply_visibility_filter(population, reliable_only=True)
    group_cols = [
        "coach_box_zone", "action_family", "event_type", "coach_possession_secured",
        "coach_clearance_outcome", "coach_block_outcome", "coach_pressure_outcome",
        "coach_duel_outcome", "coach_is_repeated_defensive_action", "competition",
    ]
    generated = {}
    conclusion_source = pd.DataFrame()
    for col in group_cols:
        if col in population.columns:
            table = _metric_table(population, col, args.bootstrap_samples, args.seed)
            _save_table(table, tables / f"by_{col}.csv")
            generated[col] = len(table)
            if conclusion_source.empty and not table.empty:
                conclusion_source = table.rename(columns={col: "subgroup"})
    if not reliable.empty:
        table = _metric_table(reliable, "coach_box_zone", args.bootstrap_samples, args.seed)
        _save_table(table, tables / "reliable_visibility_by_box_zone.csv")
    video = select_representative_events(population, n=20, reason="CB box-defence candidate for video review")
    _save_table(video, video_dir / "candidate_events.csv")
    rules = label_rules()
    _save_table(rules, tables / "tactical_label_rules.csv")
    if "coach_box_zone" in population.columns:
        zone_table = _metric_table(population, "coach_box_zone", args.bootstrap_samples, args.seed)
        if not zone_table.empty:
            metric = first_existing(zone_table, ["future_xg_per_action", "future_shot_rate", "expected_future_xg"])
            if metric:
                horizontal_metric_chart(zone_table, "coach_box_zone", metric, "CB box-defence threat by mutually exclusive box zone", figures / "zone_threat_horizontal.png")
    action_pitch_map(population, "Centre-back box-defence actions", figures / "cb_box_defence_pitch_map.png")
    metric_col = first_existing(conclusion_source, ["future_xg_per_action", "future_shot_rate", "expected_future_xg"])
    conclusions = data_derived_conclusions(conclusion_source, "subgroup", metric_col) if metric_col else []
    sections = [
        ("Population", f"Centre-back box-defence actions: {len(population)} rows across {population['match_id'].nunique() if 'match_id' in population.columns else 0} matches."),
        ("Tactical label rules", markdown_table(rules)),
        ("Generated tables", markdown_table(pd.DataFrame([{"table": k, "rows": v} for k, v in generated.items()]))),
        ("Data-derived conclusions", render_conclusions(conclusions)),
        ("Reliable-visibility sensitivity", f"Reliable-visibility subset actions: {len(reliable)}."),
        ("Video review", f"Candidate event table written with {len(video)} rows."),
    ]
    write_markdown_report(paths.output_root / "report.md", "Centre-back box-defence analysis", sections)
    write_json(
        paths.output_root / "execution_summary.json",
        execution_summary(
            "completed",
            actions=int(len(population)),
            reliable_visibility_actions=int(len(reliable)),
            generated_tables=generated,
            data_derived_conclusions=conclusions,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
