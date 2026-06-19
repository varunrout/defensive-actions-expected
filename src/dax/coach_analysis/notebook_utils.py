from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from .labels import label_box_defence, label_rule_samples
from .loaders import (
    build_coach_analysis_frame,
    ensure_output_dirs,
    read_optional_table,
    schema_inventory,
)
from .metrics import add_suppression, summary_table
from .populations import add_visibility_flag, box_defence_population, transition_population, wide_defence_population
from .representative_events import select_representative_events
from .sequences import construct_sequences
from .zones import add_pitch_zones


def repo_root(start: Path | None = None) -> Path:
    root = (start or Path.cwd()).resolve()
    for parent in [root, *root.parents]:
        if (parent / "pyproject.toml").exists() or (parent / "requirements.txt").exists():
            return parent
    return root


def prepare_coach_frame(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]]:
    root = repo_root(root)
    actions = read_optional_table("data/features/player_defensive_actions.parquet", root)
    if actions.empty:
        raise ValueError("data/features/player_defensive_actions.parquet is missing or empty")
    classification_oof = read_optional_table("outputs/oof/classification_oof.parquet", root)
    regression_oof = read_optional_table("outputs/oof/regression_oof.parquet", root)
    two_part_exploratory = read_optional_table("outputs/oof/two_part_future_xg_oof_exploratory.parquet", root)
    events = read_optional_table("data/processed/events_enriched.parquet", root)
    if not events.empty and {"match_id", "event_id"}.issubset(actions.columns):
        context_columns = [
            column
            for column in ["match_id", "id", "competition_label", "opponent_team"]
            if column in events.columns
        ]
        if {"match_id", "id"}.issubset(context_columns):
            context = events[context_columns].rename(columns={"id": "event_id"}).drop_duplicates(["match_id", "event_id"])
            actions = actions.merge(context, on=["match_id", "event_id"], how="left")
            if "competition_label" in actions.columns and "competition" not in actions.columns:
                actions["competition"] = actions["competition_label"]
    merged, coverage = build_coach_analysis_frame(actions, classification_oof, regression_oof, two_part_exploratory)
    merged = add_pitch_zones(add_visibility_flag(add_suppression(merged)))
    merged = construct_sequences(merged, events)
    return merged, events, coverage


def population_stats(df: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(len(df)),
        "matches": int(df["match_id"].nunique()) if "match_id" in df.columns else 0,
        "players": int(df["player"].nunique()) if "player" in df.columns else 0,
    }


def _safe_group_cols(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def export_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def export_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, path: Path, top_n: int = 12) -> Path | None:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    plot_df = df.sort_values(y_col, ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df[x_col].astype(str), plot_df[y_col])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def write_notebook_outputs(
    notebook_slug: str,
    population: pd.DataFrame,
    *,
    groupings: list[list[str]],
    video_n: int = 2,
    include_labels: bool = False,
) -> dict[str, object]:
    root = repo_root()
    output_dirs = ensure_output_dirs(root)
    working = population.copy()
    if include_labels:
        working = label_box_defence(working)

    tables: list[str] = []
    figures: list[str] = []
    for idx, grouping in enumerate(groupings, start=1):
        selected = _safe_group_cols(working, grouping)
        if not selected:
            continue
        table = summary_table(working, selected)
        table_path = output_dirs["tables"] / f"{notebook_slug}_table_{idx}.csv"
        export_table(table, table_path)
        tables.append(str(table_path.relative_to(root)))
        fig_path = output_dirs["figures"] / f"{notebook_slug}_figure_{idx}.png"
        saved_fig = export_bar(table, selected[0], "actions", f"{notebook_slug}: {' / '.join(selected)}", fig_path)
        if saved_fig is not None:
            figures.append(str(saved_fig.relative_to(root)))

    video = select_representative_events(working, n=video_n)
    video_path = output_dirs["video_review"] / f"{notebook_slug}_video_review.csv"
    export_table(video, video_path)

    warnings = []
    if working.empty:
        warnings.append("empty_population")
    if "coach_reliable_visibility" in working.columns:
        coverage = float(working["coach_reliable_visibility"].mean()) if len(working) else 0.0
        if coverage < 0.2:
            warnings.append("low_visibility_coverage")

    return {
        "population": population_stats(working),
        "variants": {
            "classification_primary": "b7_full_with_360",
            "classification_sensitivity": "b6_full_without_360",
            "regression_primary": "r4_full_with_360",
            "two_part": "b7_full_with_360 + conditional_tweedie",
        },
        "tables": tables,
        "figures": figures,
        "video_review": str(video_path.relative_to(root)),
        "video_rows": int(len(video)),
        "warnings": warnings,
    }


def notebook_00_readiness_summary() -> pd.DataFrame:
    root = repo_root()
    inv = schema_inventory(root)
    rows = []
    for key, payload in inv.items():
        rows.append({"input": key, "path": payload.get("path"), "rows": payload.get("rows"), "columns": len(payload.get("columns", []))})
    return pd.DataFrame(rows)


def notebook_rule_samples(population: pd.DataFrame) -> pd.DataFrame:
    return label_rule_samples(population)


def select_population(frame: pd.DataFrame, notebook_id: str) -> pd.DataFrame:
    if notebook_id == "01":
        return box_defence_population(frame, centre_backs_only=True)
    if notebook_id == "02":
        return wide_defence_population(frame)
    if notebook_id == "03":
        return transition_population(frame)
    if notebook_id == "04":
        return frame[frame["action_family"].astype(str).str.contains("pressure", case=False, na=False)].copy()
    if notebook_id == "05":
        return box_defence_population(frame, centre_backs_only=False)
    if notebook_id == "06":
        return frame.copy()
    if notebook_id == "07":
        return frame.copy()
    if notebook_id == "08":
        return frame.copy()
    return frame.copy()


