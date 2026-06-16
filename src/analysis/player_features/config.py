"""Configuration for the player-feature analysis module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalysisConfig:
    """Runtime configuration for the analysis pipeline."""

    repo_root: Path
    input_path: Path
    output_dir: Path
    figures_dir: Path
    tables_dir: Path
    report_dir: Path
    random_seed: int
    min_group_size: int
    max_missing_pct: float
    max_duplicate_event_pct: float
    min_unique_players: int
    bootstrap_iterations: int
    bootstrap_sample_frac: float
    max_feature_missing_for_core: float


def build_config(repo_root: Path, input_path: Path | None = None, output_root: Path | None = None) -> AnalysisConfig:
    data_features = repo_root / "data" / "features"
    default_input = data_features / "player_defensive_actions.parquet"
    fallback_input = data_features / "player_defensive_actions_sample2.parquet"

    resolved_input = input_path or (default_input if default_input.exists() else fallback_input)
    if not resolved_input.exists():
        raise FileNotFoundError(
            "Player defensive dataset not found. Run scripts/build_player_defense_dataset.py first."
        )

    base_out = output_root or (repo_root / "outputs" / "validation" / "analysis" / "player_features")
    figures = base_out / "figures"
    tables = base_out / "tables"
    report = base_out / "report"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)

    return AnalysisConfig(
        repo_root=repo_root,
        input_path=resolved_input,
        output_dir=base_out,
        figures_dir=figures,
        tables_dir=tables,
        report_dir=report,
        random_seed=42,
        min_group_size=40,
        max_missing_pct=35.0,
        max_duplicate_event_pct=0.5,
        min_unique_players=120,
        bootstrap_iterations=200,
        bootstrap_sample_frac=0.7,
        max_feature_missing_for_core=20.0,
    )
