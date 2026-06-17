from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from dax.features.event_context import add_event_context
from dax.features.phase_segmentation import label_defensive_phases
from dax.targets.short_horizon import add_future_shot_target, add_future_xg_target

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL = [0, 0, 120, 0, 120, 80, 0, 80]
SMALL = [40, 20, 80, 20, 80, 60, 40, 60]


def _shift_point(point: list[float] | None, dx: float) -> list[float] | None:
    if point is None:
        return None
    return [max(0.0, min(120.0, point[0] + dx)), point[1]]


def _build_fixture_events() -> pd.DataFrame:
    base_events = [
        {"period": 1, "index": 1, "minute": 0, "second": 0, "possession": 1, "team": "Team A", "possession_team": "Team A", "team_in_possession": "Team A", "home_team": "Team A", "away_team": "Team B", "event_type": "Pass", "id": "e1", "location": [80, 40], "ball_x": 80, "ball_y": 40, "has_360": True, "freeze_frame": [], "visible_area": FULL},
        {"period": 1, "index": 2, "minute": 0, "second": 2, "possession": 1, "team": "Team B", "possession_team": "Team A", "team_in_possession": "Team A", "home_team": "Team A", "away_team": "Team B", "event_type": "Pressure", "id": "pressure", "player_id": 9, "player": "Defender", "position": "Centre Back", "location": [70, 40], "ball_x": 70, "ball_y": 40, "has_360": True, "freeze_frame": [{"teammate": True, "location": [60, 40]}, {"teammate": True, "location": [80, 40]}, {"teammate": True, "location": [100, 40]}], "visible_area": FULL},
        {"period": 1, "index": 3, "minute": 0, "second": 6, "possession": 1, "team": "Team A", "possession_team": "Team A", "team_in_possession": "Team A", "home_team": "Team A", "away_team": "Team B", "event_type": "Shot", "id": "shot", "location": [108, 40], "ball_x": 108, "ball_y": 40, "shot_statsbomb_xg": 0.25, "has_360": True, "freeze_frame": [], "visible_area": FULL},
        {"period": 1, "index": 4, "minute": 0, "second": 20, "possession": 2, "team": "Team B", "possession_team": "Team B", "team_in_possession": "Team B", "home_team": "Team A", "away_team": "Team B", "event_type": "Ball Recovery", "id": "recovery", "player_id": 10, "player": "Recoverer", "position": "Midfield", "location": [62, 42], "ball_x": 62, "ball_y": 42, "has_360": True, "freeze_frame": [], "visible_area": FULL},
        {"period": 1, "index": 5, "minute": 0, "second": 30, "possession": 3, "team": "Team A", "possession_team": "Team A", "team_in_possession": "Team A", "home_team": "Team A", "away_team": "Team B", "event_type": "Pass", "id": "e5", "location": [50, 40], "ball_x": 50, "ball_y": 40, "has_360": True, "freeze_frame": [], "visible_area": FULL},
        {"period": 1, "index": 6, "minute": 0, "second": 35, "possession": 4, "team": "Team B", "possession_team": "Team B", "team_in_possession": "Team B", "home_team": "Team A", "away_team": "Team B", "event_type": "Interception", "id": "interception", "player_id": 11, "player": "Interceptor", "position": "Full Back", "location": [58, 38], "ball_x": 58, "ball_y": 38, "has_360": True, "freeze_frame": [], "visible_area": FULL},
        {"period": 1, "index": 7, "minute": 0, "second": 50, "possession": 5, "team": "Team A", "possession_team": "Team A", "team_in_possession": "Team A", "home_team": "Team A", "away_team": "Team B", "event_type": "Pass", "id": "e7", "location": [55, 40], "ball_x": 55, "ball_y": 40, "has_360": True, "freeze_frame": [], "visible_area": FULL},
        {"period": 1, "index": 8, "minute": 0, "second": 52, "possession": 5, "team": None, "possession_team": "Team A", "team_in_possession": "Team A", "home_team": "Team A", "away_team": "Team B", "event_type": "Pressure", "id": "unknown", "player_id": 12, "player": "Unknown Context", "position": "Midfield", "location": [42, 40], "ball_x": 42, "ball_y": 40, "has_360": True, "freeze_frame": [{"teammate": True, "location": [43, 40]}, {"teammate": False, "location": [44, 40]}], "visible_area": SMALL},
        {"period": 1, "index": 9, "minute": 1, "second": 0, "possession": 5, "team": "Team A", "possession_team": "Team A", "team_in_possession": "Team A", "home_team": "Team A", "away_team": "Team B", "event_type": "Duel", "id": "duel", "player_id": 13, "player": "Duel Player", "position": "Midfield", "location": [60, 44], "ball_x": 60, "ball_y": 44, "has_360": True, "freeze_frame": [{"teammate": True, "location": [59, 45]}], "visible_area": FULL},
    ]
    rows: list[dict] = []
    for match_id, shift in enumerate([0.0, 2.0, -2.0, 1.0], start=1):
        for event in base_events:
            row = dict(event)
            row["match_id"] = match_id
            row["id"] = f"{event['id']}_{match_id}"
            row["location"] = _shift_point(event.get("location"), shift)
            row["ball_x"] = max(0.0, min(120.0, float(event["ball_x"]) + shift))
            if event.get("freeze_frame"):
                row["freeze_frame"] = [
                    {
                        **frame,
                        "location": _shift_point(frame.get("location"), shift),
                    }
                    for frame in event["freeze_frame"]
                ]
            rows.append(row)
    return pd.DataFrame(rows)


def _write_fixture_targets(tmp_path: Path) -> Path:
    events = _build_fixture_events()
    context = add_event_context(events)
    phased = pd.DataFrame(label_defensive_phases(context.to_dict("records")))
    targeted = add_future_xg_target(add_future_shot_target(phased))
    input_path = tmp_path / "events_with_targets.parquet"
    targeted.to_parquet(input_path, index=False)
    return input_path


def _run_cli(*args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def test_build_features_cli_creates_output_with_corrected_targets(tmp_path: Path):
    input_path = _write_fixture_targets(tmp_path)
    output_path = tmp_path / "player_defensive_actions.parquet"

    result = _run_cli("scripts/build_features.py", "--input", str(input_path), "--output", str(output_path))

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    df = pd.read_parquet(output_path)
    assert df["target_future_shot_10s"].sum() > 0
    assert df["target_future_xg_10s"].sum() > 0
    assert "target_xt_10s" not in df.columns


def test_run_pipeline_dry_run_succeeds():
    result = _run_cli("scripts/run_pipeline.py", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "[dry-run]" in result.stdout


def test_run_pipeline_build_features_stage_succeeds(tmp_path: Path):
    input_path = _write_fixture_targets(tmp_path)
    output_path = tmp_path / "pipeline_player_defensive_actions.parquet"

    result = _run_cli(
        "scripts/run_pipeline.py",
        "--stage",
        "build-features",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    df = pd.read_parquet(output_path)
    assert df["target_future_shot_10s"].sum() > 0
    assert df["target_future_xg_10s"].sum() > 0


def test_train_validate_and_generate_reports_clis_succeed_on_fixture_data(tmp_path: Path):
    input_path = _write_fixture_targets(tmp_path)
    feature_path = tmp_path / "player_defensive_actions.parquet"

    build_result = _run_cli("scripts/build_features.py", "--input", str(input_path), "--output", str(feature_path))
    assert build_result.returncode == 0, build_result.stderr

    models_dir = tmp_path / "models"
    validation_dir = tmp_path / "validation"
    oof_dir = tmp_path / "oof"

    train_result = _run_cli(
        "scripts/train_models.py",
        "--task",
        "all",
        "--input",
        str(feature_path),
        "--models-dir",
        str(models_dir),
        "--validation-dir",
        str(validation_dir),
        "--oof-dir",
        str(oof_dir),
        "--n-splits",
        "2",
    )
    assert train_result.returncode == 0, train_result.stderr
    assert (validation_dir / "baseline" / "baseline_model_metrics.json").exists()
    assert (validation_dir / "regression" / "regression_model_metrics.json").exists()

    validate_result = _run_cli(
        "scripts/validate_models.py",
        "--task",
        "all",
        "--validation-dir",
        str(validation_dir),
        "--oof-dir",
        str(oof_dir),
    )
    assert validate_result.returncode == 0, validate_result.stderr
    assert (validation_dir / "baseline" / "baseline_model_metrics_table.csv").exists()
    assert (validation_dir / "regression" / "regression_model_metrics_table.csv").exists()

    report_path = tmp_path / "reports" / "VALIDATION_SUMMARY.md"
    report_result = _run_cli(
        "scripts/generate_reports.py",
        "--baseline-metrics",
        str(validation_dir / "baseline" / "baseline_model_metrics.json"),
        "--regression-metrics",
        str(validation_dir / "regression" / "regression_model_metrics.json"),
        "--output",
        str(report_path),
    )
    assert report_result.returncode == 0, report_result.stderr
    assert report_path.exists()
