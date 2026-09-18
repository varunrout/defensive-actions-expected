"""Smoke tests for canonical CLI entry points."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_active_cli_help_commands() -> None:
    scripts = [
        "run_pipeline.py",
        "features/build_features.py",
        "models/train_models.py",
        "models/validate_models.py",
        "analysis/generate_reports.py",
    ]
    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / script), "--help"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
