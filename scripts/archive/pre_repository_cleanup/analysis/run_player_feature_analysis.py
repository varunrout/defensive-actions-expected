"""Thin runner for src.analysis.player_features module."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis.player_features.pipeline import run_analysis


if __name__ == "__main__":
    run_analysis(repo_root=REPO_ROOT, verbose=True)

