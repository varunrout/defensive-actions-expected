from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .loaders import repo_root


@dataclass(frozen=True)
class CoachAnalysisPaths:
    root: Path
    output_root: Path

    def child(self, name: str) -> Path:
        path = self.output_root / name
        path.mkdir(parents=True, exist_ok=True)
        return path


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--repo-root", type=Path, default=None, help="Repository root. Defaults to auto-discovery.")
    parser.add_argument("--output-root", type=Path, default=None, help="Output directory for this analysis.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--min-actions", type=int, default=30)
    parser.add_argument("--min-matches", type=int, default=5)
    return parser


def resolve_paths(args: argparse.Namespace, analysis_name: str) -> CoachAnalysisPaths:
    root = repo_root(args.repo_root)
    output_root = args.output_root or root / "outputs" / "coach_analysis" / analysis_name
    output_root.mkdir(parents=True, exist_ok=True)
    return CoachAnalysisPaths(root=root, output_root=output_root)


def execution_summary(status: str, **kwargs: Any) -> dict[str, Any]:
    payload = {"status": status}
    payload.update(kwargs)
    return payload


def add_model_selection_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--classification-variant", default="b7_full_with_360")
    parser.add_argument("--classification-sensitivity", default="b6_full_without_360")
    parser.add_argument("--regression-variant", default="r4_full_with_360")
    parser.add_argument("--regression-sensitivity", default="r6_nonlinear_candidate")
    parser.add_argument("--two-part-variant", default="exploratory_two_part")
    return parser
