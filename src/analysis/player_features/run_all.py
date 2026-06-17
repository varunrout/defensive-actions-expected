"""Orchestrator for the rigorous player-feature analysis module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import build_config
from .io import load_player_features
from .m01_schema_quality import run as run_01
from .m02_target_label_audit import run as run_02
from .m03_univariate_signal import run as run_03
from .m04_feature_interactions import run as run_04
from .m05_phase_role_stratification import run as run_05
from .m06_leakage_and_confounding import run as run_06
from .m07_stability_and_shift import run as run_07
from .m08_redundancy_and_selection import run as run_08
from .m09_sanity_negative_controls import run as run_09
from .m10_decision_report import run as run_10
from .utils_plots import (
    save_barh,
    save_heatmap,
    save_negative_control_chart,
    save_pitch_heatmap,
    save_rate_ci_bar,
    save_stability_errorbar,
    save_stratified_corr_heatmap,
)


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_figures(df: pd.DataFrame, cfg) -> None:
    miss_path = cfg.tables_dir / "01_missingness.csv"
    if miss_path.exists():
        miss = pd.read_csv(miss_path)
        save_barh(miss, y="feature", x="missing_pct", title="Feature missingness (%)", out_path=cfg.figures_dir / "01_missingness.png")

    phase_path = cfg.tables_dir / "02_target_by_phase.csv"
    if phase_path.exists():
        phase = pd.read_csv(phase_path)
        save_rate_ci_bar(phase, "phase_label", "Shot rate by defensive phase (with CI)", cfg.figures_dir / "02_target_by_phase_ci.png")

    action_path = cfg.tables_dir / "02_target_by_action_family.csv"
    if action_path.exists():
        action = pd.read_csv(action_path)
        save_rate_ci_bar(action, "action_family", "Shot rate by action family (with CI)", cfg.figures_dir / "02_target_by_action_ci.png")

    role_path = cfg.tables_dir / "02_target_by_position_group.csv"
    if role_path.exists():
        role = pd.read_csv(role_path)
        save_rate_ci_bar(role, "position_group", "Shot rate by position group (with CI)", cfg.figures_dir / "02_target_by_position_ci.png")

    numeric_path = cfg.tables_dir / "03_univariate_numeric_signal.csv"
    if numeric_path.exists():
        numeric = pd.read_csv(numeric_path)
        if not numeric.empty:
            # Reuse bar helper with renamed columns for readability.
            chart_df = numeric[["feature", "corr"]].copy().rename(columns={"corr": "correlation"})
            save_barh(
                chart_df.assign(abs_corr=chart_df["correlation"].abs()).sort_values("abs_corr", ascending=False),
                y="feature",
                x="correlation",
                title="Univariate numeric signal (correlation)",
                out_path=cfg.figures_dir / "03_numeric_signal_corr.png",
                top_n=15,
            )

    inter_path = cfg.tables_dir / "04_interaction_tables.csv"
    if inter_path.exists():
        inter = pd.read_csv(inter_path)
        if not inter.empty:
            # Pick one representative interaction heatmap.
            sample = inter[inter["context"] == "phase_label"].copy()
            if not sample.empty:
                sample["feature_bin"] = sample["feature_bin"].astype(str)
                save_heatmap(
                    sample[sample["feature"] == sample["feature"].iloc[0]],
                    index="phase_label",
                    columns="feature_bin",
                    values="shot_rate",
                    title="Shot rate by phase and feature quartile",
                    out_path=cfg.figures_dir / "04_phase_feature_interaction_heatmap.png",
                )

    strat_path = cfg.tables_dir / "05_stratified_correlations.csv"
    if strat_path.exists():
        strat = pd.read_csv(strat_path)
        save_stratified_corr_heatmap(strat, cfg.figures_dir / "05_stratified_corr_heatmap.png")

    team_conf_path = cfg.tables_dir / "06_team_rate_confounding.csv"
    if team_conf_path.exists():
        team_conf = pd.read_csv(team_conf_path)
        if not team_conf.empty:
            save_barh(
                team_conf.sort_values("shot_rate", ascending=False),
                y="team",
                x="shot_rate",
                title="Team-level target rate (confounding screen)",
                out_path=cfg.figures_dir / "06_team_rate_confounding.png",
                top_n=20,
            )

    drift_path = cfg.tables_dir / "07_feature_drift_psi.csv"
    if drift_path.exists():
        drift = pd.read_csv(drift_path)
        if not drift.empty:
            save_barh(
                drift.sort_values("psi_early_vs_late", ascending=False),
                y="feature",
                x="psi_early_vs_late",
                title="Feature drift PSI (early vs late matches)",
                out_path=cfg.figures_dir / "07_feature_drift_psi.png",
                top_n=15,
            )

    stability_path = cfg.tables_dir / "07_grouped_bootstrap_stability.csv"
    if stability_path.exists():
        stability = pd.read_csv(stability_path)
        save_stability_errorbar(stability, "Grouped bootstrap stability", cfg.figures_dir / "07_bootstrap_stability.png")

    redundant_path = cfg.tables_dir / "08_redundant_feature_pairs.csv"
    if redundant_path.exists():
        redundant = pd.read_csv(redundant_path)
        if not redundant.empty:
            save_barh(
                redundant.assign(pair=redundant["feature_a"].astype(str) + " vs " + redundant["feature_b"].astype(str)),
                y="pair",
                x="abs_corr",
                title="Highly correlated feature pairs",
                out_path=cfg.figures_dir / "08_redundant_feature_pairs.png",
                top_n=15,
            )

    neg_path = cfg.tables_dir / "09_negative_control_summary.csv"
    if neg_path.exists():
        neg = pd.read_csv(neg_path)
        save_negative_control_chart(neg, cfg.figures_dir / "09_negative_control_check.png")

    # Pitch maps for tactical justification.
    save_pitch_heatmap(
        df,
        cfg.figures_dir / "pitch_action_density.png",
        title="Defensive action density map",
        statistic="count",
        normalize=True,
    )
    save_pitch_heatmap(
        df,
        cfg.figures_dir / "pitch_shot_risk_map.png",
        title="Shot-in-10s rate by defensive action location",
        statistic="mean",
        value_col="target_future_shot_10s",
    )


def run_analysis(
    repo_root: Path,
    input_path: Path | None = None,
    output_root: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the full analysis module and return consolidated results."""
    cfg = build_config(repo_root=repo_root, input_path=input_path, output_root=output_root)
    df = load_player_features(cfg.input_path)

    results: dict[str, dict[str, Any]] = {}
    results["01_schema_quality"] = run_01(df, cfg)
    results["02_target_label_audit"] = run_02(df, cfg)
    results["03_univariate_signal"] = run_03(df, cfg)
    results["04_feature_interactions"] = run_04(df, cfg)
    results["05_phase_role_stratification"] = run_05(df, cfg)
    results["06_leakage_and_confounding"] = run_06(df, cfg)
    results["07_stability_and_shift"] = run_07(df, cfg)
    results["08_redundancy_and_selection"] = run_08(df, cfg)
    results["09_sanity_negative_controls"] = run_09(df, cfg)

    decision = run_10(results, cfg)
    results["10_decision_report"] = decision

    summary = {
        "input_path": str(cfg.input_path),
        "output_dir": str(cfg.output_dir),
        "rows": int(len(df)),
        "matches": int(df["match_id"].nunique()),
        "players": int(df["player_id"].nunique()),
        "decision": decision["decision"],
        "failed_mandatory_steps": decision["failed_mandatory_steps"],
        "step_results": results,
    }

    _write_summary(cfg.output_dir / "summary.json", summary)
    _make_figures(df, cfg)

    if verbose:
        print("\n" + "=" * 72)
        print("PLAYER FEATURE ANALYSIS (RIGOROUS MODULE)")
        print("=" * 72)
        print(f"Input: {cfg.input_path}")
        print(f"Rows: {summary['rows']:,} | Matches: {summary['matches']:,} | Players: {summary['players']:,}")
        print(f"Decision: {summary['decision']}")
        if summary["failed_mandatory_steps"]:
            print("Failed gates: " + ", ".join(summary["failed_mandatory_steps"]))
        print(f"Artifacts: {cfg.output_dir}")

    return summary

