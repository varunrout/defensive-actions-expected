"""Create 360-only possession visualizations for DAx.

Outputs static PNGs in ``outputs/validation/possessions/``:
  - possession_phase_transition_heatmap.png
  - possession_phase_mix_by_outcome.png
  - sample_shot_possession_sequence.png
  - sample_no_shot_possession_sequence.png

The script uses only events with ``has_360 == True`` and the possession table built
from those events.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_FEATURES = REPO_ROOT / "data" / "features"
DATA_VALIDATION = REPO_ROOT / "outputs" / "validation" / "possessions"

PHASE_ORDER = [
    "counterpress_after_loss",
    "transition_defence",
    "box_defence",
    "settled_low_block",
    "settled_mid_block",
    "second_ball",
    "high_press",
    "wide_defending",
    "central_progression_defence",
]

PHASE_COLORS = {
    "counterpress_after_loss": "#d62728",
    "transition_defence": "#ff7f0e",
    "box_defence": "#8c564b",
    "settled_low_block": "#1f77b4",
    "settled_mid_block": "#17becf",
    "second_ball": "#9467bd",
    "high_press": "#2ca02c",
    "wide_defending": "#e377c2",
    "central_progression_defence": "#bcbd22",
}


def as_list(value: Any) -> list[Any]:
    """Normalize parquet list-like values to plain Python lists."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def draw_pitch(ax: plt.Axes) -> None:
    """Draw a StatsBomb-sized pitch (120x80)."""
    line_color = "#222222"
    ax.add_patch(Rectangle((0, 0), 120, 80, fill=False, ec=line_color, lw=1.5))
    ax.plot([60, 60], [0, 80], color=line_color, lw=1.2)
    ax.add_patch(Circle((60, 40), 10, fill=False, ec=line_color, lw=1.2))
    ax.add_patch(Circle((60, 40), 0.6, color=line_color))

    # Penalty areas
    ax.add_patch(Rectangle((0, 18), 18, 44, fill=False, ec=line_color, lw=1.2))
    ax.add_patch(Rectangle((102, 18), 18, 44, fill=False, ec=line_color, lw=1.2))
    ax.add_patch(Rectangle((0, 30), 6, 20, fill=False, ec=line_color, lw=1.2))
    ax.add_patch(Rectangle((114, 30), 6, 20, fill=False, ec=line_color, lw=1.2))
    ax.add_patch(Circle((12, 40), 0.6, color=line_color))
    ax.add_patch(Circle((108, 40), 0.6, color=line_color))
    ax.add_patch(Arc((12, 40), 20, 20, angle=0, theta1=310, theta2=50, color=line_color, lw=1.0))
    ax.add_patch(Arc((108, 40), 20, 20, angle=0, theta1=130, theta2=230, color=line_color, lw=1.0))

    ax.set_xlim(-2, 122)
    ax.set_ylim(-2, 82)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f8f8f8")
    ax.annotate(
        "Attacking direction ->",
        xy=(102, 82),
        xytext=(18, 82),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#222222"),
        ha="left",
        va="center",
        fontsize=8,
        color="#222222",
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    possessions = pd.read_parquet(DATA_FEATURES / "possessions_with_360.parquet")
    events = pd.read_parquet(DATA_PROCESSED / "events_with_targets.parquet")
    events_360 = events[events["has_360"] == True].reset_index(drop=True).copy()
    return possessions, events_360


def possession_events(events_360: pd.DataFrame, possession_row: pd.Series) -> pd.DataFrame:
    indices = possession_row["event_indices"]
    if not isinstance(indices, (list, np.ndarray)):
        return pd.DataFrame()
    idx = [int(i) for i in indices if 0 <= int(i) < len(events_360)]
    return events_360.iloc[idx].reset_index(drop=True)


def plot_transition_heatmap(possessions: pd.DataFrame, output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for transitions in possessions["phase_transitions"]:
        for from_phase, to_phase in as_list(transitions):
            rows.append({"from_phase": from_phase, "to_phase": to_phase})

    if not rows:
        return

    df = pd.DataFrame(rows)
    matrix = pd.crosstab(df["from_phase"], df["to_phase"]).reindex(
        index=PHASE_ORDER,
        columns=PHASE_ORDER,
        fill_value=0,
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap="YlGnBu", linewidths=0.5, annot=False)
    plt.title("Defensive phase transitions within 360-only possessions")
    plt.xlabel("Next phase")
    plt.ylabel("Current phase")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_phase_mix_by_outcome(possessions: pd.DataFrame, output_path: Path) -> None:
    phase_counts = {phase: [] for phase in PHASE_ORDER}
    outcomes = []

    for _, row in possessions.iterrows():
        phases = as_list(row["phases"])
        total = max(1, len(phases))
        counts = pd.Series(phases).value_counts()
        for phase in PHASE_ORDER:
            phase_counts[phase].append(float(counts.get(phase, 0)) / total)
        outcomes.append("Shot in 10s" if int(row["has_shot_in_10s"]) == 1 else "No shot in 10s")

    mix_df = pd.DataFrame(phase_counts)
    mix_df["outcome"] = outcomes
    grouped = mix_df.groupby("outcome")[PHASE_ORDER].mean()

    ax = grouped.T.plot(
        kind="barh",
        figsize=(10, 6),
        color=["#4c78a8", "#f58518"],
    )
    ax.set_title("Average phase share inside possessions (360-only)")
    ax.set_xlabel("Average share of possession events")
    ax.set_ylabel("Defensive phase")
    ax.legend(title="Outcome", loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def phase_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", color="w", label=phase, markerfacecolor=PHASE_COLORS[phase], markersize=8)
        for phase in PHASE_ORDER
    ]


def choose_sample_possession(possessions: pd.DataFrame, shot_value: int) -> pd.Series:
    subset = possessions[possessions["has_shot_in_10s"] == shot_value].copy()
    subset = subset[(subset["event_count"] >= 8) & (subset["event_count"] <= 80)]
    if subset.empty:
        subset = possessions[possessions["has_shot_in_10s"] == shot_value].copy()
    subset["score"] = (
        subset["phase_unique_count"].astype(float) * 3.0
        + subset["event_count"].astype(float) * 0.2
        + subset["opponent_count_avg"].astype(float)
    )
    return subset.sort_values("score", ascending=False).iloc[0]


def plot_trajectory(ax: plt.Axes, events_df: pd.DataFrame, title: str) -> None:
    draw_pitch(ax)
    if events_df.empty:
        ax.set_title(title)
        return

    x = pd.to_numeric(events_df["ball_x"], errors="coerce").to_numpy()
    y = pd.to_numeric(events_df["ball_y"], errors="coerce").to_numpy()
    opp = pd.to_numeric(events_df["opponent_count"], errors="coerce").fillna(0).to_numpy()

    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    opp = opp[valid]
    phases = events_df.loc[valid, "phase_label"].tolist()

    if len(x) == 0:
        ax.set_title(title)
        return

    for i in range(len(x) - 1):
        ax.annotate(
            "",
            xy=(x[i + 1], y[i + 1]),
            xytext=(x[i], y[i]),
            arrowprops=dict(arrowstyle="->", lw=1.0, color="#888888", alpha=0.6),
        )

    for i, (xi, yi, phase, pressure) in enumerate(zip(x, y, phases, opp, strict=False)):
        size = 40 + (pressure * 8)
        ax.scatter(
            xi,
            yi,
            s=size,
            color=PHASE_COLORS.get(phase, "#444444"),
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        if i in {0, len(x) - 1}:
            ax.text(xi + 1.5, yi + 1.5, "Start" if i == 0 else "End", fontsize=8, weight="bold")

    ax.set_title(title)
    ax.legend(handles=phase_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=8)


def plot_timeline(ax: plt.Axes, events_df: pd.DataFrame, title: str) -> None:
    if events_df.empty:
        ax.set_title(title)
        return

    x = np.arange(len(events_df))
    phases = events_df["phase_label"].tolist()
    opp = pd.to_numeric(events_df["opponent_count"], errors="coerce").fillna(0).to_numpy()
    secs = (pd.to_numeric(events_df["minute"], errors="coerce").fillna(0) * 60 + pd.to_numeric(events_df["second"], errors="coerce").fillna(0)).to_numpy()

    for i, phase in enumerate(phases):
        ax.bar(i, 1, color=PHASE_COLORS.get(phase, "#999999"), width=0.95)

    ax.set_yticks([])
    ax.set_xlim(-0.5, len(events_df) - 0.5)
    ax.set_xlabel("Event order inside possession")
    ax.set_title(title)

    ax2 = ax.twinx()
    ax2.plot(x, opp, color="black", marker="o", linewidth=1.3, markersize=3, label="Opponent count")
    ax2.set_ylabel("Visible opponents")
    if len(events_df) > 1:
        ax.set_xticks(np.linspace(0, len(events_df) - 1, num=min(8, len(events_df)), dtype=int))
        ax.set_xticklabels([f"{int(secs[i])}s" for i in ax.get_xticks()])


def _extract_players(frame: Any) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]]]:
    teammates: list[tuple[float, float]] = []
    opponents: list[tuple[float, float]] = []
    actors: list[tuple[float, float]] = []

    if not isinstance(frame, (list, np.ndarray)):
        return teammates, opponents, actors

    for player in frame:
        if not isinstance(player, dict):
            continue
        loc = player.get("location")
        if not isinstance(loc, (list, tuple, np.ndarray)) or len(loc) < 2:
            continue
        point = (float(loc[0]), float(loc[1]))
        if bool(player.get("actor")):
            actors.append(point)
        elif bool(player.get("teammate")):
            teammates.append(point)
        else:
            opponents.append(point)

    return teammates, opponents, actors


def plot_freeze_frames(axs: list[plt.Axes], events_df: pd.DataFrame) -> None:
    if events_df.empty:
        return

    sample_ix = sorted({0, len(events_df) // 2, len(events_df) - 1})
    for ax in axs:
        ax.axis("off")

    for ax, i in zip(axs, sample_ix, strict=False):
        ax.axis("on")
        draw_pitch(ax)
        row = events_df.iloc[i]
        teammates, opponents, actors = _extract_players(row.get("freeze_frame"))

        if teammates:
            tx, ty = zip(*teammates)
            ax.scatter(tx, ty, color="#1f77b4", s=45, label="Attacking teammates")
        if opponents:
            ox, oy = zip(*opponents)
            ax.scatter(ox, oy, color="#d62728", s=45, label="Defenders")
        if actors:
            ax.scatter(*zip(*actors), color="#ffbf00", edgecolor="black", s=75, label="Actor")

        bx = pd.to_numeric(pd.Series([row.get("ball_x")]), errors="coerce").iloc[0]
        by = pd.to_numeric(pd.Series([row.get("ball_y")]), errors="coerce").iloc[0]
        if pd.notna(bx) and pd.notna(by):
            ax.scatter([bx], [by], color="black", s=25, zorder=4)

        phase = row.get("phase_label")
        minute = int(row.get("minute") or 0)
        second = int(row.get("second") or 0)
        ax.set_title(f"{minute:02d}:{second:02d} · {phase}", fontsize=10)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", label="Attacking teammates", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", label="Defenders", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ffbf00", markeredgecolor="black", label="Actor", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black", label="Ball", markersize=6),
    ]
    axs[0].legend(handles=handles, loc="upper center", bbox_to_anchor=(1.65, -0.1), ncol=4, frameon=False, fontsize=8)


def plot_sample_possession(possession_row: pd.Series, events_360: pd.DataFrame, output_path: Path, title_prefix: str) -> None:
    events_df = possession_events(events_360, possession_row)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1.0])
    ax_pitch = fig.add_subplot(gs[0, 0:2])
    ax_timeline = fig.add_subplot(gs[1, 0:2])
    sub_gs = gs[:, 2].subgridspec(3, 1)
    freeze_axes = [fig.add_subplot(sub_gs[i, 0]) for i in range(3)]

    summary = (
        f"{title_prefix} · match {int(possession_row['match_id'])} · events={int(possession_row['event_count'])} · "
        f"duration={int(possession_row['duration'])}s · phases={int(possession_row['phase_unique_count'])} · "
        f"shot_in_10s={int(possession_row['has_shot_in_10s'])}"
    )
    fig.suptitle(summary, fontsize=14, weight="bold")

    plot_trajectory(ax_pitch, events_df, "Ball path colored by defensive phase")
    plot_timeline(ax_timeline, events_df, "Phase strip + visible defender count")
    plot_freeze_frames(freeze_axes, events_df)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    DATA_VALIDATION.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    possessions, events_360 = load_data()

    plot_transition_heatmap(
        possessions,
        DATA_VALIDATION / "possession_phase_transition_heatmap.png",
    )
    plot_phase_mix_by_outcome(
        possessions,
        DATA_VALIDATION / "possession_phase_mix_by_outcome.png",
    )

    sample_shot = choose_sample_possession(possessions, shot_value=1)
    sample_no_shot = choose_sample_possession(possessions, shot_value=0)

    plot_sample_possession(
        sample_shot,
        events_360,
        DATA_VALIDATION / "sample_shot_possession_sequence.png",
        title_prefix="Sample shot-ending possession",
    )
    plot_sample_possession(
        sample_no_shot,
        events_360,
        DATA_VALIDATION / "sample_no_shot_possession_sequence.png",
        title_prefix="Sample non-shot possession",
    )

    print("Created visualization files:")
    for path in [
        DATA_VALIDATION / "possession_phase_transition_heatmap.png",
        DATA_VALIDATION / "possession_phase_mix_by_outcome.png",
        DATA_VALIDATION / "sample_shot_possession_sequence.png",
        DATA_VALIDATION / "sample_no_shot_possession_sequence.png",
    ]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()

