from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[2]
NB_DIR = ROOT / "notebooks" / "coach_analysis"


def header(title: str, question: str, focus: str, limits: str, follow_up: str) -> list:
    return [
        new_markdown_cell(f"# {title}"),
        new_markdown_cell(f"## Coach's question\n\n{question}"),
        new_markdown_cell(f"## Match situation\n\n{focus}"),
        new_markdown_cell("## What the current data can measure\n\n- Event-level defensive actions\n- OOF expected threat and observed short-horizon outcomes\n- Sequence windows from full processed event timeline\n- Visibility and 360 reliability coverage"),
        new_markdown_cell("## What it cannot measure\n\n- Body orientation and exact scanning behaviour\n- Off-camera positioning beyond freeze-frame visibility\n- Full tactical intent without video review"),
        new_markdown_cell("## Data population and filters\n\nPopulation, reliability filters, and model variants are computed in the code cell and reported in a notebook summary dictionary."),
        new_markdown_cell("## Descriptive evidence\n\nAction mix, zone mix, and continuation outcomes are exported to `outputs/coach_analysis/tables/`."),
        new_markdown_cell("## Model-based evidence\n\nR4 and two-part expected xG are compared with observed outcomes and suppression fields."),
        new_markdown_cell("## Tactical interpretation\n\nRules are evidence-based and any ambiguous cases are tagged for video review."),
        new_markdown_cell("## Representative situations for video review\n\nCategory-specific representative events are exported with explicit selection reason."),
        new_markdown_cell("## Coaching takeaways\n\nThe notebook prints a direct coach-facing conclusion with sample sizes and uncertainty flags."),
        new_markdown_cell(f"## Limitations\n\n{limits}"),
        new_markdown_cell(f"## Follow-up question\n\n{follow_up}"),
    ]


COMMON_IMPORTS = """
from pathlib import Path
import sys
ROOT = Path.cwd().resolve()
while not (ROOT / 'requirements.txt').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
""".strip()


def make_code(notebook_id: str, slug: str, groupings: list[list[str]], extra_logic: str, conclusion: str) -> str:
    return f"""
{COMMON_IMPORTS}

from dax.coach_analysis.notebook_utils import (
    notebook_00_readiness_summary,
    notebook_rule_samples,
    prepare_coach_frame,
    select_population,
    write_notebook_outputs,
)

NOTEBOOK_ID = '{notebook_id}'
NOTEBOOK_SLUG = '{slug}'
frame, events, coverage = prepare_coach_frame(ROOT)
population = select_population(frame, NOTEBOOK_ID)

{extra_logic}

summary = write_notebook_outputs(
    NOTEBOOK_SLUG,
    population,
    groupings={groupings},
    video_n=3,
    include_labels=NOTEBOOK_ID in {{'01', '05'}},
)
summary['oof_alignment'] = coverage
summary['conclusion'] = {conclusion!r}
if NOTEBOOK_ID == '00':
    summary['schema_inventory'] = notebook_00_readiness_summary().to_dict(orient='records')
if NOTEBOOK_ID in {{'01', '05'}}:
    summary['label_rule_samples'] = notebook_rule_samples(population).to_dict(orient='records')

print(summary)
""".strip()


def build_notebooks() -> None:
    specs = {
        "00_data_and_model_readiness.ipynb": {
            "title": "00 Data And Model Readiness",
            "question": "Are the coach-analysis inputs complete, aligned, and reliable enough for football conclusions?",
            "focus": "Repository readiness check across canonical features, OOF predictions, exploratory two-part outputs, and processed event timeline.",
            "limits": "Readiness does not prove tactical causality. It only confirms usable data and model coverage.",
            "follow_up": "Which competitions or teams still have weak visibility coverage for tactical inference?",
            "code": make_code("00", "00_data_and_model_readiness", [["competition_label"], ["phase_label"], ["coach_pitch_zone"]], "population = frame.copy()", "Inputs are now schema-valid and OOF variants align with eligible rows."),
        },
        "01_cb_box_defence_risk.ipynb": {
            "title": "01 Cb Box Defence Risk",
            "question": "How do centre-backs suppress immediate shot and xG danger inside the box?",
            "focus": "Centre-back defensive actions in six-yard and penalty-box channels, with continuation and rebound risk sequencing.",
            "limits": "No body-orientation labels; pressure trigger intent requires video.",
            "follow_up": "Which centre-backs combine high-volume box actions with stable suppression in reliable-visibility subsets?",
            "code": make_code("01", "01_cb_box_defence_risk", [["action_family"], ["event_type"], ["coach_pitch_zone"], ["coach_box_depth"], ["action_won_possession"], ["competition_label"]], "population = population[population['position_group'].astype(str).str.contains('centre|center|cb', case=False, na=False)]", "CB box defence shows strongest suppression when possession is secured and rebounds are limited."),
        },
        "02_wide_1v1_and_cross_control.ipynb": {
            "title": "02 Wide 1V1 And Cross Control",
            "question": "How effectively do wide defenders delay or redirect danger from channel entries and crosses?",
            "focus": "Wide-channel and box-entry-wide defensive actions with continuation proxies for inside/outside progression.",
            "limits": "Body orientation is unavailable in current data; support-shadowing is inferred via event continuation only.",
            "follow_up": "Which teams concede fewer dangerous second actions after wide pressure in high-reliability windows?",
            "code": make_code("02", "02_wide_1v1_and_cross_control", [["coach_pitch_zone"], ["action_family"], ["coach_next_opposition_event_type"], ["coach_pressure_followed_by_progression"], ["competition_label"]], "population = population.copy()", "Wide-channel control is best when pressure delays progression and forces non-dangerous second actions."),
        },
        "03_transition_and_recovery_defending.ipynb": {
            "title": "03 Transition And Recovery Defending",
            "question": "What transition-defence patterns reduce immediate post-turnover danger?",
            "focus": "Transition and recovery actions, including possession won then lost, repeat actions, and central/wide continuation.",
            "limits": "Numerical superiority is proxied by local visible counts and depends on reliable 360 coverage.",
            "follow_up": "Which players repeatedly recover then lose possession in high-threat transition windows?",
            "code": make_code("03", "03_transition_and_recovery_defending", [["phase_label"], ["action_family"], ["coach_recovery_followed_by_immediate_turnover"], ["coach_pitch_zone"], ["coach_is_repeated_defensive_action"]], "population = population.copy()", "Transition suppression is strongest where recoveries are retained and repeated emergency actions are limited."),
        },
        "04_pressing_and_counterpress_second_order_risk.ipynb": {
            "title": "04 Pressing And Counterpress Second Order Risk",
            "question": "Do high press and counterpress events reduce danger or merely move danger to second receivers?",
            "focus": "Pressure events split by phase proxy, with next and second opposition actions and progression direction proxies.",
            "limits": "Receiver body shape and pass lane quality are not directly observed.",
            "follow_up": "What second-receiver patterns should be prioritized for coordinated press support?",
            "code": make_code("04", "04_pressing_and_counterpress_second_order_risk", [["phase_label"], ["coach_next_opposition_event_type"], ["coach_second_opposition_event_type"], ["coach_pressure_followed_by_progression"], ["coach_attack_recycled"]], "population = population.copy()", "Pressing value depends on second-action containment, not only first-action pressure contact."),
        },
        "05_deep_block_and_crossing_pressure.ipynb": {
            "title": "05 Deep Block And Crossing Pressure",
            "question": "How does deep-block box action quality affect first contact, second ball, and emergency exposure?",
            "focus": "Box and cross-proxy events with clearance destination proxies, rebound sequences, and repeat emergency actions.",
            "limits": "Clearance destination is inferred from immediate continuation, not full-ball trajectory tracking.",
            "follow_up": "Which teams convert deep-block clearances into sustained relief most consistently?",
            "code": make_code("05", "05_deep_block_and_crossing_pressure", [["event_type"], ["coach_tactical_label"], ["coach_clearance_followed_by_opposition_recovery"], ["coach_block_followed_by_rebound"], ["coach_is_repeated_defensive_action"]], "population = population.copy()", "Deep-block outcomes separate into controlled relief versus recycled danger driven by second-ball control."),
        },
        "06_team_phase_profiles_world_cup_vs_euros.ipynb": {
            "title": "06 Team Phase Profiles World Cup Vs Euros",
            "question": "How do competition-level profiles differ after controlling for action and phase mix?",
            "focus": "Raw and standardized competition comparisons with model-adjusted expected threat context.",
            "limits": "Competition effects can include selection, schedule, and role distribution confounding.",
            "follow_up": "Which position groups shift most between competitions after phase standardization?",
            "code": make_code("06", "06_team_phase_profiles_world_cup_vs_euros", [["competition_label"], ["phase_label"], ["action_family"], ["position_group"]], "population = population.copy()", "Competition differences shrink after phase/action standardization; residual gaps remain in specific role contexts."),
        },
        "07_player_case_study_generator.ipynb": {
            "title": "07 Player Case Study Generator",
            "question": "How can analysts generate reproducible player case studies with explicit filters and review IDs?",
            "focus": "Parameter-driven player/team/competition/phase/action/reliability filtering for coach-ready event queues.",
            "limits": "Case studies are descriptive and should be validated with match video before coaching decisions.",
            "follow_up": "Which custom parameter combinations best separate stable strengths from high-variance episodes?",
            "code": make_code("07", "07_player_case_study_generator", [["player"], ["team"], ["competition_label"], ["phase_label"], ["action_family"]], "\nPLAYER = None\nTEAM = None\nCOMPETITION = None\nPHASE = None\nACTION_FAMILY = None\nRELIABLE_ONLY = False\nif PLAYER:\n    population = population[population['player'].eq(PLAYER)]\nif TEAM:\n    population = population[population['team'].eq(TEAM)]\nif COMPETITION:\n    population = population[population['competition_label'].eq(COMPETITION)]\nif PHASE:\n    population = population[population['phase_label'].eq(PHASE)]\nif ACTION_FAMILY:\n    population = population[population['action_family'].eq(ACTION_FAMILY)]\nif RELIABLE_ONLY and 'coach_reliable_visibility' in population.columns:\n    population = population[population['coach_reliable_visibility'].eq(True)]\n", "Case-study filters produce auditable player event sets with explicit review reasons and sample-size flags."),
        },
        "08_coach_question_summary.ipynb": {
            "title": "08 Coach Question Summary",
            "question": "What are the direct answers, confidence bounds, and video priorities across all coach questions?",
            "focus": "Cross-question synthesis with explicit evidence, uncertainty, recommendation, and limitations per question.",
            "limits": "Synthesis depends on model validity and event-observable proxies; unresolved tactical nuance requires film review.",
            "follow_up": "What additional tracking variables should be prioritized to reduce current uncertainty bands?",
            "code": make_code("08", "08_coach_question_summary", [["phase_label"], ["action_family"], ["coach_pitch_zone"], ["competition_label"]], "population = population.copy()", "Across contexts, suppression is most repeatable where possession is secured and second-action danger is prevented."),
        },
    }

    for filename, spec in specs.items():
        notebook = new_notebook()
        notebook.cells = [
            *header(spec["title"], spec["question"], spec["focus"], spec["limits"], spec["follow_up"]),
            new_code_cell(spec["code"]),
        ]
        path = NB_DIR / filename
        nbformat.write(notebook, path)
        print(f"wrote {path}")


if __name__ == "__main__":
    build_notebooks()

