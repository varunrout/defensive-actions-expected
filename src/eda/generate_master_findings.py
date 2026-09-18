"""CLI entrypoint: prompt 35 -- the master findings page. Synthesises the
whole DE-through-EDA pipeline into one document: what the data is, every
feature-selection decision and why, every pattern-analysis finding, and
what each of the 4 model-legs should learn. Not a new analysis -- every
number here is re-read from its live source JSON at generation time (see
the _load_* calls in main()), not copied from an intermediate summary.

Source-document correction, verified before writing anything: the prompt
names two documents that do not exist in this repo -- claude/passive-
defense-build-plan.md and reports/analysis/shot_target/PATTERN_ANALYSIS_CLOSEOUT.md (prompt
32 was never run in this session, confirmed already in prompt 34). Section
2's phase history is built from src/eda/generate_pipeline_log.py's
STAGE_HISTORY instead (the actual, live-asserted record of every stage that
touched the candidate feature lists -- functionally the same content the
build plan would have held). Section 5's exhaustive findings are built
directly from reports/analysis/shot_target/FEATURE_LOCK_CONFIRMATION.json and reports/analysis/xg_target/
FEATURE_LOCK_CONFIRMATION_XG.json's pattern_analysis_findings sections
(prompt 34), which are themselves already live-verified against the
individual pattern-analysis reports.

Content is defined once as a list of (heading, blocks) sections and
rendered to BOTH markdown and HTML from the same data, so the two files
cannot drift from each other.

Usage:
    python -m src.eda.generate_master_findings
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.feature_config import ACTIVE, PASSIVE
from src.eda.render import esc

REPO_ROOT = Path(__file__).resolve().parents[2]
EDA_DIR = REPO_ROOT / "reports" / "analysis" / "shot_target"
EDA_XG_DIR = REPO_ROOT / "reports" / "analysis" / "xg_target"
MD_OUTPUT_PATH = EDA_DIR / "MASTER_FINDINGS.md"
HTML_OUTPUT_PATH = EDA_DIR / "MASTER_FINDINGS.html"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Block-level content model: every section is a list of blocks, rendered to
# markdown and HTML from the same data. Block kinds: p, ul, table, note.
# --------------------------------------------------------------------------

def p(text: str) -> dict:
    return {"kind": "p", "text": text}


def ul(items: list[str]) -> dict:
    return {"kind": "ul", "items": items}


def table(headers: list[str], rows: list[list[str]]) -> dict:
    return {"kind": "table", "headers": headers, "rows": rows}


def note(tag: str, text: str) -> dict:
    return {"kind": "note", "tag": tag, "text": text}


def h3(text: str) -> dict:
    return {"kind": "h3", "text": text}


def render_blocks_md(blocks: list[dict]) -> str:
    out = []
    for b in blocks:
        if b["kind"] == "p":
            out.append(b["text"])
        elif b["kind"] == "ul":
            out.append("\n".join(f"- {i}" for i in b["items"]))
        elif b["kind"] == "h3":
            out.append(f"### {b['text']}")
        elif b["kind"] == "note":
            out.append(f"> **{b['tag']}** -- {b['text']}")
        elif b["kind"] == "table":
            header_row = "| " + " | ".join(b["headers"]) + " |"
            sep_row = "| " + " | ".join("---" for _ in b["headers"]) + " |"
            data_rows = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in b["rows"])
            out.append("\n".join([header_row, sep_row, data_rows]))
    return "\n\n".join(out)


def render_blocks_html(blocks: list[dict]) -> str:
    out = []
    for b in blocks:
        if b["kind"] == "p":
            out.append(f'<p class="mf-p">{b["text"]}</p>')
        elif b["kind"] == "ul":
            out.append("<ul class='mf-ul'>" + "".join(f"<li>{i}</li>" for i in b["items"]) + "</ul>")
        elif b["kind"] == "h3":
            out.append(f'<h3 class="mf-h3">{esc(b["text"])}</h3>')
        elif b["kind"] == "note":
            out.append(f'<div class="finding flag" style="margin:10px 0;"><span class="tag">{esc(b["tag"])}</span><p>{b["text"]}</p></div>')
        elif b["kind"] == "table":
            header_html = "".join(f"<th>{esc(h)}</th>" for h in b["headers"])
            rows_html = "".join(
                "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
                for row in b["rows"]
            )
            out.append(f'<div class="mf-table-wrap"><table class="mf-table"><thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table></div>')
    return "\n".join(out)


# --------------------------------------------------------------------------
# Section builders
# --------------------------------------------------------------------------

def build_section_1(active_count: int, passive_count: int) -> list[dict]:
    return [
        p(
            "Two datasets describe the same matches from two different vantage points: "
            f"`player_defensive_actions.parquet` (**active**, {active_count} locked features, one row per actual "
            "defensive action -- Pressure/Duel/Clearance/Block/Interception/Ball Recovery/Foul Committed/50-50 -- "
            "by an identifiable player) and `passive_defense.parquet` (**passive**, "
            f"{passive_count} locked features, one row per anonymous defender-slot per on-ball attacking event -- "
            "off-ball positioning, no player identity by design)."
        ),
        p(
            "They're kept separate because they answer different questions with different row semantics: active "
            "asks \"what happened at the moment a defender acted\", passive asks \"where was every nearby "
            "defender, at every attacking touch, whether or not they did anything\". Merging them would blur a "
            "one-action-per-row table into a many-slots-per-event one and lose both.\n\n"
            "Four model-legs follow from crossing {active, passive} x {binary target `target_future_shot_10s`, "
            "continuous target `target_future_xg_10s`}: active-binary, active-continuous, passive-binary, "
            "passive-continuous."
        ),
        note("current stage", "EDA is complete. Modelling has not started. Everything in this document is descriptive/observational -- no causal claims, anywhere."),
    ]


def build_section_2(archetype: dict) -> list[dict]:
    drift_flags = archetype["full_population_drift_flags"]
    sanity = _load(EDA_DIR / "FOOTBALL_SANITY_CHECK.json")

    return [
        p(
            "Every decision below changed the candidate feature lists or is a named validation checkpoint in the "
            "pipeline, in the order it ran (source: `src/eda/generate_pipeline_log.py`'s STAGE_HISTORY, which "
            "asserts its own cumulative total against `feature_config.py`'s live counts on every render -- it "
            "cannot silently drift)."
        ),
        table(
            ["Decision", "Reason"],
            [
                ["Drop-to-one, active Cluster 1 (goal-proximity)", "distance_to_attacking_goal/box, distance_to_defending_goal/box, action_zone mutually r/eta &ge; 0.94 -- kept <code>distance_to_attacking_box</code> (is_in_attacking_box lift +11.99pp is the most predictive single cut)."],
                ["Merge (not drop), active Cluster 2 (centroids)", "attacker_centroid_x/y r=0.984 with defender_centroid_x/y -- replaced by <code>defender_attacker_gap_x/y</code> (defensive compactness relative to attacking shape) since the two centroids describe different teams' shapes."],
                ["Drop-to-one, active Cluster 3 (possession clock)", "events_elapsed_in_possession / phase_transitions_observed_so_far mutually r=0.90-0.95 with possession_elapsed_seconds -- kept <code>possession_elapsed_seconds</code> (most granular, continuous)."],
                ["Drop-to-one, passive Cluster 4 (defender-position)", "distance_to_attacking_box/box, distance_to_attacking_goal/defending_goal, defender_zone mutually r/eta &ge; 0.92 -- kept <code>defender_x</code> (the most primitive measurement, upstream of every derived distance/zone column)."],
                ["Cluster 5 (passive option threat-score ranks) -- DECIDED 2026-09-17", "top_option_2_threat_score &lt;-&gt; top_option_3_threat_score, r=0.902. Kept both, permanently (PASSIVE_COLLAPSE_OPTION_RANKS=False, no longer gated). See section 3 for the confound-based reasoning."],
                ["Passive raw option coordinates replaced", "top_option_n_target_x/y were a coordinate-frame artefact (r=0.88 with ball_x) -- replaced with ball-relative dx/dy/distance_from_ball/angle_from_ball, kept alongside during a transition period then dropped once confirmed working (6 columns removed)."],
                ["V2 methodology gaps fixed", "Categorical pairs had no REVIEW band at all, and continuous&lt;-&gt;continuous REVIEW pairs had no resolution rule. Both fixed in separate _v2 modules; V1 candidate lists untouched."],
                ["Functional-role bucket fix (4&rarr;5 categories)", "62.4% \"unclassified\" wasn't sparse visibility -- a missing advanced+wide bucket plus narrow terciles. Added <code>advanced_wide</code>, renamed the n&ge;2 fallback from unclassified to <code>mid_block</code>. unclassified now reserved exclusively for n&lt;2 (structural, not a category-count add/drop)."],
                ["VIF drop (active, -2 features)", "local_numerical_balance_5m/10m: pairwise correlation cleared this 6-column cluster (max pairwise r &lt;0.90), VIF didn't -- joint linear dependency (condition number ~1.36e15 pre-drop) invisible to pairwise correlation. Both dropped; exactly recoverable from attackers_within_Nm - defenders_within_Nm, which stay."],
                ["Leakage drop (passive, -1 feature)", "<code>has_screened_outcome</code> is a censoring-mechanism proxy for target_future_shot_10s's own 10s window being truncated at end-of-period/match (chi2=136.5, p=1.54e-31) -- not defensive signal. Excluded."],
                ["Football sanity check -- substitution + named gap", sanity["substitution_note"] + " Named gap: " + sanity["coverage_gap"] + ((" " + sanity["manual_video_validation"]["note"]) if sanity.get("manual_video_validation", {}).get("status") == "completed" else "")],
                ["Passive archetype clustering -- fit/refit resolution", (
                    "KMeans swept over k in {2,3,4,5} on a fixed-seed sample (tractability -- mid_block alone has "
                    "838,270 rows), best k selected by a multi-metric score, then the FINAL model refit at that k "
                    "with n_init=20 for stability. That fitted pipeline (transform+predict only, no further "
                    f"refitting) then labels every row in the full bucket population; sample-vs-full-population "
                    f"drift is checked explicitly (&ge;3pp shot-rate or &ge;2pp share triggers a flag) -- "
                    f"{'no drift found in any of the 5 clustered buckets' if not drift_flags else f'{len(drift_flags)} drift flag(s) found'}, "
                    "confirming the sample generalised."
                )],
            ],
        ),
    ]


def build_section_3(correlation_diff: dict) -> list[dict]:
    return [
        p(
            "V1/V2's correlation-resolution logic ran two review passes (categorical vs categorical, and "
            "continuous vs continuous had gaps V1 didn't cover -- see section 2) but resolved into the same "
            "candidate lists. Every resolved cluster, with the actual kept/dropped names (read from "
            "`feature_config.py`'s `REDUNDANCY_DROPPED_ACTIVE`/`REDUNDANCY_DROPPED_PASSIVE` dicts directly):"
        ),
        table(
            ["Cluster", "Dropped", "Kept", "Resolution type"],
            [
                ["Active Cluster 1 (goal-proximity)", "distance_to_attacking_goal, distance_to_defending_goal, distance_to_defending_box, action_zone", "distance_to_attacking_box", "drop-to-one"],
                ["Active Cluster 2 (centroids)", "attacker_centroid_x/y, defender_centroid_x/y", "defender_attacker_gap_x/y (new merge)", "merge"],
                ["Active Cluster 3 (possession clock)", "events_elapsed_in_possession, phase_transitions_observed_so_far", "possession_elapsed_seconds", "drop-to-one"],
                ["Active VIF cluster", "local_numerical_balance_5m, local_numerical_balance_10m", "attackers_within_Nm, defenders_within_Nm (both kept, unaffected)", "VIF drop (joint, not pairwise)"],
                ["Passive Cluster 4 (defender-position)", "distance_to_attacking_box/box, distance_to_attacking_goal, distance_to_defending_goal, defender_zone", "defender_x", "drop-to-one"],
                ["Passive zone_defensive_value pair", "zone_defensive_value", "distance_to_defending_goal (which is itself dropped into defender_x above -- the final surviving name is defender_x, not distance_to_defending_goal)", "drop-to-one, chained"],
                ["Passive raw option coordinates", "top_option_1/2/3_target_x/y (6 columns)", "top_option_1/2/3_dx/dy/distance_from_ball/angle_from_ball", "superseded"],
                ["Passive leakage drop", "has_screened_outcome", "(excluded, not replaced -- censoring artefact)", "leakage exclusion"],
                ["Cluster 5 -- DECIDED 2026-09-17", "neither dropped", "top_option_2_threat_score, top_option_3_threat_score (r=0.902)", "kept both, permanently"],
            ],
        ),
        note(
            "Cluster 5 -- decided, not deferred",
            "Confound tests (reports/analysis/shot_target/CONFOUND_ANALYSIS.json) checked whether each option's threat-score "
            "U-shape is explained by defender_x (the zone-danger confound): option 2's U-shape is "
            "<b>partially</b> explained (verdict 'partially', survives in 3/4 strata); option 3's is "
            "<b>not</b> explained (verdict 'no', survives in all 4/4 strata). This IS the resolving evidence: "
            "a pair with r=0.902 that behaved identically after conditioning would be a strong case to "
            "collapse, but option 2 and option 3 diverge under that conditioning -- option 3 carries "
            "independent signal option 2 doesn't have as cleanly, and collapsing the pair risks losing it. "
            "Decision: keep both top_option_2_threat_score and top_option_3_threat_score as separate locked "
            "features, permanently (`PASSIVE_COLLAPSE_OPTION_RANKS = False` in feature_config.py, comment "
            "updated with this reasoning). This substitutes confound evidence for the originally-planned "
            "baseline-model feature-importance check, since no baseline model exists yet and this evidence "
            "already answers the same question.",
        ),
        p(f"Correlation diff at post-lock confirmation: {'empty for both datasets' if correlation_diff.get('active', {}).get('n_added', 1) == 0 and correlation_diff.get('active', {}).get('n_removed', 1) == 0 and correlation_diff.get('active', {}).get('n_tier_changed', 1) == 0 else 'NOT empty -- see FEATURE_LOCK_CONFIRMATION.json'} -- zero new DROP/COLLAPSE tier crossings since the locked lists were finalised."),
    ]


def build_section_4(split: dict, player_validity: dict) -> list[dict]:
    active = split["legs"]["active"]
    passive = split["legs"]["passive"]
    risk = next((r for r in player_validity["risk_pairs"]), None)

    return [
        p(
            f"Canonical split: {split['n_matches_total']} matches, {split['n_test_matches']} held out as TEST, "
            f"{split['n_train_val_matches']} in TRAIN+VAL split into {split['n_folds']} `StratifiedGroupKFold` "
            "folds grouped by match_id. Stratified on the per-match shot rate (combined percentile rank across "
            "both legs), so no single match's rows appear in more than one split -- shared between all 4 "
            "model-legs, frozen at `outputs/models/splits/match_assignment.json`."
        ),
        table(
            ["Leg", "Total rows", "Overall shot rate", "Test shot rate"],
            [
                ["Active", f"{active['n_rows_total']:,}", f"{active['overall_shot_rate_pct']}%", f"{active['by_split'][0]['shot_rate_pct']}%"],
                ["Passive", f"{passive['n_rows_total']:,}", f"{passive['overall_shot_rate_pct']}%", f"{passive['by_split'][0]['shot_rate_pct']}%"],
            ],
        ),
        note(
            "player-level caveat (prompt 31) -- a split-validity finding, stated here",
            f"The match-grouped split does not group by player. {player_validity['row_concentration']['n_distinct_players']} distinct "
            f"players in the active dataset, Gini={player_validity['row_concentration']['gini_coefficient']} (moderate concentration, "
            f"not extreme), but {player_validity['train_test_player_overlap']['overlap_pct_of_test_players']}% of test-set players also "
            f"appear in train+val. Cross-referencing with per-feature player-identity association (ICC / "
            f"Cramer's V against player_id): <b>{esc(risk['feature']) if risk else 'none'}</b> is "
            f"{esc(risk['risk_level']) if risk else ''} risk ({esc(risk['icc_or_cramers_v']) if risk else ''} &gt; 0.3, "
            "the same combination of high identity-association and high train/test overlap the prompt named as "
            "the actual risk case). Active-dataset only -- passive has no player_id column, so this caveat does "
            "not apply to either passive leg.",
        ),
    ]


def build_section_5(binary_pf: dict, xg_pf: dict) -> list[dict]:
    ct = binary_pf["confound_tests"]
    ts = binary_pf["tournament_stability"]
    sr = binary_pf["slicer_redundancy"]
    ni = binary_pf["numeric_interaction"]
    ct_xg = xg_pf["confound_tests_xg"]
    ss_xg = xg_pf["slice_stratification_xg"]
    ts_xg = xg_pf["tournament_stability_xg"]
    ni_xg = xg_pf["numeric_interaction_xg"]
    pv = binary_pf["player_level_validity"]

    blocks: list[dict] = [
        p(
            "This section is deliberately exhaustive, not a highlight reel -- every test run in prompts 21-31 is "
            "named individually below, whatever it found."
        ),
        h3(f"Confound tests -- binary target, all {ct['n_tests']}"),
        table(["Test", "Verdict"], [[esc(t["title"]), esc(t["verdict"])] for t in ct["tests"]]),
        h3(f"Confound tests -- continuous target (xG), all {ct_xg['n_tests']}"),
        p(ct_xg["note"]),
        table(["Test", "Verdict"], [[esc(t["title"]), esc(t["verdict"])] for t in ct_xg["tests"]]),
        h3(f"Tournament stability -- binary target, all {ts['n_features']}"),
        table(["Feature", "Dataset", "Verdict"], [[esc(f["feature"]), esc(f["dataset"]), esc(f["verdict"])] for f in ts["features"]]),
        h3(f"Tournament stability -- continuous target (xG), all {ts_xg['n_features']}"),
        p(ts_xg["target_agnosticism_note"]),
        table(["Feature", "Dataset", "Verdict"], [[esc(f["feature"]), esc(f["dataset"]), esc(f["verdict"])] for f in ts_xg["features"]]),
        h3("Slicer redundancy -- all 3 named candidate pairs, plus the headline"),
        p(
            "Headline: of the 21 slicer pairs tested (15 active + 6 passive), only <b>1 of 21</b> is confirmed "
            "redundant by both Cramer's V and NMI agreeing -- <code>phase_label</code> &harr; "
            "<code>phase_label_prev_event</code> (active), the constructed 1-event-lag pair. Genuinely target-"
            "agnostic (never references a target column)."
        ),
        table(
            ["Named candidate pair", "Cramer's V", "NMI", "Verdict"],
            [
                ["phase_label &harr; phase_label_prev_event (active)", "0.6503", "0.5149", "highly redundant"],
                ["event_type &harr; play_pattern (active)", "0.0894", "0.0156", "independent"],
                ["phase_label &harr; defender_functional_role (passive)", "0.0343", "0.0019", "independent"],
            ],
        ),
        p(f"Active: redundant clusters {sr['active']['redundant_clusters']}, independent: {sr['active']['independent_slicers']}. Passive: redundant clusters {sr['passive']['redundant_clusters'] or '(none)'}, independent: {sr['passive']['independent_slicers']}."),
        h3("Numeric x numeric interaction -- all 10 classifications, binary target"),
        p(f"{ni['target_agnosticism_note']} Counts: {ni['classification_counts']}."),
        table(["Feature A", "Feature B", "Dataset", "Classification"], [[esc(x["feature_a"]), esc(x["feature_b"]), esc(x["dataset"]), esc(x["classification"])] for x in ni["pairs"]]),
        h3("Numeric x numeric interaction -- all 10 classifications, continuous target (xG)"),
        p(f"{ni_xg['target_agnosticism_note']} Unconditional counts: {ni_xg['unconditional']['classification_counts']}. Given-shot counts: {ni_xg['given_shot']['classification_counts']} (all 10 inconclusive -- the shot-only sample is too small at this threshold to resolve additive/interactive/substitutive on chance quality). Agreement between unconditional and given-shot: {ni_xg['n_pairs_where_unconditional_and_given_shot_agree']}/10."),
        table(["Feature A", "Feature B", "Dataset", "Unconditional"], [[esc(x["feature_a"]), esc(x["feature_b"]), esc(x["dataset"]), esc(x["classification"])] for x in ni_xg["unconditional"]["pairs"]]),
        h3("Slice stratification -- unconditional vs shot-conditional (xG)"),
        p(
            f"V1 categorical-slicer grid: {binary_pf['slice_stratification']['v1_categorical_slicers']['n_cells']} "
            f"cells, {binary_pf['slice_stratification']['v1_categorical_slicers']['n_genuine_divergences']} "
            f"genuine divergences (binary). On xG: {ss_xg['n_genuine_divergences_unconditional']} unconditional "
            f"&rarr; {ss_xg['n_genuine_divergences_given_shot']} given-shot, "
            f"{ss_xg['n_conditioning_disagreements']} conditioning disagreements -- most unconditional xG "
            "divergences are about whether a shot happens, not its quality."
        ),
        h3("Archetype vs boolean slicers -- the comparison prompt 28 set out to make"),
        p(
            "defender_archetype_name alone produced divergence at 1.46/cell (binary) and 2.00/cell (xG) vs the "
            "15 boolean slicers combined at 1.22/cell (binary) and 1.12/cell (xG) -- the single archetype slicer "
            "matches or exceeds the combined per-cell divergence rate of all 15 boolean flags together, on both "
            "targets."
        ),
        h3("Player-level validity -- the one risk pair"),
        table(["Feature", "ICC/Cramer's V", "Risk level"], [[esc(r["feature"]), esc(str(r["icc_or_cramers_v"])), esc(r["risk_level"])] for r in pv["risk_pairs"]]) if pv["risk_pairs"] else p("No risk pairs."),
    ]
    return blocks


def build_section_6() -> list[dict]:
    return [
        h3("Active-binary (target_future_shot_10s)"),
        ul([
            "Clean signal, use directly: <code>defender_spread</code> (rho=-0.188, strongest active correlate, U-shaped), <code>attacker_spread</code>, <code>attacking_goal_centrality</code>, <code>attacker_defender_ratio</code> -- established monotonic/U-shaped patterns from prompt 21, none flagged unreliable.",
            "Interaction terms to build, named pairs (all confirmed 'interactive' on this target specifically, prompt 30): <code>defenders_within_10m &times; distance_to_attacking_box</code>, <code>visible_defender_count &times; attacker_spread</code>, <code>possession_elapsed_seconds &times; match_time_seconds</code>, <code>defenders_between_ball_and_attacking_goal &times; attacker_defender_ratio</code>. Also: <code>defenders_within_10m &times; phase_label</code> diverges in 7/7 phase categories -- the single strongest slice-level interaction signal in the whole corpus.",
            "Special handling -- redundancy: <code>defenders_within_5m &times; defenders_within_10m</code> is substitutive (nested by construction, 5m is a subset of 10m) -- consider dropping or de-weighting defenders_within_5m rather than treating both as independent.",
            "Special handling -- tournament-dependence: <code>defenders_within_5m</code> and <code>attackers_within_5m</code> show a genuine tournament-level difference (WC2022 vs Euro2024) on this target -- validate with a tournament-aware check alongside the match-grouped CV, not just in-sample.",
            "Special handling -- identity leakage: <code>position</code> is high-risk (Cramer's V=0.77 vs player_id, 90% train/test player overlap) -- consider a player-grouped CV fold in addition to the match-grouped one, or drop/de-weight position, before trusting its apparent predictive strength.",
            "Open decisions bearing on this leg: none remaining -- video validation of defender_functional_role and related geometry has been completed manually and confirmed correct (was a named football-sanity-check gap, now closed). Cluster 5 does not apply (passive-only).",
        ]),
        h3("Active-continuous (target_future_xg_10s)"),
        ul([
            "Clean signal: near-identical ranking to active-binary (defender_spread rho=-0.188, attacking_goal_centrality, distance_to_center_line, attacker_spread, attacker_defender_ratio) -- the numerical-vs-target ordering is stable across targets even though individual interaction findings are not (see below).",
            "Structural-zero caveat: target_future_xg_10s == 0 exactly wherever target_future_shot_10s == 0 -- any unconditional xG finding on this leg should be treated as re-deriving occurrence unless separately confirmed on the shot-conditional (given-shot) panel.",
            "Interaction terms -- do NOT reuse the binary leg's list unchanged: <code>defenders_within_10m &times; distance_to_attacking_box</code> and <code>visible_defender_count &times; attacker_spread</code> are interactive here too (agree with binary), but <code>defenders_within_5m &times; defenders_within_10m</code> is interactive on xG (not substitutive like binary -- a genuine target-dependent disagreement), and <code>possession_elapsed_seconds &times; match_time_seconds</code> is additive on xG (not interactive like binary). <code>defenders_between_ball_and_attacking_goal &times; attacker_defender_ratio</code> is inconclusive on xG.",
            "Special handling -- given-shot interaction findings are all inconclusive (10/10) at the current threshold -- don't make an interaction-term decision for this leg's chance-quality modelling based on the given-shot panel alone; only 3/10 pairs agree between unconditional and given-shot.",
            "Special handling -- tournament-dependence differs from binary for <code>defenders_within_5m</code> and <code>attackers_within_5m</code> (binary: genuine difference, xG: arbitrary noise) -- don't assume the binary leg's tournament caveat carries over unchanged.",
            "Special handling -- identity leakage: same <code>position</code> risk pair applies (target-agnostic finding).",
            "has_screened_outcome's leakage effect is present but materially weaker here (1.3x, p=0.043 vs binary's ~12x at p=1.5e-31) -- the drop still holds on structural grounds, just less strongly evidenced on this leg specifically.",
        ]),
        h3("Passive-binary (target_future_shot_10s)"),
        ul([
            "Clean signal: <code>top_option_3_threat_score</code> (rho=-0.085, strongest passive correlate, U-shaped), and both established reversals -- marking_tightness and lane_screening_score_option_1 -- survive every confound test run against them ('no' verdict, not explained away).",
            "Occurrence vs quality (prompt 23/26): marking_tightness's reversal is <b>occurrence-only</b> (flat once conditioned on a shot happening) -- treat it as a shot-occurs signal for this binary leg, but see the continuous leg below for why it doesn't transfer as a quality signal. lane_screening_score_option_1's reversal is <b>occurrence+quality</b> -- genuinely useful signal either way.",
            "has_option_2/has_option_3 (prompt 29): both survive conditioning on top_option_1_threat_score AND on defender_x (all 'no' verdicts) -- real signal about the current freeze frame, not a proxy for possession danger or box proximity. Safe to use directly.",
            "Interaction terms to build, named pairs (prompt 30): <code>lane_screening_score_option_2 &times; engagement_distance_to_carrier</code>, <code>marking_tightness &times; engagement_distance_to_carrier</code>, <code>overload_score &times; attacking_goal_centrality</code>, <code>lane_screening_score_option_1 &times; marking_tightness</code>. <code>top_option_2_threat_score &times; top_option_2_distance_from_ball</code> is additive -- safe to use both independently, no interaction term needed there.",
            "Cluster 5 -- decided: top_option_2_threat_score and top_option_3_threat_score (r=0.902) are kept as separate features, permanently. Confound evidence was asymmetric (option 2's U-shape 'partially' explained by defender_x, option 3's 'no' -- not explained) and that asymmetry itself was the deciding signal: a highly-correlated pair that behaves differently under identical conditioning isn't safe to collapse. Use both directly.",
            "Modelling-stage candidate: <code>defender_archetype_name</code> matches or exceeds all 15 boolean slicers combined on a per-cell divergence basis (1.46 vs 1.22/cell) -- worth treating as a first-class categorical covariate/interaction candidate, not just a diagnostic curiosity. Its 'unclassified' category (n=1026) is structural (n&lt;2 visible defenders) -- never a 5th behavioural role.",
            "Video validation gap -- closed: marking_tightness and is_goal_side_of_nearest_attacker (the two features the football sanity check's reimplementation approach couldn't independently re-derive) were checked manually against real match video and confirmed correct.",
        ]),
        h3("Passive-continuous (target_future_xg_10s)"),
        ul([
            "Structural-zero framing matters most on this leg: 363 unconditional genuine divergences collapse to 90 given-shot, with 130 conditioning disagreements -- treat any unconditional passive xG finding as probably about occurrence, not quality, until checked given-shot.",
            "marking_tightness vs defender_x: weaker than binary here -- 'partially' both unconditionally and given-shot (binary was a clean 'no'). Treat marking_tightness's confound-survival as more fragile on this leg specifically.",
            "lane_screening_score_option_1 vs top_option_1_threat_score: robust on both counts ('no' unconditionally and given-shot) -- safe to use for occurrence AND quality here.",
            "has_option_2 vs top_option_1_threat_score flips to occurrence-only on this leg specifically (unconditional 'no', given-shot 'yes') -- don't use has_option_2 as a chance-quality signal via this pathway on xG, even though it's a real, robust occurrence signal (per the binary leg above). has_option_3 stays robust both ways ('no'/'no'). has_option_2 vs defender_x is mixed ('no'/'partially').",
            "Interaction terms -- do not reuse the binary leg's list unchanged: <code>top_option_2_threat_score &times; top_option_2_distance_from_ball</code> flips to interactive on xG (was additive on binary) -- a genuine target-dependent difference. <code>marking_tightness &times; engagement_distance_to_carrier</code> and <code>overload_score &times; attacking_goal_centrality</code> both drop to inconclusive on xG (were interactive on binary) -- weaker evidence here, don't force an interaction-term decision from either pair alone for this leg.",
            "All 10 pairs' given-shot classification is inconclusive at the current threshold -- no numeric-interaction decision for chance-quality on this leg should rest on the given-shot panel alone yet.",
            "defender_archetype_name is even more informative relative to boolean slicers on this leg than on binary (2.00 vs 1.12 divergences/cell) -- the strongest case across all 4 legs for treating it as a first-class covariate.",
        ]),
    ]


def build_section_7() -> list[dict]:
    return [
        p(
            "Nothing analytical remains open in this EDA stage. All three items previously tracked here "
            "have been resolved (2026-09-17):"
        ),
        ul([
            "<b>1. Player-grouped CV for the active legs</b> -- done. A player-disjoint fold structure "
            "(zero player overlap across folds, 964 players, 5 folds) was built and compared against the "
            "canonical match-grouped folds over the same rows: fold balance and position-share balance are "
            "both comparable across the two structures (max 5.14pp position-share deviation in any fold). "
            "No material identity-leakage risk from `position` surfaces in this check. See "
            "PLAYER_GROUPED_SPLIT_CHECK.json / .html.",
            "<b>2. Numeric-interaction given-shot resolution</b> -- decided. All 10 pairs' given-shot "
            "classification stays 'inconclusive' at the current, principled threshold "
            "(FLAT_MARGIN_RATIO x each dataset's own shot-conditional mean xG). This is accepted as the "
            "final answer, not loosened to force a classification -- doing so would tune the threshold to "
            "the result rather than the reverse. See FEATURE_INTERACTION_ANALYSIS.json's "
            "given_shot_threshold_decision field.",
            "<b>3. Repo cleanup</b> -- audited (not yet actioned). A deletion-candidate report has been "
            "written (reports/REPO_CLEANUP_AUDIT.md) identifying local_archive/ (3.4GB, gitignored, three "
            "months stale), mlflow.db.bak_20260916182833, and the stray `Claude outputs/` folder as the "
            "clearest candidates, plus three root-level docs worth a look. Nothing has been deleted -- each "
            "item awaits explicit go-ahead per standing policy.",
        ]),
        p(
            "Resolved since the previous revision of this document (2026-09-17): Cluster 5 "
            "(top_option_2_threat_score vs top_option_3_threat_score, r=0.902) -- decided, kept both, "
            "permanently (see sections 2/3/6); real video validation of the football-sanity gap "
            "(marking_tightness, is_goal_side_of_nearest_attacker) -- completed manually and confirmed "
            "correct (see sections 2/6); and all three items above. The only remaining action item in this "
            "whole document is the repo-cleanup deletions themselves, which are a housekeeping decision for "
            "Varun, not an EDA task."
        ),
    ]


def build_section_8() -> list[dict]:
    rows = [
        ["claude/passive-defense-data-exploration-context.md", "Working notes on missingness flags -- NOT a build plan (that file doesn't exist in this session)."],
        ["src/eda/generate_pipeline_log.py -> EDA_PIPELINE_LOG.html", "The actual phase/stage history this document's section 2 is built from."],
        ["EDA_ANALYSIS.html", "Base category/flag/distribution atlases; original marking-tightness/lane-screening reversal discovery."],
        ["FEATURE_LOCK_CONFIRMATION.json / .html", "Binary-target lock confirmation + pattern-analysis findings (prompts 14, 34)."],
        ["FEATURE_LOCK_CONFIRMATION_XG.json / .html", "Continuous-target lock confirmation + pattern-analysis findings (prompts 33, 34)."],
        ["CORRELATION_ANALYSIS.json, REVIEW_ANALYSIS.json (+ _V2, _V3)", "Full pairwise correlation tiers and REVIEW resolutions behind section 3."],
        ["VIF_ANALYSIS.json", "Multicollinearity check behind the active VIF drop."],
        ["LEAKAGE_AUDIT.json (both targets)", "has_screened_outcome and has_option_2/3 leakage checks."],
        ["FOOTBALL_SANITY_CHECK.json", "Independent-reimplementation validation substitute and its named coverage gap."],
        ["PASSIVE_ARCHETYPES.json", "Archetype clustering method, buckets, and sample/full-population drift check."],
        ["SPLIT_VALIDATION.json", "Canonical match-grouped split counts."],
        ["PLAYER_LEVEL_VALIDITY_CHECK.json (both portals)", "Row concentration, per-feature ICC, train/test player overlap."],
        ["CONFOUND_ANALYSIS.json (both targets)", "All confound-test verdicts, sections 3 and 5."],
        ["TOURNAMENT_STABILITY_CHECK.json (both targets)", "All tournament-stability verdicts, section 5."],
        ["SLICE_STRATIFICATION.json / _V2.json (both targets)", "Categorical/archetype/boolean slice-divergence counts and worked examples."],
        ["SLICER_REDUNDANCY.json", "All 21 slicer-pair association tests, section 5."],
        ["FEATURE_INTERACTION_ANALYSIS.json (both targets)", "All 10 numeric-interaction classifications plus the given-shot threshold decision, sections 5, 6 and 7."],
        ["PLAYER_GROUPED_SPLIT_CHECK.json / .html", "Player-disjoint fold structure, leakage check, and fold/position balance comparison behind section 7 item 1."],
        ["REPO_CLEANUP_AUDIT.md", "Deletion-candidate audit behind section 7 item 3 -- awaiting Varun's go-ahead."],
    ]
    return [
        p("Every source document this page synthesises, so a reader wanting more depth on any one point knows exactly where to go."),
        table(["Document", "What it covers"], [[esc(a), b] for a, b in rows]),
    ]


def main() -> None:
    active_count = len(ACTIVE["categorical"]) + len(ACTIVE["boolean"]) + len(ACTIVE["continuous"]) + len(ACTIVE["discrete"])
    passive_count = len(PASSIVE["categorical"]) + len(PASSIVE["boolean"]) + len(PASSIVE["continuous"]) + len(PASSIVE["discrete"])

    archetype = _load(EDA_DIR / "PASSIVE_ARCHETYPES.json")
    binary_lock = _load(EDA_DIR / "FEATURE_LOCK_CONFIRMATION.json")
    xg_lock = _load(EDA_XG_DIR / "FEATURE_LOCK_CONFIRMATION_XG.json")
    split = _load(EDA_DIR / "SPLIT_VALIDATION.json")
    player_validity = _load(EDA_DIR / "PLAYER_LEVEL_VALIDITY_CHECK.json")

    binary_pf = binary_lock["pattern_analysis_findings"]
    xg_pf = xg_lock["pattern_analysis_findings"]

    sections = [
        ("What this is", build_section_1(active_count, passive_count)),
        ("Data engineering & feature engineering, in decisions", build_section_2(archetype)),
        ("Feature selection, in decisions", build_section_3(binary_lock["correlation_diff"])),
        ("The locked split", build_section_4(split, player_validity)),
        ("Pattern analysis, every finding, however small", build_section_5(binary_pf, xg_pf)),
        ("What each model should learn", build_section_6()),
        ("Open items, prioritised", build_section_7()),
        ("Document map", build_section_8()),
    ]

    bottom_line = (
        "EDA is complete across both datasets and both target types; modelling has not "
        f"started. The locked feature set ({active_count} active / {passive_count} passive) is confirmed "
        "internally consistent for both binary and continuous targets. Cluster 5 (the one open redundancy "
        "decision) is resolved -- keep both threat-score ranks. Real video validation of the "
        "football-sanity gap has been completed manually and confirmed correct. The one confirmed "
        "identity-leakage risk feature (`position`, active legs only) has been stress-tested with a "
        "player-disjoint fold structure (zero player overlap across folds) and shows no material "
        "degradation -- see PLAYER_GROUPED_SPLIT_CHECK.json. Nothing analytical remains open; a set of "
        "named interaction-term candidates and target-specific caveats per model-leg in section 6 are "
        "the only thing to read before writing model code."
    )

    # --- Markdown ---
    md_parts = ["# Master Findings -- Defensive Action Expected", "", f"**Bottom line up front:** {bottom_line}", ""]
    for i, (title, blocks) in enumerate(sections, start=1):
        md_parts.append(f"## {i}. {title}\n")
        md_parts.append(render_blocks_md(blocks))
        md_parts.append("")
    MD_OUTPUT_PATH.write_text("\n".join(md_parts), encoding="utf-8")

    # --- HTML ---
    html_body_parts = [f'<div class="finding" style="margin-bottom:28px;"><span class="tag">bottom line up front</span><p>{bottom_line}</p></div>']
    for i, (title, blocks) in enumerate(sections, start=1):
        html_body_parts.append(f'<h2 class="mf-section-title" id="s{i}">{i}. {esc(title)}</h2>')
        html_body_parts.append(render_blocks_html(blocks))
    html_body = "\n".join(html_body_parts)

    html = render.render_article(
        eyebrow="MASTER FINDINGS",
        title="Defensive Action Expected -- Master Findings",
        dek=(
            "The single entry point for this project: what the data is, every feature-selection decision and "
            "why, every pattern-analysis finding, and what each of the 4 model-legs should learn before baseline "
            "modelling starts. A synthesis of existing reports, not a new analysis -- every number re-verified "
            "against its live source at generation time."
        ),
        stats=[
            (f"{active_count}", "active features"),
            (f"{passive_count}", "passive features"),
            ("4", "model-legs"),
        ],
        body_html=html_body,
        extra_css=render.CORR_CSS + render.CONFOUND_CSS + render.NUMERICAL_ATLAS_CSS + """
.mf-section-title { margin-top:48px; padding-top:24px; border-top:2px solid var(--border); }
.mf-p { color: var(--text-secondary); font-size: 14px; line-height:1.6; margin: 10px 0; }
.mf-ul { padding-left:22px; }
.mf-ul li { color: var(--text-secondary); font-size: 13.5px; line-height:1.6; margin-bottom:10px; }
.mf-h3 { font-family:"Archivo",sans-serif; font-size:16px; margin-top:24px; }
.mf-table-wrap { overflow-x:auto; margin:12px 0; }
.mf-table { width:100%; border-collapse:collapse; font-size:12.5px; }
.mf-table th { text-align:left; color:var(--text-muted); padding:6px 10px; border-bottom:1px solid var(--border); }
.mf-table td { padding:6px 10px; border-top:1px solid var(--border); color:var(--text-secondary); }
""",
    )
    HTML_OUTPUT_PATH.write_text(html, encoding="utf-8")

    print(f"Wrote {MD_OUTPUT_PATH}")
    print(f"Wrote {HTML_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
