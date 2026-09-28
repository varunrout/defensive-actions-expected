import fs from "fs";
import path from "path";

const DATA_DIR = path.join(process.cwd(), "src", "data");

function readJson<T>(rel: string): T {
  const p = path.join(DATA_DIR, rel);
  return JSON.parse(fs.readFileSync(p, "utf-8")) as T;
}

export const MATCH_IDS = ["3938643", "3857294", "3857298"] as const;
export type MatchId = (typeof MATCH_IDS)[number];

export interface LegsSummary {
  schema_version: number;
  comparability: unknown;
  groups: unknown;
  legs: Array<{
    id: string;
    group: string;
    side: string;
    plain_name: string;
    question: string;
    reference_model: string;
    model_family_plain: string;
    headline: { metric: string; value: number; random_guess_level?: number; source: string };
    supporting_metrics: Record<string, number | string>;
    baseline_to_reference: { metric: string; from: number; to: number; source: string };
    audit: { status: string; checks?: number; prompt?: number; source: string };
    one_liner: string;
  }>;
  active_vs_passive: unknown;
  cross_leg_pattern: unknown;
}

export interface MethodologySteps {
  schema_version: number;
  steps: Array<{
    id: string;
    order: number;
    stage: string;
    description: string;
    facts?: Record<string, number>;
    source: string;
  }>;
  honest_result_callouts: unknown;
}

export interface MatchExplorerEvent {
  event_id: string;
  timestamp: string;
  minute: number;
  second: number;
  period: number;
  team: string;
  player: string | null;
  phase: "active" | "passive";
  location: { x: number; y: number };
  location_match_frame: { x: number; y: number };
  on_ball_event_type: string;
  on_ball_team?: string;
  counterpress?: boolean;
  won_possession?: boolean;
  visibility_limited?: boolean;
  phase_label: string;
  phase_label_frame_consistent: boolean;
  predictions?: {
    active_binary?: { probability: number };
    active_continuous?: { expected_value: number; expected_xg_given_shot: number };
    active_xt?: { expected_delta: number; p_nonzero: number; expected_delta_given_nonzero: number };
  };
  observed?: { shot_within_10s: number; xg_within_10s: number; xt_delta: number };
  defenders?: Array<{
    defender_slot_index: number;
    location: { x: number; y: number };
    location_match_frame: { x: number; y: number };
    functional_role: string;
    predictions?: {
      passive_binary?: { probability: number };
      passive_continuous?: { expected_value: number; expected_xg_given_shot: number };
      passive_xt?: { expected_delta: number; p_nonzero: number; expected_delta_given_nonzero: number };
    };
  }>;
}

export interface MatchExplorer {
  schema_version: number;
  match_id: number;
  competition: string;
  competition_stage: string;
  teams: { home: string; away: string };
  score: { home: number; away: number };
  date: string;
  canonical_split: string;
  why_selected: string;
  reference_models: Record<string, string>;
  coordinate_frames: unknown;
  data_quality: Record<string, number>;
  events: MatchExplorerEvent[];
}

export interface MatchFeaturesEvent {
  event_id: string;
  location: { x: number; y: number };
  phase: "active" | "passive";
  features: Record<string, number | string | boolean | null>;
  defenders?: Array<{
    defender_slot_index: number;
    location: { x: number; y: number };
    features: Record<string, number | string | boolean | null>;
  }>;
}

export interface MatchFeatures {
  match_id: number;
  locked_feature_counts: Record<string, number>;
  events: MatchFeaturesEvent[];
}

export function getLegsSummary(): LegsSummary {
  return readJson<LegsSummary>("legs_summary.json");
}

export function getMethodologySteps(): MethodologySteps {
  return readJson<MethodologySteps>("methodology_steps.json");
}

export function getSelectionAudit(): unknown {
  return readJson("selection_and_frame_audit.json");
}

// --- Feature atlas (page 05) support ---------------------------------------
// Real per-feature decile-bin distributions (bin edges, n, shot_rate_pct),
// synced from reports/analysis/shot_target/*_numerical_target_atlas.json via
// scripts/sync-feature-atlas.mjs. Shape matches the source JSON exactly.

export interface FeatureAtlasBin {
  bin: string;
  n: number;
  shot_rate_pct: number;
}

export interface FeatureAtlasFeature {
  feature: string;
  status: "locked" | "dropped";
  reason: string | null;
  type: "continuous" | "discrete";
  n_rows_used: number;
  n_unique_values: number;
  binning_method: string;
  pearson_r: number;
  spearman_rho: number;
  bin_range_pp: number;
  bins: FeatureAtlasBin[];
  shape: string;
  consistency_check?: unknown;
  unreliable_note?: string | null;
}

export interface FeatureAtlas {
  dataset: "active" | "passive";
  generated_at: string;
  target: string;
  pool_construction: {
    n_locked: number;
    n_dropped: number;
    n_total: number;
    excluded_as_coordinate_duplicate: string[];
  };
  n_features_analyzed: number;
  n_features_checked_for_consistency: number;
  n_features_flagged_inconsistent: number;
  rho_threshold: number;
  range_threshold_pp: number;
  features: FeatureAtlasFeature[];
}

export function getFeatureAtlas(dataset: "active" | "passive"): FeatureAtlas {
  return readJson<FeatureAtlas>(`feature_atlas/${dataset}.json`);
}

export function getAtlasFeature(dataset: "active" | "passive", feature: string): FeatureAtlasFeature | undefined {
  return getFeatureAtlas(dataset).features.find((f) => f.feature === feature);
}

// Live counts of numerical (continuous + discrete) candidate features, computed
// straight from each dataset's atlas `type` field — never hardcoded, so the
// page 05 stat tiles can't drift out of sync with the underlying atlas again.
export function getFeatureAtlasCounts(dataset: "active" | "passive"): {
  continuous: number;
  discrete: number;
  total: number;
} {
  const { features } = getFeatureAtlas(dataset);
  const continuous = features.filter((f) => f.type === "continuous").length;
  const discrete = features.filter((f) => f.type === "discrete").length;
  return { continuous, discrete, total: continuous + discrete };
}

export function getMatchExplorer(matchId: string): MatchExplorer {
  return readJson<MatchExplorer>(`match_explorer/${matchId}.json`);
}

export function getMatchFeatures(matchId: string): MatchFeatures {
  return readJson<MatchFeatures>(`match_features/${matchId}.json`);
}

export function matchLabel(m: MatchExplorer): string {
  return `${m.teams.home} ${m.score.home}–${m.score.away} ${m.teams.away} · ${m.competition}`;
}

// --- Match Analyser (page 07) support -------------------------------------

import { ANALYSER_FEATURES, type AnalyserFeature, type AnalyserMatch, type AnalyserEvent } from "./analyserFeatures";
export { ANALYSER_FEATURES, ANALYSER_FEATURE_META, type AnalyserFeature, type AnalyserMatch, type AnalyserEvent } from "./analyserFeatures";

export function getAnalyserData(): AnalyserMatch[] {
  return MATCH_IDS.map((matchId) => {
    const explorer = getMatchExplorer(matchId);
    const features = getMatchFeatures(matchId);
    const featById = new Map(features.events.map((e) => [e.event_id, e]));

    const activeEvents = explorer.events.filter((e) => e.phase === "active");
    // Evenly sample down to a manageable, clickable set of markers.
    const maxMarkers = 60;
    const step = Math.max(1, Math.floor(activeEvents.length / maxMarkers));
    const sampled = activeEvents.filter((_, i) => i % step === 0).slice(0, maxMarkers);

    const events: AnalyserEvent[] = [];
    for (const e of sampled) {
      const f = featById.get(e.event_id);
      if (!f) continue;
      const values: Partial<Record<AnalyserFeature, number>> = {};
      let ok = true;
      for (const key of ANALYSER_FEATURES) {
        const raw = f.features[key];
        if (typeof raw !== "number") {
          ok = false;
          break;
        }
        values[key] = raw;
      }
      if (!ok) continue;
      events.push({
        id: e.event_id,
        x: e.location.x,
        y: e.location.y,
        values: values as Record<AnalyserFeature, number>,
      });
    }

    const ranges = Object.fromEntries(
      ANALYSER_FEATURES.map((key) => {
        const vals = events.map((e) => e.values[key]);
        return [key, [Math.min(...vals), Math.max(...vals)]];
      })
    ) as Record<AnalyserFeature, [number, number]>;

    return { matchId, label: matchLabel(explorer), events, ranges };
  });
}

// --- Match Explorer (page 09) support --------------------------------------

export interface ExplorerRow {
  id: string;
  x: number;
  y: number;
  phase: "active" | "passive";
  team: string;
  player: string | null;
  eventType: string;
  time: string;
  stats: Array<{ label: string; value: string; note: string }>;
}

export function getExplorerData(matchId: string): { label: string; rows: ExplorerRow[]; meta: MatchExplorer } {
  const m = getMatchExplorer(matchId);
  const rows: ExplorerRow[] = [];

  const timeStr = (min: number, sec: number) => `${String(min).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;

  for (const e of m.events) {
    if (e.phase === "active" && e.predictions?.active_binary && e.predictions?.active_continuous && e.predictions?.active_xt) {
      const p = e.predictions;
      rows.push({
        id: e.event_id,
        x: e.location.x,
        y: e.location.y,
        phase: "active",
        team: e.team,
        player: e.player,
        eventType: e.on_ball_event_type,
        time: timeStr(e.minute, e.second),
        stats: [
          {
            label: "Active — Binary (v1e)",
            value: p.active_binary!.probability.toFixed(2),
            note: "Probability this action succeeds as a defensive intervention — the tackle/block/interception lands.",
          },
          {
            label: "Active — Continuous (c1d)",
            value: p.active_continuous!.expected_value.toFixed(3),
            note: "Expected danger prevented on this action, on the leg's own scale. Higher = more threat taken off the game.",
          },
          {
            label: "Active — xT (x1c)",
            value: p.active_xt!.expected_delta.toFixed(3),
            note: "Expected-threat swing from this action. Negative = threat removed from the attacking side.",
          },
        ],
      });
    } else if (e.phase === "passive" && e.defenders?.length) {
      // Use the first defender slot with all three predictions present as the representative passive row.
      const d = e.defenders.find(
        (d) => d.predictions?.passive_binary && d.predictions?.passive_continuous && d.predictions?.passive_xt
      );
      if (!d) continue;
      const p = d.predictions!;
      rows.push({
        id: `${e.event_id}:${d.defender_slot_index}`,
        x: d.location.x,
        y: d.location.y,
        phase: "passive",
        team: e.team,
        player: `Defender slot ${d.defender_slot_index} (${d.functional_role})`,
        eventType: "Off-ball positioning",
        time: timeStr(e.minute, e.second),
        stats: [
          {
            label: "Passive — Binary (p1e)",
            value: p.passive_binary!.probability.toFixed(2),
            note: "Probability this off-ball position counts as a strong defensive stance — not an action, a state.",
          },
          {
            label: "Passive — Continuous (d1)",
            value: p.passive_continuous!.expected_value.toFixed(3),
            note: "Expected danger prevented purely by this positioning, before any action is taken.",
          },
          {
            label: "Passive — xT (y1c)",
            value: p.passive_xt!.expected_delta.toFixed(3),
            note: "Expected-threat swing attributable to holding this position. Positive here reads as residual exposure, not prevention.",
          },
        ],
      });
    }
  }

  // Sample down to a manageable, clickable set (roughly balanced active/passive).
  const maxRows = 70;
  const active = rows.filter((r) => r.phase === "active");
  const passive = rows.filter((r) => r.phase === "passive");
  const pickEvery = <T,>(arr: T[], n: number) => {
    const step = Math.max(1, Math.floor(arr.length / n));
    return arr.filter((_, i) => i % step === 0).slice(0, n);
  };
  const half = Math.floor(maxRows / 2);
  const sampled = [...pickEvery(active, half), ...pickEvery(passive, half)].sort((a, b) => a.time.localeCompare(b.time));

  return { label: matchLabel(m), rows: sampled, meta: m };
}
