// Client-safe constants for the Match Analyser (page 07) — no filesystem
// dependency, so this can be imported directly by client components.

export const ANALYSER_FEATURES = [
  "nearest_defender_distance",
  "distance_to_attacking_box",
  "defenders_within_5m",
  "attacker_defender_ratio",
] as const;
export type AnalyserFeature = (typeof ANALYSER_FEATURES)[number];

export const ANALYSER_FEATURE_META: Record<AnalyserFeature, { label: string; finding: string; format: (v: number) => string }> = {
  nearest_defender_distance: {
    label: "nearest_defender_distance",
    finding:
      "Distance (m) from the acting defender to the nearest attacker at the moment of the action. Tighter distances cluster around already-dangerous situations rather than causing them.",
    format: (v) => `${v.toFixed(1)}m`,
  },
  distance_to_attacking_box: {
    label: "distance_to_attacking_box",
    finding:
      "Distance (m) from the action to the edge of the box being attacked. Actions taken deep in the defensive third sit at the low end of this scale.",
    format: (v) => `${v.toFixed(1)}m`,
  },
  defenders_within_5m: {
    label: "defenders_within_5m",
    finding:
      "Count of teammates within 5m of the action. Higher local defensive density generally reads as cover, not vulnerability.",
    format: (v) => `${Math.round(v)}`,
  },
  attacker_defender_ratio: {
    label: "attacker_defender_ratio",
    finding:
      "Ratio of nearby attackers to nearby defenders around the action. Values above 1 mean the defending side is locally outnumbered.",
    format: (v) => v.toFixed(2),
  },
};

export interface AnalyserEvent {
  id: string;
  x: number;
  y: number;
  values: Record<AnalyserFeature, number>;
}

export interface AnalyserMatch {
  matchId: string;
  label: string;
  events: AnalyserEvent[];
  ranges: Record<AnalyserFeature, [number, number]>;
}
