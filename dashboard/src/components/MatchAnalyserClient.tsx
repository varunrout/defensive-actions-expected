"use client";

import { useMemo, useState } from "react";
import { ANALYSER_FEATURES, ANALYSER_FEATURE_META, type AnalyserFeature, type AnalyserMatch } from "@/lib/analyserFeatures";

export default function MatchAnalyserClient({ matches }: { matches: AnalyserMatch[] }) {
  const [matchIdx, setMatchIdx] = useState(0);
  const [feature, setFeature] = useState<AnalyserFeature>(ANALYSER_FEATURES[0]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const match = matches[matchIdx];
  const meta = ANALYSER_FEATURE_META[feature];
  const [lo, hi] = match.ranges[feature];

  const selected = useMemo(() => match.events.find((e) => e.id === selectedId) ?? null, [match, selectedId]);

  function opacityFor(v: number) {
    const norm = hi > lo ? (v - lo) / (hi - lo) : 0.5;
    return 0.3 + norm * 0.7;
  }

  return (
    <div className="flex flex-col gap-4 px-[88px] py-[28px] flex-1" style={{ overflow: "hidden" }}>
      <div className="flex flex-col gap-1">
        <div className="eyebrow">Match Analyser</div>
        <h2 style={{ fontSize: 24 }}>See the Analysis findings on a real match</h2>
      </div>

      <div className="banner wip" style={{ fontSize: 13 }}>
        Real feature values from <code className="mono">match_features/{"{match_id}"}.json</code>{" "}
        — sampled to {match.events.length} active-defending events for legibility. Findings are
        real (pages 05/06).{" "}
        <a href="/explorer" style={{ color: "var(--pitch)", fontWeight: 600 }}>
          For post-model output, see Match Explorer (09) →
        </a>
      </div>

      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex gap-2">
          {matches.map((m, i) => (
            <button key={m.matchId} className={`pill ${i === matchIdx ? "active" : ""}`} onClick={() => { setMatchIdx(i); setSelectedId(null); }}>
              {m.label}
            </button>
          ))}
        </div>
        <div className="flex gap-2">
          {ANALYSER_FEATURES.map((f) => (
            <button key={f} className={`pill mono ${f === feature ? "active" : ""}`} style={{ fontSize: 11.5 }} onClick={() => { setFeature(f); setSelectedId(null); }}>
              {f}
            </button>
          ))}
        </div>
      </div>

      <div className="flex gap-5 flex-1" style={{ minHeight: 0 }}>
        <div style={{ flex: 1.4, minWidth: 0, position: "relative" }}>
          <div className="relative rounded-xl overflow-hidden h-full" style={{ background: "#e9f3ec", border: "1px solid var(--border)" }}>
            <PitchSvg />
            {match.events.map((e) => (
              <button
                key={e.id}
                onClick={() => setSelectedId(e.id)}
                className="absolute rounded-full"
                style={{
                  left: `${(e.x / 120) * 100}%`,
                  top: `${(e.y / 80) * 100}%`,
                  width: 12,
                  height: 12,
                  transform: "translate(-50%, -50%)",
                  background: `rgba(47,107,70,${opacityFor(e.values[feature])})`,
                  border: selectedId === e.id ? "2px solid var(--text)" : "1px solid rgba(255,255,255,0.6)",
                }}
                aria-label={`event ${e.id}`}
              />
            ))}
            {selected && (
              <div
                className="absolute mono"
                style={{
                  left: `${(selected.x / 120) * 100}%`,
                  top: `${(selected.y / 80) * 100}%`,
                  transform: "translate(8px, -24px)",
                  fontSize: 11,
                  background: "var(--text)",
                  color: "white",
                  padding: "3px 7px",
                  borderRadius: 4,
                  whiteSpace: "nowrap",
                }}
              >
                {meta.format(selected.values[feature])}
              </div>
            )}
          </div>
        </div>

        <div className="card flex flex-col gap-3" style={{ flex: 1, minWidth: 0, overflowY: "auto" }}>
          <div className="mono" style={{ color: "var(--pitch)", fontSize: 13, fontWeight: 600 }}>{meta.label}</div>
          <p style={{ fontSize: 14, color: "var(--text)" }}>{meta.finding}</p>
          <div style={{ borderTop: "1px solid var(--border)", paddingTop: 10, fontSize: 12, color: "var(--muted)" }}>
            Click a marker to see its real value. Darker fill = higher value on this feature.
            Range in this sample: {meta.format(lo)} – {meta.format(hi)}.
          </div>
        </div>
      </div>
    </div>
  );
}

function PitchSvg() {
  return (
    <svg viewBox="0 0 120 80" className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
      <rect x="1" y="1" width="118" height="78" fill="none" stroke="white" strokeOpacity="0.6" strokeWidth="0.5" />
      <line x1="60" y1="1" x2="60" y2="79" stroke="white" strokeOpacity="0.6" strokeWidth="0.5" />
      <circle cx="60" cy="40" r="9" fill="none" stroke="white" strokeOpacity="0.6" strokeWidth="0.5" />
      <rect x="1" y="18" width="16" height="44" fill="none" stroke="white" strokeOpacity="0.6" strokeWidth="0.5" />
      <rect x="103" y="18" width="16" height="44" fill="none" stroke="white" strokeOpacity="0.6" strokeWidth="0.5" />
    </svg>
  );
}
