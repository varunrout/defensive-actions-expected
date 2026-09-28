"use client";

import { useMemo, useState } from "react";
import type { ExplorerRow } from "@/lib/data";

interface MatchBundle {
  matchId: string;
  label: string;
  rows: ExplorerRow[];
}

export default function MatchExplorerClient({ matches }: { matches: MatchBundle[] }) {
  const [matchIdx, setMatchIdx] = useState(0);
  const [phase, setPhase] = useState<"all" | "active" | "passive">("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const match = matches[matchIdx];
  const visible = useMemo(
    () => match.rows.filter((r) => phase === "all" || r.phase === phase),
    [match, phase]
  );
  const selected = useMemo(() => visible.find((r) => r.id === selectedId) ?? null, [visible, selectedId]);

  function selectMatch(i: number) {
    setMatchIdx(i);
    setSelectedId(null);
  }

  return (
    <div className="flex flex-col gap-4 px-[88px] py-[28px] flex-1" style={{ overflow: "hidden" }}>
      <div className="flex flex-col gap-1">
        <div className="eyebrow">Match Explorer</div>
        <h2 style={{ fontSize: 24 }}>Every defensive action, rated by the finished models</h2>
      </div>

      <div className="banner wip" style={{ fontSize: 13 }}>
        Real predictions from <code className="mono">match_explorer/{"{match_id}"}.json</code> —
        sampled to {match.rows.length} events for legibility. This is <b>post-model</b>: every
        applicable model&apos;s raw output is shown, not one blended score. Marker colour =
        phase (active/passive). For <b>pre-model</b> feature patterns, see{" "}
        <a href="/analyser" style={{ color: "var(--pitch)", fontWeight: 600 }}>
          Match Analyser (07) →
        </a>
      </div>

      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex gap-2">
          {matches.map((m, i) => (
            <button key={m.matchId} className={`pill ${i === matchIdx ? "active" : ""}`} onClick={() => selectMatch(i)}>
              {m.label}
            </button>
          ))}
        </div>
        <div className="flex gap-2">
          <button
            className="pill"
            style={phase === "all" ? { background: "var(--pitch)", borderColor: "var(--pitch)", color: "white" } : {}}
            onClick={() => setPhase("all")}
          >
            All
          </button>
          <button
            className="pill"
            style={phase === "active" ? { background: "var(--active-marker)", borderColor: "var(--active-marker)", color: "white" } : {}}
            onClick={() => setPhase("active")}
          >
            Active
          </button>
          <button
            className="pill"
            style={phase === "passive" ? { background: "var(--passive-marker)", borderColor: "var(--passive-marker)", color: "white" } : {}}
            onClick={() => setPhase("passive")}
          >
            Passive
          </button>
        </div>
      </div>

      <div className="flex gap-5 flex-1" style={{ minHeight: 0 }}>
        <div className="flex flex-col gap-2.5" style={{ flex: 1.4, minHeight: 0 }}>
          <div className="relative rounded-xl overflow-hidden flex-1" style={{ background: "#e9f3ec", border: "1px solid var(--border)" }}>
            <PitchSvg />
            {visible.map((r) => (
              <button
                key={r.id}
                onClick={() => setSelectedId(r.id)}
                className="absolute rounded-full"
                style={{
                  left: `${(r.x / 120) * 100}%`,
                  top: `${(r.y / 80) * 100}%`,
                  width: 12,
                  height: 12,
                  transform: "translate(-50%, -50%)",
                  background: r.phase === "active" ? "var(--active-marker)" : "var(--passive-marker)",
                  border: selectedId === r.id ? "2px solid var(--text)" : "1px solid rgba(255,255,255,0.7)",
                }}
                aria-label={`event ${r.id}`}
              />
            ))}
            <div className="absolute bottom-2 left-2 mono" style={{ fontSize: 10.5, color: "var(--muted)" }}>Attacking →</div>
            <div className="absolute bottom-2 right-2 mono" style={{ fontSize: 10.5, color: "var(--muted)" }}>
              Blue = active · Amber = passive
            </div>
          </div>
          <div className="flex gap-1.5 overflow-x-auto pb-1">
            {visible.map((r) => (
              <button
                key={r.id}
                onClick={() => setSelectedId(r.id)}
                className="pill mono flex-shrink-0"
                style={{
                  fontSize: 11,
                  padding: "5px 10px",
                  ...(selectedId === r.id
                    ? { background: r.phase === "active" ? "var(--active-marker)" : "var(--passive-marker)", borderColor: "transparent", color: "white" }
                    : {}),
                }}
              >
                {r.time}
              </button>
            ))}
          </div>
        </div>

        <div className="card" style={{ flex: 1, overflowY: "auto" }}>
          {!selected ? (
            <div className="h-full flex items-center justify-center text-center" style={{ color: "var(--muted)", fontSize: 14 }}>
              Click a marker or a time below the pitch to see its DAx rating.
            </div>
          ) : (
            <div className="flex flex-col gap-3">
              <div className="mono" style={{ fontSize: 12, color: "var(--muted)" }}>
                {selected.time} · {selected.eventType}
              </div>
              <h3 style={{ fontSize: 19 }}>{selected.player ?? selected.team}</h3>
              <div className="mono" style={{ fontSize: 11.5, color: "var(--muted)" }}>
                <b style={{ color: "var(--text)" }}>{selected.phase === "active" ? "Active" : "Passive"}</b>{" "}
                phase · {selected.stats.length} applicable models — shown individually, never blended
              </div>
              <div className="flex flex-col gap-2 mt-1">
                {selected.stats.map((s) => (
                  <div key={s.label} style={{ borderTop: "1px solid var(--border)", paddingTop: 10 }}>
                    <div className="flex items-baseline justify-between">
                      <span className="mono" style={{ fontSize: 12.5 }}>{s.label}</span>
                      <span className="mono" style={{ fontSize: 16, fontWeight: 600, color: "var(--pitch)" }}>{s.value}</span>
                    </div>
                    <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 4 }}>{s.note}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
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
