"use client";

import { useMemo, useState } from "react";
import type { FeatureJourney, FeatureJourneyFeature } from "@/lib/data";

function MiniHist({ bins }: { bins: FeatureJourneyFeature["profile"] extends null ? never : { bin?: string; value?: string; n: number; shot_rate_pct: number }[] }) {
  const max = Math.max(...bins.map((b) => b.shot_rate_pct), 0.0001);
  return (
    <div className="flex items-end gap-[2px]" style={{ height: 44 }}>
      {bins.map((b, i) => (
        <div
          key={i}
          title={`${b.bin ?? b.value} · n=${b.n.toLocaleString()} · shot rate ${b.shot_rate_pct}%`}
          style={{
            width: Math.max(4, Math.floor(140 / bins.length)),
            height: `${Math.max((b.shot_rate_pct / max) * 100, 3)}%`,
            background: "var(--pitch)",
            opacity: 0.7,
          }}
        />
      ))}
    </div>
  );
}

const STAGE_LABELS: Record<string, string> = {
  before: "Before candidate pool",
  "1": "Stage 01 · Candidate pool",
  "4": "Stage 04 · Structural redesign",
  "5": "Stage 05 · Collapse-tier resolution",
  "6": "Stage 06 · Drop raw option coordinates",
  "9": "Stage 09 · Multicollinearity (VIF)",
  "10": "Stage 10 · Leakage audit",
};

function stageKey(f: FeatureJourneyFeature): string {
  if (f.origin === "excluded_before_count") return "before";
  return String(f.stage ?? "1");
}

export default function FeatureJourneyExplorer({ journey }: { journey: FeatureJourney }) {
  const { features, stages } = journey;

  const [dataset, setDataset] = useState<"all" | "active" | "passive">("all");
  const [type, setType] = useState<string>("all");
  const [fate, setFate] = useState<"all" | "locked" | "dropped">("all");
  const [stage, setStage] = useState<string>("all");
  const [modelled, setModelled] = useState<"all" | "yes" | "no">("all");
  const [selectedName, setSelectedName] = useState<string | null>(null);

  const types = useMemo(() => Array.from(new Set(features.map((f) => f.type ?? "unresolved"))).sort(), [features]);
  const stageOptions = useMemo(() => {
    const keys = Array.from(new Set(features.map(stageKey)));
    return keys.sort((a, b) => (a === "before" ? -1 : b === "before" ? 1 : Number(a) - Number(b)));
  }, [features]);

  const filtered = useMemo(() => {
    return features.filter((f) => {
      if (dataset !== "all" && f.dataset !== dataset) return false;
      if (type !== "all" && (f.type ?? "unresolved") !== type) return false;
      if (fate !== "all" && f.fate !== fate) return false;
      if (stage !== "all" && stageKey(f) !== stage) return false;
      if (modelled !== "all" && (modelled === "yes") !== f.modelled) return false;
      return true;
    });
  }, [features, dataset, type, fate, stage, modelled]);

  const selected = useMemo(
    () => (selectedName ? features.find((f) => f.name === selectedName && f.dataset === dataset) ?? features.find((f) => f.name === selectedName) ?? null : null),
    [features, selectedName, dataset]
  );

  return (
    <div className="flex flex-col gap-4">
      {/* Journey visual */}
      <div className="grid grid-cols-2 gap-4">
        {(["active", "passive"] as const).map((ds) => {
          const s = stages[ds];
          return (
            <div key={ds} className="card flex flex-col gap-2.5" style={{ minWidth: 0 }}>
              <b style={{ fontSize: 14, textTransform: "capitalize" }}>{ds}-defence journey</b>
              <div className="mono flex items-center gap-2 flex-wrap" style={{ fontSize: 13 }}>
                <span style={{ color: "var(--muted)" }}>{s.stage_counts.excluded_before_count} excluded</span>
                <span>→</span>
                <span style={{ color: "var(--pitch)", fontWeight: 700 }}>{s.stage_counts.candidate_pool_stage01} candidates</span>
                <span>→</span>
                <span style={{ color: "var(--pitch)", fontWeight: 700 }}>{s.stage_counts.final_locked} locked</span>
              </div>
              <div className="flex flex-col gap-1.5">
                {s.ledger.map((stg) => (
                  <div key={stg.stage} style={{ borderTop: "1px solid var(--border)", paddingTop: 6, minWidth: 0 }}>
                    <div className="flex items-baseline justify-between gap-2">
                      <span style={{ fontSize: 12, fontWeight: 600 }}>{stg.label}</span>
                      <span className="mono" style={{ fontSize: 11, color: "var(--muted)", whiteSpace: "nowrap" }}>
                        {stg.count_before} → {stg.count_after}
                      </span>
                    </div>
                    {(stg.dropped.length > 0 || stg.engineered.length > 0) && (
                      <p style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>
                        {stg.dropped.length > 0 && <>dropped: {stg.dropped.join(", ")}</>}
                        {stg.dropped.length > 0 && stg.engineered.length > 0 && " · "}
                        {stg.engineered.length > 0 && <>engineered: {stg.engineered.join(", ")}</>}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* Browser */}
      <div className="card flex flex-col gap-3" style={{ minWidth: 0 }}>
        <div className="flex items-center justify-between flex-wrap gap-2">
          <b style={{ fontSize: 14 }}>
            Feature browser <span style={{ fontWeight: 400, color: "var(--muted)", fontSize: 12 }}>({filtered.length} of {features.length} records)</span>
          </b>
        </div>
        <div className="flex gap-2 flex-wrap">
          {(["all", "active", "passive"] as const).map((d) => (
            <button key={d} className={`pill mono ${dataset === d ? "active" : ""}`} style={{ fontSize: 11 }} onClick={() => setDataset(d)}>
              {d}
            </button>
          ))}
          <span style={{ width: 1, background: "var(--border)" }} />
          {(["all", "locked", "dropped"] as const).map((f) => (
            <button key={f} className={`pill mono ${fate === f ? "active" : ""}`} style={{ fontSize: 11 }} onClick={() => setFate(f)}>
              {f}
            </button>
          ))}
          <span style={{ width: 1, background: "var(--border)" }} />
          {(["all", "yes", "no"] as const).map((m) => (
            <button key={m} className={`pill mono ${modelled === m ? "active" : ""}`} style={{ fontSize: 11 }} onClick={() => setModelled(m)}>
              modelled: {m}
            </button>
          ))}
        </div>
        <div className="flex gap-2 flex-wrap">
          <select className="pill mono" style={{ fontSize: 11 }} value={type} onChange={(e) => setType(e.target.value)}>
            <option value="all">type: all</option>
            {types.map((t) => (
              <option key={t} value={t}>
                type: {t}
              </option>
            ))}
          </select>
          <select className="pill mono" style={{ fontSize: 11 }} value={stage} onChange={(e) => setStage(e.target.value)}>
            <option value="all">stage: all</option>
            {stageOptions.map((s) => (
              <option key={s} value={s}>
                {STAGE_LABELS[s] ?? `Stage ${s}`}
              </option>
            ))}
          </select>
        </div>

        <div className="flex gap-4" style={{ minHeight: 0 }}>
          <div className="flex flex-col gap-1" style={{ flex: 1.3, minWidth: 0, maxHeight: 360, overflowY: "auto" }}>
            {filtered.map((f) => (
              <button
                key={`${f.dataset}:${f.name}:${f.origin}`}
                onClick={() => setSelectedName(f.name)}
                className="flex items-center justify-between gap-2"
                style={{
                  textAlign: "left",
                  padding: "6px 8px",
                  borderRadius: 5,
                  border: "1px solid transparent",
                  background: selected?.name === f.name && selected?.dataset === f.dataset ? "var(--pitch-soft)" : "transparent",
                  cursor: "pointer",
                }}
              >
                <span className="mono" style={{ fontSize: 12, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {f.name}
                </span>
                <span className="mono" style={{ fontSize: 10, color: f.fate === "locked" ? "var(--pitch)" : "var(--blocked)", whiteSpace: "nowrap" }}>
                  {f.dataset[0]}·{f.fate}
                </span>
              </button>
            ))}
          </div>
          <div className="flex flex-col gap-2" style={{ flex: 1, minWidth: 0, borderLeft: "1px solid var(--border)", paddingLeft: 16 }}>
            {!selected ? (
              <p style={{ fontSize: 13, color: "var(--muted)" }}>Select a feature to see its real profile, fate and reason.</p>
            ) : (
              <>
                <div className="mono" style={{ fontSize: 13, fontWeight: 600, color: "var(--pitch)" }}>{selected.name}</div>
                <div className="mono flex gap-2 flex-wrap" style={{ fontSize: 10.5, color: "var(--muted)" }}>
                  <span>{selected.dataset}</span>
                  <span>{selected.type ?? "type unresolved"}</span>
                  <span>{selected.fate}</span>
                  <span>{selected.modelled ? "modelled" : "not modelled"}</span>
                </div>
                <p style={{ fontSize: 11.5, color: "var(--muted)" }}>{selected.stage_label}</p>
                <p style={{ fontSize: 12.5 }}>{selected.reason}</p>
                {selected.evidence && (
                  <p className="mono" style={{ fontSize: 11, color: "var(--pitch)" }}>
                    evidence: {selected.evidence.metric}={selected.evidence.value} vs {selected.evidence.vs}
                  </p>
                )}
                {selected.profile ? (
                  <div>
                    <MiniHist bins={selected.profile.bins} />
                    <p style={{ fontSize: 10.5, color: "var(--muted)", marginTop: 4 }}>
                      n={selected.profile.n_rows_used.toLocaleString()} · {selected.profile.binning_method}
                    </p>
                  </div>
                ) : (
                  <p style={{ fontSize: 11, color: "var(--muted)" }}>No profile: {selected.profile_reason}</p>
                )}
                <p className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 4 }}>{selected.source}</p>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
