"use client";

import { useMemo, useState } from "react";
import type { FeatureAtlasFeature } from "@/lib/data";
import { Hist, AtlasCaption } from "@/components/FeatureHist";

type DatasetFilter = "all" | "active" | "passive";
type TypeFilter = "all" | "continuous" | "discrete";

interface Row {
  dataset: "active" | "passive";
  feature: FeatureAtlasFeature;
}

export default function FeatureAtlasExplorer({
  activeFeatures,
  passiveFeatures,
}: {
  activeFeatures: FeatureAtlasFeature[];
  passiveFeatures: FeatureAtlasFeature[];
}) {
  const [query, setQuery] = useState("");
  const [dataset, setDataset] = useState<DatasetFilter>("all");
  const [type, setType] = useState<TypeFilter>("all");

  const rows: Row[] = useMemo(
    () => [
      ...activeFeatures.map((feature) => ({ dataset: "active" as const, feature })),
      ...passiveFeatures.map((feature) => ({ dataset: "passive" as const, feature })),
    ],
    [activeFeatures, passiveFeatures]
  );

  const filtered = rows.filter((r) => {
    if (dataset !== "all" && r.dataset !== dataset) return false;
    if (type !== "all" && r.feature.type !== type) return false;
    if (query && !r.feature.feature.toLowerCase().includes(query.toLowerCase())) return false;
    return true;
  });

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Filter by feature name…"
          className="mono"
          style={{
            fontSize: 12.5,
            padding: "6px 10px",
            borderRadius: 8,
            border: "1px solid var(--border)",
            minWidth: 220,
            background: "var(--surface)",
          }}
        />
        <div className="flex gap-2 flex-wrap">
          {(["all", "active", "passive"] as const).map((d) => (
            <button key={d} className={`pill ${dataset === d ? "active" : ""}`} onClick={() => setDataset(d)}>
              {d === "all" ? "All datasets" : d === "active" ? "Active" : "Passive"}
            </button>
          ))}
          {(["all", "continuous", "discrete"] as const).map((t) => (
            <button key={t} className={`pill ${type === t ? "active" : ""}`} onClick={() => setType(t)}>
              {t === "all" ? "All types" : t}
            </button>
          ))}
        </div>
      </div>

      <p style={{ fontSize: 12, color: "var(--muted)" }}>
        Showing {filtered.length} of {rows.length} numerical candidate features ({activeFeatures.length} active +{" "}
        {passiveFeatures.length} passive).
      </p>

      <div
        className="grid grid-cols-3 gap-3"
        style={{ maxHeight: 420, overflowY: "auto", paddingRight: 4 }}
      >
        {filtered.map((r) => (
          <div className="card" key={`${r.dataset}:${r.feature.feature}`}>
            <div className="flex items-center justify-between" style={{ marginBottom: 6 }}>
              <div className="mono" style={{ fontSize: 12 }}>
                {r.feature.feature}
              </div>
              <span
                className="mono"
                style={{
                  fontSize: 10,
                  color: "var(--muted)",
                  border: "1px solid var(--border)",
                  borderRadius: 999,
                  padding: "1px 6px",
                }}
              >
                {r.dataset} · {r.feature.type}
              </span>
            </div>
            <Hist atlasFeature={r.feature} />
            <AtlasCaption atlasFeature={r.feature} dataset={r.dataset} />
          </div>
        ))}
        {filtered.length === 0 && (
          <div style={{ fontSize: 13, color: "var(--muted)", padding: "12px 0" }}>
            No features match that filter.
          </div>
        )}
      </div>
    </div>
  );
}
