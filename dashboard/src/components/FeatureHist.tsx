import type { FeatureAtlasFeature } from "@/lib/data";

// Bars are scaled from shot_rate_pct, not n. These are quantile-decile bins
// (n is ~equal per bin by construction for continuous/quantile features), so a
// histogram of n would render as ~flat bars and hide the real shape. shot_rate_pct
// is the actual signal the atlas exists to show, and it varies meaningfully across
// bins (U-shaped, monotonic, etc.) for every feature in the atlas.
export function Hist({ atlasFeature }: { atlasFeature: FeatureAtlasFeature }) {
  const { bins } = atlasFeature;
  const max = Math.max(...bins.map((b) => b.shot_rate_pct), 0.0001);
  return (
    <div className="flex items-end gap-[2px]" style={{ height: 44 }}>
      {bins.map((b, i) => (
        <div
          key={i}
          title={`${b.bin} · n=${b.n.toLocaleString()} · shot rate ${b.shot_rate_pct}%`}
          style={{
            width: 6,
            height: `${Math.max((b.shot_rate_pct / max) * 100, 3)}%`,
            background: "var(--pitch)",
            opacity: 0.7,
          }}
        />
      ))}
    </div>
  );
}

export function AtlasCaption({ atlasFeature, dataset }: { atlasFeature: FeatureAtlasFeature; dataset: "active" | "passive" }) {
  return (
    <p style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>
      n={atlasFeature.n_rows_used.toLocaleString()} rows analysed, {dataset} defence dataset
    </p>
  );
}
