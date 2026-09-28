import { PageHeading, StatBar, Card, Banner } from "@/components/ui";
import { getLegsSummary } from "@/lib/data";

export default function AnalysisPage() {
  const legs = getLegsSummary();
  const byGroup = (g: string) => legs.legs.filter((l) => l.group === g);

  const passes = [
    {
      title: "Redundancy & Correlation Atlas",
      text: "Type-matched methods throughout — Spearman, phi, point-biserial, Cramér's V, correlation ratio — because one Pearson matrix would misread a mixed continuous/boolean/categorical feature set.",
      stat: "95 scanned · 4 exact dupes · 9 dropped · 3 still open",
    },
    {
      title: "Flag Ledgers",
      text: "Every boolean ranked by shot-rate lift. Active-defending had 3 flags sitting at an exact 0.00% shot rate — not weak signal, a tautology: the target window has nothing left to look into once those fire. Left out of the ranking on purpose.",
      stat: "Active: 17 booleans, 3 tautologies · Passive: 16 booleans, 0 tautologies",
    },
    {
      title: "V2 Correlation Resolution",
      text: "A gap in the original framework, caught on rebuild: no test existed for continuous↔continuous pairs, so 65 of 66 review pairs fell straight through to a human call. Fixed and resolved here, not hidden.",
      stat: "23 new collapse pairs · 65 review pairs resolved · 4 genuine human calls left",
    },
    {
      title: "Football Sanity Check",
      text: "Match video wasn't available, so the substitute was independent reimplementation — each feature's formula rebuilt from scratch, from the docstring, and checked against the real stored output.",
      stat: "4/4 features match exactly · 1.59M rows checked · 0 mismatches",
    },
    {
      title: "Passive Defence Archetypes",
      text: "KMeans clustering within each functional-role bucket, not across roles — role is already a validated signal, this asks what varies within one. No player identity anywhere in the clustering.",
      stat: "5 buckets · 10 archetypes (k=2 won every time) · silhouette 0.22–0.30",
    },
  ];

  return (
    <div className="flex flex-col gap-6 px-[88px] py-[34px] overflow-y-auto">
      <PageHeading
        eyebrow="Analysis"
        title="What the exploratory analysis actually found"
        subhead="Every number below traces to a real report in reports/analysis/ — not recalled from memory of the work."
      />

      <StatBar
        wrap
        stats={[
          { value: "56,068 / 34", label: "Active rows / locked features" },
          { value: "1.59M / 38", label: "Passive rows / locked features" },
          { value: "320", label: "Genuine divergences (binary target)" },
          { value: "0", label: "Features context-independent" },
        ]}
      />

      <section>
        <h3 style={{ fontSize: 16, marginBottom: 10 }}>Bottom line</h3>
        <div className="grid grid-cols-2 gap-3">
          <Banner tone="pitch">
            <b>Both reversals are real, not artefacts.</b> Tighter marking → more shots, and more
            lane-screening → more shots — both survive rigorous confound-conditioning. Genuine
            football signal, not a selection effect.
          </Banner>
          <Banner tone="wip">
            <b>Tournament, not football.</b> defenders_within_10m / _5m carry the largest raw
            signal (44–47pp) but are confirmed WC2022-vs-Euro2024 differences, not stable
            defensive patterns — need tournament-aware handling, not blind trust.
          </Banner>
          <Banner tone="blocked">
            <b>A leakage flag, caught early.</b> position (active-only) is flagged as a
            player-identity-leakage risk for validation — surfaced here, before it could quietly
            bias a model.
          </Banner>
          <Banner tone="neutral">
            <b>Conditioning cuts noise ~4×.</b> On the continuous target, conditioning on
            &quot;did a shot even happen&quot; cuts divergence counts roughly 4× — separating
            what predicts a shot from what predicts how good the chance is.
          </Banner>
        </div>
      </section>

      <section>
        <h3 style={{ fontSize: 16, marginBottom: 4 }}>Per-leg model analysis</h3>
        <p style={{ fontSize: 12.5, color: "var(--muted)", marginBottom: 10 }}>
          Binary and continuous/xT legs are never ranked on one axis — different metrics, different scales.
        </p>

        <div className="mb-3">
          <div className="mono" style={{ fontSize: 11, color: "var(--muted)", marginBottom: 6, textTransform: "uppercase" }}>Binary group</div>
          <div className="grid grid-cols-2 gap-3">
            {byGroup("binary").map((l) => (
              <Card key={l.id}>
                <div className="mono" style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase" }}>
                  {l.side} · Binary
                </div>
                <div className="mono" style={{ fontSize: 24, color: "var(--pitch)", margin: "4px 0" }}>
                  {l.headline.metric} {l.headline.value.toFixed(4)}
                </div>
                <p style={{ fontSize: 12.5, color: "var(--muted)" }}>
                  {l.reference_model} — {l.question}
                </p>
              </Card>
            ))}
          </div>
        </div>

        <div>
          <div className="mono" style={{ fontSize: 11, color: "var(--muted)", marginBottom: 6, textTransform: "uppercase" }}>Continuous &amp; xT group</div>
          <div className="grid grid-cols-2 gap-3">
            {[...byGroup("hurdle"), ...byGroup("xt")].map((l) => (
              <Card key={l.id}>
                <div className="mono" style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase" }}>
                  {l.side} · {l.group === "hurdle" ? "Continuous" : "xT"}
                </div>
                <div className="mono" style={{ fontSize: 24, color: "var(--pitch)", margin: "4px 0" }}>
                  {l.headline.metric} {l.headline.value.toFixed(4)}
                </div>
                <p style={{ fontSize: 12.5, color: "var(--muted)" }}>
                  {l.reference_model} — {l.question}
                </p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      <section>
        <h3 style={{ fontSize: 16, marginBottom: 10 }}>The underlying passes</h3>
        <div className="grid grid-cols-3 gap-3">
          {passes.map((p) => (
            <Card key={p.title}>
              <b style={{ fontSize: 13.5 }}>{p.title}</b>
              <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>{p.text}</p>
              <p className="mono" style={{ fontSize: 11.5, color: "var(--pitch)", marginTop: 8 }}>{p.stat}</p>
            </Card>
          ))}
          <div className="banner neutral flex items-center" style={{ fontSize: 12.5 }}>
            No causal claims anywhere — every finding on this page is descriptive:
            &quot;correlates with,&quot; never &quot;causes.&quot;
          </div>
        </div>
      </section>
    </div>
  );
}
