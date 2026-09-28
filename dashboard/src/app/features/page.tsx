import { PageHeading, StatBar, Banner, Card } from "@/components/ui";
import { getAtlasFeature, type FeatureAtlasFeature } from "@/lib/data";

// Bars are scaled from shot_rate_pct, not n. These are quantile-decile bins
// (n is ~equal per bin by construction for continuous/quantile features), so a
// histogram of n would render as ~flat bars and hide the real shape. shot_rate_pct
// is the actual signal the atlas exists to show, and it varies meaningfully across
// bins (U-shaped, monotonic, etc.) for every one of these 5 features.
function Hist({ atlasFeature }: { atlasFeature: FeatureAtlasFeature }) {
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

function AtlasCaption({ atlasFeature, dataset }: { atlasFeature: FeatureAtlasFeature; dataset: "active" | "passive" }) {
  return (
    <p style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>
      n={atlasFeature.n_rows_used.toLocaleString()} rows analysed, {dataset} defence dataset
    </p>
  );
}

export default function FeaturesPage() {
  const markingTightness = getAtlasFeature("passive", "marking_tightness")!;
  const distanceToAttackingGoal = getAtlasFeature("active", "distance_to_attacking_goal")!;
  const zoneDefensiveValue = getAtlasFeature("passive", "zone_defensive_value")!;
  const overloadScore = getAtlasFeature("passive", "overload_score")!;
  const eventsElapsedInPossession = getAtlasFeature("active", "events_elapsed_in_possession")!;

  return (
    <div className="flex flex-col gap-[22px] px-[88px] py-[34px] overflow-y-auto">
      <PageHeading
        eyebrow="Features"
        title="The initial candidate list"
        subhead="Before the correlation review and collapse-tier resolution on page 04 narrowed anything down — this is what the pipeline actually started with."
      />

      <div className="grid grid-cols-2 gap-4">
        <Card>
          <b style={{ fontSize: 14 }}>Active-defence candidates</b>
          <div className="mt-2">
            <StatBar
              stats={[
                { value: "29", label: "Numerical" },
                { value: "18", label: "Continuous" },
                { value: "11", label: "Discrete" },
                { value: "10", label: "Categorical" },
              ]}
              wrap
            />
          </div>
        </Card>
        <Card>
          <b style={{ fontSize: 14 }}>Passive-defence candidates</b>
          <div className="mt-2">
            <StatBar
              stats={[
                { value: "1.59M", label: "Rows" },
                { value: "6", label: "Categorical" },
                { value: "5.97%", label: "Base shot rate" },
                { value: "115", label: "Matches" },
              ]}
              wrap
            />
          </div>
        </Card>
      </div>

      <Banner tone="wip">
        <b>Shape only, this pass</b> — no shot-rate signal yet. One self-reference bug
        (nearest_defender_distance) was caught and fixed here; 5 columns dropped as duplicates or
        camera-coverage artifacts (action_x/y — identical to ball_x/y — plus two visible-area
        columns and freeze_frame_count).
      </Banner>

      <section>
        <h3 style={{ fontSize: 15, marginBottom: 4 }}>
          Continuous — real histograms (3 of 18){" "}
          <span style={{ fontWeight: 400, color: "var(--muted)", fontSize: 12.5 }}>
            bars = shot rate per decile bin, hover for bin edges
          </span>
        </h3>
        <div className="grid grid-cols-3 gap-3">
          <Card>
            <div className="mono" style={{ fontSize: 12.5, marginBottom: 6 }}>marking_tightness</div>
            <Hist atlasFeature={markingTightness} />
            <p style={{ fontSize: 12, color: "var(--muted)", marginTop: 6 }}>Monotonic decreasing shot rate — tightest marking decile is 8.3%, loosest is 5.4%</p>
            <AtlasCaption atlasFeature={markingTightness} dataset="passive" />
          </Card>
          <Card>
            <div className="mono" style={{ fontSize: 12.5, marginBottom: 6 }}>distance_to_attacking_goal</div>
            <Hist atlasFeature={distanceToAttackingGoal} />
            <p style={{ fontSize: 12, color: "var(--muted)", marginTop: 6 }}>U-shaped — highest shot rate right at the goal, lowest around 55–70m out</p>
            <AtlasCaption atlasFeature={distanceToAttackingGoal} dataset="active" />
          </Card>
          <Card>
            <div className="mono" style={{ fontSize: 12.5, marginBottom: 6 }}>zone_defensive_value</div>
            <Hist atlasFeature={zoneDefensiveValue} />
            <p style={{ fontSize: 12, color: "var(--muted)", marginTop: 6 }}>U-shaped — bounded 0–1, denser at both ends</p>
            <AtlasCaption atlasFeature={zoneDefensiveValue} dataset="passive" />
          </Card>
        </div>
      </section>

      <section>
        <h3 style={{ fontSize: 15, marginBottom: 4 }}>
          Discrete — real histograms (2 of 11){" "}
          <span style={{ fontWeight: 400, color: "var(--muted)", fontSize: 12.5 }}>bars = shot rate per bin, hover for bin edges</span>
        </h3>
        <div className="grid grid-cols-3 gap-3">
          <Card>
            <div className="mono" style={{ fontSize: 12.5, marginBottom: 6 }}>overload_score</div>
            <Hist atlasFeature={overloadScore} />
            <p style={{ fontSize: 12, color: "var(--muted)", marginTop: 6 }}>0–3 converging defenders, monotonic decreasing shot rate (6.1% → 4.5%)</p>
            <AtlasCaption atlasFeature={overloadScore} dataset="passive" />
          </Card>
          <Card>
            <div className="mono" style={{ fontSize: 12.5, marginBottom: 6 }}>events_elapsed_in_possession</div>
            <Hist atlasFeature={eventsElapsedInPossession} />
            <p style={{ fontSize: 12, color: "var(--muted)", marginTop: 6 }}>Monotonic increasing — shot rate rises from 4.1% early in a possession to ~10% late</p>
            <AtlasCaption atlasFeature={eventsElapsedInPossession} dataset="active" />
          </Card>
          <Card style={{ display: "flex", alignItems: "center", justifyContent: "center", borderStyle: "dashed" }}>
            <span style={{ fontSize: 12.5, color: "var(--muted)" }}>+ 9 more discrete features in the full atlas</span>
          </Card>
        </div>
      </section>

      <Banner tone="neutral">
        Small categories carry wide uncertainty — e.g. goalkeeper (n=874) or visibility
        &quot;high&quot; (n=2) in the active-defending category breakdown. Read their rates as
        noisy, not settled.
      </Banner>

      <section className="mt-2">
        <h3 style={{ fontSize: 16 }}>What actually mattered on the pitch</h3>
        <p style={{ fontSize: 13.5, color: "var(--muted)", marginBottom: 10 }}>
          Of the 29 candidates, these are the ones with a real, football-legible story — not just
          a passing correlation.
        </p>
        <div className="grid grid-cols-3 gap-3">
          <Card>
            <div className="mono" style={{ fontSize: 12.5, color: "var(--pitch)" }}>marking_tightness</div>
            <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>
              Tighter marking (0–3.6m) → 7.6% near-term shot rate; loosest (9–57m) → 4.9%. Reads
              as a selection effect — defenders mark tight because the situation is already
              dangerous.
            </p>
          </Card>
          <Card>
            <div className="mono" style={{ fontSize: 12.5, color: "var(--pitch)" }}>lane_screening_score</div>
            <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>
              Screening the top-ranked passing option correlates with a HIGHER shot rate (7.6% vs
              5.8%). A &quot;worth screening&quot; option usually means the situation is already
              threatening.
            </p>
          </Card>
          <Card>
            <div className="mono" style={{ fontSize: 12.5, color: "var(--pitch)" }}>zone_defensive_value</div>
            <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>
              A real U-shape: high press → 8.1% shot rate; calm midfield → 2.4–2.9%; deep/own-box
              → 10.4%. Both extremes carry more risk than the calm middle.
            </p>
          </Card>
          <Card>
            <div className="mono" style={{ fontSize: 12.5, color: "var(--pitch)" }}>overload_score</div>
            <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>
              The one feature that behaves exactly as designed: more converging defenders (0→3) →
              shot rate drops 6.1% → 4.8%, monotonically.
            </p>
          </Card>
          <Card>
            <div className="mono" style={{ fontSize: 12.5, color: "var(--pitch)" }}>defender_functional_role</div>
            <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>
              last_line / wide_cover / central_screen / unclassified. Small spread alone
              (4.7–6.3%) — likely more useful combined with zone/screening features.
            </p>
          </Card>
          <div className="banner pitch flex items-center" style={{ fontSize: 12.5, fontWeight: 600 }}>
            No causal claims anywhere — every relationship here is &quot;correlates with,&quot;
            never &quot;causes.&quot;
          </div>
        </div>
      </section>
    </div>
  );
}
