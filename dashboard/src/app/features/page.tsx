import { PageHeading, StatBar, Banner, Card } from "@/components/ui";
import { getFeatureJourney, getAnalysisFacts } from "@/lib/data";
import FeatureJourneyExplorer from "@/components/FeatureJourneyExplorer";

export default function FeaturesPage() {
  const journey = getFeatureJourney();
  const facts = getAnalysisFacts().facts;

  const active = journey.stages.active;
  const passive = journey.stages.passive;

  const activeLocked = journey.features.filter((f) => f.dataset === "active" && f.fate === "locked");
  const activeModelled = activeLocked.filter((f) => f.modelled);
  const passiveLocked = journey.features.filter((f) => f.dataset === "passive" && f.fate === "locked");
  const passiveModelled = passiveLocked.filter((f) => f.modelled);

  const totalCandidates = active.stage_counts.candidate_pool_stage01 + passive.stage_counts.candidate_pool_stage01;
  const totalEngineered = journey.features.filter((f) => f.origin === "engineered").length;
  const totalExclusions = journey.features.filter((f) => f.origin === "excluded_before_count").length;

  const markingTightness = journey.features.find((f) => f.name === "marking_tightness" && f.dataset === "passive")!;
  const laneScreening = journey.features.find((f) => f.name === "lane_screening_score_option_1" && f.dataset === "passive")!;
  const markingBins = markingTightness.profile!.bins;
  const laneBins = laneScreening.profile!.bins;

  const confound = facts.confound_analysis as { tests: Array<{ name: string; verdict: string }> };
  const markingTest = confound.tests.find((t) => t.name.startsWith("marking_tightness_vs"))!;
  const laneTest = confound.tests.find((t) => t.name.startsWith("lane_screening_vs"))!;

  const zoneDefensiveValue = journey.features.find((f) => f.name === "zone_defensive_value" && f.dataset === "passive")!;

  return (
    <div className="flex flex-col gap-[22px] px-[88px] py-[34px] overflow-y-auto">
      <PageHeading
        eyebrow="Features"
        title="From every candidate to what's actually locked"
        subhead="Every feature that was ever considered, tracked stage by stage from feature_journey.json — not just the survivors."
      />

      <div className="grid grid-cols-2 gap-4">
        <Card>
          <b style={{ fontSize: 14 }}>Active-defence pipeline</b>
          <div className="mt-2">
            <StatBar
              stats={[
                { value: String(active.stage_counts.candidate_pool_stage01), label: "Candidates" },
                { value: String(active.stage_counts.final_locked), label: "Locked" },
                { value: String(activeModelled.length), label: "Modelled" },
                { value: String(active.stage_counts.excluded_before_count), label: "Excluded pre-pool" },
              ]}
              wrap
            />
          </div>
        </Card>
        <Card>
          <b style={{ fontSize: 14 }}>Passive-defence pipeline</b>
          <div className="mt-2">
            <StatBar
              stats={[
                { value: String(passive.stage_counts.candidate_pool_stage01), label: "Candidates" },
                { value: String(passive.stage_counts.final_locked), label: "Locked" },
                { value: String(passiveModelled.length), label: "Modelled" },
                { value: String(passive.stage_counts.excluded_before_count), label: "Excluded pre-pool" },
              ]}
              wrap
            />
          </div>
        </Card>
      </div>

      <Banner tone="wip">
        <b>{totalCandidates} candidates in the pool</b> ({active.stage_counts.candidate_pool_stage01} active + {passive.stage_counts.candidate_pool_stage01} passive),
        plus {totalEngineered} features engineered mid-pipeline and {totalExclusions} excluded before the pool was even formed —{" "}
        {journey.features.length} feature records in total, every one browsable below.
      </Banner>

      <section>
        <h3 style={{ fontSize: 16, marginBottom: 8 }}>The journey: candidates → locked</h3>
        <FeatureJourneyExplorer journey={journey} />
      </section>

      <section className="mt-2">
        <h3 style={{ fontSize: 16 }}>What actually held up</h3>
        <p style={{ fontSize: 13.5, color: "var(--muted)", marginBottom: 10 }}>
          Findings sourced only from analysis_facts.json&apos;s confound analysis and feature_journey.json&apos;s own
          per-feature profiles — not from the earlier (pre-fix) &quot;selection effect&quot; reading, which the
          confound analysis contradicts.
        </p>
        <div className="grid grid-cols-2 gap-3">
          <Card>
            <div className="mono" style={{ fontSize: 12.5, color: "var(--pitch)" }}>marking_tightness — a reversal that survives</div>
            <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>
              Tightest marking decile: {markingBins[0].shot_rate_pct}% near-term shot rate (n={markingBins[0].n.toLocaleString()}).
              Loosest decile: {markingBins[markingBins.length - 1].shot_rate_pct}% (n={markingBins[markingBins.length - 1].n.toLocaleString()}).
              Tighter marking correlates with a <b>higher</b>, not lower, shot rate — the opposite of a pure
              selection-effect reading.
            </p>
            <p className="mono" style={{ fontSize: 11, color: "var(--pitch)", marginTop: 8 }}>
              confound test &quot;{markingTest.name}&quot;: verdict = {markingTest.verdict} (not explained away by zone_defensive_value)
            </p>
          </Card>
          <Card>
            <div className="mono" style={{ fontSize: 12.5, color: "var(--pitch)" }}>lane_screening_score_option_1 — a reversal that survives</div>
            <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>
              Lowest-screening decile: {laneBins[0].shot_rate_pct}% shot rate (n={laneBins[0].n.toLocaleString()}).
              Highest-screening decile: {laneBins[laneBins.length - 1].shot_rate_pct}% (n={laneBins[laneBins.length - 1].n.toLocaleString()}).
              More screening of the top passing option correlates with a <b>higher</b> shot rate.
            </p>
            <p className="mono" style={{ fontSize: 11, color: "var(--pitch)", marginTop: 8 }}>
              confound test &quot;{laneTest.name}&quot;: verdict = {laneTest.verdict} (not explained away by top_option_1_threat_score)
            </p>
          </Card>
        </div>
        <Banner tone="blocked" >
          <b>zone_defensive_value is not a surviving signal.</b> It was dropped at {zoneDefensiveValue.stage_label.split(" (")[0]}
          {" "}— {zoneDefensiveValue.reason}. Earlier dashboard copy that listed it as a &quot;surviving U-shape&quot; predates this
          correction and is removed here.
        </Banner>
      </section>

      <Banner tone="pitch">
        No causal claims anywhere — every relationship here is &quot;correlates with,&quot; never &quot;causes.&quot;
      </Banner>
    </div>
  );
}
