import { PageHeading, StatBar, Banner, Card } from "@/components/ui";
import { getMethodologySteps, getFeatureJourney, getAnalysisFacts, getMatchFeatures } from "@/lib/data";

export default function TransformPage() {
  const steps = getMethodologySteps();
  const dataStep = steps.steps.find((s) => s.id === "data")!;
  const viewsStep = steps.steps.find((s) => s.id === "two_views")!;

  const journey = getFeatureJourney();
  const facts = getAnalysisFacts().facts;
  const feat = (name: string, dataset: "active" | "passive") =>
    journey.features.find((f) => f.name === name && f.dataset === dataset)!;

  const ct = facts.correlation_tiers as {
    v1_historical: { datasets: Record<string, { n_features: number; tier_counts: { drop: number; collapse: number; review: number; distinct: number } }> };
    current: { datasets: Record<string, { n_features: number; tier_counts: { drop: number; collapse: number; review: number; distinct: number } }> };
  };
  const v1Active = ct.v1_historical.datasets.active.tier_counts;
  const v1Passive = ct.v1_historical.datasets.passive.tier_counts;
  const curActive = ct.current.datasets.active.tier_counts;
  const curPassive = ct.current.datasets.passive.tier_counts;
  const collapseTotals = facts.collapse_pair_totals as { current_pipeline_active_plus_passive: number; v1_historical_active_plus_passive: number };

  // Stage-06 "drop raw option coordinates" — real ledger entries (passive only; active untouched at this stage).
  const stage6Passive = journey.stages.passive.ledger.find((s) => s.stage === 6)!;
  const stage6Active = journey.stages.active.ledger.find((s) => s.stage === 6)!;

  const splitMethodology = facts.split_methodology as {
    icc_ranges: { within_possession: string; between_match: string };
    split_23_92: { test_matches: number; train_val_matches: number; total_matches: number };
    overlap_5_of_23: { value: string };
    folds: number;
  };

  const clusters = facts.collapse_clusters as {
    cluster_5: { members: string[]; correlation: { metric: string; value: number }; decision: string; decision_date: string; evidence: string };
    note: string;
  };

  // Type 1 / Type 2 review-resolution examples — every number is the locked
  // per-feature `reason`/`evidence` field from feature_journey.json.
  const isCentralLane = feat("is_central_lane", "active");
  const isWideLane = feat("is_wide_lane", "active");
  const isDeepZone = feat("is_deep_zone", "active");
  const isHighZone = feat("is_high_zone", "active");
  const distanceToCenterLine = feat("distance_to_center_line", "active");
  const actionFamily = feat("action_family", "active");
  const positionGroup = feat("position_group", "active");

  // No UI fallback for these — methodology_steps.json is the source of truth and these
  // fields are always populated for the "data"/"two_views" steps; if one is ever missing,
  // throwing here fails the static build loudly instead of silently rendering a stale number.
  const requireFact = <T,>(value: T | undefined, label: string): T => {
    if (value === undefined) throw new Error(`transform page: missing required fact "${label}" in methodology_steps.json`);
    return value;
  };
  const matchCount = requireFact(dataStep.facts?.matches, "data.matches");
  const activeRows = requireFact(viewsStep.facts?.active_rows, "two_views.active_rows");
  const passiveDefenderRows = requireFact(viewsStep.facts?.passive_defender_rows, "two_views.passive_defender_rows");

  // Two real, non-degenerate rows from match_features/*.json — one active, one passive.
  const mf = getMatchFeatures("3938643");
  const activeRow = mf.events.find((e) => e.event_id === "d9f53cc5-22e9-49fc-b824-07313815960e")!;
  const passiveEvent = mf.events.find((e) => e.event_id === "c833c15b-f1f9-45f0-96af-9a9f53203820")!;
  const passiveDefender = passiveEvent.defenders!.find((d) => d.defender_slot_index === 0)!;

  return (
    <div className="flex flex-col gap-[26px] px-[88px] py-[34px] overflow-y-auto">
      <PageHeading
        eyebrow="Methodology"
        title="How raw becomes usable"
        subhead="Not just what happened at each stage — the actual decision framework, with the evidence behind each call."
      />

      <div className="flex items-center gap-3 mono" style={{ fontSize: 13, color: "var(--muted)" }}>
        <span>01 · Raw events &amp; 360 frames</span>
        <span>→</span>
        <span>02 · Row construction</span>
        <span>→</span>
        <span>03 · Geometric features</span>
      </div>

      <Banner tone="wip">
        <b>04 · Leakage checks — caught &amp; fixed.</b> Outcome targets must be computed
        per-row from that row&apos;s own anchor timestamp — never broadcast across a possession.
        Broadcasting a possession-level future window would leak later information into earlier
        rows. A regression test guards it.
      </Banner>

      <section className="flex flex-col gap-3">
        <h3 style={{ fontSize: 18 }}>05 · Correlation review</h3>
        <StatBar
          stats={[
            { value: `${v1Active.drop + v1Passive.drop}`, label: "Exact drops (candidate pool, historical)" },
            { value: `${collapseTotals.v1_historical_active_plus_passive}`, label: "Collapse pairs (historical, candidate pool)" },
            { value: `${v1Active.review + v1Passive.review}`, label: "Review pairs (historical, candidate pool)" },
            { value: `${collapseTotals.current_pipeline_active_plus_passive}`, label: "Collapse pairs (current, locked set)" },
          ]}
          wrap
        />
        <p style={{ fontSize: 12.5, color: "var(--muted)" }}>
          The {collapseTotals.v1_historical_active_plus_passive} figure is the original candidate-pool pass over all{" "}
          {v1Active.drop + v1Active.collapse + v1Active.review + v1Active.distinct + v1Passive.drop + v1Passive.collapse + v1Passive.review + v1Passive.distinct}{" "}
          evaluated pairs ({v1Active.collapse} active + {v1Passive.collapse} passive collapse-tier), from
          CORRELATION_ANALYSIS_V1_HISTORICAL.json — not the current pipeline&apos;s number. Re-run on just the{" "}
          {curActive.review + curActive.collapse + curPassive.review + curPassive.collapse}-pair final locked set today,
          the same check finds {collapseTotals.current_pipeline_active_plus_passive} collapse-tier pairs
          (CORRELATION_ANALYSIS.json) — citing {collapseTotals.v1_historical_active_plus_passive} without the
          &quot;historical&quot; qualifier is misleading.
        </p>
        <p style={{ fontSize: 14, color: "var(--muted)" }}>
          Correlation says two features move together — never which one, or whether either, matters to the target.
          High r earns a pair a second look; a real shot-rate lift number decides the outcome, not the coefficient.
        </p>
        <div className="grid grid-cols-2 gap-3">
          <Card>
            <b style={{ fontSize: 13.5 }}>Type 1 · Boolean vs. its continuous parent</b>
            <p className="mono" style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>
              {isCentralLane.name} vs {isCentralLane.evidence!.vs}, {isCentralLane.evidence!.metric}={isCentralLane.evidence!.value} → drop the boolean.
              <br />
              {isWideLane.name} vs {isWideLane.evidence!.vs}, {isWideLane.evidence!.metric}={isWideLane.evidence!.value} → drop the boolean.
            </p>
          </Card>
          <Card>
            <b style={{ fontSize: 13.5 }}>Type 2 · Zone flag vs. its OWN box flag</b>
            <p className="mono" style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>
              {isDeepZone.name}: {isDeepZone.reason}
              <br />
              {isHighZone.name}: {isHighZone.reason}
              <br />
              Each zone flag is compared to its own box flag — not to the other zone flag.
            </p>
          </Card>
          <Card>
            <b style={{ fontSize: 13.5 }}>Exact-duplicate drops (historical, Cramér&apos;s V / Spearman = 1.0)</b>
            <p className="mono" style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>
              {actionFamily.name} vs {actionFamily.evidence!.vs}, {actionFamily.evidence!.metric}={actionFamily.evidence!.value} — kept the more granular field.
              <br />
              {positionGroup.name} vs {positionGroup.evidence!.vs}, {positionGroup.evidence!.metric}={positionGroup.evidence!.value} — kept the more granular field.
              <br />
              {distanceToCenterLine.name} vs {distanceToCenterLine.evidence!.vs}, {distanceToCenterLine.evidence!.metric}={distanceToCenterLine.evidence!.value}.
            </p>
          </Card>
          <Card>
            <b style={{ fontSize: 13.5 }}>Raw option coordinates dropped at Stage 06</b>
            <p className="mono" style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>
              Passive: {stage6Passive.dropped.length} raw target coordinates dropped ({stage6Passive.dropped.join(", ")}).
              <br />
              Active: {stage6Active.dropped.length} dropped at this stage — the active leg has no raw option-coordinate columns.
            </p>
          </Card>
        </div>
      </section>

      <section className="flex flex-col gap-3">
        <h3 style={{ fontSize: 18 }}>06 · Collapse-tier resolution</h3>
        <p style={{ fontSize: 14, color: "var(--muted)" }}>
          &quot;Collapse&quot; was a placeholder, not a verdict. Four clusters resolve into drop-to-one or merge-to-engineered;
          a fifth cluster got a different answer entirely.
        </p>
        <div className="grid grid-cols-2 gap-3">
          <Card>
            <b style={{ fontSize: 13.5 }}>Clusters 1–4 — drop-to-one or merge</b>
            <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>
              {clusters.note}
            </p>
          </Card>
          <Card>
            <b style={{ fontSize: 13.5 }}>Cluster 5 — the one that wasn&apos;t collapsed</b>
            <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>
              {clusters.cluster_5.members.join(" ↔ ")}, {clusters.cluster_5.correlation.metric}={clusters.cluster_5.correlation.value}.{" "}
              Decision ({clusters.cluster_5.decision_date}): {clusters.cluster_5.decision}
            </p>
            <p style={{ fontSize: 11.5, color: "var(--muted)", marginTop: 6 }}>{clusters.cluster_5.evidence}</p>
          </Card>
        </div>
      </section>

      <section className="flex flex-col gap-3">
        <h3 style={{ fontSize: 18 }}>07 · Match-grouped split</h3>
        <StatBar
          stats={[
            { value: `${splitMethodology.split_23_92.total_matches}`, label: "Matches both legs" },
            { value: `${splitMethodology.split_23_92.test_matches}/${splitMethodology.split_23_92.train_val_matches}`, label: "Test / train+val" },
            { value: `${splitMethodology.folds}`, label: "StratifiedGroupKFold" },
            { value: "0", label: "Matches leaked" },
          ]}
        />
        <p style={{ fontSize: 14, color: "var(--muted)" }}>
          Possession-autocorrelation analysis found the target heavily clustered within
          possessions (ICC {splitMethodology.icc_ranges.within_possession}) but barely differing
          between matches (ICC {splitMethodology.icc_ranges.between_match}) —
          grouping by match is nearly free in signal terms and the only grouping that fully rules
          out possession/event leakage. Splitting each leg independently was tried first: it put
          only {splitMethodology.overlap_5_of_23.value} matches in common between the two
          legs&apos; test sets. <b>One shared split</b> — same {splitMethodology.split_23_92.test_matches}{" "}
          test matches, same {splitMethodology.folds} folds, for both active and passive — allows a
          direct active-vs-passive comparison on identical held-out matches, and is one artifact
          to maintain instead of two that quietly drift apart. Test matches are stratified on
          each match&apos;s combined shot-rate percentile rank across both legs, not either
          leg&apos;s raw rate alone.
        </p>
      </section>

      <div className="grid grid-cols-2 gap-4">
        <Card>
          <b style={{ fontSize: 14 }}>Active-defence dataset</b>
          <div className="mono" style={{ fontSize: 22, color: "var(--pitch)", margin: "6px 0" }}>
            {activeRows.toLocaleString()} rows
          </div>
          <p style={{ fontSize: 13, color: "var(--muted)" }}>
            {matchCount} matches — one row per on-ball defensive event
          </p>
        </Card>
        <Card>
          <b style={{ fontSize: 14 }}>Passive-defence dataset</b>
          <div className="mono" style={{ fontSize: 22, color: "var(--pitch)", margin: "6px 0" }}>
            {passiveDefenderRows.toLocaleString()} rows
          </div>
          <p style={{ fontSize: 13, color: "var(--muted)" }}>
            {matchCount} matches — one row per (defender-slot, on-ball event)
            pair
          </p>
        </Card>
      </div>

      <div>
        <h3 style={{ fontSize: 16, marginBottom: 8 }}>Sample transformed row</h3>
        <p style={{ fontSize: 12, color: "var(--muted)", marginBottom: 8 }}>
          Two real, non-degenerate rows from the synced match_features export (match 3938643): event{" "}
          <code>{activeRow.event_id}</code> (active) and event <code>{passiveEvent.event_id}</code>, defender slot{" "}
          {passiveDefender.defender_slot_index} (passive).
        </p>
        <table className="w-full mono" style={{ fontSize: 12.5, borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ color: "var(--muted)", textAlign: "left" }}>
              <th className="py-2 pr-4">phase</th>
              <th className="py-2 pr-4">event_type</th>
              <th className="py-2 pr-4">marking_tightness</th>
              <th className="py-2 pr-4">attacking_goal_centrality</th>
              <th className="py-2 pr-4">defenders_within_10m</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderTop: "1px solid var(--border)" }}>
              <td className="py-2 pr-4">active</td>
              <td className="py-2 pr-4">{String(activeRow.features.event_type)}</td>
              <td className="py-2 pr-4">—</td>
              <td className="py-2 pr-4">{Number(activeRow.features.attacking_goal_centrality).toFixed(3)}</td>
              <td className="py-2 pr-4">{String(activeRow.features.defenders_within_10m)}</td>
            </tr>
            <tr style={{ borderTop: "1px solid var(--border)" }}>
              <td className="py-2 pr-4">passive</td>
              <td className="py-2 pr-4">{String(passiveEvent.features.on_ball_event_type)}</td>
              <td className="py-2 pr-4">{Number(passiveDefender.features.marking_tightness).toFixed(2)}m</td>
              <td className="py-2 pr-4">{Number(passiveDefender.features.attacking_goal_centrality).toFixed(3)}</td>
              <td className="py-2 pr-4">—</td>
            </tr>
          </tbody>
        </table>
      </div>

      <Banner tone="pitch">
        <b>No causal claims anywhere.</b> Every column is an observed, descriptive signal —
        findings are framed as &quot;correlates with,&quot; never &quot;prevents&quot; or
        &quot;causes.&quot;
      </Banner>
    </div>
  );
}
