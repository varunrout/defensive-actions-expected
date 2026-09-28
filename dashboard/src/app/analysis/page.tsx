import { PageHeading, StatBar, Card, Banner } from "@/components/ui";
import { getLegsSummary, getFeatureJourney, getAnalysisFacts, getModelLadders } from "@/lib/data";

export default function AnalysisPage() {
  const legs = getLegsSummary();
  const byGroup = (g: string) => legs.legs.filter((l) => l.group === g);

  const journey = getFeatureJourney();
  const facts = getAnalysisFacts().facts;
  const ladders = getModelLadders();

  const activeLocked = journey.features.filter((f) => f.dataset === "active" && f.fate === "locked");
  const activeModelled = activeLocked.filter((f) => f.modelled);
  const passiveLocked = journey.features.filter((f) => f.dataset === "passive" && f.fate === "locked");
  const passiveModelled = passiveLocked.filter((f) => f.modelled);

  const hurdle = ladders.legs.active_continuous.rungs.find((r) => r.name === "c1d_random_forest")!.hurdle_pipeline_headline!;

  const dupes = facts.exact_duplicate_pairs as { by_version: { v1_historical: { count: number } }; note: string };
  const activeCandBooleans = journey.stages.active.type_breakdown.candidate_stage01.boolean;
  const passiveCandBooleans = journey.stages.passive.type_breakdown.candidate_stage01.boolean;
  const tautologyNote = (facts.boolean_tautology as { note: string }).note;

  const v2 = (facts.correlation_tiers as {
    v2: { datasets: Record<string, { n_features: number; tier_counts: { collapse: number; review: number } }> };
  }).v2.datasets;
  const v2ReviewTotal = v2.active.tier_counts.review + v2.passive.tier_counts.review;
  const v2ActiveRisky = v2.active.tier_counts.collapse + v2.active.tier_counts.review;
  const v2PassiveRisky = v2.passive.tier_counts.collapse + v2.passive.tier_counts.review;

  const sanity = facts.football_sanity_check as { n_checks: number; checks: Array<{ check: string; n_rows_checked: number; known_result: string }>; note: string };
  const lockConfirmation = facts.feature_lock_confirmation as { verdict: string; diff_is_empty: boolean; any_newly_risky_pairs_found: boolean };

  const passes = [
    {
      title: "Redundancy & Correlation Atlas",
      text: "Type-matched methods throughout — Spearman, phi, point-biserial, Cramér's V, correlation ratio — because one Pearson matrix would misread a mixed continuous/boolean/categorical feature set.",
      stat: `${dupes.by_version.v1_historical.count} exact dupes (historical) · ${lockConfirmation.verdict}`,
    },
    {
      title: "Flag Ledgers",
      text: "Every boolean ranked by shot-rate lift. No verified tautology count exists in this repo's own analysis for either leg — a previously-stated tautology figure could not be substantiated here and is not repeated.",
      stat: `Active: ${activeCandBooleans} candidate booleans · Passive: ${passiveCandBooleans} candidate booleans`,
    },
    {
      title: "V2 Correlation Resolution",
      text: `A gap in the original framework, caught on rebuild: no test existed for continuous↔continuous pairs. Re-run under the V2 methodology across the (pre-lock) ${v2.active.n_features}/${v2.passive.n_features}-feature pool.`,
      stat: `${v2ReviewTotal} review pairs (V2) · ${v2ActiveRisky} active + ${v2PassiveRisky} passive collapse+review pairs`,
    },
    {
      title: "Football Sanity Check",
      text: "Match video wasn't available, so the substitute was independent reimplementation — each feature's formula rebuilt from scratch, from the docstring, and checked against the real stored output. Each check has its own row-count scope; they are not additive into one blanket total.",
      stat: sanity.checks.map((c) => `${c.check}: n=${c.n_rows_checked.toLocaleString()}`).join(" · "),
    },
    {
      title: "Passive Defence Archetypes",
      text: "KMeans clustering within each functional-role bucket, not across roles — role is already a validated signal, this asks what varies within one. No player identity anywhere in the clustering.",
      stat: `${(facts.passive_archetypes as { buckets_clustered: string[] }).buckets_clustered.length} buckets clustered`,
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
          { value: `${activeModelled.length} / ${activeLocked.length}`, label: "Active — modelled / locked" },
          { value: `${passiveModelled.length} / ${passiveLocked.length}`, label: "Passive — modelled / locked" },
          { value: `${dupes.by_version.v1_historical.count}`, label: "Exact-dupe pairs (historical)" },
          { value: sanity.n_checks.toString(), label: "Sanity checks (passive, own scopes)" },
        ]}
      />

      <Banner tone="wip">
        <b>c1d_random_forest hurdle-pipeline R² = {hurdle.value}</b> is a pre-coordinate-fix number
        (<code>pre_fix: {String(hurdle.pre_fix)}</code>) that was <b>not re-scored</b> (
        <code>rescored: {String(hurdle.rescored)}</code>) as part of the coordinate-frame fix — only the
        rung&apos;s own continuous-component metric was refit post-fix. Read it as pre-fix, not current.
      </Banner>

      <section>
        <h3 style={{ fontSize: 16, marginBottom: 10 }}>Bottom line</h3>
        <div className="grid grid-cols-2 gap-3">
          <Banner tone="pitch">
            <b>Both reversals are real, not artefacts.</b> Tighter marking → more shots, and more
            lane-screening → more shots — both survive confound-conditioning (see Features page).
            Genuine football signal, not a selection effect.
          </Banner>
          <Banner tone="wip">
            <b>Tournament, not football.</b> defenders_within_10m / _5m carry large raw signal but
            are confirmed interactive/substitutive with each other and with pitch position, not
            independent stable defensive patterns — need careful handling, not blind trust (see
            feature_interaction_analysis).
          </Banner>
          <Banner tone="blocked">
            <b>A leakage flag, caught early.</b> has_screened_outcome (passive) was dropped for
            leakage: chi2={(facts.leakage_audit as { raw: { part_c_has_screened_outcome: { chi2: number } } }).raw.part_c_has_screened_outcome.chi2.toFixed(2)},
            p={(facts.leakage_audit as { raw: { part_c_has_screened_outcome: { p_value: number } } }).raw.part_c_has_screened_outcome.p_value.toExponential(3)} — a
            censoring-mechanism proxy, not defensive signal.
          </Banner>
          <Banner tone="neutral">
            <b>{lockConfirmation.verdict}</b>{" "}
            No new DROP/COLLAPSE crossings found in the post-fix re-check
            (diff_is_empty={String(lockConfirmation.diff_is_empty)}).
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
                  {l.id === "active_continuous" && (
                    <span className="mono" style={{ fontSize: 11, color: "var(--wip)", marginLeft: 8 }}>
                      pre-fix
                    </span>
                  )}
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
              <p className="mono" style={{ fontSize: 11, color: "var(--pitch)", marginTop: 8 }}>{p.stat}</p>
            </Card>
          ))}
          <div className="banner neutral flex items-center" style={{ fontSize: 12.5 }}>
            No causal claims anywhere — every finding on this page is descriptive:
            &quot;correlates with,&quot; never &quot;causes.&quot;
          </div>
        </div>
        <p style={{ fontSize: 11, color: "var(--muted)", marginTop: 10 }}>{tautologyNote}</p>
      </section>
    </div>
  );
}
