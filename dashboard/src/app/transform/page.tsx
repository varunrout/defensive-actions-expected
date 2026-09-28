import { PageHeading, StatBar, Banner, Card } from "@/components/ui";
import { getMethodologySteps } from "@/lib/data";

export default function TransformPage() {
  const steps = getMethodologySteps();
  const dataStep = steps.steps.find((s) => s.id === "data")!;
  const viewsStep = steps.steps.find((s) => s.id === "two_views")!;
  const splitStep = steps.steps.find((s) => s.id === "split")!;

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
            { value: "4", label: "Relationship types" },
            { value: "21", label: "Pairs resolved" },
            { value: "9", label: "Dropped on evidence" },
            { value: "1", label: "Left as redesign" },
          ]}
        />
        <p style={{ fontSize: 14, color: "var(--muted)" }}>
          Correlation says two features move together — never which one, or whether either,
          matters to the target. High r earns a pair a second look; it never decides the outcome
          alone. Every drop traces to an actual shot-rate lift number, not the correlation
          coefficient.
        </p>
        <div className="grid grid-cols-2 gap-3">
          <Card>
            <b style={{ fontSize: 13.5 }}>01 · Boolean vs. its continuous parent</b>
            <p className="mono" style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>
              is_central_lane ↔ attacking_goal_centrality, r=0.854 → drop the boolean.
            </p>
          </Card>
          <Card>
            <b style={{ fontSize: 13.5 }}>02 · Boolean vs. boolean, same concept</b>
            <p className="mono" style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>
              is_deep_zone 7.222pp vs is_high_zone 7.224pp → both drop. −1.97pp vs +0.72pp
              despite r=0.847 → both kept.
            </p>
          </Card>
          <Card>
            <b style={{ fontSize: 13.5 }}>03 · Categorical vs. correlated continuous</b>
            <p className="mono" style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>
              phase_label ↔ distance features, r up to 0.857 — tactical label, not the same field
              → kept.
            </p>
          </Card>
          <Card>
            <b style={{ fontSize: 13.5 }}>04 · State-persistence pairs</b>
            <p className="mono" style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>
              phase_label ↔ phase_label_prev_event, r=0.65 → kept, not even really a redundancy
              question.
            </p>
          </Card>
        </div>
        <Banner tone="blocked">
          <b>The one that didn&apos;t resolve.</b> Passive-side attacking-option family (12 raw
          columns) — not one problem but two. The coordinate-frame issue is fixed now: target
          coordinates recast ball-relative (option_N_dx = target_x − ball_x) instead of absolute.
          The aggregation question (collapse ranks 2–3 or keep raw) is deferred until a baseline
          model&apos;s feature importance can say whether they pull real weight — not decided
          from correlation alone.
        </Banner>
      </section>

      <section className="flex flex-col gap-3">
        <h3 style={{ fontSize: 18 }}>06 · Collapse-tier resolution</h3>
        <StatBar
          stats={[
            { value: "5", label: "Clusters resolved" },
            { value: "15", label: "Dropped" },
            { value: "2", label: "Engineered" },
            { value: "2", label: "Blocked on redesign" },
          ]}
        />
        <p style={{ fontSize: 14, color: "var(--muted)" }}>
          &quot;Collapse&quot; was a placeholder, not a verdict — 30 pairs got that label without
          saying how. Every cluster resolves into one of two real operations, per-pair.
        </p>
        <div className="grid grid-cols-2 gap-3">
          <Card>
            <b style={{ fontSize: 13.5 }}>Drop to one — same entity, measured N times</b>
            <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>
              Goal-proximity cluster: 5 members, r/η ≥ 0.94. distance_to_attacking_box kept — its
              flag showed +11.99pp shot-rate lift vs ~7.2pp for the broader zone flags. Backed by
              evidence, not convenience. <b>5 → 1 kept, 4 dropped.</b>
            </p>
          </Card>
          <Card>
            <b style={{ fontSize: 13.5 }}>Merge — different entities that co-vary</b>
            <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>
              attacker_centroid ↔ defender_centroid, r=0.94–0.984. Merged into
              defender_attacker_gap_x/y — defensive compactness relative to the attacking shape.{" "}
              <b>4 raw → 0 kept, 4 dropped, 2 engineered.</b>
            </p>
          </Card>
        </div>
      </section>

      <section className="flex flex-col gap-3">
        <h3 style={{ fontSize: 18 }}>07 · Match-grouped split</h3>
        <StatBar
          stats={[
            { value: "115", label: "Matches both legs" },
            { value: `${splitStep.facts?.held_out_test_matches ?? 23}/92`, label: "Test / train+val" },
            { value: `${splitStep.facts?.cv_folds ?? 5}`, label: "StratifiedGroupKFold" },
            { value: "0", label: "Matches leaked" },
          ]}
        />
        <p style={{ fontSize: 14, color: "var(--muted)" }}>
          Possession-autocorrelation analysis found the target heavily clustered within
          possessions (ICC 0.27–0.49) but barely differing between matches (ICC 0.006–0.008) —
          grouping by match is nearly free in signal terms and the only grouping that fully rules
          out possession/event leakage. Splitting each leg independently was tried first: it put
          only 5 of 23 matches in common between the two legs&apos; test sets. <b>One shared
          split</b> — same 23 test matches, same 5 folds, for both active and passive — allows a
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
            {viewsStep.facts?.active_rows?.toLocaleString() ?? "56,068"} rows
          </div>
          <p style={{ fontSize: 13, color: "var(--muted)" }}>
            {dataStep.facts?.matches ?? 115} matches — one row per on-ball defensive event
          </p>
        </Card>
        <Card>
          <b style={{ fontSize: 14 }}>Passive-defence dataset</b>
          <div className="mono" style={{ fontSize: 22, color: "var(--pitch)", margin: "6px 0" }}>
            {viewsStep.facts?.passive_defender_rows?.toLocaleString() ?? "1,593,181"} rows
          </div>
          <p style={{ fontSize: 13, color: "var(--muted)" }}>
            {dataStep.facts?.matches ?? 115} matches — one row per (defender-slot, on-ball event)
            pair
          </p>
        </Card>
      </div>

      <div>
        <h3 style={{ fontSize: 16, marginBottom: 8 }}>Sample transformed row</h3>
        <table className="w-full mono" style={{ fontSize: 13, borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ color: "var(--muted)", textAlign: "left" }}>
              <th className="py-2 pr-4">defender_slot</th>
              <th className="py-2 pr-4">on_ball_event</th>
              <th className="py-2 pr-4">marking_tightness</th>
              <th className="py-2 pr-4">zone_defensive_value</th>
              <th className="py-2 pr-4">target_future_shot_10s</th>
            </tr>
          </thead>
          <tbody>
            {[
              ["4", "Pass", "2.1m", "0.031", "0"],
              ["6", "Carry", "5.8m", "0.047", "1"],
            ].map((row, i) => (
              <tr key={i} style={{ borderTop: "1px solid var(--border)" }}>
                {row.map((cell, j) => (
                  <td key={j} className="py-2 pr-4">
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
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
