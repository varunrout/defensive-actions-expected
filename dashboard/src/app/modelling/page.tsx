import { PageHeading, Card, StatusBadge, Banner, type LadderStatus } from "@/components/ui";
import { getModelLadders } from "@/lib/data";

const LEG_ORDER = ["active_binary", "passive_binary", "active_continuous", "passive_continuous", "active_xt", "passive_xt"] as const;

export default function ModellingPage() {
  const ladders = getModelLadders();

  return (
    <div className="flex flex-col gap-6 px-[88px] py-[34px] overflow-y-auto">
      <PageHeading
        eyebrow="Modelling"
        title="From findings to models"
        subhead="The reasoning behind what got built and how — not the scoreboard. Full per-leg results live on the Analysis page."
      />

      <div className="grid grid-cols-2 gap-3">
        <Card>
          <b style={{ fontSize: 13.5 }}>What we&apos;re predicting</b>
          <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>Six legs, target defined once, applied consistently across every one.</p>
        </Card>
        <Card>
          <b style={{ fontSize: 13.5 }}>Why a two-stage architecture (xT legs)</b>
          <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>
            Two-stage means P(threat changes at all) × expected change given it does — not one model
            doing both jobs badly.
          </p>
        </Card>
        <Card>
          <b style={{ fontSize: 13.5 }}>Why ladder discipline</b>
          <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>
            Baseline → quadratic terms → random forest → gradient boosting. Rungs are skipped only on
            cited evidence — the c1c and d1c interaction rungs on the two continuous legs, each with
            an independently-derived rationale, not run for form&apos;s sake.
          </p>
        </Card>
        <Card>
          <b style={{ fontSize: 13.5 }}>Which features survived, and why</b>
          <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>Carried over directly from the redundancy/correlation review and collapse-tier resolution on the Transform page.</p>
        </Card>
      </div>

      <div>
        <h3 style={{ fontSize: 16, marginBottom: 4 }}>The ladder, per leg — every rung climbed, not just the one that won</h3>
        <p className="mono" style={{ fontSize: 11.5, color: "var(--muted)", marginBottom: 12 }}>
          Baseline → Quadratic/Interactions → Random Forest → Gradient Boosting
        </p>
        <div className="flex flex-col gap-4">
          {LEG_ORDER.map((legId) => {
            const leg = ladders.legs[legId];
            return (
              <Card key={legId}>
                <b style={{ fontSize: 14 }}>{leg.label}</b>
                <p style={{ fontSize: 11.5, color: "var(--muted)", marginTop: 4 }}>{leg.reference_model_status}</p>
                <div className="grid grid-cols-4 gap-2.5 mt-2.5">
                  {leg.rungs.map((r) => (
                    <div key={r.name} style={{ border: "1px solid var(--border)", borderRadius: 6, padding: "10px 12px", minWidth: 0 }}>
                      <div className="mono" style={{ fontSize: 11.5, marginBottom: 4, overflowWrap: "break-word" }}>{r.name}</div>
                      <StatusBadge status={r.status as LadderStatus} />
                      {r.headline_metric_name && r.headline_metric_value !== null && (
                        <div className="mono" style={{ fontSize: 12, color: "var(--pitch)", marginTop: 6 }}>
                          {r.headline_metric_name} {r.headline_metric_value}
                          {r.pre_fix && <span style={{ color: "var(--wip)", marginLeft: 6, fontSize: 10 }}>pre-fix</span>}
                        </div>
                      )}
                      <p style={{ fontSize: 11.5, color: "var(--muted)", marginTop: 6 }}>{r.note}</p>
                      {r.caveat && (
                        <p style={{ fontSize: 11, color: "var(--wip)", marginTop: 6, borderTop: "1px solid var(--border)", paddingTop: 6 }}>
                          {r.caveat}
                        </p>
                      )}
                      {r.hurdle_pipeline_headline && (
                        <p style={{ fontSize: 11, color: "var(--wip)", marginTop: 6, borderTop: "1px solid var(--border)", paddingTop: 6 }}>
                          Hurdle-pipeline {r.hurdle_pipeline_headline.metric}={r.hurdle_pipeline_headline.value} — pre_fix=
                          {String(r.hurdle_pipeline_headline.pre_fix)}, rescored={String(r.hurdle_pipeline_headline.rescored)}.
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </Card>
            );
          })}
        </div>
      </div>

      <div className="flex gap-4 flex-wrap mono" style={{ fontSize: 12 }}>
        <span><StatusBadge status="current_reference" /></span>
        <span><StatusBadge status="tried_not_promoted" /></span>
        <span><StatusBadge status="mixed" /></span>
        <span><StatusBadge status="superseded" /></span>
        <span><StatusBadge status="skipped_by_design" /> — not built at all, on cited evidence (distinct from a tried-and-rejected rung)</span>
      </div>

      <Banner tone="pitch">
        Every rung shown here is read live from model_ladders.json — the project&apos;s own ladder
        history, nothing invented to fill a grid. Full per-leg headline results (AUC / R² for all six,
        binary and continuous/xT kept as separate groups) live on the{" "}
        <a href="/analysis" style={{ fontWeight: 700 }}>
          Analysis page →
        </a>
        . This page is the reasoning, not the scoreboard.
      </Banner>
    </div>
  );
}
