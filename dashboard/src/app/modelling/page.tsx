import { PageHeading, Card, StatusBadge, Banner } from "@/components/ui";

type Status = "promoted" | "blocked" | "mixed" | "reference";
interface Rung {
  name: string;
  status: Status;
  note: string;
}
interface Ladder {
  title: string;
  rungs: Rung[];
}

const LADDERS: Ladder[] = [
  {
    title: "Active · Binary",
    rungs: [
      { name: "v1_unweighted", status: "reference", note: "Baseline, unweighted logistic regression." },
      { name: "v1b/v1c", status: "mixed", note: "v1b quadratic rejected outright; v1c hand-built interactions + L1 promoted, held-out PR-AUC 0.3747." },
      { name: "v1d_random_forest", status: "mixed", note: "PR-AUC 0.4085 (+0.0338 vs v1c, p=0.0015) but ECE 0.0183 — ~1.8× v1c's, overconfident." },
      { name: "v1e_gradient_boosting", status: "promoted", note: "Matches/beats v1d on ranking, recovers calibration close to v1c's. Standing reference." },
    ],
  },
  {
    title: "Passive · Binary",
    rungs: [
      { name: "p1_unweighted", status: "reference", note: "Baseline, PR-AUC 0.1678 (WC2022) / 0.1821 (Euro2024)." },
      { name: "p1b/p1c", status: "mixed", note: "Improved monotonically, unlike active's v1b — p1c is the best linear candidate." },
      { name: "p1d_random_forest", status: "mixed", note: "Directionally ahead of p1c but not significant, p=0.275." },
      { name: "p1e_gradient_boosting_calibrated", status: "promoted", note: "PR-AUC 0.2162, clears p1c specifically (p=0.036). Calibrated variant beat the raw one on both ranking and calibration." },
    ],
  },
  {
    title: "Active · Continuous",
    rungs: [
      { name: "c1_lognormal_glm", status: "reference", note: "Baseline hurdle-style, E[xg|shot] on log scale." },
      { name: "c1c_systematic_interactions", status: "blocked", note: "Explicitly not built — skipped on cited evidence rather than run for form's sake." },
      { name: "c1d_random_forest", status: "promoted", note: "Beats GLM on log RMSE, 5/5 CV folds p=0.0126. distance_to_attacking_box jumped rank #21→#2." },
      { name: "c1e_gradient_boosting", status: "blocked", note: "Tried, not promoted — tested head-to-head vs c1d, didn't clear a real edge." },
    ],
  },
  {
    title: "Passive · Continuous",
    rungs: [
      { name: "d1_lognormal_glm", status: "promoted", note: "Never dislodged — still the reference after every later rung tried against it." },
      { name: "quadratic/interactions", status: "blocked", note: "Failed to beat GLM — first honest null." },
      { name: "random forest", status: "blocked", note: "Also failed — second honest null." },
      { name: "gradient boosting", status: "blocked", note: "Third independent attempt, third honest null — GLM stands after all three." },
    ],
  },
  {
    title: "Active · xT",
    rungs: [
      { name: "x1_two_stage_huber", status: "reference", note: "Baseline superseded, R² 0.1514 CV / 0.1519 held-out, extreme-bin predictions under-shoot 3×+." },
      { name: "x1b_quadratic", status: "blocked", note: "Tried, not promoted — x1c's promotion is measured against x1 directly, not x1b." },
      { name: "x1c_random_forest", status: "promoted", note: "R² 0.374 (corrected post coordinate-fix), replaced x1." },
      { name: "x1d_gradient_boosting", status: "mixed", note: "RMSE win but calibration regressed on tails (p=0.0031, Spearman held-out −0.0255) — x1c stayed." },
    ],
  },
  {
    title: "Passive · xT",
    rungs: [
      { name: "y1_two_stage", status: "reference", note: "Baseline superseded, systematic positive prediction bias +0.00278." },
      { name: "y1b_quadratic", status: "blocked", note: "Tried, not promoted — bias got worse: +0.00298." },
      { name: "y1c_random_forest", status: "promoted", note: "Bias essentially eliminated (−0.000018), R² 0.331 (corrected post coordinate-fix)." },
      { name: "y1d_gradient_boosting", status: "blocked", note: "Tried, not promoted — paired test not significant (p=0.0635; Wilcoxon p=0.125), held-out Spearman fell 0.3154→0.2732." },
    ],
  },
];

export default function ModellingPage() {
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
          <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>Shot-sequence-follows, then expected threat — not one model doing both jobs badly.</p>
        </Card>
        <Card>
          <b style={{ fontSize: 13.5 }}>Why ladder discipline</b>
          <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>Baseline → quadratic terms → random forest → gradient boosting. Nothing skips rungs.</p>
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
          {LADDERS.map((l) => (
            <Card key={l.title}>
              <b style={{ fontSize: 14 }}>{l.title}</b>
              <div className="grid grid-cols-4 gap-2.5 mt-2.5">
                {l.rungs.map((r) => (
                  <div key={r.name} style={{ border: "1px solid var(--border)", borderRadius: 6, padding: "10px 12px" }}>
                    <div className="mono" style={{ fontSize: 11.5, marginBottom: 4 }}>{r.name}</div>
                    <StatusBadge status={r.status} />
                    <p style={{ fontSize: 11.5, color: "var(--muted)", marginTop: 6 }}>{r.note}</p>
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </div>
      </div>

      <div className="flex gap-4 flex-wrap mono" style={{ fontSize: 12 }}>
        <span><StatusBadge status="promoted" /> Promoted</span>
        <span><StatusBadge status="blocked" /> Tried, not promoted</span>
        <span><StatusBadge status="mixed" /> Mixed / trade-off</span>
        <span><StatusBadge status="reference" /> Reference / skipped by design</span>
      </div>

      <Banner tone="pitch">
        Every rung shown here is a real, sourced step from the project&apos;s own ladder history —
        nothing invented to fill a grid. Full per-leg headline results (AUC / R² for all six,
        binary and continuous/xT kept as separate groups) live on the{" "}
        <a href="/analysis" style={{ fontWeight: 700 }}>
          Analysis page →
        </a>
        . This page is the reasoning, not the scoreboard.
      </Banner>
    </div>
  );
}
