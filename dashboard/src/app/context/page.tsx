import { PageHeading, Banner, Card } from "@/components/ui";

export default function ContextPage() {
  return (
    <div className="flex flex-col gap-5 px-[88px] py-[34px] overflow-y-auto">
      <PageHeading eyebrow="Context" title={'What "defending well" actually means'} />

      <div className="grid grid-cols-3 gap-4">
        <Card>
          <b style={{ fontSize: 14 }}>What&apos;s a defensive action?</b>
          <p style={{ fontSize: 13.5, color: "var(--muted)", marginTop: 6 }}>
            A tackle. An interception. A block. Or simply standing in the right spot before the
            ball ever arrives — positioning that denies space without a single visible event.
          </p>
        </Card>
        <Card>
          <b style={{ fontSize: 14 }}>Active vs. passive defending</b>
          <p style={{ fontSize: 13.5, color: "var(--muted)", marginTop: 6 }}>
            Active defending is what makes the highlight reel — the tackle, the interception.
            Passive defending is invisible: body position that quietly removes a passing lane,
            with no event to point to.
          </p>
        </Card>
        <Card>
          <b style={{ fontSize: 14 }}>Why measure defense at all?</b>
          <p style={{ fontSize: 13.5, color: "var(--muted)", marginTop: 6 }}>
            Public football analytics is attack-heavy — xG is everywhere. Defensive value is
            comparatively underserved, and passive defending almost entirely unmeasured.
          </p>
        </Card>
      </div>

      <div className="flex gap-5 mt-2">
        <Card style={{ flex: 1 }}>
          <div className="flex items-center gap-2 mb-2">
            <span style={{ width: 8, height: 8, borderRadius: 999, background: "var(--active-marker)" }} />
            <b style={{ fontSize: 14 }}>Active — held-out events</b>
          </div>
          <div
            className="relative rounded-lg overflow-hidden mb-2"
            style={{ aspectRatio: "3 / 2", background: "var(--surface-2)" }}
          >
            <PitchOutline />
          </div>
          <p style={{ fontSize: 13, color: "var(--muted)" }}>
            A visible, on-ball event — something happens, and it&apos;s clearly attributable to
            one defender.
          </p>
        </Card>
        <Card style={{ flex: 1 }}>
          <div className="flex items-center gap-2 mb-2">
            <span style={{ width: 8, height: 8, borderRadius: 999, background: "var(--passive-marker)" }} />
            <b style={{ fontSize: 14 }}>Passive — held-out events</b>
          </div>
          <div
            className="relative rounded-lg overflow-hidden mb-2"
            style={{ aspectRatio: "3 / 2", background: "var(--surface-2)" }}
          >
            <PitchOutline />
          </div>
          <p style={{ fontSize: 13, color: "var(--muted)" }}>
            No event at all — the value is in the body shape blocking the lane before anything
            happens.
          </p>
        </Card>
      </div>
      <div className="mono text-center" style={{ fontSize: 11, color: "var(--muted)" }}>
        Illustrative held-out moments — not final match data.
      </div>

      <Banner tone="pitch">
        <b>The throughline across all six models: threat removed, not just events counted.</b>
      </Banner>
    </div>
  );
}

function PitchOutline() {
  return (
    <svg viewBox="0 0 120 80" className="absolute inset-0 w-full h-full">
      <rect x="2" y="2" width="116" height="76" fill="none" stroke="#dedad2" strokeWidth="0.6" />
      <line x1="60" y1="2" x2="60" y2="78" stroke="#dedad2" strokeWidth="0.6" />
      <circle cx="60" cy="40" r="9" fill="none" stroke="#dedad2" strokeWidth="0.6" />
      <rect x="2" y="22" width="14" height="36" fill="none" stroke="#dedad2" strokeWidth="0.6" />
      <rect x="104" y="22" width="14" height="36" fill="none" stroke="#dedad2" strokeWidth="0.6" />
    </svg>
  );
}
