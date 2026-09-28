import { PageHeading, Banner, Card } from "@/components/ui";

export default function DataPage() {
  return (
    <div className="flex flex-col gap-[18px] px-[88px] py-[34px] overflow-y-auto">
      <PageHeading eyebrow="The data" title="Two raw sources, before any transformation" />

      <div className="grid grid-cols-2 gap-4">
        <Card>
          <h3 style={{ fontSize: 16, marginBottom: 8 }}>Event data</h3>
          <p style={{ fontSize: 14, color: "var(--muted)" }}>
            Every on-ball action, timestamped and typed: Pass, Carry, Dribble, Shot, Pressure,
            Duel, Interception, Block, Clearance. Each carries player, team, minute, pitch
            location (x, y), duration — plus fields specific to its type (a pass has its own
            end-location and outcome; a shot has its own xG and technique).
          </p>
        </Card>
        <Card>
          <h3 style={{ fontSize: 16, marginBottom: 8 }}>360 tracking data</h3>
          <p style={{ fontSize: 14, color: "var(--muted)" }}>
            A freeze-frame attached to selected events: the x, y position of every player the
            broadcast camera could see at that instant, each flagged teammate / opponent /
            goalkeeper — plus a visible-area polygon marking how much of the pitch that camera
            actually covered.
          </p>
        </Card>
      </div>

      <Banner tone="pitch">
        <b>Scope:</b> 115 matches with 360 coverage — World Cup 2022 + Euro 2024, 44 national
        teams. International tournament football only, no club football.
      </Banner>

      <h3 style={{ fontSize: 16, marginTop: 8 }}>How granular this actually is</h3>
      <div className="grid grid-cols-3 gap-4">
        <Card>
          <b style={{ fontSize: 14 }}>Event-triggered, not continuous</b>
          <p style={{ fontSize: 13.5, color: "var(--muted)", marginTop: 6 }}>
            360 freeze-frames only exist at StatsBomb&apos;s tagged on-ball events — not a
            fixed-rate tracking feed running throughout the match.
          </p>
        </Card>
        <Card>
          <b style={{ fontSize: 14 }}>Visibility varies by frame</b>
          <p style={{ fontSize: 13.5, color: "var(--muted)", marginTop: 6 }}>
            Only players inside the broadcast camera&apos;s visible area are captured. That
            coverage differs shot to shot — which is exactly why visibility has to be tracked as
            its own signal downstream.
          </p>
        </Card>
        <Card>
          <b style={{ fontSize: 14 }}>Every event, richly typed</b>
          <p style={{ fontSize: 13.5, color: "var(--muted)", marginTop: 6 }}>
            The schema branches by event type — a pass&apos;s fields aren&apos;t a shot&apos;s
            fields. Nothing here is a flat, uniform table yet.
          </p>
        </Card>
      </div>

      <div className="mt-2">
        <h3 style={{ fontSize: 16, marginBottom: 4 }}>
          Sample raw event <span style={{ fontWeight: 400, color: "var(--muted)", fontSize: 13 }}>(illustrative — not pulled from the real data)</span>
        </h3>
        <table className="w-full mono" style={{ fontSize: 13, borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ color: "var(--muted)", textAlign: "left" }}>
              <th className="py-2 pr-4">type</th>
              <th className="py-2 pr-4">player</th>
              <th className="py-2 pr-4">team</th>
              <th className="py-2 pr-4">location (x, y)</th>
              <th className="py-2 pr-4">minute</th>
            </tr>
          </thead>
          <tbody>
            {[
              ["Pressure", "#4", "Defending team", "(88.2, 34.5)", "23"],
              ["Pass", "#8", "Attacking team", "(70.1, 40.2)", "23"],
              ["Interception", "#6", "Defending team", "(75.4, 38.9)", "23"],
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
    </div>
  );
}
