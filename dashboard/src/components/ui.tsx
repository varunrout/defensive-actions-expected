import type { ReactNode, CSSProperties } from "react";

export function Eyebrow({ children }: { children: ReactNode }) {
  return <div className="eyebrow">{children}</div>;
}

export function PageHeading({ eyebrow, title, subhead }: { eyebrow: string; title: ReactNode; subhead?: string }) {
  return (
    <div className="flex flex-col gap-2 mb-2">
      <Eyebrow>{eyebrow}</Eyebrow>
      <h2 style={{ fontSize: 28 }}>{title}</h2>
      {subhead && <p style={{ color: "var(--muted)", fontSize: 14.5, maxWidth: 760 }}>{subhead}</p>}
    </div>
  );
}

export function StatBar({ stats, wrap = false }: { stats: Array<{ value: string; label: string }>; wrap?: boolean }) {
  return (
    <div className="statbar" style={{ flexWrap: wrap ? "wrap" : "nowrap" }}>
      {stats.map((s, i) => (
        <div className="stat" key={i}>
          <span className="value">{s.value}</span>
          <span className="label">{s.label}</span>
        </div>
      ))}
    </div>
  );
}

export function Banner({ tone, children }: { tone: "pitch" | "wip" | "blocked" | "neutral"; children: ReactNode }) {
  return <div className={`banner ${tone}`}>{children}</div>;
}

export type LadderStatus = "current_reference" | "superseded" | "tried_not_promoted" | "mixed" | "skipped_by_design";

export function StatusBadge({ status }: { status: LadderStatus }) {
  const text: Record<LadderStatus, string> = {
    current_reference: "Current reference",
    superseded: "Superseded",
    tried_not_promoted: "Tried, not promoted",
    mixed: "Mixed / trade-off",
    skipped_by_design: "Skipped by design",
  };
  const cls: Record<LadderStatus, string> = {
    current_reference: "promoted",
    superseded: "reference",
    tried_not_promoted: "blocked",
    mixed: "mixed",
    skipped_by_design: "skipped",
  };
  return <span className={`status-badge ${cls[status]}`}>{text[status]}</span>;
}

export function Card({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return (
    <div className="card" style={style}>
      {children}
    </div>
  );
}
