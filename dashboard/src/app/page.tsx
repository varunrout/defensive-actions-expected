import Link from "next/link";
import { Eyebrow } from "@/components/ui";

export default function HeroPage() {
  const events = [
    { id: "a", x: 22, y: 30, v: "+0.014" },
    { id: "b", x: 55, y: 20, v: "+0.006" },
    { id: "c", x: 70, y: 55, v: "+0.011" },
    { id: "d", x: 40, y: 65, v: "+0.002" },
    { id: "e", x: 85, y: 40, v: "+0.004" },
  ];

  return (
    <div className="flex-1 flex items-center px-[88px] gap-14">
      <div className="flex flex-col gap-5" style={{ maxWidth: 660 }}>
        <Eyebrow>Portfolio case study · Football analytics</Eyebrow>
        <h1 style={{ fontSize: 50, lineHeight: 1.08 }}>
          Most football analytics measures attack.
          <br />
          This measures defense.
        </h1>
        <p style={{ fontSize: 17.5, color: "var(--muted)" }}>
          Every tackle, block and off-ball position — scored for the danger it actually
          prevented. Six models, two phases of defending, evaluated against held-out World Cup
          and Euro data.
        </p>
        <div className="flex gap-3 mt-2">
          <Link
            href="/transform"
            className="px-5 py-2.5 rounded-lg text-white"
            style={{ background: "var(--pitch)", fontWeight: 600, fontSize: 14 }}
          >
            Start the story →
          </Link>
          <Link
            href="/explorer"
            className="px-5 py-2.5 rounded-lg"
            style={{ border: "1px solid var(--border)", fontWeight: 600, fontSize: 14 }}
          >
            Jump to Match Explorer →
          </Link>
        </div>
        <div className="flex items-center gap-5 mt-4 mono" style={{ fontSize: 13 }}>
          <span><b>6</b> models</span>
          <span style={{ color: "var(--border)" }}>|</span>
          <span><b>2</b> phases</span>
          <span style={{ color: "var(--border)" }}>|</span>
          <span><b>1.59M</b> rows (passive)</span>
          <span style={{ color: "var(--border)" }}>|</span>
          <span>WC2022 + Euro2024</span>
        </div>
      </div>

      <div
        className="relative flex-shrink-0 rounded-xl overflow-hidden"
        style={{ width: 480, height: 320, background: "var(--surface)", border: "1px solid var(--border)" }}
      >
        <svg viewBox="0 0 120 80" className="absolute inset-0 w-full h-full">
          <rect x="2" y="2" width="116" height="76" fill="none" stroke="#dedad2" strokeWidth="0.6" />
          <line x1="60" y1="2" x2="60" y2="78" stroke="#dedad2" strokeWidth="0.6" />
          <circle cx="60" cy="40" r="9" fill="none" stroke="#dedad2" strokeWidth="0.6" />
          <rect x="2" y="22" width="14" height="36" fill="none" stroke="#dedad2" strokeWidth="0.6" />
          <rect x="104" y="22" width="14" height="36" fill="none" stroke="#dedad2" strokeWidth="0.6" />
          <rect x="2" y="32" width="6" height="16" fill="none" stroke="#dedad2" strokeWidth="0.6" />
          <rect x="112" y="32" width="6" height="16" fill="none" stroke="#dedad2" strokeWidth="0.6" />
        </svg>
        {events.map((e) => (
          <div
            key={e.id}
            className="absolute flex items-center gap-1.5"
            style={{ left: `${e.x}%`, top: `${e.y}%` }}
          >
            <span style={{ width: 6, height: 6, borderRadius: 999, background: "var(--marker, #a9631a)" }} />
            <span
              className="mono"
              style={{
                fontSize: 10.5,
                padding: "2px 6px",
                borderRadius: 999,
                background: "var(--surface)",
                border: "1px solid var(--border)",
              }}
            >
              DAx {e.v}
            </span>
          </div>
        ))}
        <div
          className="absolute bottom-2 left-2 mono"
          style={{ fontSize: 9.5, color: "var(--muted)" }}
        >
          Illustrative held-out events — not final match_explorer data
        </div>
      </div>
    </div>
  );
}
