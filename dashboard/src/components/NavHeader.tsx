"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS = [
  { idx: "01", label: "Hero", href: "/" },
  { idx: "02", label: "Data", href: "/data" },
  { idx: "03", label: "Context", href: "/context" },
  { idx: "04", label: "Transform", href: "/transform" },
  { idx: "05", label: "Features", href: "/features" },
  { idx: "06", label: "Analysis", href: "/analysis" },
  { idx: "07", label: "Analyser", href: "/analyser" },
  { idx: "08", label: "Modelling", href: "/modelling" },
  { idx: "09", label: "Explorer", href: "/explorer" },
];

export default function NavHeader() {
  const pathname = usePathname();

  return (
    <header
      className="flex items-center justify-between px-10 h-16 border-b sticky top-0 z-50"
      style={{ background: "var(--surface)", borderColor: "var(--border)" }}
    >
      <div className="flex items-center gap-3">
        <div
          className="w-[26px] h-[26px] rounded-[6px] flex items-center justify-center text-white"
          style={{ background: "var(--pitch)", fontFamily: "var(--mono)", fontWeight: 600, fontSize: 11 }}
        >
          Dx
        </div>
        <span style={{ fontFamily: "var(--display)", fontWeight: 700, fontSize: 14.5 }}>
          Defensive Actions Expected
        </span>
      </div>

      <nav className="flex items-center gap-1">
        {TABS.map((tab) => {
          const active = pathname === tab.href;
          return (
            <Link
              key={tab.href}
              href={tab.href}
              className="flex flex-col items-center px-3 py-1.5 rounded-lg leading-tight"
              style={{
                background: active ? "var(--pitch-soft)" : "transparent",
              }}
            >
              <span
                className="mono"
                style={{
                  fontSize: 10,
                  color: active ? "var(--pitch)" : "var(--muted)",
                  fontWeight: active ? 600 : 400,
                }}
              >
                {tab.idx}
              </span>
              <span
                style={{
                  fontSize: 12.5,
                  color: active ? "var(--pitch)" : "var(--text)",
                  fontWeight: active ? 700 : 400,
                }}
              >
                {tab.label}
              </span>
            </Link>
          );
        })}
      </nav>

      <a href="https://github.com/varunrout" className="mono" style={{ fontSize: 12, color: "var(--muted)" }}>
        GitHub ↗
      </a>
    </header>
  );
}
