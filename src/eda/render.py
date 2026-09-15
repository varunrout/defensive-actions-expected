"""Shared HTML rendering for the three EDA report types.

Every output file is fully self-contained (inline CSS/JS, Google Fonts only).
Language discipline: generated finding text uses "correlates with" / "is
associated with" -- never causal language.
"""

from __future__ import annotations

import html

CSS = """
:root {
  --bg: #f4f5f1;
  --plane: #eceee8;
  --surface: #ffffff;
  --text-primary: #10151c;
  --text-secondary: #4c525b;
  --text-muted: #868d96;
  --border: rgba(16,21,28,0.09);
  --gridline: #e2e4de;
  --accent: #e34948;
  --pos: #e34948;
  --neg: #2a78d6;
  --amber: #eda100;
  --amber-wash: rgba(237,161,0,0.10);
  --good: #2f9e5b;
  --good-wash: rgba(47,158,91,0.10);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0d1117;
    --plane: #090c10;
    --surface: #161b22;
    --text-primary: #f3f5f7;
    --text-secondary: #b6bdc7;
    --text-muted: #7c8591;
    --border: rgba(255,255,255,0.09);
    --gridline: #232a34;
    --accent: #e66767;
    --pos: #e66767;
    --neg: #3987e5;
    --amber: #c98500;
    --amber-wash: rgba(201,133,0,0.14);
    --good: #3fbf78;
    --good-wash: rgba(63,191,120,0.14);
  }
}
:root[data-theme="dark"] {
  --bg: #0d1117;
  --plane: #090c10;
  --surface: #161b22;
  --text-primary: #f3f5f7;
  --text-secondary: #b6bdc7;
  --text-muted: #7c8591;
  --border: rgba(255,255,255,0.09);
  --gridline: #232a34;
  --accent: #e66767;
  --pos: #e66767;
  --neg: #3987e5;
  --amber: #c98500;
  --amber-wash: rgba(201,133,0,0.14);
  --good: #3fbf78;
  --good-wash: rgba(63,191,120,0.14);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text-primary);
  font-family: "Public Sans", -apple-system, sans-serif;
  font-size: 15px;
  line-height: 1.5;
}
.mono { font-family: "JetBrains Mono", monospace; }
.masthead {
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  padding: 40px 24px 32px;
}
.masthead-inner { max-width: 1180px; margin: 0 auto; }
.eyebrow {
  font-family: "JetBrains Mono", monospace;
  text-transform: uppercase;
  font-size: 12px;
  letter-spacing: 0.08em;
  color: var(--accent);
  font-weight: 600;
  margin: 0 0 10px;
}
h1.title {
  font-family: "Archivo", sans-serif;
  font-weight: 800;
  font-size: clamp(26px, 4vw, 38px);
  margin: 0 0 12px;
  color: var(--text-primary);
}
.dek {
  color: var(--text-secondary);
  max-width: 64ch;
  margin: 0 0 24px;
  font-size: 15.5px;
}
.statbar { display: flex; flex-wrap: wrap; gap: 28px 40px; }
.stat b {
  display: block;
  font-family: "Archivo", sans-serif;
  font-weight: 700;
  font-size: 22px;
  color: var(--text-primary);
}
.stat span {
  display: block;
  font-family: "JetBrains Mono", monospace;
  text-transform: uppercase;
  font-size: 11px;
  color: var(--text-muted);
  letter-spacing: 0.04em;
  margin-top: 2px;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 40px 24px 80px; }
h2.section-title {
  font-family: "Archivo", sans-serif;
  font-weight: 700;
  font-size: 21px;
  margin: 48px 0 4px;
}
.section-title:first-of-type { margin-top: 0; }
.section-note { color: var(--text-muted); font-size: 13.5px; margin: 0 0 20px; }
.findings {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin-bottom: 8px;
}
@media (max-width: 760px) { .findings { grid-template-columns: 1fr; } }
.finding {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 18px;
}
.finding.flag { background: var(--amber-wash); border-color: rgba(237,161,0,0.35); }
.tag {
  display: inline-block;
  font-family: "JetBrains Mono", monospace;
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  padding: 3px 8px;
  border-radius: 999px;
  background: rgba(42,120,214,0.12);
  color: var(--neg);
  font-weight: 600;
  margin-bottom: 8px;
}
.finding.flag .tag { background: rgba(237,161,0,0.18); color: var(--amber); }
.finding p { margin: 0; color: var(--text-secondary); font-size: 13.5px; }
.card-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}
@media (max-width: 980px) { .card-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 620px) { .card-grid { grid-template-columns: 1fr; } }
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px 20px;
}
.card h3 {
  font-family: "Archivo", sans-serif;
  font-size: 15.5px;
  margin: 0 0 4px;
}
.card .subnote { color: var(--text-muted); font-size: 12px; margin: 0 0 14px; }
.catbar-row { display: grid; grid-template-columns: 1fr; gap: 6px; margin-bottom: 10px; }
.catbar-label {
  display: flex; justify-content: space-between; font-size: 12.5px;
  color: var(--text-secondary);
}
.catbar-label .n { color: var(--text-muted); font-family: "JetBrains Mono", monospace; font-size: 11px; }
.catbar-track {
  position: relative; height: 10px; background: var(--gridline);
  border-radius: 5px; overflow: visible;
}
.catbar-fill { position: absolute; top: 0; left: 0; height: 100%; border-radius: 5px; background: var(--neg); }
.catbar-refline {
  position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--amber);
}
.rate-val { font-family: "JetBrains Mono", monospace; font-weight: 600; color: var(--text-primary); }
.ledger { display: flex; flex-direction: column; gap: 2px; }
.ledger-row {
  display: grid;
  grid-template-columns: 220px 1fr 200px;
  align-items: center;
  gap: 16px;
  padding: 10px 14px;
  border-radius: 8px;
  border-bottom: 1px solid var(--border);
}
.ledger-row:hover { background: var(--surface); }
@media (max-width: 760px) {
  .ledger-row { grid-template-columns: 1fr; gap: 6px; }
}
.ledger-label { font-size: 13.5px; }
.ledger-label .pct { color: var(--text-muted); font-family: "JetBrains Mono", monospace; font-size: 11.5px; display: block; }
.divaxis { position: relative; height: 20px; background: var(--gridline); border-radius: 4px; }
.divaxis .zero { position: absolute; left: 50%; top: -4px; bottom: -4px; width: 1px; background: var(--border); }
.divbar { position: absolute; top: 2px; bottom: 2px; border-radius: 3px; }
.divbar.pos { background: var(--pos); left: 50%; }
.divbar.neg { background: var(--neg); right: 50%; }
.ledger-figs { text-align: right; font-family: "JetBrains Mono", monospace; font-size: 12px; color: var(--text-secondary); }
.ledger-figs .lift { font-weight: 700; font-size: 14px; color: var(--text-primary); }
.small-n-badge {
  font-family: "JetBrains Mono", monospace; font-size: 10px; color: var(--amber);
  border: 1px solid var(--amber); border-radius: 4px; padding: 1px 5px; margin-left: 6px;
}
.hist { display: flex; align-items: flex-end; gap: 1.5px; height: 64px; margin-bottom: 10px; }
.hist .bin { flex: 1; background: var(--neg); border-radius: 2px 2px 0 0; min-height: 1px; }
.hist .bin.tail { background: var(--amber); }
.stat-row {
  display: flex; justify-content: space-between; font-family: "JetBrains Mono", monospace;
  font-size: 11.5px; color: var(--text-secondary); border-top: 1px solid var(--border);
  padding-top: 8px; margin-top: 8px;
}
.stat-row div b { display: block; color: var(--text-primary); font-size: 13px; }
.extreme-note { font-size: 11.5px; color: var(--amber); margin-top: 8px; }
footer {
  border-top: 1px solid var(--border);
  padding: 28px 24px 60px;
  max-width: 1180px;
  margin: 0 auto;
}
footer p { color: var(--text-muted); font-size: 12px; max-width: 780px; margin: 0 0 8px; }
footer ul { color: var(--text-muted); font-size: 12px; max-width: 780px; padding-left: 18px; margin: 0 0 8px; }
"""

ARTICLE_CSS = """
.article { max-width: 860px; }
.article > p:first-child {
  font-family: "JetBrains Mono", monospace;
  font-size: 12.5px;
  color: var(--text-muted);
  margin: 0 0 32px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--border);
}
.article h2 {
  font-family: "Archivo", sans-serif;
  font-weight: 700;
  font-size: 22px;
  margin: 44px 0 6px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}
.article h2:first-of-type { margin-top: 0; }
.article h3 {
  font-family: "Archivo", sans-serif;
  font-weight: 700;
  font-size: 17px;
  margin: 30px 0 4px;
  color: var(--accent);
}
.article h4 {
  font-family: "Public Sans", sans-serif;
  font-weight: 700;
  font-size: 14.5px;
  margin: 22px 0 8px;
}
.article p { color: var(--text-secondary); margin: 0 0 14px; font-size: 14.5px; }
.article strong { color: var(--text-primary); font-weight: 700; }
.article em { color: var(--text-secondary); }
.article code {
  font-family: "JetBrains Mono", monospace;
  font-size: 0.88em;
  background: var(--plane);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1px 5px;
  color: var(--neg);
}
.article ul, .article ol {
  color: var(--text-secondary);
  font-size: 14.5px;
  padding-left: 22px;
  margin: 0 0 16px;
}
.article li { margin-bottom: 6px; }
.article li > p { margin: 0; display: inline; }
.article hr { border: none; border-top: 1px solid var(--border); margin: 40px 0; }
.article table {
  width: 100%;
  border-collapse: collapse;
  margin: 4px 0 22px;
  font-size: 13px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
}
.article th {
  text-align: left;
  font-family: "JetBrains Mono", monospace;
  text-transform: uppercase;
  font-size: 10.5px;
  letter-spacing: 0.03em;
  color: var(--text-muted);
  background: var(--plane);
  padding: 9px 12px;
  border-bottom: 1px solid var(--border);
}
.article td {
  padding: 9px 12px;
  border-bottom: 1px solid var(--border);
  color: var(--text-secondary);
  vertical-align: top;
}
.article tr:last-child td { border-bottom: none; }
.article td:first-child, .article th:first-child { color: var(--text-primary); font-weight: 600; }
.article td code, .article th code { font-weight: 400; }
"""

CORR_CSS = """
.dataset-block { margin-top: 56px; padding-top: 36px; border-top: 2px solid var(--border); }
.dataset-block:first-of-type { margin-top: 0; padding-top: 0; border-top: none; }
.dataset-title {
  font-family: "Archivo", sans-serif;
  font-weight: 800;
  font-size: 26px;
  margin: 0 0 4px;
}
.dataset-substat { color: var(--text-muted); font-family: "JetBrains Mono", monospace; font-size: 12.5px; margin: 0 0 18px; }
.method-strip {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px;
  margin: 16px 0 40px;
}
@media (max-width: 980px) { .method-strip { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 560px) { .method-strip { grid-template-columns: 1fr; } }
.method-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
}
.method-card .m-name {
  font-family: "JetBrains Mono", monospace;
  font-weight: 600;
  font-size: 12.5px;
  color: var(--accent);
  margin-bottom: 6px;
  display: block;
}
.method-card p { margin: 0; font-size: 12.5px; color: var(--text-secondary); }
.tier-section { margin: 12px 0 44px; }
.tier-heading {
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: "Archivo", sans-serif;
  font-weight: 700;
  font-size: 18px;
  margin: 0 0 4px;
}
.tier-dot { width: 11px; height: 11px; border-radius: 50%; flex: none; }
.tier-dot.drop { background: var(--pos); }
.tier-dot.collapse { background: var(--amber); }
.tier-dot.review { background: var(--neg); }
.tier-dot.distinct { background: var(--good); }
.tier-note { color: var(--text-muted); font-size: 13px; margin: 0 0 14px; }
.corr-ledger { display: flex; flex-direction: column; border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
.corr-row {
  display: grid;
  grid-template-columns: 1fr 100px 160px 76px 96px;
  align-items: center;
  gap: 10px;
  padding: 9px 14px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  font-size: 12.5px;
}
.corr-row:last-child { border-bottom: none; }
.corr-row:hover { background: var(--plane); }
@media (max-width: 760px) {
  .corr-row { grid-template-columns: 1fr; gap: 4px; padding: 10px 14px; }
}
.corr-pair { color: var(--text-primary); font-weight: 600; }
.corr-pair .arrow { color: var(--text-muted); font-weight: 400; margin: 0 4px; }
.corr-method {
  font-family: "JetBrains Mono", monospace;
  font-size: 10.5px;
  text-transform: uppercase;
  color: var(--text-muted);
}
.corr-mag-track { position: relative; height: 8px; background: var(--gridline); border-radius: 4px; }
.corr-mag-fill { position: absolute; top: 0; left: 0; height: 100%; border-radius: 4px; }
.corr-mag-fill.drop { background: var(--pos); }
.corr-mag-fill.collapse { background: var(--amber); }
.corr-mag-fill.review { background: var(--neg); }
.corr-mag-fill.distinct { background: var(--good); }
.corr-value { font-family: "JetBrains Mono", monospace; font-weight: 600; text-align: right; }
.verdict-chip {
  justify-self: end;
  font-family: "JetBrains Mono", monospace;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  padding: 3px 9px;
  border-radius: 999px;
  font-weight: 700;
  white-space: nowrap;
}
.verdict-chip.drop { background: rgba(227,73,72,0.14); color: var(--pos); }
.verdict-chip.collapse { background: var(--amber-wash); color: var(--amber); }
.verdict-chip.review { background: rgba(42,120,214,0.12); color: var(--neg); }
.verdict-chip.distinct { background: var(--good-wash); color: var(--good); }
.closing-note {
  margin-top: 24px;
  background: var(--plane);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 18px 22px;
  font-size: 14px;
  color: var(--text-secondary);
}
.closing-note b { color: var(--text-primary); }

.rule-banner {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 4px solid var(--accent);
  border-radius: 8px;
  padding: 16px 20px;
  font-size: 14.5px;
  color: var(--text-secondary);
  margin: 0 0 8px;
}
.rule-banner b { color: var(--text-primary); }
.framework-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 20px 0 40px; }
@media (max-width: 760px) { .framework-grid { grid-template-columns: 1fr; } }
.framework-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; }
.framework-card .f-type {
  font-family: "JetBrains Mono", monospace;
  font-size: 11px;
  text-transform: uppercase;
  color: var(--accent);
  font-weight: 700;
  margin-bottom: 6px;
  display: block;
}
.framework-card h3 { font-family: "Archivo", sans-serif; font-size: 16px; margin: 0 0 8px; }
.framework-card p { font-size: 13.5px; color: var(--text-secondary); margin: 0 0 10px; }
.framework-card .example {
  font-size: 12.5px;
  background: var(--plane);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  color: var(--text-secondary);
}
.framework-card .example b { color: var(--text-primary); }

.hcall-panel {
  background: var(--amber-wash);
  border: 1px solid rgba(237,161,0,0.35);
  border-radius: 12px;
  padding: 20px 24px;
  margin: 20px 0 40px;
}
.hcall-panel > .hp-title {
  font-family: "Archivo", sans-serif;
  font-weight: 700;
  font-size: 17px;
  color: var(--text-primary);
  margin: 0 0 4px;
}
.hcall-panel > .hp-note { font-size: 13px; color: var(--text-secondary); margin: 0 0 16px; }
.hcall-item {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 10px;
  font-size: 13px;
}
.hcall-item:last-child { margin-bottom: 0; }
.hcall-item .hp-pair { font-weight: 700; color: var(--text-primary); margin-bottom: 4px; }
.hcall-item .hp-reason { color: var(--text-secondary); }

.table-scroll { overflow-x: auto; border-radius: 10px; border: 1px solid var(--border); }
.evidence-ledger { width: 100%; min-width: 560px; border-collapse: collapse; font-size: 12.5px; border: none; border-radius: 0; }
.evidence-ledger th {
  text-align: left;
  font-family: "JetBrains Mono", monospace;
  text-transform: uppercase;
  font-size: 10px;
  color: var(--text-muted);
  background: var(--plane);
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
}
.evidence-ledger td { padding: 8px 12px; border-bottom: 1px solid var(--border); color: var(--text-secondary); background: var(--surface); }
.evidence-ledger tr:last-child td { border-bottom: none; }
.evidence-ledger td:first-child { color: var(--text-primary); font-weight: 600; }

.path-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 14px 0; }
@media (max-width: 760px) { .path-grid { grid-template-columns: 1fr; } }
.path-card { border-radius: 10px; padding: 16px 18px; border: 1px solid var(--border); }
.path-card.do-now { background: var(--good-wash); border-color: rgba(47,158,91,0.35); }
.path-card.defer { background: var(--amber-wash); border-color: rgba(237,161,0,0.35); }
.path-card .p-label {
  font-family: "JetBrains Mono", monospace;
  text-transform: uppercase;
  font-size: 11px;
  font-weight: 700;
  margin-bottom: 8px;
  display: block;
}
.path-card.do-now .p-label { color: var(--good); }
.path-card.defer .p-label { color: var(--amber); }
.path-card p { font-size: 13.5px; color: var(--text-secondary); margin: 0; }
.recommendation {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 4px solid var(--good);
  border-radius: 8px;
  padding: 14px 18px;
  font-size: 13.5px;
  color: var(--text-secondary);
  margin-top: 6px;
}
.recommendation b { color: var(--text-primary); }
"""

CONFOUND_CSS = """
.test-block { margin-top: 48px; padding-top: 36px; border-top: 2px solid var(--border); }
.test-block:first-of-type { margin-top: 0; padding-top: 0; border-top: none; }
.test-title { font-family: "Archivo", sans-serif; font-weight: 800; font-size: 24px; margin: 0 0 6px; }
.test-subnote { color: var(--text-muted); font-size: 13px; margin: 0 0 24px; }
.qchart-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px 22px; margin-bottom: 20px; }
.qchart-card h4 { font-family: "Archivo", sans-serif; font-size: 14.5px; margin: 0 0 4px; }
.qchart-card .qc-note { color: var(--text-muted); font-size: 12px; margin: 0 0 16px; }
.qbar-row { display: flex; align-items: flex-end; gap: 14px; height: 140px; padding: 0 4px; }
.qbar-col { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 100%; }
.qbar-val { font-family: "JetBrains Mono", monospace; font-size: 12px; font-weight: 700; color: var(--text-primary); margin-bottom: 4px; }
.qbar-fill { width: 100%; max-width: 56px; background: var(--neg); border-radius: 4px 4px 0 0; min-height: 2px; }
.qbar-label { font-size: 10.5px; color: var(--text-muted); margin-top: 8px; text-align: center; line-height: 1.3; }
.qbar-n { font-family: "JetBrains Mono", monospace; font-size: 9.5px; color: var(--text-muted); }
.stratum-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
@media (max-width: 700px) { .stratum-grid { grid-template-columns: 1fr; } }
.stratum-card { background: var(--plane); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; }
.stratum-card h5 { font-family: "JetBrains Mono", monospace; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.02em; color: var(--accent); margin: 0 0 4px; }
.stratum-delta { font-size: 11.5px; color: var(--text-muted); margin: 0 0 12px; }
.stratum-delta b { color: var(--text-primary); }
.qbar-row.small { height: 90px; gap: 8px; }
.qbar-row.small .qbar-fill { max-width: 40px; }
.verdict-banner { border-radius: 10px; padding: 16px 20px; margin: 8px 0 28px; font-size: 14px; }
.verdict-banner.v-no { background: var(--good-wash); border: 1px solid rgba(47,158,91,0.35); }
.verdict-banner.v-yes { background: var(--amber-wash); border: 1px solid rgba(237,161,0,0.35); }
.verdict-banner.v-partially { background: rgba(42,120,214,0.10); border: 1px solid rgba(42,120,214,0.3); }
.verdict-banner b { color: var(--text-primary); }
.caveat-box { background: var(--amber-wash); border: 1px solid rgba(237,161,0,0.35); border-radius: 12px; padding: 20px 24px; margin-top: 32px; }
.caveat-box h3 { font-family: "Archivo", sans-serif; font-size: 16px; margin: 0 0 8px; }
.caveat-box ul { margin: 8px 0 0; padding-left: 20px; font-size: 13.5px; color: var(--text-secondary); }
.caveat-box li { margin-bottom: 4px; }
"""

VIF_CSS = """
.vif-dataset-block { margin-top: 48px; padding-top: 36px; border-top: 2px solid var(--border); }
.vif-dataset-block:first-of-type { margin-top: 0; padding-top: 0; border-top: none; }
.vif-chart { display: flex; flex-direction: column; gap: 6px; position: relative; padding: 8px 0; }
.vif-row { display: grid; grid-template-columns: 180px 1fr 70px; align-items: center; gap: 10px; font-size: 12px; }
.vif-label { color: var(--text-primary); font-weight: 600; text-align: right; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.vif-track { position: relative; height: 14px; background: var(--gridline); border-radius: 3px; }
.vif-fill { position: absolute; top: 0; left: 0; height: 100%; border-radius: 3px; background: var(--neg); }
.vif-fill.warn { background: var(--amber); }
.vif-fill.severe { background: var(--pos); }
.vif-value { font-family: "JetBrains Mono", monospace; font-weight: 700; }
.vif-threshold { position: absolute; top: -4px; bottom: -4px; width: 1px; background: var(--text-muted); opacity: 0.5; }
.vif-threshold-label { position: absolute; top: -18px; font-size: 9px; color: var(--text-muted); font-family: "JetBrains Mono", monospace; transform: translateX(-50%); }
.vif-legend { display: flex; gap: 20px; font-size: 11.5px; color: var(--text-muted); margin: 8px 0 20px; }
.vif-legend span { display: inline-flex; align-items: center; gap: 6px; }
.vif-legend .dot { width: 9px; height: 9px; border-radius: 50%; }
.cond-badge { display: inline-block; font-family: "JetBrains Mono", monospace; font-size: 12px; background: var(--plane); border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px; margin-bottom: 16px; }
.before-after { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }
@media (max-width: 700px) { .before-after { grid-template-columns: 1fr; } }
.ba-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; }
.ba-card h4 { font-family: "Archivo", sans-serif; font-size: 14px; margin: 0 0 12px; }
.ba-card.before h4 { color: var(--pos); }
.ba-card.after h4 { color: var(--good); }
"""

FONT_LINKS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&'
    "family=Public+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600"
    '&display=swap" rel="stylesheet">'
)


def esc(x) -> str:
    return html.escape(str(x))


def finding_card(tag: str, title: str, text: str, flag: bool = False) -> str:
    cls = "finding flag" if flag else "finding"
    return (
        f'<div class="{cls}"><span class="tag">{esc(tag)}</span>'
        f"<p><b>{esc(title)}</b> &mdash; {text}</p></div>"
    )


def findings_grid(cards: list[str]) -> str:
    return f'<div class="findings">{"".join(cards)}</div>'


def html_shell(*, eyebrow: str, title: str, dek: str, stats: list[tuple[str, str]], body: str, footer: str) -> str:
    stat_html = "".join(f'<div class="stat"><b>{esc(v)}</b><span>{esc(l)}</span></div>' for v, l in stats)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
{FONT_LINKS}
<style>{CSS}</style>
</head>
<body>
<div class="masthead"><div class="masthead-inner">
  <p class="eyebrow">{esc(eyebrow)}</p>
  <h1 class="title">{esc(title)}</h1>
  <p class="dek">{dek}</p>
  <div class="statbar">{stat_html}</div>
</div></div>
<div class="wrap">
{body}
</div>
<footer>
{footer}
</footer>
</body>
</html>"""


def render_article(*, eyebrow: str, title: str, dek: str, stats: list[tuple[str, str]], body_html: str, extra_css: str = "") -> str:
    """Render a long-form narrative report (prose + tables) in the same shared design system."""
    stat_html = "".join(f'<div class="stat"><b>{esc(v)}</b><span>{esc(l)}</span></div>' for v, l in stats)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
{FONT_LINKS}
<style>{CSS}{ARTICLE_CSS}{extra_css}</style>
</head>
<body>
<div class="masthead"><div class="masthead-inner">
  <p class="eyebrow">{esc(eyebrow)}</p>
  <h1 class="title">{esc(title)}</h1>
  <p class="dek">{dek}</p>
  <div class="statbar">{stat_html}</div>
</div></div>
<div class="wrap">
<div class="article">
{body_html}
</div>
</div>
</body>
</html>"""


def footer_html(dataset_cfg: dict) -> str:
    excluded_items = "".join(f"<li><b>{esc(c)}</b> &mdash; {esc(reason)}</li>" for c, reason in dataset_cfg["excluded"].items())
    return f"""
<p><b>Lift</b> = shot rate when a flag is True minus shot rate when it is False, in percentage points,
against the dataset base rate of {dataset_cfg['_base_rate']:.2f}%. <b>Skew</b> is the sample skewness
(scipy.stats.skew) of the full (unclipped) column. Continuous histograms use 24 equal-width bins from the
column minimum to its 99th percentile; rows above p99 are counted as "extreme" but never dropped from the
reported mean/median/skew. Boolean groups (True or False) with n &lt; 500 are flagged as small-n rather than
excluded. All findings describe correlation/association, never causation.</p>
<p><b>Excluded from this report:</b></p>
<ul>{excluded_items}</ul>
"""


# ---------------------------------------------------------------------------
# Category Atlas
# ---------------------------------------------------------------------------

def render_category_atlas(dataset_cfg: dict, base_rate: float, n_rows: int, tables: dict[str, list[dict]]) -> str:
    dataset_cfg = {**dataset_cfg, "_base_rate": base_rate}
    cards = []
    for col, rows in tables.items():
        max_rate = max([r["rate"] for r in rows] + [base_rate]) * 1.15 or 1.0
        bar_rows = []
        for r in rows:
            fill_pct = max(0.0, min(100.0, (r["rate"] / max_rate) * 100))
            ref_pct = max(0.0, min(100.0, (base_rate / max_rate) * 100))
            bar_rows.append(f"""
<div class="catbar-row" title="{esc(r['category'])}: n={r['n']}, shots={r['shots']}, rate={r['rate']:.2f}%">
  <div class="catbar-label"><span>{esc(r['category'])}</span><span class="n">n={r['n']}</span></div>
  <div class="catbar-track">
    <div class="catbar-fill" style="width:{fill_pct:.2f}%"></div>
    <div class="catbar-refline" style="left:{ref_pct:.2f}%"></div>
  </div>
  <span class="rate-val">{r['rate']:.2f}%</span>
</div>""")
        cards.append(f"""
<div class="card" style="margin-bottom:16px;">
<h3>{esc(col)}</h3>
<p class="subnote">shot rate by category &middot; dashed marker = dataset base rate ({base_rate:.2f}%)</p>
{''.join(bar_rows)}
</div>""")

    top_col, top_row = _top_categorical_signal(tables, base_rate)
    findings = [
        finding_card(
            "signal",
            f"{top_col}: {top_row['category']}",
            f"correlates with a {top_row['rate']:.2f}% shot rate (n={top_row['n']}), vs a {base_rate:.2f}% dataset base rate.",
        ),
    ]
    if dataset_cfg["key"] == "active":
        findings.append(
            finding_card(
                "data issue",
                "action_x / action_y duplicate ball_x / ball_y",
                "verified 100%-identical as of this pass -- a known unresolved data issue, so both columns are excluded from every report rather than silently double-counted.",
                flag=True,
            )
        )
    findings.append(
        finding_card(
            "confound",
            "Camera-visibility columns excluded",
            "visibility/coverage columns show real statistical lift, but that reflects camera coverage of crowded, dangerous zones -- not a defensive signal -- so they are excluded from every report.",
            flag=True,
        )
    )

    body = findings_grid(findings) + "\n<h2 class=\"section-title\">Category Atlas</h2>" \
        "<p class=\"section-note\">One card per categorical column. Bars = groupby-mean shot rate against the full row population per category (never filter-to-shots-then-distribute).</p>" \
        + "".join(cards)

    return html_shell(
        eyebrow=f"CATEGORY ATLAS · {dataset_cfg['label'].upper()}",
        title=f"{dataset_cfg['label']}: Category Atlas",
        dek=f"Shot rate by category across every categorical column, {esc(dataset_cfg['row_description'])}.",
        stats=[
            (f"{n_rows:,}", "rows"),
            (f"{base_rate:.2f}%", "base rate"),
            (str(len(tables)), "categorical columns"),
        ],
        body=body,
        footer=footer_html(dataset_cfg),
    )


def _top_categorical_signal(tables: dict[str, list[dict]], base_rate: float):
    best_col, best_row, best_gap = None, None, -1
    for col, rows in tables.items():
        for r in rows:
            if r["n"] < 30:
                continue
            gap = abs(r["rate"] - base_rate)
            if gap > best_gap:
                best_gap, best_col, best_row = gap, col, r
    return best_col, best_row


# ---------------------------------------------------------------------------
# Flag Ledger
# ---------------------------------------------------------------------------

def render_flag_ledger(dataset_cfg: dict, base_rate: float, n_rows: int, lifts: list[dict]) -> str:
    dataset_cfg = {**dataset_cfg, "_base_rate": base_rate}
    ranked = sorted(lifts, key=lambda r: abs(r["lift"]), reverse=True)
    max_abs_lift = max([abs(r["lift"]) for r in ranked] + [1.0])

    rows_html = []
    for r in ranked:
        pos_pct = max(0.0, min(50.0, (max(r["lift"], 0) / max_abs_lift) * 50))
        neg_pct = max(0.0, min(50.0, (max(-r["lift"], 0) / max_abs_lift) * 50))
        badge = '<span class="small-n-badge">small n</span>' if r["small_n"] else ""
        rows_html.append(f"""
<div class="ledger-row" title="True: {r['rate_true']:.2f}% (n={r['n_true']})  False: {r['rate_false']:.2f}% (n={r['n_false']})">
  <div class="ledger-label">{esc(r['column'])}{badge}<span class="pct">{r['pct_true']:.1f}% True</span></div>
  <div class="divaxis"><span class="zero"></span>
    <div class="divbar neg" style="width:{neg_pct:.2f}%"></div>
    <div class="divbar pos" style="width:{pos_pct:.2f}%"></div>
  </div>
  <div class="ledger-figs"><span class="lift">{r['lift']:+.2f}pp</span><br>T {r['rate_true']:.2f}% &middot; F {r['rate_false']:.2f}%</div>
</div>""")

    findings = []
    for r in ranked[:2]:
        findings.append(
            finding_card(
                "lift",
                r["column"],
                f"is associated with a {r['lift']:+.2f}pp shift in shot rate (True {r['rate_true']:.2f}% vs False {r['rate_false']:.2f}%).",
            )
        )
    small_n = [r for r in ranked if r["small_n"]]
    for r in small_n:
        findings.append(
            finding_card(
                "small n",
                r["column"],
                f"has a True or False group under 500 rows (n_true={r['n_true']}, n_false={r['n_false']}) -- lift estimate is noisy, interpret with caution.",
                flag=True,
            )
        )

    body = findings_grid(findings) + "\n<h2 class=\"section-title\">Flag Ledger</h2>" \
        "<p class=\"section-note\">Diverging bars ranked by absolute lift (percentage points), centered on zero.</p>" \
        f'<div class="ledger">{"".join(rows_html)}</div>'

    return html_shell(
        eyebrow=f"FLAG LEDGER · {dataset_cfg['label'].upper()}",
        title=f"{dataset_cfg['label']}: Flag Ledger",
        dek=f"Every boolean column ranked by how far its True/False shot rates diverge, {esc(dataset_cfg['row_description'])}.",
        stats=[
            (f"{n_rows:,}", "rows"),
            (f"{base_rate:.2f}%", "base rate"),
            (str(len(ranked)), "boolean columns"),
        ],
        body=body,
        footer=footer_html(dataset_cfg),
    )


# ---------------------------------------------------------------------------
# Distribution Atlas
# ---------------------------------------------------------------------------

def render_distribution_atlas(
    dataset_cfg: dict,
    base_rate: float,
    n_rows: int,
    continuous: dict[str, dict],
    discrete: dict[str, dict],
) -> str:
    dataset_cfg = {**dataset_cfg, "_base_rate": base_rate}
    cont_cards = [_continuous_card(col, d) for col, d in continuous.items()]
    disc_cards = [_discrete_card(col, d) for col, d in discrete.items()]

    most_skewed = max(continuous.items(), key=lambda kv: abs(kv[1]["skew"])) if continuous else None
    findings = []
    if most_skewed:
        col, d = most_skewed
        findings.append(
            finding_card(
                "shape",
                col,
                f"is the most skewed continuous feature (skew={d['skew']:.2f}); its p99-clipped view keeps the readable range legible.",
            )
        )
    extreme_cols = [(c, d) for c, d in continuous.items() if d["n_extreme"] > 0]
    if extreme_cols:
        col, d = max(extreme_cols, key=lambda kv: kv[1]["n_extreme"])
        findings.append(
            finding_card(
                "extremes",
                col,
                f"has {d['n_extreme']} rows above its p99 ({d['p99']:.2f}) up to a true max of {d['max']:.2f} -- these are counted, not dropped.",
                flag=True,
            )
        )

    body = findings_grid(findings)
    if cont_cards:
        body += '\n<h2 class="section-title">Continuous Distributions</h2>' \
            '<p class="section-note">24 equal-width bins from the column min to its 99th percentile. Rows above p99 are counted as extreme, never dropped.</p>' \
            f'<div class="card-grid">{"".join(cont_cards)}</div>'
    if disc_cards:
        body += '\n<h2 class="section-title">Discrete Distributions</h2>' \
            '<p class="section-note">Raw value counts, capped at 20 distinct values; any remaining tail is bucketed and marked.</p>' \
            f'<div class="card-grid">{"".join(disc_cards)}</div>'

    return html_shell(
        eyebrow=f"DISTRIBUTION ATLAS · {dataset_cfg['label'].upper()}",
        title=f"{dataset_cfg['label']}: Distribution Atlas",
        dek=f"One card per numerical column, {esc(dataset_cfg['row_description'])}.",
        stats=[
            (f"{n_rows:,}", "rows"),
            (f"{base_rate:.2f}%", "base rate"),
            (str(len(continuous)), "continuous"),
            (str(len(discrete)), "discrete"),
        ],
        body=body,
        footer=footer_html(dataset_cfg),
    )


def _continuous_card(col: str, d: dict) -> str:
    max_count = max(d["bin_counts"] + [1])
    bins_html = "".join(
        f'<div class="bin" style="height:{max(2, int(c / max_count * 64))}px" title="{c}"></div>'
        for c in d["bin_counts"]
    )
    extreme_note = (
        f'<div class="extreme-note">{d["n_extreme"]} rows above p99 ({d["p99"]:.2f}); true max {d["max"]:.2f}</div>'
        if d["n_extreme"] > 0 else ""
    )
    return f"""
<div class="card">
<h3>{esc(col)}</h3>
<p class="subnote">continuous &middot; p99-clipped histogram</p>
<div class="hist">{bins_html}</div>
<div class="stat-row">
  <div><b>{d['mean']:.2f}</b>mean</div>
  <div><b>{d['median']:.2f}</b>median</div>
  <div><b>{d['skew']:.2f}</b>skew</div>
</div>
{extreme_note}
</div>"""


def _discrete_card(col: str, d: dict) -> str:
    max_n = max([b["n"] for b in d["bars"]] + [1])
    bars_html = "".join(
        f'<div class="bin{" tail" if b["is_tail"] else ""}" style="height:{max(2, int(b["n"] / max_n * 64))}px" title="{esc(b["value"])}: {b["n"]}"></div>'
        for b in d["bars"]
    )
    return f"""
<div class="card">
<h3>{esc(col)}</h3>
<p class="subnote">discrete &middot; value counts{' (tail bucketed, amber)' if any(b['is_tail'] for b in d['bars']) else ''}</p>
<div class="hist">{bars_html}</div>
<div class="stat-row">
  <div><b>{d['mean']:.2f}</b>mean</div>
  <div><b>{d['median']:.2f}</b>median</div>
  <div><b>{d['n']:,}</b>n</div>
</div>
</div>"""
