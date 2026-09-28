// Copies the repo's dashboard_data/ (the real, versioned data export) into
// src/data/ so the Next.js app has a stable local read path. Run before dev/build.
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.join(__dirname, "..", "..", "dashboard_data");
const DEST = path.join(__dirname, "..", "src", "data");

function copyRecursive(src, dest) {
  const stat = fs.statSync(src);
  if (stat.isDirectory()) {
    fs.mkdirSync(dest, { recursive: true });
    for (const entry of fs.readdirSync(src)) {
      copyRecursive(path.join(src, entry), path.join(dest, entry));
    }
  } else {
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.copyFileSync(src, dest);
  }
}

const files = [
  "legs_summary.json",
  "methodology_steps.json",
  "selection_and_frame_audit.json",
  "match_explorer",
  "match_features",
  "feature_journey.json",
  "analysis_facts.json",
  "model_ladders.json",
];

for (const f of files) {
  const s = path.join(SRC, f);
  const d = path.join(DEST, f);
  if (!fs.existsSync(s)) {
    console.warn(`[sync-data] missing source: ${s}`);
    continue;
  }
  copyRecursive(s, d);
}
console.log(`[sync-data] synced dashboard_data/ -> dashboard/src/data/`);
