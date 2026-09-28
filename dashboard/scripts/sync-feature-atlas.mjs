// Copies the real per-feature decile-bin distributions (bin edges, n, shot_rate_pct)
// from reports/analysis/shot_target/*.json into src/data/feature_atlas/ so the
// Features page (05) can render real histograms instead of illustrative ones.
// Run before dev/build, same pattern as sync-data.mjs.
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.join(__dirname, "..", "..", "reports", "analysis", "shot_target");
const DEST = path.join(__dirname, "..", "src", "data", "feature_atlas");

const files = [
  ["active_numerical_target_atlas.json", "active.json"],
  ["passive_numerical_target_atlas.json", "passive.json"],
];

fs.mkdirSync(DEST, { recursive: true });

for (const [srcName, destName] of files) {
  const s = path.join(SRC, srcName);
  const d = path.join(DEST, destName);
  if (!fs.existsSync(s)) {
    console.warn(`[sync-feature-atlas] missing source: ${s}`);
    continue;
  }
  fs.copyFileSync(s, d);
}
console.log(`[sync-feature-atlas] synced reports/analysis/shot_target/ -> dashboard/src/data/feature_atlas/`);
