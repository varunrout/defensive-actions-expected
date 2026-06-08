# DAx Data Pipeline Status

##  Current Progress

The pipeline is **actively downloading** all StatsBomb data for the three target tournaments.

### Stage 1: Raw Data Fetching (IN PROGRESS)

**Completed:**
- ✅ 1a: competitions.json loaded (3 competitions registered)
- ✅ 1b: matches loaded and saved
  - Euro 2020 (55-43): 51 matches
  - World Cup 2022 (43-106): 64 matches  
  - Euro 2024 (55-282): 51 matches
  - **Total: 166 matches**
-  1c: Events and 360 freeze-frames downloading
  - Events: ~62 / 166 downloaded
  - 360 frames: ~11 / ~100+ frames for WC 2022 + Euro 2024

### Data Directory Structure

```
data/
  raw/                           # Stage 1 outputs (fetching)
    ├── competitions.json
    ├── matches/
    │   ├── 43_106.json         (WC 2022 match list)
    │   ├── 55_282.json         (Euro 2024 match list)
    │   └── 55_43.json          (Euro 2020 match list)
    ├── events/                 (JSON files per match)
    │   ├── 3788741.json        ← Downloaded
    │   ├── 3788742.json        ← Downloaded
    │   └── ...
    └── three-sixty/            (360 frames, WC 2022 + Euro 2024 only)
        ├── 3857255.json        ← Downloaded
        └── ...
  
  processed/                     # Stage 2 outputs (awaiting Stage 1)
    (empty until Stage 1 complete)
    - Will contain:
      - events_enriched.parquet
      - events_with_phases.parquet
      - events_with_targets.parquet
      - summary.json
  
  features/                      # Stage 3 subdirs
    ├── player_context/
    ├── team_context/
    ├── threat_models/
    └── attributions/
  
  models/                        # Future model artifacts
  validation/                    # Future validation outputs
```

##  Next Steps

The pipeline will continue running in the background:

1. **Stage 1 completes** when all 166 event JSON files + 360 frames are downloaded
2. **Stage 2 begins** automatically: 
   - Build enriched event DataFrames
   - Add defensive phase labels
   - Add attacking threat targets
   - Save to Parquet
3. **Stage 3 completes**: Initialize feature engineering directories

##  Expected Final Output

Once complete, `data/processed/` will contain:

| File | Rows | Columns | Content |
|------|------|---------|---------|
| `events_enriched.parquet` | ~1.2M | 50+ | Raw enriched events with freeze-frame counts, team context |
| `events_with_phases.parquet` | ~1.2M | 51+ | Events + defensive phase labels (9 categories) |
| `events_with_targets.parquet` | ~1.2M | 52+ | Events + phases + `target_shot_in_10s` label (~2.5% positive rate) |
| `summary.json` | — | — | Metadata: row count, competition count, 360 availability, etc. |

##  How to Monitor

Check current progress:

```powershell
# Count downloaded raw files
$ec = @(Get-ChildItem "data\raw\events\" -ErrorAction SilentlyContinue).Count 
$ff = @(Get-ChildItem "data\raw\three-sixty\" -ErrorAction SilentlyContinue).Count
Write-Host "Events: $ec / 166,  360 frames: $ff"

# Check if Parquet files exist yet
Get-ChildItem "data\processed\*.parquet" -ErrorAction SilentlyContinue
```

##  Notes

- **No outputs to notebooks yet** — only raw and processed files
- **Data remains platform-agnostic** — JSON + Parquet are language/tool independent
- **Pipeline is idempotent** — safe to re-run without data loss
- **360 data** present only for:
  - World Cup 2022 (most matches have 360)
  - Euro 2024 (most matches have 360)
  - Euro 2020 has NO 360 data (0% availability)

##  If Pipeline Stalls

Monitor the background process:

```powershell
# List Python processes
Get-Process python | Select-Object Name, Id, CPU, Memory

# If hung: Stop and restart manually
Stop-Process -Name python -Force
.\.venv\Scripts\python.exe scripts\pipeline.py
```

---

**Started:** June 8, 2026
**Status:** Data download in progress (~37% complete)
**ETA:** depends on network; typically 5-15 minutes for 166 matches
