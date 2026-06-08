"""Quick validation of pipeline structure."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import warnings
warnings.filterwarnings("ignore")

from dax.data.statsbomb_loader import (
    load_target_competitions,
    load_matches,
    TARGET_COMPETITIONS,
    COMPETITIONS_WITH_360,
)

print("="*70)
print("DAx Pipeline Validation")
print("="*70)

# Test 1: Load competitions
print("\n1. Loading target competitions...")
try:
    comps = load_target_competitions()
    print(f"   OK: {len(comps)} competitions loaded")
    for _, row in comps.iterrows():
        print(f"     - {row['competition_name']} {row['season_name']}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

# Test 2: Load matches per competition
print("\n2. Loading matches...")
try:
    total_matches = 0
    for comp in TARGET_COMPETITIONS:
        cid = comp["competition_id"]
        sid = comp["season_id"]
        label = comp["label"]
        matches = load_matches(cid, sid)
        print(f"   {label}: {len(matches)} matches")
        total_matches += len(matches)
    print(f"   Total: {total_matches} matches")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

# Test 3: Check directory structure
print("\n3. Checking data directories...")
data_dirs = [
    REPO_ROOT / "data" / "raw",
    REPO_ROOT / "data" / "processed",
    REPO_ROOT / "data" / "features",
]
for d in data_dirs:
    d.mkdir(parents=True, exist_ok=True)
    print(f"   {d.relative_to(REPO_ROOT)}/: OK")

# Test 4: Verify 360 availability
print("\n4. 360 freeze-frame availability:")
print(f"   Competitions with 360: {COMPETITIONS_WITH_360}")

print("\n" + "="*70)
print("SUCCESS: All validations passed!")
print("="*70)
print("\nPipeline is ready. Run:")
print(r"  .\.venv\Scripts\python.exe scripts\pipeline.py")
