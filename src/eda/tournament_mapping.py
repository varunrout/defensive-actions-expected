"""Match-id -> tournament mapping, built directly from the raw StatsBomb
match lists this repo already downloaded (data/raw/matches/{competition_id}_
{season_id}.json), not hand-typed. Only the two tournaments actually in
scope (configs/competitions.yaml's TARGET_COMPETITIONS, minus Euro 2020
which src/dax/data/statsbomb_loader.py keeps commented out): World Cup 2022
(43/106) and Euro 2024 (55/282).

Verified: 64 WC2022 + 51 Euro2024 = 115 matches, zero overlap, covers every
match_id present in both feature parquets exactly once.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MATCHES_DIR = REPO_ROOT / "data" / "raw" / "matches"

TOURNAMENTS = {
    "WC2022": "43_106.json",
    "Euro2024": "55_282.json",
}


def load_match_tournament_map() -> dict[int, str]:
    mapping: dict[int, str] = {}
    for label, filename in TOURNAMENTS.items():
        matches = json.loads((MATCHES_DIR / filename).read_text(encoding="utf-8"))
        for m in matches:
            mapping[m["match_id"]] = label
    return mapping
