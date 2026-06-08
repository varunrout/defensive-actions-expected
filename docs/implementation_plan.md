# Defensive Actions Expected - Implementation Plan

This file tracks practical delivery from MVP to advanced DAx.

## Current status

- [x] Phase 1 framing translated into DAx formula and scope
- [x] Phase 2 baseline data ingestion for StatsBomb Open Data + 360 joins
- [x] Phase 3 rule-based phase segmentation
- [x] Phase 4 xT-style attacking threat baseline (`shot_in_10s` target)
- [ ] Phase 5 attacking option tree
- [ ] Phase 6 player defensive features
- [ ] Phase 7 team defensive features
- [ ] Phase 8 suppression model
- [ ] Phase 9 attribution framework
- [ ] Phase 10 phase-specific models
- [ ] Phase 11 validation suite and case studies
- [ ] Phase 12 dashboard and storytelling outputs
- [ ] Phase 13 portfolio packaging

## MVP objective (now)

Build reproducible tables with these columns:

- `match_id`, `possession_id`, `event_id`, `timestamp`
- `team_in_possession`, `defending_team`
- ball location (`ball_x`, `ball_y`)
- `event_type`, `event_outcome`
- 360 freeze-frame counts and visibility flags
- defensive `phase_label`
- attacking baseline `threat_base_score`

## Next implementation step

Phase 5 option tree prototype:

1. Enumerate candidate options by zone and body orientation proxies.
2. Score option feasibility using lane openness and pressure proxies.
3. Attach expected threat to each branch.
4. Add branch-level responsibility placeholders for attribution.

