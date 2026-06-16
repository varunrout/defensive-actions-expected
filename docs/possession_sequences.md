# Possession Sequence Enrichment

## Overview

This module extracts **possession-level features** from your event data to deeply understand how defensive phases evolve during attacking sequences.

### Key Innovation

Instead of labeling defensive phases at the **event level** (point-in-time), we now analyze defense at the **possession level** (narrative arc).

**Old approach:**
```
Event 1: phase = "settled_mid_block"
Event 2: phase = "high_press"
Event 3: phase = "box_defence"
→ Three independent labels
```

**New approach:**
```
Possession #1023:
  Events: 3 events across 8 seconds
  Phase trajectory: settled_mid_block → high_press → box_defence
  Pressure: 4 opponents (start) → 7 opponents (end) = 75% intensification
  Zone: midfield → attacking third
  Outcome: Shot in 10s? YES
  
→ Unified tactical narrative with spatial + temporal + pressure dynamics
```

---

## What Gets Extracted

### Per Possession

| Field | Type | Meaning |
|-------|------|---------|
| **possession_id** | str | Unique identifier (match_period_team_index) |
| **match_id** | int | Match identifier |
| **period** | int | Match period (1-2 normal, 3-4 ET) |
| **team_in_possession** | str | Attacking team |
| **defending_team** | str | Defending team |

### Possession Structure

| Field | Type | Meaning |
|-------|------|---------|
| **event_count** | int | How many events in this possession |
| **event_indices** | list[int] | Raw event row indices |
| **start_time** / **end_time** | int | Seconds in period |
| **duration** | int | Total possession time in seconds |

### Phase Trajectory

| Field | Type | Meaning |
|-------|------|---------|
| **phases** | list[str] | All defensive phases during possession |
| **phase_transitions** | list[tuple] | When defense changed tactics (from_phase, to_phase) |
| **phase_unique_count** | int | How many different phases did defense use |

**Example:**
```python
phases: ["settled_low_block", "high_press", "high_press", "box_defence"]
phase_transitions: [
    ("settled_low_block", "high_press"),
    ("high_press", "box_defence")
]
phase_unique_count: 3
```

### Ball Progression

| Field | Type | Meaning |
|-------|------|---------|
| **start_ball_x / start_ball_y** | float | Ball position at possession start |
| **end_ball_x / end_ball_y** | float | Ball position at possession end |
| **start_zone** | str | Pitch zone at start (e.g., "defensive_third_center") |
| **end_zone** | str | Pitch zone at end |
| **zones_visited** | list[str] | Sequential zones the ball moved through |

**Zones:**
```
X-axis: defensive_third (0-40) | middle_third (40-80) | attacking_third (80-120)
Y-axis: left_flank (0-20)     | center (20-60)        | right_flank (60-80)
Result: 9 zones (e.g., "attacking_third_right_flank")
```

### 360 Pressure Dynamics

| Field | Type | Meaning |
|-------|------|---------|
| **events_with_360** | int | How many events in possession had 360 data |
| **opponent_count_start** | int | Opponents visible at first 360 event |
| **opponent_count_end** | int | Opponents visible at last 360 event |
| **opponent_count_avg** | float | Average visible opponents throughout |
| **opponent_count_max** | int | Peak opponent count |
| **opponent_count_min** | int | Minimum opponent count |
| **opponent_count_decay** | float | (start - end) / start (0.0 = no change, 1.0 = all cleared) |
| **teammate_count_avg** | float | Average visible teammates |
| **teammate_count_max** | int | Peak teammate visibility |

**Interpretation:**
- `opponent_count_decay = 0.5` means defense lost 50% of their pressure midway
- `opponent_count_max = 10` means at least 10 defenders were visible at some point
- High decay + shot outcome = "Breaking down a pressing attack"

### Outcome

| Field | Type | Meaning |
|-------|------|---------|
| **has_shot_in_10s** | int (0/1) | Did this possession lead to a shot within 10 seconds? |

---

## Data Format

Output: **`data/features/possessions_with_360.parquet`**

Each row = one possession with 360 data

```python
import pandas as pd

df = pd.read_parquet("data/features/possessions_with_360.parquet")
print(f"Shape: {df.shape}")  # e.g., (85000, 30)
print(df.columns)
```

---

## Usage Examples

### Run Extraction

```bash
python scripts/extract_possessions.py
```

### Visualize Results

```bash
python scripts/visualize_possessions.py
```

This creates 360-only possession visuals in `outputs/validation/possessions/`:
- `possession_phase_transition_heatmap.png`
- `possession_phase_mix_by_outcome.png`
- `sample_shot_possession_sequence.png`
- `sample_no_shot_possession_sequence.png`

### Custom Analysis

```python
import pandas as pd

df = pd.read_parquet("data/features/possessions_with_360.parquet")

# Example 1: Which possessions led to shots?
shots = df[df['has_shot_in_10s'] == 1]
print(f"Shot rate: {len(shots) / len(df) * 100:.2f}%")

# Example 2: Defensive transitions before shots
shots_with_transitions = shots[shots['phase_transitions'].apply(len) > 0]
print(f"Shots from possessions with phase changes: {len(shots_with_transitions)}")

# Example 3: Pressure patterns
high_pressure = df[df['opponent_count_avg'] > 6]
print(f"High-pressure defense: {len(high_pressure)} possessions")
print(f"Shot rate under high pressure: {high_pressure['has_shot_in_10s'].mean() * 100:.2f}%")

# Example 4: Progressive possessions
progressive = df[df['start_zone'].str.contains('defensive')] & \
              df[df['end_zone'].str.contains('attacking')]
print(f"Ball progressed from defense to attack: {len(progressive)} times")
```

---

## Insights You Discover

### 1. Defensive Structure Dynamics
```
Q: "How does defense adapt during an attacking sequence?"
A: Look at `phase_transitions`
   - Frequent transitions = reactive defense
   - Few transitions = organized defense
```

### 2. Pressure Effectiveness
```
Q: "Did defensive pressure actually reduce the threat?"
A: Look at `opponent_count_decay`
   - High decay = defense got pulled apart
   - Low decay = sustained organization
```

### 3. Progressive Threats
```
Q: "Which zone progressions are most dangerous?"
A: Group by `zones_visited` and filter by `has_shot_in_10s == 1`
   - "defensive_third_center → middle_third_center → attacking_third_center"
   - = dangerous central progessions
```

### 4. Phase-Outcome Coupling
```
Q: "Which defensive phases struggle with which possession types?"
A: Compare `phases` vs `has_shot_in_10s`
   - Possessions with `high_press` only = 2% shot rate (good press)
   - Possessions with `settled_low_block` only = 5% shot rate (deep concedes)
```

### 5. 360 Validation
```
Q: "Do our phase labels match spatial reality?"
A: Correlation between:
   - `phases` (our labels) vs `opponent_count_*` (objective 360 data)
   - If high_press has opponent_count_avg=8, good alignment
   - If high_press has opponent_count_avg=2, mislabeled
```

---

## Integration with Model

### Feature Engineering

These possession-level features are **raw material** for ML:

```python
# Features from possessions
X = df[[
    'phase_unique_count',           # How many defensive tactics
    'opponent_count_decay',         # Pressure change
    'duration',                     # Time possessed
    'zones_visited_length',         # Distance traveled
    'event_count',                  # Possession complexity
    'events_with_360',              # Data quality
]]

# Target
y = df['has_shot_in_10s']

# Build defensive action expectancy model
model.fit(X, y)
```

### Interpretability

When model predicts "High defensive action needed":

```python
# You can trace back:
poss_id = sample['possession_id']
events_in_poss = events[events_indices.isin(df.loc[poss_id, 'event_indices'])]

# See:
# - What phases did defense attempt?
# - How many opponents were visible on 360?
# - Which zone did the threat come from?
# - How did pressure evolve?
```

---

## Data Quality Notes

- **Only possessions with 360 data** are included (WC 2022 + Euro 2024)
- **Euro 2020 is excluded** (no 360 data available)
- ~80-85% event coverage with 360 per match
- Each possession has at least one event with 360 frame

---

## Files

| File | Purpose |
|------|---------|
| `src/dax/features/possession_sequences.py` | Core possession extraction logic |
| `scripts/extract_possessions.py` | CLI to run extraction |
| `scripts/visualize_possessions.py` | Static 360-only possession visualizations |
| `data/features/possessions_with_360.parquet` | Output (created by extraction) |

---

## Next Steps

1. **Run extraction:** `python scripts/extract_possessions.py`
2. **Visualize:** `python scripts/visualize_possessions.py`
3. **Build features:** Use possession-level data to train models
4. **Validate:** Compare predictions against actual shot outcomes

---

## Technical Details

### Possession Definition
- Starts: First event of a team
- Ends: Just before first event of opposing team (turnover)
- Can be single event (e.g., turnover immediately after pass)
- Resets at match/period boundaries

### Phase Labels
9 possible defensive phases (see `phase_segmentation.py`):
- `counterpress_after_loss`
- `transition_defence`
- `box_defence`
- `settled_low_block`
- `settled_mid_block`
- `second_ball`
- `high_press`
- `wide_defending`
- `central_progression_defence`

### 360 Aggregation
For each possession, we track:
- Opponent count (# of visible opposing players on each 360 frame)
- Teammate count (# of visible own players)
- Derived: decay rate, max/min/avg counts

---

## Questions?

See docstrings in:
- `src/dax/features/possession_sequences.py`
- `scripts/extract_possessions.py`

