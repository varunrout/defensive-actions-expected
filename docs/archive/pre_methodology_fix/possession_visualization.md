# Possession Visualization Guide (360-only)

This project visualizes possessions using only events where `has_360 == True`.

## Outputs

Running `scripts/visualization/visualize_possessions.py` creates:

- `outputs/validation/possessions/possession_phase_transition_heatmap.png`
- `outputs/validation/possessions/possession_phase_mix_by_outcome.png`
- `outputs/validation/possessions/sample_shot_possession_sequence.png`
- `outputs/validation/possessions/sample_no_shot_possession_sequence.png`

## How to read them

### 1. Transition heatmap
- Rows = current defensive phase
- Columns = next defensive phase inside the same possession
- Darker cells = more common phase-to-phase switches
- Use it to see how defensive structure adapts as the possession progresses

### 2. Phase mix by outcome
- Compares the average phase composition of possessions that end with:
  - `Shot in 10s`
  - `No shot in 10s`
- This shows whether successful attacks are associated with different defensive-phase blends

### 3. Sample possession sequence panels
Each sample possession has three views:

- **Pitch trajectory**
  - Ball path through the possession
  - Point color = defensive phase at that event
  - Point size = visible opponent count from 360

- **Timeline strip**
  - X-axis = event order / elapsed time within the possession
  - Colored strip = phase label sequence
  - Black line = visible defender count

- **Freeze-frame snapshots**
  - Three snapshots: start, middle, end
  - Blue = attacking teammates
  - Red = defenders
  - Yellow = actor
  - Black = ball

## Interpretation

With this setup, a possession is not just "one phase". It becomes:

- a spatial story (where the ball moved),
- a phase story (how the defense changed),
- and a pressure story (how many defenders were visible around the action).

That means defensive phases are seen in relation to the possession as:

- **phase occupancy**: how much of the possession is spent in each phase,
- **phase transitions**: how often the defense switches shape or intent,
- **phase timing**: when the switch happens inside the sequence,
- **phase-pressure coupling**: whether certain phases coincide with more visible defenders.

## Important note

Because this workflow is intentionally restricted to `has_360 == True`, the result is best interpreted as **360-visible possession sequences** rather than full all-event possessions.
