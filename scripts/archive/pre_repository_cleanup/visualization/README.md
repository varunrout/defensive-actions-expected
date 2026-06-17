# Visualization Scripts

Create visual outputs for data exploration and validation.

## Scripts

- **visualize_possessions.py** - Generate possession-level visualizations
  - Creates heatmaps showing phase transitions, phase mix by outcome
  - Generates sample possession sequence diagrams
  - Output: PNG charts in `outputs/validation/possessions/`

## Usage

```bash
python scripts/visualization/visualize_possessions.py
```

## Output

- `outputs/validation/possessions/possession_phase_transition_heatmap.png` - Phase change patterns
- `outputs/validation/possessions/possession_phase_mix_by_outcome.png` - Phase distribution (shot vs no-shot)
- `outputs/validation/possessions/sample_shot_possession_sequence.png` - Example scoring possession
- `outputs/validation/possessions/sample_no_shot_possession_sequence.png` - Example non-scoring possession
