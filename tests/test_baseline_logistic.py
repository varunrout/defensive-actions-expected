import sys
from pathlib import Path
import unittest

import pandas as pd

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dax.models.baseline_logistic import (  # noqa: E402
    build_pipeline,
    default_variant_specs,
    grouped_cv_scores,
    prepare_xyg,
    resolve_columns,
)


class BaselineLogisticTests(unittest.TestCase):
    def test_baseline_variant_pipeline_smoke(self) -> None:
        df = pd.DataFrame(
            {
                "match_id": [1, 1, 2, 2, 3, 3],
                "target_future_shot_10s": [0, 1, 0, 1, 0, 1],
                "phase_label": ["high_press", "high_press", "settled_low_block", "settled_low_block", "transition_defence", "transition_defence"],
                "action_zone": ["middle_third_center"] * 6,
                "action_family": ["pressure", "intervention", "pressure", "intervention", "pressure", "intervention"],
                "position_group": ["centre_back", "centre_back", "fullback_wingback", "fullback_wingback", "defensive_midfielder", "defensive_midfielder"],
                "action_x": [50, 52, 40, 42, 55, 58],
                "action_y": [35, 36, 42, 43, 30, 31],
                "nearest_goal_distance": [60, 58, 70, 68, 50, 48],
                "distance_to_center_line": [5, 4, 2, 3, 10, 9],
            }
        )

        spec = resolve_columns(df, default_variant_specs()[1])
        x, y, groups = prepare_xyg(df, spec)
        pipeline = build_pipeline(spec)
        out = grouped_cv_scores(x, y, groups, pipeline, n_splits=3)

        self.assertIn("roc_auc", out)
        self.assertIn("avg_precision", out)
        self.assertEqual(len(out["fold_metrics"]), 3)


if __name__ == "__main__":
    unittest.main()
