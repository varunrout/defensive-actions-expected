import unittest

import pandas as pd

from dax.models.baseline_logistic import (  # noqa: E402
    VariantSpec,
    build_pipeline,
    grouped_cv_scores,
    prepare_xyg,
    resolve_columns,
)


class BaselineLogisticTests(unittest.TestCase):
    def test_baseline_variant_pipeline_smoke(self) -> None:
        # Uses the active-binary locked feature set (phase_label categorical,
        # event_type/play_pattern omitted for brevity in this smoke fixture;
        # action_zone/action_family/position_group/action_x/action_y/
        # distance_to_center_line are pre-lock/dropped columns and are no
        # longer part of any variant spec -- see feature_config.py's
        # ACTIVE['excluded'] for the drop reasons).
        df = pd.DataFrame(
            {
                "match_id": [1, 1, 2, 2, 3, 3],
                "target_future_shot_10s": [0, 1, 0, 1, 0, 1],
                "phase_label": ["high_press", "high_press", "settled_low_block", "settled_low_block", "transition_defence", "transition_defence"],
                "distance_to_attacking_box": [60, 58, 70, 68, 50, 48],
                "attacking_goal_centrality": [5, 4, 2, 3, 10, 9],
            }
        )

        spec = resolve_columns(
            df,
            VariantSpec(
                name="smoke_test_variant",
                categorical=["phase_label"],
                numeric=["distance_to_attacking_box", "attacking_goal_centrality"],
            ),
        )
        x, y, groups = prepare_xyg(df, spec)
        pipeline = build_pipeline(spec)
        out = grouped_cv_scores(x, y, groups, pipeline, n_splits=3)

        self.assertIn("roc_auc", out)
        self.assertIn("avg_precision", out)
        self.assertEqual(len(out["fold_metrics"]), 3)


if __name__ == "__main__":
    unittest.main()
