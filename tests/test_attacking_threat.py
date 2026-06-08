import unittest
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dax.models.attacking_threat import GridThreatModel, add_shot_in_10s_target


class AttackingThreatTests(unittest.TestCase):
    def test_target_labeling(self):
        rows = [
            {
                "period": 1,
                "minute": 10,
                "second": 0,
                "team_in_possession": "A",
                "event_type": "Pass",
            },
            {
                "period": 1,
                "minute": 10,
                "second": 6,
                "team_in_possession": "A",
                "event_type": "Shot",
            },
        ]

        out = add_shot_in_10s_target(rows)
        self.assertEqual(out[0]["target_shot_in_10s"], 1)

    def test_grid_model_scores_higher_in_positive_cell(self):
        rows = [
            {"ball_x": 100.0, "ball_y": 40.0, "target_shot_in_10s": 1},
            {"ball_x": 100.0, "ball_y": 42.0, "target_shot_in_10s": 1},
            {"ball_x": 20.0, "ball_y": 40.0, "target_shot_in_10s": 0},
            {"ball_x": 22.0, "ball_y": 38.0, "target_shot_in_10s": 0},
        ]
        model = GridThreatModel(n_x=12, n_y=8, smoothing=1.0).fit(rows)

        high = model.predict_point(100.0, 41.0)
        low = model.predict_point(21.0, 39.0)
        self.assertGreater(high, low)


if __name__ == "__main__":
    unittest.main()


