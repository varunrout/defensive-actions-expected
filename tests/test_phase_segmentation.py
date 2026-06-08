import unittest
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dax.features.phase_segmentation import label_defensive_phases


class PhaseSegmentationTests(unittest.TestCase):
    def test_counterpress_after_turnover(self):
        rows = [
            {
                "minute": 1,
                "second": 0,
                "period": 1,
                "team_in_possession": "Team A",
                "event_type": "Pass",
                "ball_x": 60.0,
                "ball_y": 40.0,
            },
            {
                "minute": 1,
                "second": 3,
                "period": 1,
                "team_in_possession": "Team B",
                "event_type": "Pass",
                "ball_x": 58.0,
                "ball_y": 42.0,
            },
        ]

        labeled = label_defensive_phases(rows)
        self.assertEqual(labeled[1]["phase_label"], "counterpress_after_loss")

    def test_box_defence_from_clearance(self):
        rows = [
            {
                "minute": 2,
                "second": 10,
                "period": 1,
                "team_in_possession": "Team A",
                "event_type": "Clearance",
                "ball_x": 20.0,
                "ball_y": 40.0,
            }
        ]

        labeled = label_defensive_phases(rows)
        self.assertEqual(labeled[0]["phase_label"], "box_defence")


if __name__ == "__main__":
    unittest.main()


