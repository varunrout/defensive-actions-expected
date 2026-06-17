import unittest

from dax.models.attacking_threat import GridThreatModel, add_shot_in_10s_target


class AttackingThreatTests(unittest.TestCase):
    def test_target_labeling(self):
        rows = [
            {
                "match_id": 1,
                "period": 1,
                "minute": 10,
                "second": 0,
                "team_in_possession": "A",
                "event_type": "Pass",
            },
            {
                "match_id": 1,
                "period": 1,
                "minute": 10,
                "second": 6,
                "team_in_possession": "A",
                "event_type": "Shot",
            },
        ]

        out = add_shot_in_10s_target(rows)
        self.assertEqual(out[0]["target_future_shot_10s"], 1)

    def test_target_labeling_supports_type_column(self):
        rows = [
            {
                "match_id": 1,
                "period": 1,
                "minute": 10,
                "second": 0,
                "team_in_possession": "A",
                "type": "Pass",
            },
            {
                "match_id": 1,
                "period": 1,
                "minute": 10,
                "second": 6,
                "team_in_possession": "A",
                "type": "Shot",
            },
        ]

        out = add_shot_in_10s_target(rows)
        self.assertEqual(out[0]["target_future_shot_10s"], 1)

    def test_target_labeling_does_not_cross_match_boundary(self):
        rows = [
            {
                "match_id": 1,
                "period": 1,
                "minute": 89,
                "second": 55,
                "team_in_possession": "A",
                "type": "Pass",
            },
            {
                "match_id": 2,
                "period": 1,
                "minute": 0,
                "second": 2,
                "team_in_possession": "A",
                "type": "Shot",
            },
        ]

        out = add_shot_in_10s_target(rows)
        self.assertEqual(out[0]["target_future_shot_10s"], 0)

    def test_grid_model_scores_higher_in_positive_cell(self):
        rows = [
            {"ball_x": 100.0, "ball_y": 40.0, "target_future_shot_10s": 1},
            {"ball_x": 100.0, "ball_y": 42.0, "target_future_shot_10s": 1},
            {"ball_x": 20.0, "ball_y": 40.0, "target_future_shot_10s": 0},
            {"ball_x": 22.0, "ball_y": 38.0, "target_future_shot_10s": 0},
        ]
        model = GridThreatModel(n_x=12, n_y=8, smoothing=1.0).fit(rows)

        high = model.predict_point(100.0, 41.0)
        low = model.predict_point(21.0, 39.0)
        self.assertGreater(high, low)


if __name__ == "__main__":
    unittest.main()

