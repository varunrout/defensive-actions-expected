import unittest
from unittest.mock import patch

import pandas as pd

from dax.data.statsbomb_loader import build_enriched_events


class PipelineTests(unittest.TestCase):
    @patch("dax.data.statsbomb_loader.load_360_json")
    @patch("dax.data.statsbomb_loader.load_events")
    def test_build_enriched_events_adds_360_context(self, mock_load_events, mock_load_360_json):
        mock_load_events.return_value = pd.DataFrame(
            [
                {
                    "id": "event-1",
                    "location": [100.0, 35.0],
                    "possession_team": "Team A",
                    "event_type": "Pass",
                },
                {
                    "id": "event-2",
                    "location": [60.0, 40.0],
                    "possession_team": "Team B",
                    "event_type": "Carry",
                },
            ]
        )
        mock_load_360_json.return_value = [
            {
                "event_uuid": "event-1",
                "visible_area": [0, 0, 120, 0, 120, 80, 0, 80],
                "freeze_frame": [
                    {"teammate": True, "location": [102.0, 36.0]},
                    {"teammate": False, "location": [96.0, 34.0]},
                ],
            }
        ]

        enriched = build_enriched_events(
            match_id=1,
            competition_id=43,
            season_id=106,
            competition_label="World Cup 2022",
            home_team="Team A",
            away_team="Team B",
        )

        self.assertEqual(len(enriched), 2)
        self.assertEqual(enriched.loc[0, "ball_x"], 100.0)
        self.assertEqual(enriched.loc[0, "ball_y"], 35.0)
        self.assertEqual(enriched.loc[0, "freeze_frame_count"], 2)
        self.assertEqual(enriched.loc[0, "teammate_count"], 1)
        self.assertEqual(enriched.loc[0, "opponent_count"], 1)
        self.assertTrue(enriched.loc[0, "has_360"])
        self.assertEqual(enriched.loc[0, "defending_team"], "Team B")
        self.assertEqual(enriched.loc[1, "defending_team"], "Team A")
        self.assertTrue(enriched.loc[1, "visibility_limited"])

    @patch("dax.data.statsbomb_loader.load_events")
    def test_build_enriched_events_handles_competition_without_360(self, mock_load_events):
        mock_load_events.return_value = pd.DataFrame(
            [
                {
                    "id": "event-1",
                    "location": [25.0, 15.0],
                    "possession_team": "Team A",
                    "event_type": "Clearance",
                }
            ]
        )

        enriched = build_enriched_events(
            match_id=2,
            competition_id=55,
            season_id=43,
            competition_label="Euro 2020",
            home_team="Team A",
            away_team="Team B",
        )

        self.assertEqual(enriched.loc[0, "freeze_frame_count"], 0)
        self.assertFalse(enriched.loc[0, "has_360"])
        self.assertTrue(enriched.loc[0, "visibility_limited"])
        self.assertEqual(enriched.loc[0, "competition_label"], "Euro 2020")


if __name__ == "__main__":
    unittest.main()

