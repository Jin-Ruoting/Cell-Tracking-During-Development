import csv
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
from watch_competition_submission import leaderboard_record, submission_record


class SubmissionObservationTests(unittest.TestCase):
    def test_pending_is_not_a_zero_score_and_reference_is_exact(self):
        text = 'ref,date,description,status,publicScore\n10,2026-09-22,"candidate, frozen",SubmissionStatus.PENDING,\n'
        self.assertIsNone(submission_record(text, 10)["public_score"])
        with self.assertRaises(ValueError):
            submission_record(text, 11)

    def test_complete_score_is_read(self):
        text = 'ref,date,description,status,publicScore\n10,2026-09-22,candidate,SubmissionStatus.COMPLETE,0.948\n'
        self.assertEqual(submission_record(text, 10)["public_score"], .948)

    def test_rank_controls_top_ten_percent_even_when_scores_tie(self):
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Rank", "TeamId", "TeamName", "Score"])
        for rank in range(1, 22):
            writer.writerow([rank, rank, "team-" + str(rank), .947])
        self.assertTrue(leaderboard_record(output.getvalue(), 2)["top_10_percent_reached"])
        outside = leaderboard_record(output.getvalue(), 3)
        self.assertFalse(outside["top_10_percent_reached"])
        self.assertEqual(outside["teams"], 21)
        self.assertEqual(outside["top_10_percent_cutoff_rank"], 2)


if __name__ == "__main__":
    unittest.main()
