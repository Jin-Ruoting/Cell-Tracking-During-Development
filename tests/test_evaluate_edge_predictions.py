from __future__ import annotations

import unittest

from scripts.evaluate_edge_predictions import edge_counts, score_counts


class EdgePredictionEvaluationTest(unittest.TestCase):
    def test_edge_counts_match_only_unique_ground_truth_edges(self) -> None:
        predicted_edges = [(10, 11), (10, 11), (20, 21), (30, 31)]
        ground_truth_edges = [(1, 2), (3, 4), (5, 6)]
        mapping = {
            10: 1,
            11: 2,
            20: 3,
            21: 6,
            30: 99,
            31: 100,
        }

        counts = edge_counts(
            predicted_edges,
            ground_truth_edges,
            mapping,
        )

        self.assertEqual(counts, (1, 2, 2))

    def test_unmatched_edge_without_gt_context_is_not_false_positive(self) -> None:
        counts = edge_counts(
            [(10, 11)],
            [(1, 2)],
            {},
        )

        self.assertEqual(counts, (0, 0, 1))

    def test_adjusted_edge_jaccard_matches_node_penalty_formula(self) -> None:
        result = score_counts(
            8,
            1,
            1,
            predicted_nodes=110,
            estimated_total_nodes=100,
        )

        self.assertAlmostEqual(result["edge_jaccard"], 0.8)
        self.assertAlmostEqual(result["node_delta_ratio"], 0.1)
        self.assertAlmostEqual(result["adjusted_edge_jaccard"], 0.792)


if __name__ == "__main__":
    unittest.main()
