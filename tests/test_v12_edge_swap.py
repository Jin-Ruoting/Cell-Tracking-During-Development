"""Data-free semantic checks for fixed-node association swaps."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
from v12_edge_swap import swap_frame


class SwapTests(unittest.TestCase):
    def run_case(self, logits=None, pairs=None, dst=None):
        src = np.array([[10, 20, 20], [10, 24, 20]])
        dst = src.copy() if dst is None else dst
        logits = np.array([[0., 4.], [4., 0.]]) if logits is None else np.asarray(logits)
        pairs = np.array([[1, 3], [2, 4]]) if pairs is None else np.asarray(pairs)
        original = pairs.copy()
        result = swap_frame([1, 2], [3, 4], src, dst, pairs, logits)
        np.testing.assert_array_equal(pairs, original)
        return result

    def test_reciprocal_cycle_swaps_without_mutating(self):
        result, changes, stats = self.run_case()
        np.testing.assert_array_equal(result, [[1, 4], [2, 3]])
        self.assertEqual(len(changes), 1)
        self.assertEqual(stats["changed_edges"], 2)

    def test_weak_pair_margin_rejected(self):
        result, _, stats = self.run_case(logits=[[1., 1.5], [1.5, 1.]])
        self.assertEqual(stats["margin_rejected"], 1)
        np.testing.assert_array_equal(result, [[1, 3], [2, 4]])

    def test_nonreciprocal_and_exact_tie_rejected(self):
        for logits in ([[0., 5.], [1., 4.]], [[0., 4.], [4., 4.]]):
            _, changes, _ = self.run_case(logits=logits)
            self.assertEqual(changes, [])

    def test_no_link_competition_rejected(self):
        _, changes, _ = self.run_case(logits=[[-5., -1.], [-1., -5.]])
        self.assertEqual(changes, [])

    def test_long_exchange_rejected_in_physical_units(self):
        _, changes, stats = self.run_case(dst=np.array([[50, 20, 20], [50, 24, 20]]))
        self.assertEqual(changes, [])
        self.assertEqual(stats["distance_rejected"], 1)

    def test_division_source_excluded(self):
        _, changes, _ = self.run_case(pairs=[[1, 3], [1, 4]])
        self.assertEqual(changes, [])

    def test_all_nodes_participate_in_ranking(self):
        # An unmatched destination wins; excluding it would falsely produce a swap.
        out, changes, _ = swap_frame([1, 2], [3, 4, 5], [[10, 20, 20], [10, 24, 20]],
                                    [[10, 20, 20], [10, 24, 20], [10, 22, 20]],
                                    [[1, 3], [2, 4]], [[0, 4, 6], [4, 0, 0]])
        self.assertEqual(changes, [])
        np.testing.assert_array_equal(out, [[1, 3], [2, 4]])

    def test_three_cycle_is_not_rewritten(self):
        out, changes, _ = swap_frame([1, 2, 3], [4, 5, 6], [[10, 10, 10]] * 3,
                                    [[10, 10, 10]] * 3, [[1, 4], [2, 5], [3, 6]],
                                    [[0, 5, 0], [0, 0, 5], [5, 0, 0]])
        self.assertEqual(changes, [])
        np.testing.assert_array_equal(out, [[1, 4], [2, 5], [3, 6]])

    def test_malformed_graph_or_logits_fails(self):
        for kwargs in ({"logits": [[0, np.nan], [4, 0]]}, {"pairs": [[1, 3], [2, 3]]},
                       {"pairs": [[1, 3], [2, 99]]}, {"pairs": [[1.5, 3], [2, 4]]},
                       {"dst": np.array([[-1, 20, 20], [10, 24, 20]])}):
            with self.assertRaises(ValueError):
                self.run_case(**kwargs)

    def test_node_ids_and_matrix_order_are_independent(self):
        out, changes, _ = swap_frame([20, 10], [80, 70], [[10, 20, 20]] * 2,
                                    [[10, 20, 20]] * 2, [[10, 70], [20, 80]], [[0, 4], [4, 0]])
        np.testing.assert_array_equal(out, [[10, 80], [20, 70]])
        self.assertEqual(len(changes), 1)


if __name__ == "__main__":
    unittest.main()
