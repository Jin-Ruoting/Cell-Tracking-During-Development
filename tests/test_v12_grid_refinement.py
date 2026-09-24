import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import numpy as np
import v12_grid_refinement as policy


class GridRefinementTests(unittest.TestCase):
    def test_native_grid_and_raw_units(self):
        points = np.array([[10, 21, 25], [63, 255, 255]])
        anchors = policy.native_anchors(points)
        np.testing.assert_array_equal(anchors, [[10, 5, 6], [63, 63, 63]])
        proposed = (anchors + [[.25, -.125, .375], [0, 0, 0]]) * policy.DOWNSAMPLE
        output, audit = policy.safe_proposal(points, proposed, anchors)
        np.testing.assert_array_equal(output, [[10, 20, 26], [63, 252, 252]])
        self.assertEqual(audit['changed_nodes'], 2)

    def test_zero_residual_at_integer_anchor(self):
        points = np.array([[8, 20, 28]])
        out, audit = policy.safe_proposal(points, points, policy.native_anchors(points))
        np.testing.assert_array_equal(out, points)
        self.assertEqual(audit['changed_nodes'], 0)

    def test_continuous_and_rounded_bounds_revert_without_clipping(self):
        points = np.array([[3, 8, 8], [12, 16, 16], [40, 20, 20]])
        proposed = [[-.0001, 8, 8], [12, 255.75, 16], [1e30, 20, 20]]
        out, audit = policy.safe_proposal(points, proposed, policy.native_anchors(points))
        np.testing.assert_array_equal(out, points)
        self.assertEqual(audit['bounds_rejected_nodes'], 3)

    def test_shared_anchor_and_original_duplicate_preserved(self):
        points = np.array([[5, 20, 20], [5, 21, 20], [10, 40, 40], [10, 40, 40]])
        out, audit = policy.safe_proposal(points, points + 2, policy.native_anchors(points))
        np.testing.assert_array_equal(out, points)
        self.assertEqual(audit['ambiguous_anchor_nodes'], 4)

    def test_simultaneous_collision_and_original_occupancy(self):
        points = np.array([[5, 20, 20], [5, 40, 40], [5, 60, 60]])
        out, audit = policy.safe_proposal(points, [[5, 30, 30], [5, 30, 30], points[0]], policy.native_anchors(points))
        np.testing.assert_array_equal(out, points)
        self.assertEqual(audit['proposal_collision_nodes'], 2)
        self.assertEqual(audit['original_occupancy_rejected_nodes'], 1)

    def test_permutation_and_empty_frame(self):
        points = np.array([[5, 20, 20], [5, 40, 40], [5, 60, 60]])
        proposed = np.array([[5, 30, 30], [5, 30, 30], [5, 61, 61]])
        order = [2, 0, 1]
        a, _ = policy.safe_proposal(points, proposed, policy.native_anchors(points))
        b, _ = policy.safe_proposal(points[order], proposed[order], policy.native_anchors(points[order]))
        np.testing.assert_array_equal(b, a[order])
        out, audit = policy.safe_proposal(np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)))
        self.assertEqual(out.shape, (0, 3))
        self.assertEqual(audit['nodes'], 0)

    def test_invalid_input_fails(self):
        with self.assertRaises(ValueError):policy.native_anchors([[1, 1.5, 1]])
        with self.assertRaises(ValueError):policy.safe_proposal([[1, 4, 4]], [[np.nan, 4, 4]], [[1, 1, 1]])

    def test_graph_fields_and_new_collision_rejected(self):
        fields = ['row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
        rows = [dict(zip(fields, ['node', '7', '0', '5', '20', '20', '-1', '-1'])),
                dict(zip(fields, ['node', '8', '0', '5', '40', '40', '-1', '-1'])),
                dict(zip(fields, ['edge', '-1', '-1', '-1', '-1', '-1', '7', '9']))]
        changed = [r.copy() for r in rows];changed[0]['y'] = '21'
        policy.validate_rewrite(rows, changed, fields, 1)
        changed[2]['target_id'] = '10'
        with self.assertRaises(ValueError):policy.validate_rewrite(rows, changed, fields, 1)
        changed = [r.copy() for r in rows];changed[1].update(z='5', y='20', x='20')
        with self.assertRaises(ValueError):policy.validate_rewrite(rows, changed, fields, 1)


if __name__ == '__main__':
    unittest.main()
