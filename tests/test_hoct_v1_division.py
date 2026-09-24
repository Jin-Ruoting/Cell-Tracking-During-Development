import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
from run_hoct_v1_division import audit_intersection


def fixture():
    rows = []
    for node, t, x in ((1, 0, 0), (2, 1, 1), (3, 1, 2), (4, 0, 3), (5, 0, 3), (6, 1, 4)):
        rows.append(dict(dataset="movie", row_type="node", node_id=node, t=t, z=0, y=0, x=x, source_id=-1, target_id=-1))
    for source, target in ((1, 2), (1, 3), (4, 6)):
        rows.append(dict(dataset="movie", row_type="edge", node_id=-1, t=-1, z=-1, y=-1, x=-1, source_id=source, target_id=target))
    frame = pd.DataFrame(rows)
    frame.insert(0, "id", range(len(frame)))
    return frame


def remove(frame, source, target):
    return frame.loc[~((frame.row_type == "edge") & (frame.source_id == source) & (frame.target_id == target))].copy()


class DivisionIntersectionTests(unittest.TestCase):
    def test_unconfirmed_division_child_is_removed_but_ambiguous_edge_survives(self):
        original = fixture()
        candidate = remove(original, 1, 3)
        result = audit_intersection(original, candidate, {"movie": {(1, 2)}}, {"movie": {4, 5}})
        self.assertTrue(result["passed"])
        self.assertEqual(result["movies"][0]["removed_division_edges"], 1)
        self.assertEqual(result["movies"][0]["affected_division_parents"], 1)

    def test_unsupported_protected_division_arm_is_not_a_valid_veto(self):
        original = fixture()
        with self.assertRaisesRegex(ValueError, "prescribed edge intersection"):
            audit_intersection(original, original, {"movie": {(1, 2)}}, {"movie": {4, 5}})

    def test_extra_deletion_or_duplicate_is_rejected(self):
        original = fixture()
        candidate = remove(original, 1, 3)
        extra_removed = remove(candidate, 4, 6)
        duplicate = pd.concat([candidate, candidate.iloc[-1:]], ignore_index=True)
        for broken in (extra_removed, duplicate):
            with self.assertRaisesRegex(ValueError, "prescribed edge intersection"):
                audit_intersection(original, broken, {"movie": {(1, 2)}}, {"movie": {4, 5}})

    def test_node_movement_is_rejected(self):
        original = fixture()
        candidate = remove(original, 1, 3)
        candidate.loc[candidate.node_id == 2, "x"] += 1
        with self.assertRaisesRegex(ValueError, "Original nodes"):
            audit_intersection(original, candidate, {"movie": {(1, 2)}}, {"movie": {4, 5}})


if __name__ == "__main__":
    unittest.main()
