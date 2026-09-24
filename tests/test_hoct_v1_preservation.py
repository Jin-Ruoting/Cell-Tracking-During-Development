import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
from run_hoct_v1_consensus import audit_preservation


def graph():
    rows = []
    # 1 divides into 2/3; 4/5 have exactly coincident identity positions.
    for node, t, x in ((1, 0, 0), (2, 1, 1), (3, 1, 2), (4, 0, 3), (5, 0, 3), (6, 1, 4), (7, 2, 5)):
        rows.append(dict(dataset="movie", row_type="node", node_id=node, t=t, z=0, y=0, x=x, source_id=-1, target_id=-1))
    for source, target in ((1, 2), (1, 3), (4, 6), (6, 7)):
        rows.append(dict(dataset="movie", row_type="edge", node_id=-1, t=-1, z=-1, y=-1, x=-1, source_id=source, target_id=target))
    frame = pd.DataFrame(rows)
    frame.insert(0, "id", range(len(frame)))
    return frame


def remove(frame, edge):
    return frame.loc[~((frame.row_type == "edge") & (frame.source_id == edge[0]) & (frame.target_id == edge[1]))].copy()


class ExportPreservationTests(unittest.TestCase):
    def test_ordinary_edge_removal_and_row_renumbering_are_allowed(self):
        original = graph()
        candidate = remove(original, (6, 7))
        candidate["id"] = range(len(candidate))
        result = audit_preservation(original, candidate)
        self.assertTrue(result["passed"])
        self.assertEqual(result["movies"][0]["removed_edges"], 1)

    def test_both_division_edges_are_protected_even_without_labels(self):
        original = graph()
        for edge in ((1, 2), (1, 3)):
            with self.assertRaisesRegex(ValueError, "division or coincident"):
                audit_preservation(original, remove(original, edge))

    def test_coincident_identity_incident_edge_is_protected(self):
        original = graph()
        with self.assertRaisesRegex(ValueError, "division or coincident"):
            audit_preservation(original, remove(original, (4, 6)))

    def test_new_or_duplicate_edges_are_rejected(self):
        original = graph()
        moved = original.copy()
        moved.loc[moved.source_id == 6, "source_id"] = 2
        duplicate = pd.concat([original, original.iloc[-1:]], ignore_index=True)
        for candidate in (moved, duplicate):
            with self.assertRaisesRegex(ValueError, "new or duplicate"):
                audit_preservation(original, candidate)

    def test_coordinate_change_or_node_deletion_is_rejected(self):
        original = graph()
        moved = original.copy()
        moved.loc[moved.node_id == 7, "x"] += 1
        deleted = original.loc[original.node_id != 7].copy()
        for candidate in (moved, deleted):
            with self.assertRaisesRegex(ValueError, "node field"):
                audit_preservation(original, candidate)


if __name__ == "__main__":
    unittest.main()
