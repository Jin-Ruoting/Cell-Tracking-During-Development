"""Data-free contracts for full-movie fixed-node input/output handling."""
import csv
import sys
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
from run_v12_association import fixed_graph, validate_rewrite


class GraphContracts(unittest.TestCase):
    fields = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]

    def row(self, kind, **values):
        return {**{k: "-1" for k in self.fields}, "row_type": kind, "dataset": "example", **{k: str(v) for k, v in values.items()}}

    def test_fixed_graph_keeps_frame_identity_and_empty_frames(self):
        rows = [self.row("node", node_id=20, t=0, z=1, y=2, x=3),
                self.row("node", node_id=10, t=0, z=4, y=5, x=6),
                self.row("node", node_id=30, t=1, z=7, y=8, x=9),
                self.row("edge", source_id=20, target_id=30)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "control.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fields)
                writer.writeheader()
                writer.writerows(rows)
            snapshot = path.read_bytes()
            _, result, ids, points, edge_rows = fixed_graph(path)
            self.assertEqual(result, rows)
            self.assertEqual(ids[:2], [[10, 20], [30]])
            self.assertEqual(points[0].tolist(), [[4, 5, 6], [1, 2, 3]])
            self.assertEqual(points[99].shape, (0, 3))
            self.assertEqual(edge_rows[0], [3])
            self.assertEqual(len(edge_rows), 99)
            self.assertEqual(path.read_bytes(), snapshot)

    def ordinary_rows(self):
        return [self.row("node", node_id=1, t=0, z=1, y=2, x=3),
                self.row("edge", source_id=1, target_id=3), self.row("edge", source_id=2, target_id=4)]

    def test_ordinary_swap_preserves_full_original_fields(self):
        before = self.ordinary_rows()
        after = [r.copy() for r in before]
        after[1]["target_id"], after[2]["target_id"] = "4", "3"
        validate_rewrite(before, after, self.fields, 2)

    def test_node_or_degree_change_fails(self):
        before = self.ordinary_rows()
        for index, key, value in ((0, "x", "9"), (1, "source_id", "2"), (1, "target_id", "9")):
            after = [r.copy() for r in before]
            after[index][key] = value
            with self.assertRaises(ValueError):
                validate_rewrite(before, after, self.fields, 1)

    def test_division_edge_change_rejected_even_when_degrees_match(self):
        before = [self.row("edge", source_id=1, target_id=3), self.row("edge", source_id=1, target_id=4),
                  self.row("edge", source_id=2, target_id=5)]
        after = [r.copy() for r in before]
        after[0]["target_id"], after[2]["target_id"] = "5", "3"
        with self.assertRaisesRegex(ValueError, "division"):
            validate_rewrite(before, after, self.fields, 2)


if __name__ == "__main__":
    unittest.main()
