import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import fuse_division_evidence as fusion


class DivisionEvidenceTests(unittest.TestCase):
    def test_adds_matched_division_only_to_an_unparented_child(self):
        result, audit = fusion.fuse_divisions({(1, 2)}, {(11, 12), (11, 13)}, {11: 1, 12: 2, 13: 3})
        self.assertEqual(result, {(1, 2), (1, 3)})
        self.assertEqual(audit["added_division_edges"], [[1, 3]])

    def test_existing_child_parent_is_preserved(self):
        base = {(1, 2), (4, 3)}
        result, audit = fusion.fuse_divisions(base, {(11, 12), (11, 13)}, {11: 1, 12: 2, 13: 3})
        self.assertEqual(result, base)
        self.assertEqual(audit["added_division_edges"], [])

    def test_removes_only_a_division_branch_with_a_mapped_alternative_parent(self):
        base = {(1, 2), (1, 3), (4, 5)}
        ref = {(11, 12), (14, 13)}
        mapping = {11: 1, 12: 2, 13: 3, 14: 4}
        result, audit = fusion.fuse_divisions(base, ref, mapping)
        self.assertEqual(result, {(1, 2), (4, 5)})
        self.assertEqual(audit["removed_division_edges"], [[1, 3]])

    def test_absence_of_reference_support_alone_cannot_remove_a_branch(self):
        base = {(1, 2), (1, 3)}
        result, _ = fusion.fuse_divisions(base, {(11, 12)}, {11: 1, 12: 2, 13: 3})
        self.assertEqual(result, base)

    def test_unmapped_alternative_parent_cannot_remove_a_branch(self):
        base = {(1, 2), (1, 3)}
        result, _ = fusion.fuse_divisions(base, {(11, 12), (14, 13)}, {11: 1, 12: 2, 13: 3})
        self.assertEqual(result, base)

    def test_invalid_input_topology_and_ambiguous_mapping_are_rejected(self):
        cases = [({(1, 2), (3, 2)}, {}, {}), ({(1, 2), (1, 3), (1, 4)}, {}, {}),
                 (set(), set(), {11: 1, 12: 1})]
        for base, ref, mapping in cases:
            with self.subTest(base=base), self.assertRaises(ValueError):
                fusion.fuse_divisions(base, ref, mapping)


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("numpy", "scipy", "pandas")),
                     "Physical matching tests need the pinned server runtime")
class PhysicalFusionTests(unittest.TestCase):
    def test_mutual_matches_are_close_unique_and_within_frame(self):
        base = {1: (0, 0, 0, 0), 2: (0, 0, 0, 10), 3: (1, 0, 0, 0)}
        ref = {11: (0, 0, 0, .4), 12: (0, 0, 0, .7), 13: (0, 0, 0, 13),
               14: (1, 0, 0, .4), 15: (2, 0, 0, 0)}
        self.assertEqual(fusion.mutual_mapping(base, ref, (1, 1, 1)), {11: 1, 14: 3})

    def test_csv_preserves_all_base_node_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_rows = [[0, "movie", "node", 1, 0, 2, 2, 2, -1, -1],
                         [1, "movie", "node", 2, 1, 2, 2, 2, -1, -1],
                         [2, "movie", "node", 3, 1, 2, 8, 2, -1, -1],
                         [3, "movie", "edge", -1, -1, -1, -1, -1, 1, 2]]
            ref_rows = [row[:] for row in base_rows]
            for row in ref_rows:
                for index in (3, 8, 9):
                    if row[index] >= 0:
                        row[index] += 10
            ref_rows.append([4, "movie", "edge", -1, -1, -1, -1, -1, 11, 13])
            for name, rows in (("base.csv", base_rows), ("ref.csv", ref_rows)):
                with (root / name).open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(fusion.CSV_COLUMNS)
                    writer.writerows(rows)
            with mock.patch.object(fusion, "image_scale", return_value=(1, 1, 1)):
                fusion.fuse_csv(root / "base.csv", root / "ref.csv", root / "out.csv", root)
            with (root / "out.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            actual_nodes = [[row[column] for column in fusion.CSV_COLUMNS[1:]]
                            for row in rows if row["row_type"] == "node"]
            self.assertEqual(actual_nodes, [[str(x) for x in row[1:]] for row in base_rows[:3]])
            actual_edges = {(int(row["source_id"]), int(row["target_id"]))
                            for row in rows if row["row_type"] == "edge"}
            self.assertEqual(actual_edges, {(1, 2), (1, 3)})


if __name__ == "__main__":
    unittest.main()
