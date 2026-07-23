from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_submission import EXPECTED_COLUMNS, audit_submission


def node_row(
    row_id: int,
    dataset: str,
    node_id: int,
    t: int,
    z: int = 0,
    y: int = 0,
    x: int = 0,
) -> dict[str, object]:
    return {
        "id": row_id,
        "dataset": dataset,
        "row_type": "node",
        "node_id": node_id,
        "t": t,
        "z": z,
        "y": y,
        "x": x,
        "source_id": -1,
        "target_id": -1,
    }


def edge_row(
    row_id: int,
    dataset: str,
    source_id: int,
    target_id: int,
) -> dict[str, object]:
    return {
        "id": row_id,
        "dataset": dataset,
        "row_type": "edge",
        "node_id": -1,
        "t": -1,
        "z": -1,
        "y": -1,
        "x": -1,
        "source_id": source_id,
        "target_id": target_id,
    }


def write_submission(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class SubmissionAuditTest(unittest.TestCase):
    def test_valid_track_and_connectivity_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            submission = Path(temporary_dir) / "submission.csv"
            write_submission(
                submission,
                [
                    node_row(0, "clip", 10, 0),
                    node_row(1, "clip", 11, 1),
                    node_row(2, "clip", 12, 2),
                    node_row(3, "clip", 20, 0),
                    edge_row(4, "clip", 10, 11),
                    edge_row(5, "clip", 11, 12),
                ],
            )

            report = audit_submission(submission)

        self.assertTrue(report["valid"])
        self.assertEqual(report["nodes"], 4)
        self.assertEqual(report["edges"], 2)
        self.assertEqual(report["per_dataset"][0]["components"], 2)
        self.assertEqual(report["per_dataset"][0]["largest_component_nodes"], 3)
        self.assertEqual(report["per_dataset"][0]["singleton_components"], 1)

    def test_detects_graph_and_schema_violations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            submission = Path(temporary_dir) / "submission.csv"
            write_submission(
                submission,
                [
                    node_row(0, "clip", 1, 0),
                    node_row(1, "clip", 2, 2),
                    edge_row(2, "clip", 1, 2),
                    edge_row(3, "clip", 1, 2),
                    edge_row(4, "clip", 1, 99),
                ],
            )

            report = audit_submission(submission)

        self.assertFalse(report["valid"])
        counts = report["violations"]["counts"]
        self.assertEqual(counts["duplicate_edge"], 1)
        self.assertEqual(counts["dangling_edge"], 1)
        self.assertEqual(counts["nonconsecutive_edge"], 1)

    def test_coordinate_bounds_from_zarr_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            submission = root / "submission.csv"
            write_submission(
                submission,
                [node_row(0, "clip", 1, 0, z=4, y=0, x=0)],
            )
            metadata_dir = root / "test" / "clip.zarr" / "0"
            metadata_dir.mkdir(parents=True)
            (metadata_dir / "zarr.json").write_text(
                json.dumps({"shape": [3, 4, 5, 6]}),
                encoding="utf-8",
            )

            report = audit_submission(submission, root)

        self.assertFalse(report["valid"])
        self.assertEqual(
            report["violations"]["counts"]["out_of_bounds_node"],
            1,
        )

    def test_rejects_unexpected_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            submission = Path(temporary_dir) / "submission.csv"
            submission.write_text("id,dataset\n0,clip\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unexpected CSV columns"):
                audit_submission(submission)


if __name__ == "__main__":
    unittest.main()
