from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_frozen_edges import analyze_frozen_edges
from tests.test_audit_submission import edge_row, node_row, write_submission


def write_frozen_report(
    path: Path,
    clip: str,
    pairs: list[tuple[int, int]],
) -> None:
    path.write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "clip": clip,
                        "duplicate_pairs": [
                            {"source_t": source_t, "target_t": target_t}
                            for source_t, target_t in pairs
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


class FrozenEdgeAnalysisTest(unittest.TestCase):
    def test_separates_frozen_and_normal_edge_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            predictions = root / "predictions.csv"
            frozen = root / "frozen.json"
            write_submission(
                predictions,
                [
                    node_row(0, "6bba_clip", 1, 0, x=3),
                    node_row(1, "6bba_clip", 2, 1, x=3),
                    node_row(2, "6bba_clip", 3, 2, x=4),
                    edge_row(3, "6bba_clip", 1, 2),
                    edge_row(4, "6bba_clip", 2, 3),
                ],
            )
            write_frozen_report(frozen, "6bba_clip", [(0, 1)])

            report = analyze_frozen_edges(predictions, frozen)

        self.assertEqual(report["overall"]["frozen"]["edges"], 1)
        self.assertEqual(report["overall"]["frozen"]["exact_zero_edges"], 1)
        self.assertAlmostEqual(report["overall"]["normal"]["mean_um"], 0.40625)
        summary = report["frozen_detection_and_link_summary"]
        self.assertEqual(summary["exact_position_overlap"], 1)
        self.assertEqual(summary["exact_zero_edge_fraction"], 1.0)

    def test_detects_cross_links_despite_identical_detection_sets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            predictions = root / "predictions.csv"
            frozen = root / "frozen.json"
            write_submission(
                predictions,
                [
                    node_row(0, "6bba_clip", 1, 0, x=0),
                    node_row(1, "6bba_clip", 2, 0, x=2),
                    node_row(2, "6bba_clip", 3, 1, x=0),
                    node_row(3, "6bba_clip", 4, 1, x=2),
                    edge_row(4, "6bba_clip", 1, 4),
                    edge_row(5, "6bba_clip", 2, 3),
                ],
            )
            write_frozen_report(frozen, "6bba_clip", [(0, 1)])

            report = analyze_frozen_edges(predictions, frozen)

        summary = report["frozen_detection_and_link_summary"]
        self.assertEqual(summary["exact_position_overlap"], 2)
        self.assertEqual(summary["source_overlap_fraction"], 1.0)
        self.assertEqual(summary["exact_zero_edges"], 0)
        self.assertEqual(summary["exact_zero_edge_fraction"], 0.0)

    def test_dataset_without_frozen_pairs_is_all_normal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            predictions = root / "predictions.csv"
            frozen = root / "frozen.json"
            write_submission(
                predictions,
                [
                    node_row(0, "44b6_clip", 1, 0),
                    node_row(1, "44b6_clip", 2, 1),
                    edge_row(2, "44b6_clip", 1, 2),
                ],
            )
            write_frozen_report(frozen, "other_clip", [(0, 1)])

            report = analyze_frozen_edges(predictions, frozen)

        self.assertEqual(report["overall"]["frozen"]["edges"], 0)
        self.assertEqual(report["overall"]["normal"]["edges"], 1)
        self.assertEqual(
            report["frozen_detection_and_link_summary"]["transitions"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
