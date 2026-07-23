from __future__ import annotations

import itertools
import unittest

from scripts.relink_frozen_transitions import relink_rows
from tests.test_audit_submission import edge_row, node_row


def brute_assignment(costs):
    row_count = len(costs)
    column_count = len(costs[0])
    candidates = (
        (
            list(enumerate(columns)),
            sum(costs[row][column] for row, column in enumerate(columns)),
        )
        for columns in itertools.permutations(range(column_count), row_count)
    )
    return min(candidates, key=lambda item: item[1])[0]


def prepare_rows(rows):
    return [
        {key: str(value) for key, value in row.items()}
        for row in rows
    ]


def prepare_nodes(rows):
    nodes = {}
    for row in rows:
        if row["row_type"] != "node":
            continue
        nodes.setdefault(row["dataset"], {})[int(row["node_id"])] = {
            "t": int(row["t"]),
            "position": (
                float(row["z"]),
                float(row["y"]),
                float(row["x"]),
            ),
        }
    return nodes


class FrozenRelinkTest(unittest.TestCase):
    def test_replaces_cross_links_with_minimum_motion(self) -> None:
        rows = prepare_rows(
            [
                node_row(0, "6bba_clip", 1, 0, x=0),
                node_row(1, "6bba_clip", 2, 0, x=10),
                node_row(2, "6bba_clip", 3, 1, x=0),
                node_row(3, "6bba_clip", 4, 1, x=10),
                edge_row(4, "6bba_clip", 1, 4),
                edge_row(5, "6bba_clip", 2, 3),
            ]
        )

        output, report = relink_rows(
            rows,
            prepare_nodes(rows),
            {"6bba_clip": {(0, 1)}},
            3.0,
            assignment_solver=brute_assignment,
        )

        edges = {
            (int(row["source_id"]), int(row["target_id"]))
            for row in output
            if row["row_type"] == "edge"
        }
        self.assertEqual(edges, {(1, 3), (2, 4)})
        self.assertEqual(report["totals"]["removed_edges"], 2)
        self.assertEqual(report["totals"]["matched_edges"], 2)
        self.assertEqual(report["totals"]["exact_zero_edges"], 2)

    def test_gate_leaves_distant_nodes_unmatched(self) -> None:
        rows = prepare_rows(
            [
                node_row(0, "6bba_clip", 1, 0, x=0),
                node_row(1, "6bba_clip", 2, 1, x=20),
                edge_row(2, "6bba_clip", 1, 2),
            ]
        )

        output, report = relink_rows(
            rows,
            prepare_nodes(rows),
            {"6bba_clip": {(0, 1)}},
            3.0,
            assignment_solver=brute_assignment,
        )

        self.assertFalse(
            any(row["row_type"] == "edge" for row in output)
        )
        self.assertEqual(report["totals"]["removed_edges"], 1)
        self.assertEqual(report["totals"]["matched_edges"], 0)
        self.assertEqual(report["totals"]["unmatched_sources"], 1)
        self.assertEqual(report["totals"]["unmatched_targets"], 1)

    def test_preserves_edges_outside_detected_pairs(self) -> None:
        rows = prepare_rows(
            [
                node_row(0, "44b6_clip", 1, 0),
                node_row(1, "44b6_clip", 2, 1),
                edge_row(2, "44b6_clip", 1, 2),
            ]
        )

        output, report = relink_rows(
            rows,
            prepare_nodes(rows),
            {"other_clip": {(0, 1)}},
            3.0,
            assignment_solver=brute_assignment,
        )

        edges = [
            row for row in output if row["row_type"] == "edge"
        ]
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["source_id"], "1")
        self.assertEqual(edges[0]["target_id"], "2")
        self.assertEqual(report["totals"]["removed_edges"], 0)


if __name__ == "__main__":
    unittest.main()
