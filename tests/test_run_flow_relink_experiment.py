import ast
from collections import Counter
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import run_flow_relink_experiment as flow


class FlowBoundaryTests(unittest.TestCase):
    def test_extracts_only_named_function_without_running_notebook_statements(self):
        source = "raise RuntimeError('private model required')\ndef motion_relink_edges():\n    return 7\n"
        node = flow.function_node(source)
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "selected", "exec"), namespace)
        self.assertEqual(namespace["motion_relink_edges"](), 7)

    def test_patch_preserves_all_other_processing_and_the_writer(self):
        source = "A = 1\ndef motion_relink_edges():\n    return 1\ndef gap():\n    return 2\nwrite_test_submission('base')\n"
        patched = ast.parse(flow.patch_postprocessing(source, "def motion_relink_edges():\n    return 3\n"))
        body = patched.body[len(flow.FLOW_CONFIG):]
        original = ast.parse(source).body
        for index in (0, 2, 3):
            self.assertEqual(ast.dump(body[index]), ast.dump(original[index]))
        self.assertEqual(body[1].body[0].value.value, 3)

    def test_missing_duplicated_or_decorated_motion_function_is_rejected(self):
        for source in ("pass", "def motion_relink_edges(): pass\ndef motion_relink_edges(): pass",
                       "@external\ndef motion_relink_edges(): pass"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                flow.function_node(source)

    def test_reference_checksum_is_checked_before_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "source.ipynb"
            p.write_text("not JSON")
            with self.assertRaisesRegex(ValueError, "checksum"):
                flow.read_flow_function(p)

    def test_raw_graph_fingerprint_binds_file_identity_and_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "movie.geff"
            p.mkdir()
            (p / "a").write_bytes(b"abc")
            first = flow.graph_tree_sha256([p])
            (p / "a").write_bytes(b"xyz")
            self.assertNotEqual(first, flow.graph_tree_sha256([p]))
            (p / "a").write_bytes(b"abc")
            (p / "a").rename(p / "b")
            self.assertNotEqual(first, flow.graph_tree_sha256([p]))

    def test_subset_renumbers_rows_and_parity_rejects_changed_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, subset, other = (root / name for name in ("source.csv", "subset.csv", "other.csv"))
            source.write_text("id,dataset,x\n0,a,1\n1,b,2\n2,b,3\n")
            flow.write_subset(source, subset, ["b"])
            self.assertEqual(subset.read_text(), "id,dataset,x\n0,b,2\n1,b,3\n")
            other.write_bytes(subset.read_bytes())
            flow.require_replay_parity(subset, other)
            other.write_text("id,dataset,x\n0,b,9\n1,b,3\n")
            with self.assertRaisesRegex(ValueError, "byte-identical"):
                flow.require_replay_parity(subset, other)


@unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("scipy")
                     and os.environ.get("BIOHUB_FLOW_REFERENCE_TEST")
                     and os.environ.get("BIOHUB_GEOMETRIC_REFERENCE_TEST"),
                     "Semantic tests require the pinned public sources and server array runtime")
class ActualFlowSemanticsTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial import cKDTree
        import math
        base = flow.reference.read_reference(Path(os.environ["BIOHUB_GEOMETRIC_REFERENCE_TEST"]))
        self.control_source = ast.unparse(flow.function_node(base[5]))
        self.candidate_source = flow.read_flow_function(Path(os.environ["BIOHUB_FLOW_REFERENCE_TEST"]))
        self.namespace = {
            "np": np, "math": math, "cKDTree": cKDTree,
            "linear_sum_assignment": linear_sum_assignment,
            "_position_um": lambda n: np.array([n["z"], n["y"], n["x"]], dtype=float),
            "OUTPUT_MOTION_RELINK": True, "MOTION_RELINK_MAX_FRAME_NODES": 2600,
            "MOTION_RELINK_TIGHT_UM": 5.5, "MOTION_RELINK_RELAXED_UM": 10.0,
            "MOTION_RELINK_VELOCITY_WEIGHT": 0.25, "MOTION_RELINK_LEARNED_BONUS": 0.75,
            **flow.FLOW_CONFIG,
        }
        self.nodes = {}
        for i, (y, x) in enumerate([(40, 40), (40, 60), (60, 40), (60, 60), (50, 50)]):
            self.nodes[i] = {"node_id": i, "t": 0, "z": 5, "y": y, "x": x}
            j = 100 + i
            self.nodes[j] = {"node_id": j, "t": 1, "z": 5, "y": y, "x": x + (6 if i == 4 else 4)}
        self.nodes[200] = {"node_id": 200, "t": 1, "z": 5, "y": 51, "x": 50}

    def edges(self, source, nodes):
        namespace = dict(self.namespace)
        exec(compile(source, "verified-motion-function", "exec"), namespace)
        stats = Counter()
        edges = namespace["motion_relink_edges"](nodes, stats)
        return {e["source_id"]: e["target_id"] for e in edges}, stats

    def test_coherent_neighbors_resolve_a_distractor(self):
        before = repr(self.nodes)
        control, _ = self.edges(self.control_source, self.nodes)
        candidate, stats = self.edges(self.candidate_source, self.nodes)
        self.assertEqual(control[4], 200)
        self.assertEqual(candidate, {i: i + 100 for i in range(5)})
        self.assertGreater(stats["motion_relink_flow_predicted"], 0)
        self.assertEqual(len(set(candidate.values())), len(candidate))
        self.assertEqual(repr(self.nodes), before)

    def test_insufficient_seeds_leave_flow_inactive(self):
        nodes = {i: n for i, n in self.nodes.items() if i in (0, 1, 100, 101)}
        control, _ = self.edges(self.control_source, nodes)
        candidate, stats = self.edges(self.candidate_source, nodes)
        self.assertEqual(candidate, control)
        self.assertEqual(stats["motion_relink_flow_frames"], 0)


if __name__ == "__main__":
    unittest.main()
