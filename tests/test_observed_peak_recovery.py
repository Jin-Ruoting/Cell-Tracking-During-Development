import ast
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import observed_peak_recovery as recovery


@unittest.skipUnless(os.environ.get("BIOHUB_FLOW_REFERENCE_TEST") and os.environ.get("BIOHUB_GEOMETRIC_REFERENCE_TEST")
                     and importlib.util.find_spec("numpy") and importlib.util.find_spec("scipy"),
                     "Pinned sources and array runtime required")
class ObservedPeakTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial import cKDTree
        self.np = np
        self.source = recovery.flow.reference.read_reference(Path(os.environ["BIOHUB_GEOMETRIC_REFERENCE_TEST"]))[5]
        self.helpers = recovery.selected_functions(Path(os.environ["BIOHUB_FLOW_REFERENCE_TEST"]))
        self.namespace = {"np": np, "os": os, "Path": Path, "cKDTree": cKDTree,
            "linear_sum_assignment": linear_sum_assignment, "VOXEL_SCALE_UM": (1, 1, 1),
            "node_point": lambda n: (n["z"], n["y"], n["x"]),
            "_next_node_id": lambda nodes: max(nodes, default=-1) + 1,
            "edge_distance_um": lambda a, b: float(np.linalg.norm(
                np.array([a[k] - b[k] for k in ("z", "y", "x")]))), **recovery.CONFIG}
        exec(compile(self.helpers, "isolated-helpers", "exec"), self.namespace)

    @staticmethod
    def node(nid, t, x):
        return {"node_id": nid, "t": t, "z": 1., "y": 10., "x": float(x)}

    def test_only_one_branch_inserted_and_other_functions_unchanged(self):
        for arm, wanted, unwanted in (("readmit", "readmit_discarded_detections", "fill_gaps_from_low_detections"),
                                     ("gapfill", "fill_gaps_from_low_detections", "readmit_discarded_detections")):
            patched = ast.parse(recovery.patch_postprocessing(self.source, self.helpers, arm))
            original = ast.parse(self.source)
            self.assertEqual([ast.dump(n) for n in original.body if not
                              (isinstance(n, ast.FunctionDef) and n.name == "filter_output_graph")],
                             [ast.dump(n) for n in patched.body if not
                              (isinstance(n, ast.FunctionDef) and n.name in (*recovery.FUNCTIONS, "filter_output_graph"))
                              and not (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                       and n.targets[0].id in recovery.CONFIG)])
            function = recovery.flow.function_node(ast.unparse(patched), "filter_output_graph")
            calls = [n.func.id for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
            self.assertEqual(calls.count(wanted), 1)
            self.assertNotIn(unwanted, calls)

    def test_missing_and_corrupt_cache_fail_instead_of_returning_original_graph(self):
        nodes = {0: self.node(0, 0, 10), 1: self.node(1, 1, 11)}
        edges = [{"source_id": 0, "target_id": 1}]
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"BIOHUB_CACHE_DIR": directory}):
            for contents in (None, b"not an npz"):
                if contents is not None:
                    (Path(directory) / "movie.npz").write_bytes(contents)
                with self.assertRaises((RuntimeError, ValueError)):
                    self.namespace["readmit_discarded_detections"](dict(nodes), edges, {}, "movie")

    def test_readmit_requires_probability_and_open_endpoint_neighborhood(self):
        nodes = {0: self.node(0, 0, 10), 1: self.node(1, 1, 11)}
        edges = [{"source_id": 0, "target_id": 1}]
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"BIOHUB_CACHE_DIR": directory}):
            path = Path(directory) / "movie.npz"
            self.np.savez(path, low_coords=self.np.array([[2, 1, 10, 14], [2, 1, 10, 12], [2, 1, 10, 30]]),
                          low_score=self.np.array([0.99, 0.8, 0.99]))
            stats = {}
            result = self.namespace["readmit_discarded_detections"](dict(nodes), edges, stats, "movie")
            self.assertEqual(len(result), 3)
            self.assertEqual(result[2]["x"], 14)
            self.assertEqual(stats["readmitted_nodes"], 1)
            self.assertEqual(stats["peak_cache_loads"], 1)

    def test_gapfill_requires_actual_peak_and_creates_consecutive_edges_within_budget(self):
        nodes = {i: self.node(i, t, x) for i, (t, x) in enumerate(((0, 8), (1, 10), (3, 14), (4, 16)))}
        nodes.update({i: self.node(i, 0, 100 + 20 * i) for i in range(4, 40)})
        edges = [{"source_id": 0, "target_id": 1}, {"source_id": 2, "target_id": 3}]
        empty_nodes, empty_edges = self.namespace["fill_gaps_from_low_detections"](dict(nodes), edges, {}, pool={})
        self.assertEqual(len(empty_nodes), 40)
        self.assertEqual(len(empty_edges), 2)
        pool = {2: {"vox": self.np.array([[1., 10., 12.]]), "um": self.np.array([[1., 10., 12.]]),
                    "score": self.np.array([0.8])}}
        stats = {}
        result, links = self.namespace["fill_gaps_from_low_detections"](dict(nodes), edges, stats, pool=pool)
        self.assertEqual(len(result), 41)
        self.assertEqual(len(links), 4)
        self.assertEqual(stats["gapfill_peak_nodes"], 1)
        self.assertEqual(stats.get("gapfill_synthetic_nodes", 0), 0)
        self.assertLessEqual(stats["gapfill_added_nodes"], round(len(nodes) * 0.03))
        for link in links:
            self.assertEqual(result[link["target_id"]]["t"] - result[link["source_id"]]["t"], 1)


if __name__ == "__main__":
    unittest.main()
