import ast
from collections import Counter
import importlib.util
import math
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import run_motion_ema_experiment as ema


@unittest.skipUnless(os.environ.get("BIOHUB_GEOMETRIC_REFERENCE_TEST") and importlib.util.find_spec("scipy"),
                     "Pinned E029 source and array runtime required")
class MotionEMATests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        source = ema.reference.read_reference(Path(os.environ["BIOHUB_GEOMETRIC_REFERENCE_TEST"]))[5]
        self.control = ast.unparse(ema.flow.function_node(source))
        self.candidate = ema.motion_function(source)
        self.namespace = {"np": np, "math": math, "linear_sum_assignment": linear_sum_assignment,
            "_position_um": lambda n: np.array([n['z'], n['y'], n['x']], dtype=float),
            "OUTPUT_MOTION_RELINK": True, "MOTION_RELINK_MAX_FRAME_NODES": 2600,
            "MOTION_RELINK_TIGHT_UM": 5.5, "MOTION_RELINK_RELAXED_UM": 10.,
            "MOTION_RELINK_LEARNED_BONUS": 1., "MOTION_RELINK_VELOCITY_WEIGHT": 0.25}

    def apply(self, function, trajectory):
        namespace = dict(self.namespace)
        exec(compile(function, "motion", "exec"), namespace)
        nodes = {i: {"t": t, "z": 0, "y": 0, "x": x} for i, (t, x) in enumerate(trajectory)}
        before = repr(nodes)
        stats = Counter()
        result = namespace["motion_relink_edges"](nodes, stats)
        self.assertEqual(before, repr(nodes))
        self.assertEqual(len({e['target_id'] for e in result}), len(result))
        return {e['source_id']: e['target_id'] for e in result}, stats

    def test_temporal_smoothing_changes_a_jitter_ambiguous_assignment(self):
        trajectory = [(0, 0), (1, 4), (2, 4), (3, 4), (3, 5)]
        before, _ = self.apply(self.control, trajectory)
        after, stats = self.apply(self.candidate, trajectory)
        self.assertEqual(before[2], 3)
        self.assertEqual(after[2], 4)
        self.assertGreater(stats['motion_ema_changed_velocities'], 0)

    def test_constant_velocity_and_new_track_start_remain_unchanged(self):
        for trajectory in ([(0, 0), (1, 1), (2, 2), (3, 3)], [(0, 0), (1, 1)], [(0, 0), (2, 2)]):
            before, _ = self.apply(self.control, trajectory)
            after, stats = self.apply(self.candidate, trajectory)
            self.assertEqual(before, after)
            self.assertEqual(stats['motion_ema_changed_velocities'], 0)


if __name__ == "__main__":
    unittest.main()
