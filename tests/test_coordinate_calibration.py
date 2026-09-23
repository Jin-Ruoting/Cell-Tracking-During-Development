import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import coordinate_calibration as calibration
import run_coordinate_calibration_experiment as experiment


class CalibrationContractTests(unittest.TestCase):
    def test_official_reindexed_nodes_must_match_original_csv_row_geometry(self):
        rows = [{"node_id": 1, "t": 1, "z": 4, "y": 5, "x": 6},
                {"node_id": 0, "t": 0, "z": 1, "y": 2, "x": 3}]
        mapping = experiment.geff_to_csv_index(rows, [0, 1], [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(mapping, {0: 0, 1: 1})
        with self.assertRaisesRegex(ValueError, "do not align"):
            experiment.geff_to_csv_index(rows, [0, 1], [[4, 5, 6], [1, 2, 3]])
        rows[0]["node_id"] = 99
        with self.assertRaisesRegex(ValueError, "reindexing"):
            experiment.geff_to_csv_index(rows, [0, 1], [[1, 2, 3], [4, 5, 6]])

    def test_folds_are_movie_disjoint_and_balance_both_embryos(self):
        names = [f"{embryo}_{index:02}" for embryo in ("44b6", "6bba") for index in range(32)]
        folds = experiment.fold_assignment(names[::-1])
        for fold in (0, 1):
            held = {name for name in names if folds[name] == fold}
            train = {name for name in names if folds[name] != fold}
            self.assertFalse(held & train)
            self.assertEqual(len(held), 32)
            self.assertEqual(sum(name.startswith("44b6_") for name in held), 16)
        with self.assertRaises(ValueError):
            experiment.fold_assignment(["unknown_movie"])

    def test_identity_fingerprint_ignores_only_node_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.csv"
            text = "id,dataset,row_type,node_id,t,z,y,x,source_id,target_id\n0,a,node,5,0,1,2,3,-1,-1\n1,a,edge,-1,-1,-1,-1,-1,5,8\n"
            path.write_text(text)
            before = experiment.identity_sha256(path)
            path.write_text(text.replace(",1,2,3,", ",2,3,4,"))
            self.assertEqual(experiment.identity_sha256(path), before)
            for edited in (text.replace(",5,8", ",5,9"), text.replace(",5,0,", ",5,1,")):
                path.write_text(edited)
                self.assertNotEqual(experiment.identity_sha256(path), before)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "Requires server NumPy runtime")
class RidgeAndBoundTests(unittest.TestCase):
    def test_equal_movie_weighting_prevents_large_movie_from_dominating(self):
        import numpy as np
        model = calibration.fit_ridge([(np.zeros((1, 2)), np.array([[1., 0., 0.]])),
                                       (np.zeros((100, 2)), np.tile([-1., 0., 0.], (100, 1)))])
        np.testing.assert_allclose(model["intercept"], 0, atol=1e-12)
        np.testing.assert_allclose(model["coefficient"], 0)
        with self.assertRaises(ValueError):
            calibration.fit_ridge([(np.zeros((1, 2)), np.full((1, 3), np.nan))])

    def test_zero_regression_preserves_positions_and_huge_shift_remains_bounded(self):
        import numpy as np
        positions = np.array([[0, 0, 0], [5, 5, 5], [9, 19, 19]])
        features = np.zeros((3, 2))
        scale = np.array([1.625, .40625, .40625])
        model = {"mean": np.zeros(2), "scale": np.ones(2),
                 "coefficient": np.zeros((2, 3)), "intercept": np.zeros(3)}
        unchanged = calibration.calibrated_positions(model, features, positions, scale, (10, 20, 20))
        np.testing.assert_array_equal(unchanged, positions)
        for shift in ([1.9, .5, .5], [999., -999., 999.], [-999., -999., -999.]):
            model["intercept"] = np.array(shift)
            moved = calibration.calibrated_positions(model, features, positions, scale, (10, 20, 20))
            self.assertTrue((np.linalg.norm((moved-positions)*scale, axis=1) <= 2).all())
            self.assertTrue((moved >= 0).all())
            self.assertTrue((moved < [10, 20, 20]).all())
        model["intercept"] = np.full(3, np.nan)
        with self.assertRaises(ValueError):
            calibration.calibrated_positions(model, features, positions, scale, (10, 20, 20))


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires server PyTorch runtime")
class FeatureGeometryTests(unittest.TestCase):
    def test_trilinear_features_respect_axis_order_and_directional_offsets(self):
        import torch
        z, y, x = torch.meshgrid(*(torch.arange(5.) for _ in range(3)), indexing="ij")
        feature = (100*z + 10*y + x)[None]
        result = calibration.sample_features(feature, torch.tensor([[1.5, 2., 2.]]))
        torch.testing.assert_close(result, torch.tensor([[172., -100., 100., -10., 10., -1., 1.]]))
        border = calibration.sample_features(feature, torch.tensor([[0., 0., 0.]]))
        torch.testing.assert_close(border, torch.tensor([[0., 0., 100., 0., 10., 0., 1.]]))


if __name__ == "__main__":
    unittest.main()
