from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1] / "kaggle" / "validate_e006_postprocess.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_e006_postprocess",
    MODULE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {MODULE_PATH}")
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def e025_namespace() -> dict[str, object]:
    return {
        "OUTPUT_SAFE_DIVISIONS": True,
        "OUTPUT_MOTION_RELINK": True,
        "MOTION_RELINK_TIGHT_UM": 6.0,
        "MOTION_RELINK_RELAXED_UM": 10.0,
        "MOTION_RELINK_LEARNED_BONUS": 1.0,
        "OUTPUT_GAP_CLOSE": True,
        "GAP_CLOSE_MAX_GAP": 2,
        "GAP_CLOSE_UM": 5.8,
        "GAP_DENSITY_ADAPTIVE": True,
        "OUTPUT_FILTER_SHORT_TRACKS": True,
        "OUTPUT_MIN_TRACK_LEN": 6,
        "ADAPTIVE_SHORT_TRACK_RESCUE": False,
        "OUTPUT_LINEFIT_SMOOTH": True,
        "OUTPUT_PRUNE_ISOLATED": True,
        "USE_DEEPCENTER_VETO": True,
        "REQUIRE_DEEPCENTER_VETO": True,
        "DEEPCENTER_GAP_VETO": True,
        "DEEPCENTER_SAFE_DIV_VETO": False,
        "DEEPCENTER_GAP_THRESHOLD": 0.25,
        "DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM": 8.5,
        "DEEPCENTER_EXPECTED_EPOCH": 500,
        "DEEPCENTER_EXPECTED_SHA256": (
            VALIDATOR.PUBLIC_V40_DEEPCENTER_SHA256
        ),
        "DEEPCENTER_SAFE_DIV_THRESHOLD": 0.12,
    }


class E025ExactModeTests(unittest.TestCase):
    def test_offline_namespace_provides_notebook_checksum_helper(self) -> None:
        notebook = Path("/tmp/e025.ipynb")
        namespace = VALIDATOR.notebook_execution_namespace(notebook)

        self.assertEqual(namespace["__file__"], str(notebook))
        self.assertIs(namespace["_sha256_file"], VALIDATOR.file_sha256)

    def test_e025_config_accepts_frozen_values(self) -> None:
        VALIDATOR.validate_e025_config(e025_namespace())

    def test_e025_config_rejects_drift(self) -> None:
        namespace = e025_namespace()
        namespace["MOTION_RELINK_RELAXED_UM"] = 9.5
        with self.assertRaisesRegex(RuntimeError, "E025 configuration mismatch"):
            VALIDATOR.validate_e025_config(namespace)

    def test_e025_mode_builds_one_exact_variant(self) -> None:
        specs = VALIDATOR.build_variant_specs(
            e025_namespace(),
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            None,
            False,
            True,
        )

        self.assertEqual(set(specs), {"e025_exact"})
        spec = specs["e025_exact"]
        self.assertTrue(spec["safe_divisions"])
        self.assertTrue(spec["deepcenter"])
        self.assertTrue(spec["deepcenter_gap_veto"])
        self.assertFalse(spec["deepcenter_safe_div_veto"])
        self.assertEqual(spec["deepcenter_gap_threshold"], 0.25)
        self.assertEqual(spec["deepcenter_gap_confirm_min_span_um"], 8.5)

    def test_e025_hash_fails_before_exec_or_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notebook = root / "untrusted.ipynb"
            notebook.write_text('{"cells": []}\n', encoding="utf-8")
            output_dir = root / "output"
            command = [
                sys.executable,
                str(MODULE_PATH),
                "--notebook",
                str(notebook),
                "--baseline-dir",
                str(root / "baseline"),
                "--image-dir",
                str(root / "images"),
                "--ground-truth-dir",
                str(root / "ground-truth"),
                "--runtime-dir",
                str(root / "runtime"),
                "--support-src",
                str(root / "support"),
                "--scorer-dir",
                str(root / "scorer"),
                "--output-dir",
                str(output_dir),
                "--e025-exact",
                "--deepcenter-checkpoint",
                str(root / "deepcenter.pt"),
            ]

            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("E025 notebook SHA256 mismatch", result.stderr)
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
