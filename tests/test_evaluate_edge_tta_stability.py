from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "kaggle"
    / "evaluate_edge_tta_stability.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "evaluate_edge_tta_stability",
    MODULE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {MODULE_PATH}")
STABILITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STABILITY)


class EdgeTtaManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.names = ["44b6_a", "6bba_a"]

    def common_manifest(self, disappearance: float) -> dict[str, object]:
        return {
            "reference_notebook_sha256": (
                STABILITY.common.EXPECTED_REFERENCE_NOTEBOOK_SHA256
            ),
            "support_predictor_sha256": (
                STABILITY.common.EXPECTED_SUPPORT_PREDICTOR_SHA256
            ),
            "primary_weight_sha256": (
                STABILITY.common.EXPECTED_PRIMARY_WEIGHT_SHA256
            ),
            "secondary_weight_sha256": (
                STABILITY.common.EXPECTED_SECONDARY_WEIGHT_SHA256
            ),
            "datasets": self.names,
            "det_threshold": 0.96875,
            "ilp_disappearance_weight": disappearance,
            "blend": {
                "edge_weight": 0.15,
                "detection_weight": 0.475,
                "link_mode": "low_margin_consensus",
                "mix_temperature": 1.0,
                "low_margin_max": 0.35,
                "edge_threshold": 0.48,
            },
            "results": {
                "blend": {
                    "datasets": 2,
                    "output_dir": "/tmp/raw",
                }
            },
        }

    def common_edge_tta(self) -> dict[str, object]:
        return {
            "implementation_version": "dual_seed_edge_tta_v2",
            "reference_notebook_sha256": (
                STABILITY.EXPECTED_EDGE_TTA_NOTEBOOK_SHA256
            ),
            "models": "all_loaded",
            "aggregation_domain": "raw_edge_logits",
            "aggregation_stage": "per_seed_before_seed_calibration",
            "node_policy": "shared_canonical_detection_nodes",
            "feature_alignment": "inverse_map_to_canonical_zyx",
            "detection_tta_changed": False,
        }

    def test_e023_contract_accepts_frozen_legacy_views(self) -> None:
        control = self.common_manifest(1.575)
        candidate = self.common_manifest(1.575)
        candidate["edge_tta"] = {
            **self.common_edge_tta(),
            "mode": "pilkwang_legacy_d4",
            "application": "global",
            "requested_view_count": 8,
            "unique_spatial_view_count": 7,
            "legacy_anti_transpose_duplicates_flip_x": True,
            "view_names": [
                "identity",
                "flip_x",
                "flip_y",
                "flip_xy",
                "rot90",
                "rot270",
                "transpose",
                "legacy_anti_transpose",
            ],
            "view_weights": {
                "identity": 0.125,
                "flip_x": 0.125,
                "flip_y": 0.125,
                "flip_xy": 0.125,
                "rot90": 0.125,
                "rot270": 0.125,
                "transpose": 0.125,
                "legacy_anti_transpose": 0.125,
            },
        }

        output = STABILITY.validate_common_manifest(
            candidate,
            self.names,
            1.575,
            "candidate",
        )
        STABILITY.validate_edge_tta_contract(
            "e023",
            control,
            candidate,
            self.names,
        )

        self.assertEqual(output, Path("/tmp/raw").resolve())

    def test_e027_contract_requires_selected_targets(self) -> None:
        control = self.common_manifest(1.5)
        control["edge_tta"] = {
            "mode": "original",
            "requested_view_count": 1,
            "unique_spatial_view_count": 1,
            "view_names": ["identity"],
            "view_weights": {"identity": 1.0},
            "detection_tta_changed": False,
        }
        candidate = self.common_manifest(1.5)
        views = (
            "identity",
            "flip_x",
            "flip_y",
            "flip_xy",
            "rot90",
            "rot270",
            "transpose",
            "anti_transpose",
        )
        candidate["edge_tta"] = {
            **self.common_edge_tta(),
            "mode": "corrected_d4",
            "application": "ambiguous_parent_consensus",
            "requested_view_count": 8,
            "unique_spatial_view_count": 8,
            "legacy_anti_transpose_duplicates_flip_x": False,
            "ambiguous_parent_margin_max": 0.35,
            "ambiguous_parent_policy": {
                "replacement_scope": "selected_target_logit_columns",
                "requires_identity_seed_disagreement": True,
                "requires_multiview_seed_consensus": True,
                "requires_primary_identity_low_margin": True,
                "requires_primary_parent_change": True,
                "unselected_policy": "identity_logits_exact",
            },
            "view_names": list(views),
            "view_weights": {
                view: 0.5 if view == "identity" else 1.0 / 14.0
                for view in views
            },
        }
        candidate["results"]["blend"]["parent_rerank"] = {
            "selected_targets": 3,
            "datasets": {name: {} for name in self.names},
        }

        STABILITY.validate_edge_tta_contract(
            "e027",
            control,
            candidate,
            self.names,
        )

        candidate["results"]["blend"]["parent_rerank"][
            "selected_targets"
        ] = 0
        with self.assertRaisesRegex(RuntimeError, "did not select"):
            STABILITY.validate_edge_tta_contract(
                "e027",
                control,
                candidate,
                self.names,
            )


class E000ReportTests(unittest.TestCase):
    def test_e000_report_requires_strict_topology(self) -> None:
        names = ["44b6_a", "6bba_a"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            output = root / "post"
            geffs = output / "score_e000_safe" / "geffs"
            report = {
                "e000_only": True,
                "e025_exact": False,
                "notebook_sha256": (
                    STABILITY.common.EXPECTED_E025_NOTEBOOK_SHA256
                ),
                "deepcenter_checkpoint": None,
                "baseline_dir": str(raw),
                "datasets": names,
                "max_outdegree": 2,
                "output_dir": str(output),
                "variants": {
                    "e000_safe": {
                        name: {
                            "duplicate_edges": 0,
                            "dangling_edges": 0,
                            "nonconsecutive_edges": 0,
                            "max_indegree": 1,
                            "max_outdegree": 2,
                            "nonbinary_sources": 0,
                        }
                        for name in names
                    }
                },
            }

            STABILITY.validate_e000_report(
                report,
                raw,
                geffs,
                names,
                "control",
            )

            report["variants"]["e000_safe"]["44b6_a"][
                "max_indegree"
            ] = 2
            with self.assertRaisesRegex(RuntimeError, "topology gate"):
                STABILITY.validate_e000_report(
                    report,
                    raw,
                    geffs,
                    names,
                    "control",
                )


if __name__ == "__main__":
    unittest.main()
