from __future__ import annotations

import hashlib
import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1] / "kaggle" / "evaluate_ab_stability.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_ab_stability",
    MODULE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {MODULE_PATH}")
STABILITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STABILITY)


class StabilityPartitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.names = [
            "44b6_a",
            "44b6_b",
            "44b6_c",
            "44b6_d",
            "6bba_a",
            "6bba_b",
            "6bba_c",
            "6bba_d",
        ]

    def test_names_hash_includes_extension_and_trailing_newline(self) -> None:
        expected_payload = "".join(
            f"{name}.geff\n" for name in sorted(self.names)
        ).encode()
        self.assertEqual(
            STABILITY.movie_names_sha256(self.names),
            hashlib.sha256(expected_payload).hexdigest(),
        )

    def test_partitions_are_embryo_balanced_and_alternating(self) -> None:
        partitions = STABILITY.build_partitions(self.names)

        self.assertEqual(
            partitions["44b6"],
            ["44b6_a", "44b6_b", "44b6_c", "44b6_d"],
        )
        self.assertEqual(
            partitions["half0"],
            ["44b6_a", "44b6_c", "6bba_a", "6bba_c"],
        )
        self.assertEqual(
            partitions["half1"],
            ["44b6_b", "44b6_d", "6bba_b", "6bba_d"],
        )

    def test_partitions_reject_unknown_or_unbalanced_embryos(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Unexpected embryo IDs"):
            STABILITY.build_partitions([*self.names, "other_a"])
        with self.assertRaisesRegex(RuntimeError, "not balanced"):
            STABILITY.build_partitions(self.names[:-1])


class StabilityManifestTests(unittest.TestCase):
    def manifest(self) -> dict[str, object]:
        return {
            "reference_notebook_sha256": (
                STABILITY.EXPECTED_REFERENCE_NOTEBOOK_SHA256
            ),
            "support_predictor_sha256": (
                STABILITY.EXPECTED_SUPPORT_PREDICTOR_SHA256
            ),
            "primary_weight_sha256": (
                STABILITY.EXPECTED_PRIMARY_WEIGHT_SHA256
            ),
            "secondary_weight_sha256": (
                STABILITY.EXPECTED_SECONDARY_WEIGHT_SHA256
            ),
            "datasets": ["44b6_a", "6bba_a"],
            "det_threshold": 0.96875,
            "ilp_disappearance_weight": 1.5,
            "blend": {
                "edge_weight": 0.15,
                "detection_weight": 0.475,
                "link_mode": "low_margin_consensus",
                "mix_temperature": 1.0,
                "low_margin_max": 0.35,
                "edge_threshold": 0.48,
            },
            "edge_tta": {"mode": "original"},
            "retention_guard": {
                "implementation_version": (
                    "dual_seed_frame_retention_guard_v1"
                ),
                "activation_variant": "blend_guard",
                "fallback_scope": "individual_frame",
                "reference_field": "primary_detection_logits",
                "candidate_extractor": "_detect_cells_pooled",
                "label_free": True,
                "minimum_candidate_retention": 0.90,
                "ab_check": {
                    "paired_frames": 4,
                    "candidate_counts_identical": True,
                    "control_fallback_frames": 0,
                    "candidate_fallback_frames": 2,
                    "guard_activated": True,
                },
            },
            "results": {
                "blend": {
                    "output_dir": "/tmp/control",
                    "patched_predictor_sha256": "abc",
                    "retention_guard": {"evaluated_frames": 4},
                },
                "blend_guard": {
                    "output_dir": "/tmp/candidate",
                    "patched_predictor_sha256": "abc",
                    "retention_guard": {"evaluated_frames": 4},
                },
            },
        }

    def test_manifest_accepts_the_single_variable_contract(self) -> None:
        control, candidate = STABILITY.validate_inference_manifest(
            self.manifest(),
            ["44b6_a", "6bba_a"],
        )

        self.assertEqual(control, Path("/tmp/control"))
        self.assertEqual(candidate, Path("/tmp/candidate"))

    def test_manifest_rejects_parameter_drift_or_no_activation(self) -> None:
        drift = self.manifest()
        drift["blend"]["detection_weight"] = 0.65
        with self.assertRaisesRegex(RuntimeError, "detection_weight mismatch"):
            STABILITY.validate_inference_manifest(
                drift,
                ["44b6_a", "6bba_a"],
            )

        inactive = self.manifest()
        inactive["retention_guard"]["ab_check"]["guard_activated"] = False
        with self.assertRaisesRegex(RuntimeError, "guard_activated mismatch"):
            STABILITY.validate_inference_manifest(
                inactive,
                ["44b6_a", "6bba_a"],
            )

        predictor_drift = self.manifest()
        predictor_drift["results"]["blend_guard"][
            "patched_predictor_sha256"
        ] = "def"
        with self.assertRaisesRegex(RuntimeError, "predictor SHA256 changed"):
            STABILITY.validate_inference_manifest(
                predictor_drift,
                ["44b6_a", "6bba_a"],
            )


class PairedOutcomeTests(unittest.TestCase):
    def test_outcomes_and_paired_summary(self) -> None:
        records = [
            {"delta_score": 0.01, "outcome": "win"},
            {"delta_score": 0.00, "outcome": "tie"},
            {"delta_score": -0.02, "outcome": "loss"},
        ]

        summary = STABILITY.paired_stats(records)

        self.assertEqual(STABILITY.outcome(1e-3, 1e-12), "win")
        self.assertEqual(STABILITY.outcome(0.0, 1e-12), "tie")
        self.assertEqual(STABILITY.outcome(-1e-3, 1e-12), "loss")
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["ties"], 1)
        self.assertEqual(summary["losses"], 1)
        self.assertEqual(summary["median_delta"], 0.0)

    def test_per_movie_diagnostic_handles_no_division_events(self) -> None:
        metrics = STABILITY.per_movie_diagnostic(
            {
                "adj_edge_jaccard": 0.8,
                "edge_jaccard": 0.75,
                "node_recall": 0.9,
                "total_node_ratio": 1.1,
                "edge_tp": 8,
                "edge_fp": 1,
                "edge_fn": 1,
                "division_tp": 0,
                "division_fp": 0,
                "division_fn": 0,
                "num_pred_nodes": 20,
            }
        )

        self.assertEqual(metrics["division_jaccard"], 0.0)
        self.assertEqual(metrics["score"], 0.8)


if __name__ == "__main__":
    unittest.main()
