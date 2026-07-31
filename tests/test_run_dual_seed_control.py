from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = (
    Path(__file__).parents[1] / "kaggle" / "run_dual_seed_control.py"
)
SPEC = importlib.util.spec_from_file_location("run_dual_seed_control", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {MODULE_PATH}")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class RetentionGuardPatchTests(unittest.TestCase):
    def test_patch_replaces_exact_blend_once_and_compiles(self) -> None:
        source = (
            f"{RUNNER.RETENTION_GUARD_HELPER_ANCHOR}):\n"
            "    if True:\n"
            "        if True:\n"
            "            if True:\n"
            "                for f in range(1):\n"
            f"{RUNNER.RETENTION_GUARD_BLEND_BLOCK}\n"
        )

        patched = RUNNER.apply_retention_guard_patch(source)

        self.assertIn(RUNNER.RETENTION_GUARD_IMPLEMENTATION_VERSION, patched)
        self.assertIn("BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION", patched)
        self.assertIn("primary_det if use_primary_detection", patched)
        compile(patched, "<retention-guard-test>", "exec")

    def test_patch_rejects_missing_or_duplicate_anchor(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "found 0"):
            RUNNER.apply_retention_guard_patch("pass\n")
        duplicate = (
            f"{RUNNER.RETENTION_GUARD_HELPER_ANCHOR}):\n"
            + RUNNER.RETENTION_GUARD_BLEND_BLOCK
            + "\n"
            + RUNNER.RETENTION_GUARD_BLEND_BLOCK
        )
        with self.assertRaisesRegex(RuntimeError, "found 2"):
            RUNNER.apply_retention_guard_patch(duplicate)

    def test_patch_accepts_edge_tta_blend_layout(self) -> None:
        self.assertEqual(
            RUNNER.EDGE_TTA_ENCODE_REPLACEMENT.count(
                RUNNER.RETENTION_GUARD_BLEND_BLOCK
            ),
            1,
        )
        patched = RUNNER.apply_retention_guard_patch(
            RUNNER.RETENTION_GUARD_HELPER_ANCHOR
            + "):\n"
            + RUNNER.EDGE_TTA_ENCODE_REPLACEMENT
        )
        self.assertIn(RUNNER.RETENTION_GUARD_IMPLEMENTATION_VERSION, patched)

    def test_edge_tta_and_retention_patches_compose(self) -> None:
        predictor = (
            RUNNER.EDGE_TTA_HELPER_ANCHOR
            + "):\n"
            + RUNNER.EDGE_TTA_ENCODE_START
            + RUNNER.EDGE_TTA_ENCODE_END
            + "\n"
            + RUNNER.EDGE_TTA_EDGE_START
            + RUNNER.EDGE_TTA_EDGE_END
            + "\n"
            + RUNNER.EDGE_TTA_CLEANUP_ANCHOR
        )

        edge_patched = RUNNER.apply_edge_tta_patch(predictor)
        combined = RUNNER.apply_retention_guard_patch(edge_patched)

        self.assertIn(RUNNER.EDGE_TTA_IMPLEMENTATION_VERSION, combined)
        self.assertIn(RUNNER.RETENTION_GUARD_IMPLEMENTATION_VERSION, combined)

    def test_injected_policy_uses_strict_retention_boundary(self) -> None:
        namespace: dict[str, object] = {}
        exec(RUNNER.RETENTION_GUARD_HELPERS, namespace)
        policy = namespace["_retention_guard_uses_primary"]

        self.assertFalse(policy(10, 9, 0.90))
        self.assertTrue(policy(10, 8, 0.90))
        self.assertFalse(policy(0, 0, 0.90))
        self.assertFalse(policy(10, 0, 0.0))


class RetentionDiagnosticsTests(unittest.TestCase):
    def write_log(self, text: str) -> Path:
        temporary = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
        )
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        with temporary:
            temporary.write(text)
        return Path(temporary.name)

    def test_parse_summarize_and_compare_paired_records(self) -> None:
        candidate_path = self.write_log(
            "BIOHUB_RETENTION_FRAME\t44b6_a\t0\t10\t8\t1\n"
            "BIOHUB_RETENTION_FRAME\t6bba_b\t1\t8\t8\t0\n"
        )
        control_path = self.write_log(
            "noise\n"
            "BIOHUB_RETENTION_FRAME\t44b6_a\t0\t10\t8\t0\n"
            "BIOHUB_RETENTION_FRAME\t6bba_b\t1\t8\t8\t0\n"
        )

        control = RUNNER.parse_retention_records(control_path)
        candidate = RUNNER.parse_retention_records(candidate_path)
        summary = RUNNER.summarize_retention_records(candidate)
        comparison = RUNNER.compare_retention_controls(control, candidate)

        self.assertEqual(summary["evaluated_frames"], 2)
        self.assertEqual(summary["fallback_frames"], 1)
        self.assertEqual(summary["minimum_retention"], 0.8)
        self.assertEqual(comparison["paired_frames"], 2)
        self.assertTrue(comparison["candidate_counts_identical"])
        self.assertTrue(comparison["guard_activated"])

    def test_parser_rejects_duplicates_and_missing_records(self) -> None:
        empty = self.write_log("unrelated\n")
        with self.assertRaisesRegex(RuntimeError, "No retention diagnostics"):
            RUNNER.parse_retention_records(empty)

        duplicate = self.write_log(
            "BIOHUB_RETENTION_FRAME\t44b6_a\t0\t10\t9\t0\n"
            "BIOHUB_RETENTION_FRAME\t44b6_a\t0\t10\t9\t0\n"
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate retention key"):
            RUNNER.parse_retention_records(duplicate)

    def test_comparison_rejects_pre_fallback_count_drift(self) -> None:
        control = {("44b6_a", 0): (10, 9, False)}
        candidate = {("44b6_a", 0): (10, 8, True)}
        with self.assertRaisesRegex(RuntimeError, "changed before fallback"):
            RUNNER.compare_retention_controls(control, candidate)


class RetentionEnvironmentTests(unittest.TestCase):
    def test_guard_environment_is_the_only_blend_difference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                runtime_dir=root / "runtime",
                output_dir=root / "output",
                export_preilp=False,
                edge_tta_mode="original",
                edge_tta_original_weight=None,
                secondary_weights=root / "secondary.pth",
                blend_edge_weight=0.15,
                blend_detection_weight=0.475,
                blend_link_mode="low_margin_consensus",
                blend_mix_temperature=1.0,
                blend_low_margin_max=0.35,
                blend_edge_threshold=0.48,
                guard_min_candidate_retention=0.90,
            )
            repo = root / "repo"

            control = RUNNER.variant_environment(
                args,
                "blend",
                "0",
                repo,
            )
            candidate = RUNNER.variant_environment(
                args,
                "blend_guard",
                "1",
                repo,
            )

        ignored = {"CUDA_VISIBLE_DEVICES", "USER"}
        control_filtered = {
            key: value for key, value in control.items() if key not in ignored
        }
        candidate_filtered = {
            key: value
            for key, value in candidate.items()
            if key not in ignored
            and key != "BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION"
        }
        self.assertEqual(control_filtered, candidate_filtered)
        self.assertNotIn(
            "BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION",
            control,
        )
        self.assertEqual(
            candidate["BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION"],
            "0.9",
        )


if __name__ == "__main__":
    unittest.main()
