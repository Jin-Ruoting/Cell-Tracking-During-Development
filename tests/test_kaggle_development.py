import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import build_kaggle_development as build
import capture_point_detector_peaks as capture
import frozen_development_corpus as corpus
import run_kaggle_development as cloud


class KaggleDevelopmentTests(unittest.TestCase):
    def test_historical_selection_is_balanced_deterministic_and_excludes_visible_movies(self):
        stats = {}
        for embryo in ("44b6", "6bba"):
            for i in range(17):
                stats[f"{embryo}_p{i:02}"] = {"divisions": 20 - i, "edges": 100 + i}
                stats[f"{embryo}_n{i:02}"] = {"divisions": 0, "edges": 100 + i}
        stats.update({n: {"divisions": 999, "edges": 999} for n in corpus.EXCLUDED})
        names = corpus.select_names(stats)
        self.assertEqual(len(names), 64)
        self.assertEqual(names, corpus.select_names(dict(reversed(list(stats.items())))))
        self.assertFalse(set(names) & corpus.EXCLUDED)
        for embryo in ("44b6", "6bba"):
            self.assertNotIn(embryo + "_p16", names)
            self.assertNotIn(embryo + "_n16", names)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            corpus.select_names({n: v for n, v in stats.items() if not n.startswith("6bba_")})

    def test_corpus_cannot_be_replaced_with_another_64_movies(self):
        names = [f"44b6_{i:08}" for i in range(64)]
        with self.assertRaisesRegex(ValueError, "checksum"):
            corpus.validate_manifest({"all_datasets": names, "datasets": names}, "full")

    def test_full_run_requires_successful_smoke_of_the_same_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "receipt.json"
            value = {"mode": "smoke", "technical_check_passed": True, "control_byte_parity": True,
                     "control_score_reproduced": True, "protocol_sha256": "frozen"}
            path.write_text(json.dumps(value))
            build.require_smoke_receipt(path, "frozen")
            for key, changed in (("control_byte_parity", False), ("mode", "full"), ("protocol_sha256", "other")):
                path.write_text(json.dumps({**value, key: changed}))
                with self.assertRaises(ValueError):
                    build.require_smoke_receipt(path, "frozen")
            with self.assertRaises(ValueError):
                build.require_smoke_receipt(None, "frozen")

    def test_runtime_rejects_mutated_bundled_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            source = root / "module.py"
            source.write_text("pass\n")
            manifest = {"mode": "smoke", "files": {"module.py": hashlib.sha256(source.read_bytes()).hexdigest()}}
            (root / "bundle_manifest.json").write_text(json.dumps(manifest))
            cloud.verified_bundle(root)
            source.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                cloud.verified_bundle(root)

    def test_unreviewed_scorer_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "wrong.tar.gz"
            archive.write_bytes(b"not the pinned scorer")
            with self.assertRaisesRegex(ValueError, "checksum"):
                build.scoring_files(archive)

    def test_existing_server_cache_entry_preserves_selection_and_sharding(self):
        args = SimpleNamespace(control_dir=Path("control"), mode="full", shard=1)
        with patch.object(capture.reference, "selected_names", return_value=["a", "b", "c", "d"]) as selection, \
             patch.object(capture.subprocess, "check_output", return_value="commit\n"), \
             patch.object(capture, "run_movies") as run_movies:
            capture.run(args)
        selection.assert_called_once_with(Path("control"), "full")
        run_movies.assert_called_once_with(args, ["b", "d"], "commit")


if __name__ == "__main__":
    unittest.main()
