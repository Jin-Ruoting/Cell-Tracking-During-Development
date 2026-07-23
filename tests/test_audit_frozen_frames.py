from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_frozen_frames import audit_clip, discover_clips, summarise


def make_clip(
    root: Path,
    name: str,
    shape: tuple[int, int, int, int],
    chunk_shape: tuple[int, int, int, int],
    chunks: dict[tuple[int, int, int, int], bytes],
) -> Path:
    clip_dir = root / f"{name}.zarr"
    array_dir = clip_dir / "0"
    metadata = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": list(shape),
        "data_type": "uint16",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(chunk_shape)},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
    }
    array_dir.mkdir(parents=True)
    (array_dir / "zarr.json").write_text(json.dumps(metadata), encoding="utf-8")
    for coordinate, payload in chunks.items():
        chunk_path = array_dir / "c"
        for value in coordinate:
            chunk_path /= str(value)
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_path.write_bytes(payload)
    return clip_dir


class FrozenFrameAuditTest(unittest.TestCase):
    def test_detects_one_exact_duplicate_transition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            clip = make_clip(
                root,
                "6bba_example",
                (3, 1, 1, 1),
                (1, 1, 1, 1),
                {
                    (0, 0, 0, 0): b"same",
                    (1, 0, 0, 0): b"same",
                    (2, 0, 0, 0): b"different",
                },
            )

            result = audit_clip(clip)

        self.assertEqual(result["duplicate_transition_count"], 1)
        self.assertEqual(
            result["duplicate_pairs"],
            [{"source_t": 0, "target_t": 1}],
        )

    def test_compares_every_spatial_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            clip = make_clip(
                root,
                "44b6_example",
                (3, 2, 1, 1),
                (1, 1, 1, 1),
                {
                    (0, 0, 0, 0): b"a",
                    (0, 1, 0, 0): b"b",
                    (1, 0, 0, 0): b"a",
                    (1, 1, 0, 0): b"b",
                    (2, 0, 0, 0): b"a",
                    (2, 1, 0, 0): b"changed",
                },
            )

            result = audit_clip(clip)

        self.assertEqual(result["duplicate_transition_count"], 1)
        self.assertEqual(result["compared_transitions"], 2)

    def test_rejects_time_chunks_larger_than_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            clip = make_clip(
                root,
                "44b6_invalid",
                (2, 1, 1, 1),
                (2, 1, 1, 1),
                {(0, 0, 0, 0): b"combined"},
            )

            with self.assertRaisesRegex(ValueError, "time chunk size"):
                audit_clip(clip)

    def test_discovers_split_and_summarises_by_embryo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            train_dir = root / "train"
            train_dir.mkdir()
            for name in ("44b6_a", "6bba_b"):
                make_clip(
                    train_dir,
                    name,
                    (2, 1, 1, 1),
                    (1, 1, 1, 1),
                    {
                        (0, 0, 0, 0): name.encode(),
                        (1, 0, 0, 0): name.encode(),
                    },
                )

            clips = discover_clips(root, "train")
            summary = summarise([audit_clip(path) for path in clips])

        self.assertEqual(len(clips), 2)
        self.assertEqual(summary["clips_with_duplicates"], 2)
        self.assertEqual(summary["by_embryo"]["44b6"]["clips"], 1)
        self.assertEqual(summary["by_embryo"]["6bba"]["duplicate_transitions"], 1)


if __name__ == "__main__":
    unittest.main()
