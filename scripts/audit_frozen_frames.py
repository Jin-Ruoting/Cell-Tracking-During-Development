#!/usr/bin/env python3
"""Audit exact consecutive-frame duplicates in Biohub Zarr v3 arrays."""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable


READ_SIZE = 1024 * 1024


def _files_equal(left: Path, right: Path) -> bool:
    left_stat = left.stat()
    right_stat = right.stat()
    if left_stat.st_size != right_stat.st_size:
        return False

    with left.open("rb") as left_file, right.open("rb") as right_file:
        while True:
            left_block = left_file.read(READ_SIZE)
            right_block = right_file.read(READ_SIZE)
            if left_block != right_block:
                return False
            if not left_block:
                return True


def _chunk_paths(
    array_dir: Path,
    time_index: int,
    spatial_chunk_counts: Iterable[int],
) -> list[Path]:
    coordinates = product(*(range(count) for count in spatial_chunk_counts))
    return [
        array_dir.joinpath("c", str(time_index), *(str(value) for value in coord))
        for coord in coordinates
    ]


def read_layout(clip_dir: Path) -> tuple[Path, tuple[int, ...], tuple[int, ...]]:
    array_dir = clip_dir / "0"
    metadata_path = array_dir / "zarr.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    shape = tuple(int(value) for value in metadata["shape"])
    chunk_shape = tuple(
        int(value)
        for value in metadata["chunk_grid"]["configuration"]["chunk_shape"]
    )
    if len(shape) != 4 or len(chunk_shape) != 4:
        raise ValueError(f"{clip_dir.name}: expected 4D tzyx array")
    if chunk_shape[0] != 1:
        raise ValueError(
            f"{clip_dir.name}: time chunk size must be 1, got {chunk_shape[0]}"
        )

    encoding = metadata.get("chunk_key_encoding", {})
    encoding_name = encoding.get("name", "default")
    separator = encoding.get("configuration", {}).get("separator", "/")
    if encoding_name != "default" or separator != "/":
        raise ValueError(
            f"{clip_dir.name}: unsupported chunk encoding "
            f"{encoding_name!r} with separator {separator!r}"
        )

    return array_dir, shape, chunk_shape


def audit_clip(clip_dir: Path) -> dict[str, Any]:
    array_dir, shape, chunk_shape = read_layout(clip_dir)
    spatial_chunk_counts = tuple(
        math.ceil(size / chunk)
        for size, chunk in zip(shape[1:], chunk_shape[1:])
    )

    duplicate_pairs: list[dict[str, int]] = []
    compared_chunk_pairs = 0
    for target_t in range(1, shape[0]):
        source_paths = _chunk_paths(
            array_dir,
            target_t - 1,
            spatial_chunk_counts,
        )
        target_paths = _chunk_paths(array_dir, target_t, spatial_chunk_counts)
        missing = [
            str(path)
            for path in (*source_paths, *target_paths)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{clip_dir.name}: missing {len(missing)} chunks; first={missing[0]}"
            )

        equal = True
        for source_path, target_path in zip(source_paths, target_paths):
            compared_chunk_pairs += 1
            if not _files_equal(source_path, target_path):
                equal = False
                break
        if equal:
            duplicate_pairs.append(
                {"source_t": target_t - 1, "target_t": target_t}
            )

    clip_name = clip_dir.name.removesuffix(".zarr")
    embryo = clip_name.split("_", 1)[0]
    return {
        "clip": clip_name,
        "embryo": embryo,
        "frames": shape[0],
        "frame_shape": list(shape[1:]),
        "chunk_shape": list(chunk_shape),
        "compared_transitions": max(shape[0] - 1, 0),
        "compared_chunk_pairs": compared_chunk_pairs,
        "duplicate_transition_count": len(duplicate_pairs),
        "duplicate_pairs": duplicate_pairs,
    }


def discover_clips(data_root: Path, split: str) -> list[Path]:
    split_dir = data_root / split
    search_dir = split_dir if split_dir.is_dir() else data_root
    clips = sorted(
        path
        for path in search_dir.iterdir()
        if path.is_dir() and path.name.endswith(".zarr")
    )
    if not clips:
        raise FileNotFoundError(f"no .zarr clips found under {search_dir}")
    return clips


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    embryo_rows: dict[str, dict[str, int]] = {}
    for row in results:
        embryo = str(row["embryo"])
        group = embryo_rows.setdefault(
            embryo,
            {
                "clips": 0,
                "clips_with_duplicates": 0,
                "duplicate_transitions": 0,
                "compared_transitions": 0,
            },
        )
        group["clips"] += 1
        group["clips_with_duplicates"] += int(
            int(row["duplicate_transition_count"]) > 0
        )
        group["duplicate_transitions"] += int(row["duplicate_transition_count"])
        group["compared_transitions"] += int(row["compared_transitions"])

    return {
        "clips": len(results),
        "clips_with_duplicates": sum(
            int(int(row["duplicate_transition_count"]) > 0) for row in results
        ),
        "duplicate_transitions": sum(
            int(row["duplicate_transition_count"]) for row in results
        ),
        "compared_transitions": sum(
            int(row["compared_transitions"]) for row in results
        ),
        "by_embryo": embryo_rows,
    }


def run_audit(data_root: Path, split: str, workers: int) -> dict[str, Any]:
    clips = discover_clips(data_root, split)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(audit_clip, clips))
    results.sort(key=lambda row: str(row["clip"]))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root.resolve()),
        "split": split,
        "comparison": "exact encoded Zarr v3 chunk bytes",
        "summary": summarise(results),
        "clips": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_root",
        type=Path,
        help="Competition root containing train/, or the train directory itself.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; otherwise print to stdout.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    report = run_audit(args.data_root, args.split, args.workers)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        print(f"Wrote {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
