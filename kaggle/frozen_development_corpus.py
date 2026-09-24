"""Reconstruct, then checksum-check the pre-existing 64-movie development set.

Selection is the historical 394154a division-calibration rule, not a new split.
Only this setup step reads label metadata; prediction receives image-only links.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import run_geometric_reference as reference

EXCLUDED = {"44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"}


def select_names(stats: dict) -> list[str]:
    selected = []
    for embryo in ("44b6", "6bba"):
        names = [n for n in stats if n.startswith(embryo + "_") and n not in EXCLUDED]
        positives = sorted((n for n in names if stats[n]["divisions"] > 0),
                           key=lambda n: (-stats[n]["divisions"], -stats[n]["edges"], n))[:16]
        available = {n for n in names if stats[n]["divisions"] == 0}
        if len(positives) != 16 or len(available) < 16:
            raise ValueError("Incomplete historical selection pool")
        selected.extend(positives)
        for positive in positives:
            negative = min(available, key=lambda n: (abs(stats[n]["edges"] - stats[positive]["edges"]), n))
            selected.append(negative)
            available.remove(negative)
    return sorted(selected)


def validate_manifest(manifest: dict, mode: str) -> list[str]:
    names = manifest["all_datasets"]
    if (mode not in ("smoke", "full") or names != sorted(set(names)) or len(names) != 64
            or reference.stability.movie_names_sha256(names) != reference.CORPUS_SHA256):
        raise ValueError("Historical 64-movie corpus checksum mismatch")
    selected = list(reference.SMOKE_NAMES) if mode == "smoke" else names
    if not set(selected) <= set(names) or manifest.get("datasets") != selected:
        raise ValueError("Selected movies differ from the frozen run mode")
    return selected


def reconstruct(train_dir: Path, mode: str) -> dict:
    import zarr

    stats = {}
    for path in sorted(train_dir.glob("*.geff")):
        graph = zarr.open_group(str(path), mode="r")
        times = dict(zip(map(int, graph["nodes/ids"][:]), map(int, graph["nodes/props/t/values"][:])))
        edges = graph["edges/ids"][:]
        children = defaultdict(list)
        for source, target in edges:
            source, target = int(source), int(target)
            if source in times and target in times:
                children[source].append(target)
        divisions = sum(len(kids) == 2 and all(times[k] == times[parent] + 1 for k in kids)
                        for parent, kids in children.items())
        stats[path.stem] = {"edges": len(edges), "divisions": divisions}
    names = select_names(stats)
    manifest = {"all_datasets": names, "datasets": list(reference.SMOKE_NAMES) if mode == "smoke" else names,
                "corpus_sha256": reference.CORPUS_SHA256, "historical_selection_commit": "394154a",
                "selection_reads_ground_truth": True, "predictions_read_ground_truth": False,
                "excluded": sorted(EXCLUDED), "selected_statistics": {n: stats[n] for n in names}}
    validate_manifest(manifest, mode)
    return manifest
