#!/usr/bin/env python3
"""Run pinned primary, secondary, and dual-seed inference controls.

The public dual-seed notebook is treated as a pinned external implementation:
its SHA256 is verified, and only its guarded predictor-patch prefix is executed
against temporary copies of the public support repository. The server checkout
and support artifact are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


REFERENCE_NOTEBOOK_SHA256 = (
    "70e0c300ceae3cd7ee2cf1650c4a5f74463543e3aae1b486ba5f729a76281656"
)
EDGE_TTA_REFERENCE_SHA256 = (
    "fd4d166ef72afc8db2e191df6e7dad661b18151f6faf9fa303e97531b6de892c"
)
SUPPORT_PREDICTOR_SHA256 = (
    "c44e771ba5980b820f93091e03a303c25dfe8f3232e501f54dc9565731c234b9"
)
PRIMARY_WEIGHT_SHA256 = (
    "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
)
SECONDARY_WEIGHT_SHA256 = (
    "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"
)
PATCH_CELL_INDEX = 9
PATCH_END_MARKER = "\ndef list_test_stems() -> list[str]:\n"
VARIANTS = ("primary", "secondary", "blend", "blend_guard")
BLEND_VARIANTS = ("blend", "blend_guard")
LINK_MODES = ("raw", "calibrated", "adaptive", "low_margin_consensus")
EDGE_TTA_MODES = ("original", "pilkwang_legacy_d4", "corrected_d4")
EDGE_TTA_IMPLEMENTATION_VERSION = "dual_seed_edge_tta_v1"
RETENTION_GUARD_IMPLEMENTATION_VERSION = (
    "dual_seed_frame_retention_guard_v1"
)
RETENTION_LOG_PREFIX = "BIOHUB_RETENTION_FRAME\t"
RETENTION_GUARD_HELPER_ANCHOR = "@torch.no_grad()\ndef predict_video("
RETENTION_GUARD_HELPERS = """# --- dual_seed_frame_retention_guard_v1 ---
def _retention_guard_uses_primary(
    primary_candidates: int,
    blended_candidates: int,
    minimum_retention: float,
) -> bool:
    if primary_candidates < 0 or blended_candidates < 0:
        raise ValueError("Retention candidate counts must be non-negative")
    if not 0.0 <= minimum_retention <= 1.0:
        raise ValueError(
            "BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION must be in [0, 1]"
        )
    if minimum_retention == 0.0 or primary_candidates == 0:
        return False
    return (
        blended_candidates / primary_candidates
    ) < minimum_retention


"""
LEGACY_EDGE_TTA_VIEWS = (
    "identity",
    "flip_x",
    "flip_y",
    "flip_xy",
    "rot90",
    "rot270",
    "transpose",
    "legacy_anti_transpose",
)
CORRECTED_EDGE_TTA_VIEWS = (
    "identity",
    "flip_x",
    "flip_y",
    "flip_xy",
    "rot90",
    "rot270",
    "transpose",
    "anti_transpose",
)

EDGE_TTA_HELPER_ANCHOR = "@torch.no_grad()\ndef predict_video("
EDGE_TTA_HELPERS = """# --- dual_seed_edge_tta_v1 ---
_EDGE_TTA_VIEW_ORDER = (
    "identity", "flip_x", "flip_y", "flip_xy",
    "rot90", "rot270", "transpose",
    "legacy_anti_transpose", "anti_transpose",
)
_LEGACY_PSEUDO_D4 = (
    "identity", "flip_x", "flip_y", "flip_xy",
    "rot90", "rot270", "transpose", "legacy_anti_transpose",
)
_CORRECTED_D4 = (
    "identity", "flip_x", "flip_y", "flip_xy",
    "rot90", "rot270", "transpose", "anti_transpose",
)


def _edge_tta_names(mode: str) -> tuple[str, ...]:
    mode = mode.strip().lower()
    if mode == "original":
        return ("identity",)
    if mode == "pilkwang_legacy_d4":
        return _LEGACY_PSEUDO_D4
    if mode == "corrected_d4":
        return _CORRECTED_D4
    raise ValueError(f"Unsupported edge TTA mode: {mode!r}")


def _apply_edge_xy_tta(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if name == "identity":
        return tensor
    if name == "flip_x":
        return tensor.flip((-1,))
    if name == "flip_y":
        return tensor.flip((-2,))
    if name == "flip_xy":
        return tensor.flip((-2, -1))
    if name == "rot90":
        return torch.rot90(tensor, 1, dims=(-2, -1))
    if name == "rot270":
        return torch.rot90(tensor, 3, dims=(-2, -1))
    if name == "transpose":
        return tensor.transpose(-1, -2)
    if name == "legacy_anti_transpose":
        return torch.rot90(tensor, 1, dims=(-2, -1)).transpose(-1, -2)
    if name == "anti_transpose":
        return tensor.transpose(-1, -2).flip((-2, -1))
    raise ValueError(f"Unsupported edge TTA transform: {name!r}")


def _invert_edge_xy_tta(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if name in {
        "identity", "flip_x", "flip_y", "flip_xy",
        "transpose", "anti_transpose",
    }:
        return _apply_edge_xy_tta(tensor, name)
    if name == "rot90":
        return torch.rot90(tensor, -1, dims=(-2, -1))
    if name == "rot270":
        return torch.rot90(tensor, 1, dims=(-2, -1))
    if name == "legacy_anti_transpose":
        return torch.rot90(
            tensor.transpose(-1, -2), -1, dims=(-2, -1)
        )
    raise ValueError(f"Unsupported edge TTA transform: {name!r}")


def _ordered_edge_tta_union(*groups: tuple[str, ...]) -> tuple[str, ...]:
    requested = set().union(*(set(group) for group in groups))
    unknown = requested - set(_EDGE_TTA_VIEW_ORDER)
    if unknown:
        raise ValueError(f"Unknown edge TTA transforms: {sorted(unknown)}")
    return tuple(name for name in _EDGE_TTA_VIEW_ORDER if name in requested)


def _edge_tta_weights(
    names: tuple[str, ...],
    original_weight: float | None,
) -> dict[str, float]:
    if not names or names[0] != "identity":
        raise ValueError("Edge TTA views must start with identity")
    if len(names) == 1:
        return {"identity": 1.0}
    if original_weight is None:
        original_weight = 1.0 / len(names)
    if not 0.0 < original_weight < 1.0:
        raise ValueError("Edge TTA original weight must be in (0, 1)")
    other_weight = (1.0 - original_weight) / (len(names) - 1)
    return {
        name: float(original_weight if name == "identity" else other_weight)
        for name in names
    }


def _encode_aligned_xy_views(
    model,
    imgs: torch.Tensor,
    feature_names: tuple[str, ...],
    detection_names: tuple[str, ...],
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor] | None]:
    names = _ordered_edge_tta_union(feature_names, detection_names)
    if not names:
        raise ValueError("At least one TTA view is required")
    canonical_spatial = tuple(imgs.shape[-3:])
    feature_name_set = set(feature_names)
    detection_name_set = set(detection_names)
    feature_maps_cpu: dict[str, torch.Tensor] = {}
    detection_sums: list[torch.Tensor] | None = None

    for name in names:
        imgs_view = (
            imgs if name == "identity" else _apply_edge_xy_tta(imgs, name)
        )
        feature_view, detection_view = model.encode(imgs_view)

        if name in feature_name_set:
            aligned_feature = _invert_edge_xy_tta(feature_view, name)
            if tuple(aligned_feature.shape[-3:]) != canonical_spatial:
                raise RuntimeError(
                    f"Aligned feature shape mismatch for {name}: "
                    f"expected {canonical_spatial}, "
                    f"got {tuple(aligned_feature.shape[-3:])}"
                )
            feature_maps_cpu[name] = (
                aligned_feature.detach().cpu().contiguous()
            )

        if name in detection_name_set:
            aligned_detection = [
                _invert_edge_xy_tta(frame_logits, name)
                for frame_logits in detection_view
            ]
            for frame_logits in aligned_detection:
                if tuple(frame_logits.shape[-3:]) != canonical_spatial:
                    raise RuntimeError(
                        f"Aligned detection shape mismatch for {name}: "
                        f"expected {canonical_spatial}, "
                        f"got {tuple(frame_logits.shape[-3:])}"
                    )
            if detection_sums is None:
                detection_sums = aligned_detection
            else:
                if len(detection_sums) != len(aligned_detection):
                    raise RuntimeError(
                        "Detection TTA window-length mismatch"
                    )
                detection_sums = [
                    total + current
                    for total, current in zip(
                        detection_sums,
                        aligned_detection,
                    )
                ]

        del feature_view, detection_view, imgs_view

    missing = feature_name_set - set(feature_maps_cpu)
    if missing:
        raise RuntimeError(
            f"Missing aligned feature views: {sorted(missing)}"
        )
    if detection_names:
        if detection_sums is None:
            raise RuntimeError("Detection TTA produced no logits")
        detection_sums = [
            logits / len(detection_names) for logits in detection_sums
        ]
    return feature_maps_cpu, detection_sums


def _predict_edge_tta_logits(
    model,
    feature_maps_cpu: dict[str, torch.Tensor],
    names: tuple[str, ...],
    weights: dict[str, float],
    f_idx: int,
    p_coords_src: torch.Tensor,
    p_coords_tgt: torch.Tensor,
    p_pos_src: torch.Tensor,
    p_pos_tgt: torch.Tensor,
    p_mask_src: torch.Tensor,
    p_mask_tgt: torch.Tensor,
    ds_arr_t: torch.Tensor,
) -> torch.Tensor:
    device = p_coords_src.device
    coords_src_cpu = p_coords_src.detach().cpu()
    coords_tgt_cpu = p_coords_tgt.detach().cpu()
    mask_src_cpu = p_mask_src.detach().cpu()
    mask_tgt_cpu = p_mask_tgt.detach().cpu()
    expected_shape = (
        p_coords_src.shape[0],
        p_coords_src.shape[1],
        p_coords_tgt.shape[1],
    )
    result: torch.Tensor | None = None

    for name in names:
        aligned_map = feature_maps_cpu[name]
        feature_src = model._index_features(
            aligned_map[:, f_idx],
            coords_src_cpu,
            mask_src_cpu,
        ).to(device)
        feature_tgt = model._index_features(
            aligned_map[:, f_idx + 1],
            coords_tgt_cpu,
            mask_tgt_cpu,
        ).to(device)
        logits = model.predict_edges(
            feature_src,
            feature_tgt,
            p_coords_src * ds_arr_t,
            p_coords_tgt * ds_arr_t,
            p_pos_src,
            p_pos_tgt,
            p_mask_src,
            p_mask_tgt,
        )
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                f"Edge TTA logit shape mismatch for {name}: "
                f"expected {expected_shape}, got {tuple(logits.shape)}"
            )
        weighted = logits * weights[name]
        result = weighted if result is None else result + weighted

    if result is None:
        raise RuntimeError("Edge TTA produced no transformer logits")
    return result


"""
EDGE_TTA_ENCODE_START = "        unet_out, det_logits = model.encode(imgs)\n"
EDGE_TTA_ENCODE_END = (
    "        del imgs\n\n"
    "        # --- Detect cells in each frame (dedup across windows) ---"
)
EDGE_TTA_ENCODE_REPLACEMENT = """        edge_tta_mode = os.environ.get(
            "BIOHUB_EDGE_TTA_MODE", "original"
        ).strip()
        edge_names = _edge_tta_names(edge_tta_mode)
        edge_original_weight_text = os.environ.get(
            "BIOHUB_EDGE_TTA_ORIGINAL_WEIGHT", ""
        ).strip()
        edge_original_weight = (
            float(edge_original_weight_text)
            if edge_original_weight_text
            else None
        )
        edge_weights = _edge_tta_weights(
            edge_names,
            edge_original_weight,
        )
        det_names = (
            _LEGACY_PSEUDO_D4 if cfg.det_tta else ("identity",)
        )
        primary_maps, det_logits = _encode_aligned_xy_views(
            model,
            imgs,
            edge_names,
            det_names,
        )
        if det_logits is None:
            raise RuntimeError("Primary detection TTA produced no logits")

        secondary_maps = None
        if secondary_model is not None:
            secondary_det_names = (
                det_names if secondary_detection_weight > 0.0 else ()
            )
            secondary_maps, secondary_det_logits = (
                _encode_aligned_xy_views(
                    secondary_model,
                    imgs,
                    edge_names,
                    secondary_det_names,
                )
            )

            if secondary_detection_weight > 0.0:
                if secondary_det_logits is None:
                    raise RuntimeError(
                        "Secondary detection TTA produced no logits"
                    )
                for f in range(W):
                    primary_det = det_logits[f]
                    secondary_det = secondary_det_logits[f]
                    primary_mean = primary_det.mean()
                    secondary_mean = secondary_det.mean()
                    primary_scale = primary_det.float().std(
                        unbiased=False
                    ).clamp_min(1e-4)
                    secondary_scale = secondary_det.float().std(
                        unbiased=False
                    ).clamp_min(1e-4)
                    scale_ratio = (
                        primary_scale / secondary_scale
                    ).clamp(0.5, 2.0)
                    secondary_det_aligned = (
                        (secondary_det - secondary_mean) * scale_ratio
                        + primary_mean
                    )
                    det_logits[f] = (
                        (1.0 - secondary_detection_weight) * primary_det
                        + secondary_detection_weight * secondary_det_aligned
                    )

            del secondary_det_logits

"""
EDGE_TTA_EDGE_START = (
    "            unet_feat_src = model._index_features(\n"
)
EDGE_TTA_EDGE_END = '                if secondary_link_mode == "raw":\n'
EDGE_TTA_EDGE_REPLACEMENT = """            edge_logits_pair = (
                _predict_edge_tta_logits(
                    model,
                    primary_maps,
                    edge_names,
                    edge_weights,
                    f_idx,
                    p_coords_src,
                    p_coords_tgt,
                    p_pos_src,
                    p_pos_tgt,
                    p_mask_src,
                    p_mask_tgt,
                    ds_arr_t,
                )
            )

            if secondary_model is not None:
                if secondary_maps is None:
                    raise RuntimeError(
                        "Secondary model is loaded but its feature maps "
                        "are missing"
                    )
                secondary_logits_pair = _predict_edge_tta_logits(
                    secondary_model,
                    secondary_maps,
                    edge_names,
                    edge_weights,
                    f_idx,
                    p_coords_src,
                    p_coords_tgt,
                    p_pos_src,
                    p_pos_tgt,
                    p_mask_src,
                    p_mask_tgt,
                    ds_arr_t,
                )

"""
EDGE_TTA_CLEANUP_ANCHOR = """        del unet_out
        if secondary_unet_out is not None:
            del secondary_unet_out
"""
EDGE_TTA_CLEANUP_REPLACEMENT = """        del primary_maps
        if secondary_maps is not None:
            del secondary_maps
"""
RETENTION_GUARD_BLEND_BLOCK = """                    det_logits[f] = (
                        (1.0 - secondary_detection_weight) * primary_det
                        + secondary_detection_weight * secondary_det_aligned
                    )"""
RETENTION_GUARD_BLEND_REPLACEMENT = """\
                    blended_det = (
                        (1.0 - secondary_detection_weight) * primary_det
                        + secondary_detection_weight * secondary_det_aligned
                    )
                    primary_candidates = len(_detect_cells_pooled(
                        primary_det[0],
                        int(frame_indices[f]),
                        cfg.det_threshold,
                        pool_k,
                    ))
                    blended_candidates = len(_detect_cells_pooled(
                        blended_det[0],
                        int(frame_indices[f]),
                        cfg.det_threshold,
                        pool_k,
                    ))
                    minimum_retention = float(os.environ.get(
                        "BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION",
                        "0",
                    ))
                    use_primary_detection = _retention_guard_uses_primary(
                        primary_candidates,
                        blended_candidates,
                        minimum_retention,
                    )
                    det_logits[f] = (
                        primary_det if use_primary_detection else blended_det
                    )
                    if int(frame_indices[f]) not in seen_frames:
                        print(
                            "BIOHUB_RETENTION_FRAME\\t"
                            f"{ds_path.stem}\\t"
                            f"{int(frame_indices[f])}\\t"
                            f"{primary_candidates}\\t"
                            f"{blended_candidates}\\t"
                            f"{int(use_primary_detection)}",
                            flush=True,
                        )"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-notebook", type=Path, required=True)
    parser.add_argument("--support-repo", type=Path, required=True)
    parser.add_argument("--primary-weights", type=Path, required=True)
    parser.add_argument("--secondary-weights", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=["secondary", "blend"],
    )
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument("--det-threshold", type=float, default=0.96875)
    parser.add_argument(
        "--ilp-disappearance-weight",
        type=float,
        default=1.575,
    )
    parser.add_argument("--blend-edge-weight", type=float, default=0.15)
    parser.add_argument("--blend-detection-weight", type=float, default=0.65)
    parser.add_argument(
        "--blend-link-mode",
        choices=LINK_MODES,
        default="low_margin_consensus",
    )
    parser.add_argument("--blend-mix-temperature", type=float, default=1.0)
    parser.add_argument("--blend-low-margin-max", type=float, default=0.35)
    parser.add_argument("--blend-edge-threshold", type=float, default=0.48)
    parser.add_argument(
        "--guard-min-candidate-retention",
        type=float,
        default=0.90,
        help=(
            "Framewise blended/primary candidate-count floor used only by "
            "the blend_guard variant."
        ),
    )
    parser.add_argument(
        "--edge-tta-reference-notebook",
        type=Path,
    )
    parser.add_argument(
        "--edge-tta-mode",
        choices=EDGE_TTA_MODES,
        default="original",
    )
    parser.add_argument("--edge-tta-original-weight", type=float)
    parser.add_argument("--export-preilp", action="store_true")
    parser.add_argument(
        "--expected-reference-sha256",
        default=REFERENCE_NOTEBOOK_SHA256,
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def verify_sha256(path: Path, expected: str) -> str:
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"SHA256 mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


def notebook_source(cell: dict[str, object]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source)


def edge_tta_views(mode: str) -> tuple[str, ...]:
    if mode == "original":
        return ("identity",)
    if mode == "pilkwang_legacy_d4":
        return LEGACY_EDGE_TTA_VIEWS
    if mode == "corrected_d4":
        return CORRECTED_EDGE_TTA_VIEWS
    raise ValueError(f"Unsupported edge TTA mode: {mode!r}")


def edge_tta_view_weights(
    mode: str,
    original_weight: float | None,
) -> dict[str, float]:
    views = edge_tta_views(mode)
    if len(views) == 1:
        return {"identity": 1.0}
    if original_weight is None:
        original_weight = 1.0 / len(views)
    if not 0.0 < original_weight < 1.0:
        raise ValueError("--edge-tta-original-weight must be in (0, 1)")
    other_weight = (1.0 - original_weight) / (len(views) - 1)
    return {
        view: float(
            original_weight if view == "identity" else other_weight
        )
        for view in views
    }


def replace_between_once(
    source: str,
    start: str,
    end: str,
    replacement: str,
    label: str,
) -> str:
    start_count = source.count(start)
    end_count = source.count(end)
    if start_count != 1 or end_count != 1:
        raise RuntimeError(
            f"{label} patch anchors must occur exactly once: "
            f"start={start_count}, end={end_count}"
        )
    start_index = source.index(start)
    end_index = source.index(end, start_index + len(start))
    return source[:start_index] + replacement + source[end_index:]


def apply_edge_tta_patch(predictor_source: str) -> str:
    if EDGE_TTA_IMPLEMENTATION_VERSION in predictor_source:
        raise RuntimeError("Edge TTA predictor patch is already present")
    helper_count = predictor_source.count(EDGE_TTA_HELPER_ANCHOR)
    if helper_count != 1:
        raise RuntimeError(
            "Edge TTA helper anchor must occur exactly once: "
            f"found {helper_count}"
        )
    predictor_source = predictor_source.replace(
        EDGE_TTA_HELPER_ANCHOR,
        EDGE_TTA_HELPERS + EDGE_TTA_HELPER_ANCHOR,
        1,
    )
    predictor_source = replace_between_once(
        predictor_source,
        EDGE_TTA_ENCODE_START,
        EDGE_TTA_ENCODE_END,
        EDGE_TTA_ENCODE_REPLACEMENT,
        "Edge TTA encode",
    )
    predictor_source = replace_between_once(
        predictor_source,
        EDGE_TTA_EDGE_START,
        EDGE_TTA_EDGE_END,
        EDGE_TTA_EDGE_REPLACEMENT,
        "Edge TTA association",
    )
    cleanup_count = predictor_source.count(EDGE_TTA_CLEANUP_ANCHOR)
    if cleanup_count != 1:
        raise RuntimeError(
            "Edge TTA cleanup anchor must occur exactly once: "
            f"found {cleanup_count}"
        )
    return predictor_source.replace(
        EDGE_TTA_CLEANUP_ANCHOR,
        EDGE_TTA_CLEANUP_REPLACEMENT,
        1,
    )


def apply_retention_guard_patch(predictor_source: str) -> str:
    if RETENTION_GUARD_IMPLEMENTATION_VERSION in predictor_source:
        raise RuntimeError("Frame-retention guard patch is already present")
    helper_count = predictor_source.count(RETENTION_GUARD_HELPER_ANCHOR)
    if helper_count != 1:
        raise RuntimeError(
            "Frame-retention guard helper anchor must occur exactly once: "
            f"found {helper_count}"
        )
    anchor_count = predictor_source.count(RETENTION_GUARD_BLEND_BLOCK)
    if anchor_count != 1:
        raise RuntimeError(
            "Frame-retention guard blend anchor must occur exactly once: "
            f"found {anchor_count}"
        )
    predictor_source = predictor_source.replace(
        RETENTION_GUARD_HELPER_ANCHOR,
        RETENTION_GUARD_HELPERS + RETENTION_GUARD_HELPER_ANCHOR,
        1,
    )
    return predictor_source.replace(
        RETENTION_GUARD_BLEND_BLOCK,
        RETENTION_GUARD_BLEND_REPLACEMENT,
        1,
    )


def parse_retention_records(
    log_path: Path,
) -> dict[tuple[str, int], tuple[int, int, bool]]:
    records: dict[tuple[str, int], tuple[int, int, bool]] = {}
    for line_number, line in enumerate(
        log_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.startswith(RETENTION_LOG_PREFIX):
            continue
        parts = line.split("\t")
        if len(parts) != 6:
            raise RuntimeError(
                f"{log_path}:{line_number}: malformed retention record"
            )
        _, dataset, frame_text, primary_text, blended_text, fallback_text = (
            parts
        )
        frame = int(frame_text)
        primary_candidates = int(primary_text)
        blended_candidates = int(blended_text)
        fallback_value = int(fallback_text)
        if (
            frame < 0
            or primary_candidates < 0
            or blended_candidates < 0
            or fallback_value not in (0, 1)
        ):
            raise RuntimeError(
                f"{log_path}:{line_number}: invalid retention values"
            )
        key = (dataset, frame)
        if key in records:
            raise RuntimeError(
                f"{log_path}:{line_number}: duplicate retention key {key}"
            )
        records[key] = (
            primary_candidates,
            blended_candidates,
            bool(fallback_value),
        )
    if not records:
        raise RuntimeError(f"No retention diagnostics found in {log_path}")
    return records


def summarize_retention_records(
    records: dict[tuple[str, int], tuple[int, int, bool]],
) -> dict[str, object]:
    by_dataset: dict[str, dict[str, object]] = {}
    all_retentions: list[float] = []
    for (dataset, _), (
        primary_candidates,
        blended_candidates,
        fallback,
    ) in sorted(records.items()):
        retention = (
            blended_candidates / primary_candidates
            if primary_candidates
            else 1.0
        )
        all_retentions.append(retention)
        stats = by_dataset.setdefault(
            dataset,
            {
                "evaluated_frames": 0,
                "fallback_frames": 0,
                "retentions": [],
            },
        )
        stats["evaluated_frames"] = int(stats["evaluated_frames"]) + 1
        stats["fallback_frames"] = int(stats["fallback_frames"]) + int(
            fallback
        )
        stats["retentions"].append(retention)

    dataset_summary = {}
    for dataset, stats in sorted(by_dataset.items()):
        retentions = list(stats.pop("retentions"))
        dataset_summary[dataset] = {
            **stats,
            "minimum_retention": min(retentions),
            "median_retention": statistics.median(retentions),
        }
    return {
        "evaluated_frames": len(records),
        "fallback_frames": sum(
            int(fallback) for _, _, fallback in records.values()
        ),
        "minimum_retention": min(all_retentions),
        "median_retention": statistics.median(all_retentions),
        "datasets": dataset_summary,
    }


def compare_retention_controls(
    control: dict[tuple[str, int], tuple[int, int, bool]],
    candidate: dict[tuple[str, int], tuple[int, int, bool]],
) -> dict[str, object]:
    if set(control) != set(candidate):
        raise RuntimeError(
            "Retention A/B diagnostic coverage mismatch: "
            f"missing={sorted(set(control) - set(candidate))}, "
            f"extra={sorted(set(candidate) - set(control))}"
        )
    mismatched_counts = [
        key
        for key in sorted(control)
        if control[key][:2] != candidate[key][:2]
    ]
    if mismatched_counts:
        raise RuntimeError(
            "Retention A/B candidate counts changed before fallback: "
            f"{mismatched_counts[:10]}"
        )
    control_fallbacks = sum(
        int(fallback) for _, _, fallback in control.values()
    )
    if control_fallbacks:
        raise RuntimeError(
            "Disabled retention control unexpectedly used fallback on "
            f"{control_fallbacks} frames"
        )
    candidate_fallbacks = sum(
        int(fallback) for _, _, fallback in candidate.values()
    )
    return {
        "paired_frames": len(control),
        "candidate_counts_identical": True,
        "control_fallback_frames": 0,
        "candidate_fallback_frames": candidate_fallbacks,
        "guard_activated": candidate_fallbacks > 0,
    }


def apply_reference_patch(
    reference_notebook: Path,
    temporary_repo: Path,
    edge_tta_mode: str,
    export_preilp: bool,
) -> str:
    notebook = json.loads(reference_notebook.read_text(encoding="utf-8"))
    cells = notebook.get("cells")
    if not isinstance(cells, list) or len(cells) <= PATCH_CELL_INDEX:
        raise RuntimeError("Pinned reference notebook has no patch cell")
    source = notebook_source(cells[PATCH_CELL_INDEX])
    if source.count(PATCH_END_MARKER) != 1:
        raise RuntimeError("Pinned reference patch boundary changed")
    patch_prefix = source.split(PATCH_END_MARKER, 1)[0]
    namespace = {
        "__name__": "biohub_pinned_dual_seed_patch",
        "REPO_DIR": temporary_repo,
    }
    exec(
        compile(
            patch_prefix,
            f"{reference_notebook}:cell-{PATCH_CELL_INDEX}",
            "exec",
        ),
        namespace,
    )
    predictor = (
        temporary_repo / "scripts" / "predict_unet_transformer.py"
    )
    predictor_source = predictor.read_text(encoding="utf-8")
    if edge_tta_mode != "original":
        predictor_source = apply_edge_tta_patch(predictor_source)
    predictor_source = apply_retention_guard_patch(predictor_source)
    predictor.write_text(predictor_source, encoding="utf-8")
    if export_preilp:
        marker = "        graph = build_graph(coords, edges)\n"
        if predictor_source.count(marker) != 1:
            raise RuntimeError(
                "Pre-ILP export patch anchor must occur exactly once"
            )
        predictor_source = predictor_source.replace(
            marker,
            marker
            + """        preilp_dir = os.environ.get("BIOHUB_PREILP_DIR")
        if preilp_dir:
            save_graph(graph, Path(preilp_dir) / f"{name}.geff")

""",
            1,
        )
        predictor.write_text(predictor_source, encoding="utf-8")
    compile(predictor_source, str(predictor), "exec")
    return file_sha256(predictor)


def selected_datasets(args: argparse.Namespace) -> list[str]:
    available = sorted(path.stem for path in args.baseline_dir.glob("*.geff"))
    if not available:
        raise FileNotFoundError(
            f"No frozen baseline GEFFs under {args.baseline_dir}"
        )
    if args.dataset:
        requested = list(dict.fromkeys(args.dataset))
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"Requested datasets not in baseline: {missing}")
        selected = requested
    else:
        selected = available
    if args.max_datasets < 0:
        raise ValueError("--max-datasets must be non-negative")
    if args.max_datasets:
        selected = selected[: args.max_datasets]
    if not selected:
        raise ValueError("No datasets selected")
    return selected


def variant_environment(
    args: argparse.Namespace,
    variant: str,
    gpu: str,
    temporary_repo: Path,
) -> dict[str, str]:
    python_path = os.pathsep.join(
        [
            str(args.runtime_dir.resolve()),
            str((temporary_repo / "src").resolve()),
            os.environ.get("PYTHONPATH", ""),
        ]
    )
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpu,
        "PYTHONPATH": python_path,
        "USER": f"dual_seed_{variant}",
    }
    if args.export_preilp:
        env["BIOHUB_PREILP_DIR"] = str(
            (args.output_dir / variant / "preilp").resolve()
        )
    else:
        env.pop("BIOHUB_PREILP_DIR", None)
    for key in (
        "BIOHUB_SECONDARY_WEIGHTS",
        "BIOHUB_SECONDARY_EDGE_WEIGHT",
        "BIOHUB_SECONDARY_DETECTION_WEIGHT",
        "BIOHUB_SECONDARY_LINK_MODE",
        "BIOHUB_SECONDARY_MIX_TEMPERATURE",
        "BIOHUB_SECONDARY_LOW_MARGIN_MAX",
        "BIOHUB_DUAL_SEED_EDGE_THRESHOLD",
        "BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION",
        "BIOHUB_EDGE_TTA_MODE",
        "BIOHUB_EDGE_TTA_ORIGINAL_WEIGHT",
    ):
        env.pop(key, None)
    if args.edge_tta_mode != "original":
        edge_weights = edge_tta_view_weights(
            args.edge_tta_mode,
            args.edge_tta_original_weight,
        )
        env.update(
            {
                "BIOHUB_EDGE_TTA_MODE": args.edge_tta_mode,
                "BIOHUB_EDGE_TTA_ORIGINAL_WEIGHT": str(
                    edge_weights["identity"]
                ),
            }
        )
    if variant in BLEND_VARIANTS:
        env.update(
            {
                "BIOHUB_SECONDARY_WEIGHTS": str(
                    args.secondary_weights.resolve()
                ),
                "BIOHUB_SECONDARY_EDGE_WEIGHT": str(
                    args.blend_edge_weight
                ),
                "BIOHUB_SECONDARY_DETECTION_WEIGHT": str(
                    args.blend_detection_weight
                ),
                "BIOHUB_SECONDARY_LINK_MODE": args.blend_link_mode,
                "BIOHUB_SECONDARY_MIX_TEMPERATURE": str(
                    args.blend_mix_temperature
                ),
                "BIOHUB_SECONDARY_LOW_MARGIN_MAX": str(
                    args.blend_low_margin_max
                ),
                "BIOHUB_DUAL_SEED_EDGE_THRESHOLD": str(
                    args.blend_edge_threshold
                ),
            }
        )
    if variant == "blend_guard":
        env["BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION"] = str(
            args.guard_min_candidate_retention
        )
    return env


def prediction_command(
    args: argparse.Namespace,
    variant: str,
    temporary_repo: Path,
    split_path: Path,
) -> list[str]:
    weights = (
        args.secondary_weights
        if variant == "secondary"
        else args.primary_weights
    )
    return [
        sys.executable,
        "scripts/predict_unet_transformer.py",
        "--method",
        variant,
        "--data-dir",
        str(args.data_dir.resolve()),
        "--splits",
        str(split_path.resolve()),
        "--split",
        "0",
        "--weights",
        str(weights.resolve()),
        "--unet-batch-size",
        "4",
        "--det-threshold",
        str(args.det_threshold),
        "--use-ilp",
        "--ilp-edge-weight",
        "-1.0",
        "--ilp-appearance-weight",
        "0.0",
        "--ilp-disappearance-weight",
        str(args.ilp_disappearance_weight),
        "--ilp-division-weight",
        "1.0",
    ]


def prepare_variant(
    args: argparse.Namespace,
    variant: str,
    datasets: list[str],
) -> dict[str, object]:
    variant_root = args.output_dir / variant
    temporary_repo = variant_root / "support_repo"
    shutil.copytree(args.support_repo, temporary_repo)
    base_predictor = (
        temporary_repo / "scripts" / "predict_unet_transformer.py"
    )
    verify_sha256(base_predictor, SUPPORT_PREDICTOR_SHA256)
    patched_sha256 = apply_reference_patch(
        args.reference_notebook,
        temporary_repo,
        args.edge_tta_mode,
        args.export_preilp,
    )
    preilp_dir = variant_root / "preilp"
    if args.export_preilp:
        preilp_dir.mkdir()
    split_path = variant_root / "split.json"
    split_path.write_text(
        json.dumps(
            [{"split": 0, "train": [], "test": datasets}],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = (
        temporary_repo
        / "predictions"
        / f"dual_seed_{variant}"
        / variant
        / "split_0"
    )
    return {
        "variant": variant,
        "root": variant_root,
        "repo": temporary_repo,
        "split": split_path,
        "output": output_dir,
        "preilp": preilp_dir if args.export_preilp else None,
        "patched_predictor_sha256": patched_sha256,
    }


def main() -> None:
    args = parse_args()
    if len(args.variants) != len(set(args.variants)):
        raise ValueError("Duplicate variants are not allowed")
    if len(args.gpus) != len(args.variants):
        raise ValueError("--gpus must have one entry per variant")
    if not 0.0 < args.blend_edge_weight < 1.0:
        raise ValueError("--blend-edge-weight must be in (0, 1)")
    if not 0.0 <= args.blend_detection_weight < 1.0:
        raise ValueError("--blend-detection-weight must be in [0, 1)")
    if not 0.5 <= args.blend_mix_temperature <= 2.0:
        raise ValueError("--blend-mix-temperature must be in [0.5, 2]")
    if not 0.0 < args.blend_low_margin_max <= 1.0:
        raise ValueError("--blend-low-margin-max must be in (0, 1]")
    if not 0.0 < args.blend_edge_threshold < 1.0:
        raise ValueError("--blend-edge-threshold must be in (0, 1)")
    if not 0.0 < args.guard_min_candidate_retention <= 1.0:
        raise ValueError(
            "--guard-min-candidate-retention must be in (0, 1]"
        )
    if (
        args.edge_tta_mode == "original"
        and args.edge_tta_original_weight is not None
    ):
        raise ValueError(
            "--edge-tta-original-weight requires a non-original "
            "--edge-tta-mode"
        )
    edge_views = edge_tta_views(args.edge_tta_mode)
    edge_weights = edge_tta_view_weights(
        args.edge_tta_mode,
        args.edge_tta_original_weight,
    )
    if (
        args.edge_tta_mode != "original"
        and args.edge_tta_reference_notebook is None
    ):
        raise ValueError(
            "--edge-tta-reference-notebook is required for non-original "
            "edge TTA"
        )
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    require_file(args.reference_notebook)
    if args.edge_tta_reference_notebook is not None:
        require_file(args.edge_tta_reference_notebook)
    require_file(args.primary_weights)
    require_file(args.secondary_weights)
    require_file(args.primary_weights.parent / "config.json")
    require_file(args.secondary_weights.parent / "config.json")
    if not args.support_repo.is_dir():
        raise NotADirectoryError(args.support_repo)
    if not args.data_dir.is_dir():
        raise NotADirectoryError(args.data_dir)
    if not args.runtime_dir.is_dir():
        raise NotADirectoryError(args.runtime_dir)

    reference_sha256 = verify_sha256(
        args.reference_notebook,
        args.expected_reference_sha256,
    )
    edge_tta_reference_sha256 = None
    if args.edge_tta_reference_notebook is not None:
        edge_tta_reference_sha256 = verify_sha256(
            args.edge_tta_reference_notebook,
            EDGE_TTA_REFERENCE_SHA256,
        )
    primary_sha256 = verify_sha256(
        args.primary_weights,
        PRIMARY_WEIGHT_SHA256,
    )
    secondary_sha256 = verify_sha256(
        args.secondary_weights,
        SECONDARY_WEIGHT_SHA256,
    )
    datasets = selected_datasets(args)
    args.output_dir.mkdir(parents=True)

    prepared = [
        prepare_variant(args, variant, datasets)
        for variant in args.variants
    ]
    processes: dict[str, subprocess.Popen[str]] = {}
    log_handles = {}
    started_at = time.time()
    commands = {}
    try:
        for item, gpu in zip(prepared, args.gpus):
            variant = str(item["variant"])
            temporary_repo = Path(item["repo"])
            command = prediction_command(
                args,
                variant,
                temporary_repo,
                Path(item["split"]),
            )
            commands[variant] = command
            log_path = Path(item["root"]) / "predict.log"
            handle = log_path.open("w", encoding="utf-8")
            log_handles[variant] = handle
            print(
                f"Launching {variant} on CUDA_VISIBLE_DEVICES={gpu}",
                flush=True,
            )
            processes[variant] = subprocess.Popen(
                command,
                cwd=temporary_repo,
                env=variant_environment(
                    args,
                    variant,
                    gpu,
                    temporary_repo,
                ),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )

        while processes:
            for variant, process in list(processes.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                del processes[variant]
                if return_code != 0:
                    for other in processes.values():
                        other.terminate()
                    for other in processes.values():
                        other.wait()
                    raise subprocess.CalledProcessError(
                        return_code,
                        commands[variant],
                    )
                print(f"Completed {variant}", flush=True)
            if processes:
                time.sleep(2)
    finally:
        for handle in log_handles.values():
            handle.close()

    results = {}
    retention_records: dict[
        str,
        dict[tuple[str, int], tuple[int, int, bool]],
    ] = {}
    expected = set(datasets)
    for item in prepared:
        variant = str(item["variant"])
        output_dir = Path(item["output"])
        found = {path.stem for path in output_dir.glob("*.geff")}
        if found != expected:
            raise RuntimeError(
                f"{variant} output mismatch: "
                f"missing={sorted(expected - found)}, "
                f"extra={sorted(found - expected)}"
            )
        results[variant] = {
            "output_dir": str(output_dir),
            "datasets": len(found),
            "patched_predictor_sha256": item[
                "patched_predictor_sha256"
            ],
            "command": commands[variant],
        }
        if variant in BLEND_VARIANTS:
            log_path = Path(item["root"]) / "predict.log"
            records = parse_retention_records(log_path)
            retention_records[variant] = records
            results[variant]["retention_guard"] = (
                summarize_retention_records(records)
            )
        if args.export_preilp:
            preilp_dir = Path(item["preilp"])
            found_preilp = {
                path.stem for path in preilp_dir.glob("*.geff")
            }
            if found_preilp != expected:
                raise RuntimeError(
                    f"{variant} pre-ILP output mismatch: "
                    f"missing={sorted(expected - found_preilp)}, "
                    f"extra={sorted(found_preilp - expected)}"
                )
            results[variant]["preilp_dir"] = str(preilp_dir)

    retention_ab_check = None
    if {"blend", "blend_guard"} <= set(retention_records):
        retention_ab_check = compare_retention_controls(
            retention_records["blend"],
            retention_records["blend_guard"],
        )

    manifest = {
        "reference_notebook": str(args.reference_notebook.resolve()),
        "reference_notebook_sha256": reference_sha256,
        "support_predictor_sha256": SUPPORT_PREDICTOR_SHA256,
        "primary_weights": str(args.primary_weights.resolve()),
        "primary_weight_sha256": primary_sha256,
        "secondary_weights": str(args.secondary_weights.resolve()),
        "secondary_weight_sha256": secondary_sha256,
        "data_dir": str(args.data_dir.resolve()),
        "frozen_baseline_dir": str(args.baseline_dir.resolve()),
        "datasets": datasets,
        "det_threshold": args.det_threshold,
        "ilp_disappearance_weight": args.ilp_disappearance_weight,
        "export_preilp": args.export_preilp,
        "blend": {
            "edge_weight": args.blend_edge_weight,
            "detection_weight": args.blend_detection_weight,
            "link_mode": args.blend_link_mode,
            "mix_temperature": args.blend_mix_temperature,
            "low_margin_max": args.blend_low_margin_max,
            "edge_threshold": args.blend_edge_threshold,
        },
        "retention_guard": {
            "implementation_version": (
                RETENTION_GUARD_IMPLEMENTATION_VERSION
            ),
            "activation_variant": "blend_guard",
            "minimum_candidate_retention": (
                args.guard_min_candidate_retention
            ),
            "fallback_scope": "individual_frame",
            "reference_field": "primary_detection_logits",
            "candidate_extractor": "_detect_cells_pooled",
            "label_free": True,
            "ab_check": retention_ab_check,
        },
        "edge_tta": {
            "mode": args.edge_tta_mode,
            "implementation_version": EDGE_TTA_IMPLEMENTATION_VERSION,
            "reference_notebook": (
                str(args.edge_tta_reference_notebook.resolve())
                if args.edge_tta_reference_notebook is not None
                else None
            ),
            "reference_notebook_sha256": edge_tta_reference_sha256,
            "models": "all_loaded",
            "aggregation_domain": "raw_edge_logits",
            "aggregation_stage": "per_seed_before_seed_calibration",
            "node_policy": "shared_canonical_detection_nodes",
            "feature_alignment": "inverse_map_to_canonical_zyx",
            "view_names": list(edge_views),
            "view_weights": edge_weights,
            "requested_view_count": len(edge_views),
            "unique_spatial_view_count": (
                7
                if args.edge_tta_mode == "pilkwang_legacy_d4"
                else len(edge_views)
            ),
            "legacy_anti_transpose_duplicates_flip_x": (
                args.edge_tta_mode == "pilkwang_legacy_d4"
            ),
            "detection_tta_changed": False,
        },
        "elapsed_seconds": time.time() - started_at,
        "results": results,
    }
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
