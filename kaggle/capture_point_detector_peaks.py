#!/usr/bin/env python3
"""Run the reviewed v5 point detector on fixed movies without reading labels.

The author uses strided Y/X sampling and metadata quantiles, not block means
or quantiles recomputed after sampling. This creates a diagnostic peak cache,
not a tracking graph or a competition submission.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import check_point_detector_runtime as runtime
import run_geometric_reference as reference

THRESHOLD = 0.2
DOWNSAMPLE = (1, 4, 4)


def normalize_volume(raw, low, high):
    import numpy as np

    if not np.isfinite([low, high]).all() or high <= low or raw.ndim != 3:
        raise ValueError("Invalid volume or metadata quantiles")
    volume = raw[::1, ::4, ::4].astype(np.float32)
    volume = np.clip((volume - low) / (high - low + 1e-6), 0.0, None)
    if not np.isfinite(volume).all():
        raise ValueError("Nonfinite normalized volume")
    return np.ascontiguousarray(volume, dtype=np.float32)


def extract_peaks(logits, frame):
    import numpy as np
    import torch
    from torch.nn import functional as F

    if logits.ndim != 4 or logits.shape[0] != 1 or not torch.isfinite(logits).all():
        raise ValueError("Expected one finite BZYX logit volume")
    probability = logits.float().sigmoid().unsqueeze(1)
    pooled = F.max_pool3d(probability, kernel_size=3, stride=1, padding=1)
    locations = torch.nonzero((probability[0, 0] >= pooled[0, 0]) &
                              (probability[0, 0] >= THRESHOLD), as_tuple=False)
    scores = probability[0, 0][tuple(locations.T)].cpu().numpy().astype(np.float32)
    coords = locations.cpu().numpy().astype(np.int32) * np.asarray(DOWNSAMPLE, dtype=np.int32)
    return np.column_stack((np.full(len(coords), frame, dtype=np.int32), coords)), scores


def run(args):
    names = reference.selected_names(args.control_dir, args.mode)
    if args.shard is not None:
        names = names[args.shard::2]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "peaks").mkdir()
    manifest = {"purpose": "reviewed_v5_real_image_peaks", "mode": args.mode,
                "datasets": names, "device": args.device, "shard": args.shard,
                "downsample": list(DOWNSAMPLE), "sampling": "strided",
                "quantiles": ["0.001", "0.999"], "clip": "minimum_zero_only",
                "peak_threshold": THRESHOLD, "peak_kernel": 3,
                "ground_truth_accessed": False, "submission_created": False,
                "quality_score": None, "complete": False, "movies": {},
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    manifest_path = args.output_dir / "run_manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    try:
        if args.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        sys.path[:0] = [str(args.runtime_dir), str(args.tracking_repo / "src")]
        import numpy as np
        import torch
        import zarr
        from tracking_cellmot.io import open_dataset

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        manifest["torch_version"] = str(torch.__version__)
        manifest["dataset_loader_sha256"] = reference.stability.file_sha256(Path(inspect.getfile(open_dataset)))
        manifest["verified_sha256"] = {}
        for name in ("model_v5.py", "00000030.pth"):
            digest = reference.stability.file_sha256(args.reference_dir / name)
            if digest != runtime.PINNED[name]:
                raise ValueError("Reviewed source/checkpoint bytes changed")
            manifest["verified_sha256"][name] = digest
        weight = args.reference_dir / "00000030.pth"
        if torch.serialization.get_unsafe_globals_in_checkpoint(weight):
            raise ValueError("Checkpoint no longer satisfies the completed restricted CPU check")
        checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
        source = runtime.load_reviewed(args.reference_dir, "model_v5")
        model = source.StrongUNet3D3Level(in_channels=1, channels=(64, 128, 256),
                                         node_channels=1, gradient_checkpointing=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        del checkpoint
        model = model.eval().to(args.device)
        save()
        for name in names:
            started = time.monotonic()
            image_path = args.data_dir / "train" / f"{name}.zarr"
            dataset = open_dataset(image_path, normalize=False, load_image=False, require_tracks=False)
            image = zarr.open_group(str(image_path), mode="r")["0"]
            if len(image.shape) != 4 or image.shape[0] != 100 or image.dtype != np.uint16:
                raise ValueError("Expected a complete 100-frame uint16 movie")
            low, high = (float(dataset.quantiles[key]) for key in ("0.001", "0.999"))
            all_coords, all_scores = [], []
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                for t in range(image.shape[0]):
                    volume = normalize_volume(np.asarray(image[t]), low, high)
                    pyramid, logits = model(torch.from_numpy(volume)[None].to(args.device))
                    if tuple(logits.shape[1:]) != volume.shape:
                        raise ValueError("Model output and sampled voxel coordinates disagree")
                    coords, scores = extract_peaks(logits, t)
                    all_coords.append(coords)
                    all_scores.append(scores)
                    del pyramid, logits
                    if (t + 1) % 20 == 0:
                        print(f"FRAMES {name} {t + 1}/100", flush=True)
            coords, scores = np.concatenate(all_coords), np.concatenate(all_scores)
            if (coords.shape != (len(scores), 4) or not np.isfinite(scores).all()
                    or (scores < THRESHOLD).any() or (scores > 1).any()
                    or (coords < 0).any() or (coords >= np.asarray(image.shape)).any()
                    or len(np.unique(coords, axis=0)) != len(coords)):
                raise ValueError("Invalid peak coordinates/probabilities")
            path = args.output_dir / "peaks" / f"{name}.npz"
            np.savez_compressed(path, coords=coords, scores=scores,
                                processed_frames=np.arange(100), image_shape=np.asarray(image.shape))
            record = {"frames": 100, "shape": list(image.shape), "peaks": len(coords),
                      "peaks_ge_050": int((scores >= 0.5).sum()), "quantile_low": low, "quantile_high": high,
                      "cache_sha256": reference.stability.file_sha256(path),
                      "coordinate_sha256": hashlib.sha256(coords.astype("<i4").tobytes()).hexdigest(),
                      "seconds": time.monotonic() - started,
                      "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else 0}
            manifest["movies"][name] = record
            save()
            print(json.dumps({"dataset": name, **record}), flush=True)
        manifest["complete"] = True
    except Exception as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
        (args.output_dir / "run_summary.md").write_text("# V5 real-image peak capture\n\n```json\n" +
            json.dumps(manifest, indent=2) + "\n```\n\nNo labels read; no tracking quality or formal submission.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data-dir", "control-dir", "reference-dir", "runtime-dir", "tracking-repo", "output-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1))
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    run(args)


if __name__ == "__main__":
    main()
