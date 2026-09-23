#!/usr/bin/env python3
"""Verify the reviewed v5 detector and its complete checkpoint on CPU only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

PINNED = {
    "model_v5.py": "68e13d3424721ee43e6d247f97d9f533dbeb735b46c038ec3c636035ee3a4e23",
    "00000030.pth": "e177eba1e161efc63b4b5385740066ef2c824134585613c2d4719be5bd337065",
    "model_v12.py": "586ac08e4429fa0ca0db0c06afdf6d7ef77d459c0260bbf286bf47169d484082",
    "loss_and_metric_v12.py": "40cc8013c47cc4d2f2684dfccb153016e9ef9372ad2db10ef376aa3e04e3133a",
}


def load_reviewed(directory, name):
    path = directory / (name + ".py")
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINNED[path.name]:
        raise ValueError("Reviewed reference source changed")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "device": "cpu", "quality_score": None,
              "real_image_inference": False, "tracking_run": False,
              "input": "synthetic zero tensor [1,64,64,64]", "verified_sha256": {}}
    started = time.monotonic()
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        for name in ("model_v5.py", "00000030.pth"):
            digest = hashlib.sha256((args.reference_dir / name).read_bytes()).hexdigest()
            if digest != PINNED[name]:
                raise ValueError("Source/checkpoint bytes differ from the verified acquisition")
            report["verified_sha256"][name] = digest
        weight = args.reference_dir / "00000030.pth"
        external = torch.serialization.get_unsafe_globals_in_checkpoint(weight)
        report["checkpoint_extra_globals"] = external
        allowed = []
        for name in external:
            if name not in ("model_v12.DotDict", "loss_and_metric_v12.DotDict"):
                raise ValueError(f"Checkpoint contains an unreviewed serialized type: {name}")
            module_name, _ = name.rsplit(".", 1)
            allowed.append(load_reviewed(args.reference_dir, module_name).DotDict)
        with torch.serialization.safe_globals(allowed):
            checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
        report["checkpoint_keys"] = sorted(checkpoint)
        source = load_reviewed(args.reference_dir, "model_v5")
        model = source.StrongUNet3D3Level(in_channels=1, channels=(64, 128, 256),
                                         node_channels=1, gradient_checkpointing=False).eval()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        with torch.inference_mode():
            pyramid, logits = model(torch.zeros(1, 64, 64, 64))
        if tuple(logits.shape) != (1, 64, 64, 64) or not torch.isfinite(logits).all():
            raise ValueError("Point-detector output contract failed")
        if any(t.device.type != "cpu" or not torch.isfinite(t).all() for t in pyramid):
            raise ValueError("Nonfinite or non-CPU reference features")
        report.update({"passed": True, "strict_model_load": True,
                       "parameters": sum(p.numel() for p in model.parameters()),
                       "pyramid_shapes": [list(t.shape) for t in pyramid],
                       "logit_shape": list(logits.shape), "torch_version": str(torch.__version__)})
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (args.output_dir / "runtime_receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output_dir / "run_summary.md").write_text("# Point detector CPU compatibility\n\n```json\n" +
            json.dumps(report, indent=2) + "\n```\n\nSynthetic tensor check only; no tracking-quality result.\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
