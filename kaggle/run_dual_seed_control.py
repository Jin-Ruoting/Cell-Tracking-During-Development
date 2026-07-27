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
import subprocess
import sys
import time
from pathlib import Path


REFERENCE_NOTEBOOK_SHA256 = (
    "70e0c300ceae3cd7ee2cf1650c4a5f74463543e3aae1b486ba5f729a76281656"
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
VARIANTS = ("primary", "secondary", "blend")
LINK_MODES = ("raw", "calibrated", "adaptive", "low_margin_consensus")


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


def apply_reference_patch(
    reference_notebook: Path,
    temporary_repo: Path,
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
    ):
        env.pop(key, None)
    if variant == "blend":
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
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    require_file(args.reference_notebook)
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
