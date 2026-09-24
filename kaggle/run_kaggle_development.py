#!/usr/bin/env python3
"""Private Kaggle compute adapter for the frozen E029/E038 development test.

Each phase runs in a fresh process. No local image inference, competition-test
prediction, parameter search, or competition submission is provided here.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import frozen_development_corpus as corpus
import run_geometric_reference as reference

WORK = Path("/kaggle/working/logs/e038-development")
EXPECTED = {
    "smoke": {"csv_sha256": "702008b6546b843582ad9277c3c9984484c3b1caeb458ef9e41270cc5edd0035",
              "score": 0.9638193643220809},
    "full": {"csv_sha256": "1d4fd28c02cb54d2279a120794b26e21fa0741d743bf62781d8ed17d5a26e2ce",
             "score": 0.9090442379185286},
}
# Exact predictor saved by the submitted E029 Kaggle v1. Substituting its two
# log-directory literals with the historical server directory reproduces the
# server's e5851002... checksum. Only these log destinations may be relocated.
PREDICTOR_SHA256 = "ddca518b0e838b2123f30cfcb31819fb159b4e68fc20ef6b3a42003cc2d528ed"


def save(path: Path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def mounted(owner: str, slug: str, competition=False) -> Path:
    paths = [Path("/kaggle/input") / slug,
             Path("/kaggle/input/competitions" if competition else "/kaggle/input/datasets") /
             (slug if competition else owner + "/" + slug)]
    found = {p.resolve() for p in paths if p.is_dir()}
    if len(found) != 1:
        raise ValueError(f"Expected one mounted source for {slug}: {sorted(map(str, found))}")
    return found.pop()


def verified_bundle(bundle: Path) -> dict:
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        path = (bundle / name).resolve()
        if bundle not in path.parents or reference.stability.file_sha256(path) != digest:
            raise ValueError(f"Bundle source checksum mismatch: {name}")
    if manifest["mode"] not in EXPECTED:
        raise ValueError("Unknown frozen run mode")
    return manifest


def cohort(manifest: dict) -> tuple[dict, list[str]]:
    selection = json.loads((WORK / "cohort.json").read_text())
    return selection, corpus.validate_manifest(selection, manifest["mode"])


def relocated_sources(sources: list[str], input_root: Path, run_dir: Path, support: Path) -> list[str]:
    # Relocate original literals before injecting absolute Kaggle paths. Doing
    # this afterwards would recursively prefix the new /kaggle/working paths.
    sources = list(sources)
    for i in range(2, 6):
        sources[i] = sources[i].replace("/kaggle/working", str(run_dir))
    sources[2] = reference.adapt_cell(sources[2], {
        "COMP_DIR": f"Path({str(input_root)!r})", "WORKING_DIR": f"Path({str(run_dir)!r})"})
    sources[3] = reference.adapt_cell(sources[3], {"ARTIFACTS": f"Path({str(support)!r})"})
    sources[4] = reference.adapt_worker_paths(sources[4])
    return sources


def verify_predictor(path: Path, run_dir: Path) -> dict:
    source = path.read_text()
    anchor = f'Path("{run_dir}")'
    if source.count(anchor) != 2 or source.count(str(run_dir)) != 2:
        raise ValueError("Expected exactly two relocated diagnostic log paths")
    normalized = source.replace(anchor, 'Path("/kaggle/working")')
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    if digest != PREDICTOR_SHA256:
        raise ValueError("E029 predictor changed beyond its two diagnostic log paths")
    return {"raw_sha256": reference.stability.file_sha256(path), "canonical_sha256": digest,
            "log_path_relocations": 2, "source_matches_submitted_e029": True}


def guard_before_inference(source: str) -> str:
    tree = ast.parse(source)
    found = 0
    body = []
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "test_stems"):
            body.extend(ast.parse("_verify_cloud_predictor()").body)
            found += 1
        body.append(node)
    if found != 1:
        raise ValueError("Pre-inference predictor audit anchor changed")
    tree.body = body
    return ast.unparse(ast.fix_missing_locations(tree))


def control(bundle: Path, manifest: dict):
    sources = reference.read_reference(bundle / "reference.ipynb")
    data = mounted("", "biohub-cell-tracking-during-development", competition=True)
    support = mounted("pilkwang", "biohub-tracking-support-pack-50ep-v1")
    secondary = mounted("pilkwang", "biohub-temporal-unet3d-seed314159-v1")
    center = mounted("pilkwang", "biohub-deepcenter-unet3d-center-prior-v1")
    run_dir = WORK / "control"
    run_dir.mkdir(parents=True, exist_ok=False)
    input_root = WORK / "input"
    for split in ("test", "train"):
        (input_root / split).mkdir(parents=True)
    for key in list(os.environ):
        if key.startswith("BIOHUB_"):
            del os.environ[key]
    os.chdir(run_dir)
    namespace = {"__name__": "biohub_e029_kaggle_development"}
    for i in (0, 1):
        exec(compile(sources[i], f"reference:cell-{i}", "exec"), namespace)
    os.environ.update({"BIOHUB_" + k: str(v) for k, v in reference.FROZEN_OVERRIDES.items()})
    os.environ.update({"BIOHUB_PRIMARY_ARTIFACT_MANIFEST": str(support / "ARTIFACT_MANIFEST.json"),
                       "BIOHUB_SECONDARY_ARTIFACT_MANIFEST": str(secondary / "ARTIFACT_MANIFEST.json"),
                       "BIOHUB_DEEPCENTER_ARTIFACT_MANIFEST": str(center / "ARTIFACT_MANIFEST.json"),
                       "BIOHUB_DEEPCENTER_CHECKPOINT": str(center / "weights/full_frame_center/best.pt"),
                       "BIOHUB_VALIDATOR_ENABLE": "0"})
    # Kaggle needs the reference's offline dependency installation, unlike the
    # already-provisioned server. No inference or policy statement is replaced.
    sources = relocated_sources(sources, input_root, run_dir, support)
    sources[4] = guard_before_inference(sources[4])
    namespace["_verify_cloud_predictor"] = lambda: save(
        run_dir / "predictor_source_audit.json",
        verify_predictor(run_dir / "tracking_repo/scripts/predict_unet_transformer.py", run_dir))
    for i in (2, 3):
        exec(compile(sources[i], f"reference:cell-{i}", "exec"), namespace)
    selection = corpus.reconstruct(data / "train", manifest["mode"])
    names = corpus.validate_manifest(selection, manifest["mode"])
    save(WORK / "cohort.json", selection)
    for name in names:
        source = data / "train" / f"{name}.zarr"
        if not source.is_dir():
            raise FileNotFoundError(source)
        for split in ("test", "train"):
            (input_root / split / source.name).symlink_to(source, target_is_directory=True)
    receipt = {"git_commit": manifest["git_commit"], "mode": manifest["mode"], "datasets": names,
               "data_dir": str(data), "reference_sha256": reference.REFERENCE_SHA256,
               "frozen_overrides": reference.FROZEN_OVERRIDES, "runtime_parameter_search": False,
               "selection_reads_ground_truth": True, "predictions_read_ground_truth": False,
               "evidence": "development only; checkpoint training overlap is unresolved"}
    save(WORK / "run_manifest.json", receipt)
    recovered = manifest.get("completed_control")
    if recovered:
        if manifest["mode"] != "smoke" or recovered["datasets"] != names:
            raise ValueError("Completed control is not the frozen smoke pair")
        path = bundle / "completed-control/raw_submission.csv"
        if reference.stability.file_sha256(path) != recovered["packaged_csv_sha256"]:
            raise ValueError("Completed control CSV bytes changed")
        predictor = verify_predictor(bundle / "completed-control/predict_unet_transformer.py", run_dir)
        (run_dir / "raw_submission.csv").write_bytes(path.read_bytes())
        receipt["completed_control_reused"] = recovered
        receipt["control_predicted_in_this_run"] = False
        print("RESUMING COMPLETED CONTROL EXPORT; no repeated E029 inference", flush=True)
    else:
        for i in (4, 5):
            print(f"EXECUTING FROZEN REFERENCE CELL {i}", flush=True)
            exec(compile(sources[i], f"reference:cell-{i}", "exec"), namespace)
        predictor = verify_predictor(run_dir / "tracking_repo/scripts/predict_unet_transformer.py", run_dir)
        (run_dir / "submission.csv").rename(run_dir / "raw_submission.csv")
        receipt["control_predicted_in_this_run"] = True
    bounds = reference.normalize_export_boundary(run_dir / "raw_submission.csv", run_dir / "control.csv",
                                                  input_root / "test", names)
    audit = reference.validate_submission(run_dir / "control.csv", input_root / "test", names)
    save(run_dir / "export_boundary_audit.json", bounds)
    save(run_dir / "topology_audit.json", audit)
    expected = EXPECTED[manifest["mode"]]
    parity = {"actual_csv_sha256": audit["submission_sha256"], "expected_csv_sha256": expected["csv_sha256"],
              "byte_parity": audit["submission_sha256"] == expected["csv_sha256"],
              "predictor_source_audit": predictor, "rows": audit["rows"]}
    save(WORK / "control_parity.json", parity)
    receipt["effective_environment"] = {k: v for k, v in os.environ.items() if k.startswith("BIOHUB_")}
    receipt["control_prediction_complete"] = True
    save(WORK / "run_manifest.json", receipt)


def peaks(bundle: Path, manifest: dict):
    import capture_point_detector_peaks as capture

    _, names = cohort(manifest)
    if not json.loads((WORK / "control_parity.json").read_text())["byte_parity"]:
        raise ValueError("E029 byte parity is required before E038 inference")
    args = SimpleNamespace(mode=manifest["mode"], device="cuda", shard=None,
                           data_dir=WORK / "input", output_dir=WORK / "v5",
                           reference_dir=mounted("hengck23", "hengck23-cell-point-detector-demo"),
                           runtime_dir=bundle / "kaggle", tracking_repo=WORK / "control/tracking_repo")
    capture.run_movies(args, names, manifest["git_commit"])


def bridges(bundle: Path, manifest: dict):
    import numpy as np
    import pandas as pd
    import point_gap_bridge as bridge
    from run_point_gap_experiment import append_bridges

    _, names = cohort(manifest)
    cache_receipt = json.loads((WORK / "v5/run_manifest.json").read_text())
    if (not cache_receipt["complete"] or cache_receipt["ground_truth_accessed"]
            or cache_receipt["datasets"] != names or cache_receipt["device"] != "cuda"):
        raise ValueError("Incomplete or incompatible real-image cache")
    frame = pd.read_csv(WORK / "control/control.csv")
    outputs, records, totals = [], {}, Counter()
    output_dir = WORK / "candidate"
    output_dir.mkdir(exist_ok=False)
    for name in names:
        path = WORK / "v5/peaks" / f"{name}.npz"
        record = cache_receipt["movies"][name]
        if reference.stability.file_sha256(path) != record["cache_sha256"] or record["gpu_peak_allocated_bytes"] <= 0:
            raise ValueError("CUDA cache changed or incomplete")
        group = frame[frame.dataset == name]
        nodes, edges = group[group.row_type == "node"], group[group.row_type == "edge"]
        with np.load(path, allow_pickle=False) as data:
            bridge.validate_peaks(data["coords"], data["scores"], data["processed_frames"], data["image_shape"])
            if data["image_shape"].tolist() != record["shape"] or len(data["coords"]) != record["peaks"]:
                raise ValueError("Cache shape/count disagrees with receipt")
            proposals, counts = bridge.bridge_one_frame(
                nodes.node_id.to_numpy(), nodes[["t", "z", "y", "x"]].to_numpy(),
                edges[["source_id", "target_id"]].to_numpy(), data["coords"], data["scores"])
        output = append_bridges(group, proposals)
        if not output.iloc[:len(group)].reset_index(drop=True).equals(group.reset_index(drop=True)):
            raise ValueError("E038 changed original E029 fields")
        outputs.append(output)
        records[name] = {"counts": counts, "bridges": proposals}
        totals.update(counts)
    output = pd.concat(outputs, ignore_index=True)
    output["id"] = np.arange(len(output))
    output.to_csv(output_dir / "submission.csv", index=False)
    audit = reference.validate_submission(output_dir / "submission.csv", WORK / "input/test", names)
    save(output_dir / "topology_audit.json", audit)
    save(output_dir / "bridge_audit.json", records)
    save(output_dir / "run_manifest.json", {"config": bridge.CONFIG, "component_counts": dict(totals),
         "all_candidate_predictions_completed_before_scoring": True,
         "preserve_all_original_nodes_coordinates_edges": True, "runtime_parameter_search": False})


def score(bundle: Path, manifest: dict, diagnose_only=False):
    _, names = cohort(manifest)
    scorer = bundle / "official"
    os.environ["PYTHONPATH"] = os.pathsep.join([str(scorer / "src"), str(scorer / "scripts")])
    control_dir = WORK / "control/geffs"
    subprocess.run([sys.executable, str(scorer / "scripts/csv_to_geffs.py"), "--csv",
                    str(WORK / "control/control.csv"), "--out-dir", str(control_dir), "--no-overwrite"], check=True)
    if sorted(reference.stability.movie_paths(control_dir)) != names:
        raise ValueError("Control conversion coverage mismatch")
    data = Path(json.loads((WORK / "run_manifest.json").read_text())["data_dir"])
    if diagnose_only:
        official = reference.stability.load_official_scorer(bundle / "kaggle", scorer)
        rows = {n: reference.stability.score_one(official, n, control_dir / f"{n}.geff", data / "train")[0]
                for n in names}
        result = reference.stability.official_summary(official, rows, names)
        save(WORK / "control_diagnostic.json", {"observed": result, "expected": EXPECTED[manifest["mode"]],
                                              "promotion_allowed": False, "candidate_run": False})
        return
    args = SimpleNamespace(runtime_dir=bundle / "kaggle", scorer_dir=scorer, control_dir=control_dir,
                           data_dir=data, output_dir=WORK / "candidate", minimum_pooled_delta=0.001,
                           expected_control_score=EXPECTED[manifest["mode"]]["score"],
                           experiment="E038 fixed observation-supported bridges; Kaggle development")
    result = reference.evaluate(args, names)
    reproduced = math.isclose(result["groups"]["all"]["control"]["score"],
                             EXPECTED[manifest["mode"]]["score"], rel_tol=0, abs_tol=1e-9)
    receipt = {"mode": manifest["mode"], "git_commit": manifest["git_commit"],
               "protocol_sha256": manifest["protocol_sha256"],
               "control_byte_parity": json.loads((WORK / "control_parity.json").read_text())["byte_parity"],
               "control_score_reproduced": reproduced, "official_control_score": result["groups"]["all"]["control"]["score"],
               "technical_check_passed": reproduced, "development_promotion_passed": result["promotion_passed"],
               "groups": result["groups"], "gates": result["gates"], "paired": result["paired"],
               "public_score": None, "formal_submission_created": False}
    save(WORK / "development_receipt.json", receipt)
    (WORK / "run_summary.md").write_text("# E038 Kaggle development\n\nFrozen development evidence; "
        "smoke does not select parameters or authorize promotion.\n\n```json\n" + json.dumps(receipt, indent=2) + "\n```\n")
    print(json.dumps(receipt, indent=2), flush=True)
    if not reproduced:
        raise ValueError("Official E029 control score changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--phase", choices=("control", "peaks", "bridges", "score", "diagnose-control"), required=True)
    args = parser.parse_args()
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("This adapter may run only on Kaggle compute")
    bundle = args.bundle.resolve()
    manifest = verified_bundle(bundle)
    WORK.mkdir(parents=True, exist_ok=True)
    os.environ.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                       "POLARS_MAX_THREADS": "4", "PYTHONUNBUFFERED": "1"})
    try:
        if args.phase == "diagnose-control":
            score(bundle, manifest, diagnose_only=True)
        else:
            globals()[args.phase](bundle, manifest)
    except Exception as exc:
        (WORK / "run_summary.md").write_text(
            f"# E038 Kaggle development failed\n\nPhase: {args.phase}\n\n{type(exc).__name__}: {exc}\n\nNo promotion.\n")
        raise


if __name__ == "__main__":
    main()
