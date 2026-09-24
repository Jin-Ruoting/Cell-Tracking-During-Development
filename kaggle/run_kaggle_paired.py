#!/usr/bin/env python3
"""E039: append the frozen bridge policy to one immutable Kaggle control graph.

This separate experiment does not certify historical E038 server parity.
Every candidate uses the exact control generated in the same cloud run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import capture_point_detector_peaks as capture
import run_kaggle_development as cloud

WORK = Path("/kaggle/working/logs/e039-development")
reference = cloud.reference


def graph_fingerprints(path: Path, names: list[str]) -> dict[str, str]:
    """Compare graph content, allowing only export row IDs and a node-ID offset."""
    graphs = {name: {"nodes": {}, "edges": []} for name in names}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] not in graphs:
                continue
            graph = graphs[row["dataset"]]
            if row["row_type"] == "node":
                node = int(row["node_id"])
                if node in graph["nodes"]:
                    raise ValueError("Duplicate anchor node ID")
                graph["nodes"][node] = [float(row[k]) for k in ("t", "z", "y", "x")]
            elif row["row_type"] == "edge":
                graph["edges"].append((int(row["source_id"]), int(row["target_id"])))
            else:
                raise ValueError("Unknown anchor row type")
    result = {}
    for name, graph in graphs.items():
        nodes = graph["nodes"]
        if not nodes or not graph["edges"]:
            raise ValueError("Missing anchor graph")
        # Concatenating preceding movies can offset every node ID equally.
        offset = min(nodes)
        mapping = {node: node - offset for node in nodes}
        payload = {"nodes": [[mapping[n], *nodes[n]] for n in sorted(nodes)],
                   "edges": sorted((mapping[a], mapping[b]) for a, b in graph["edges"])}
        result[name] = hashlib.sha256(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return result


def verify_repeat_evidence(diagnostic: dict, run: dict, csv_sha256: str) -> dict:
    repeated = diagnostic.get("cloud_repeatability", {})
    parity = diagnostic.get("control_parity", {})
    if (diagnostic.get("candidate_run") is not False
            or diagnostic.get("promotion_allowed") is not False
            or diagnostic["observed"]["n"] != 2
            or repeated.get("csv_bytes_reproduced") is not True
            or repeated.get("score_reproduced") is not True
            or parity.get("actual_csv_sha256") != csv_sha256
            or repeated["reference"]["csv_sha256"] != csv_sha256
            or not math.isclose(diagnostic["observed"]["score"], repeated["reference"]["score"], rel_tol=0, abs_tol=1e-9)
            or parity["predictor_source_audit"]["canonical_sha256"] != cloud.PREDICTOR_SHA256
            or run.get("mode") != "smoke" or run.get("datasets") != list(reference.SMOKE_NAMES)
            or run.get("reference_sha256") != reference.REFERENCE_SHA256
            or run.get("frozen_overrides") != reference.FROZEN_OVERRIDES
            or run.get("control_predicted_in_this_run") is not True
            or run.get("control_prediction_complete") is not True
            or run.get("predictions_read_ground_truth") is not False
            or not run.get("runtime_versions")):
        raise ValueError("E039 requires a completed independent cloud control repeat with identical bytes and score")
    return {"csv_sha256": csv_sha256, "score": diagnostic["observed"]["score"],
            "runtime_versions": run["runtime_versions"], "source_git_commit": run["git_commit"]}


def frozen_control() -> dict:
    receipt = json.loads((WORK / "frozen_control.json").read_text())
    if reference.stability.file_sha256(WORK / "control/control.csv") != receipt["csv_sha256"]:
        raise ValueError("Frozen paired control changed between phases")
    return receipt


def control(bundle: Path, manifest: dict):
    cloud.control(bundle, manifest)
    run = json.loads((WORK / "run_manifest.json").read_text())
    baseline = manifest["cloud_baseline"]
    if run["runtime_versions"] != baseline["runtime_versions"]:
        raise ValueError("E039 runtime versions differ from the repeated cloud control")
    import torch
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(devices) != 2 or any("T4" not in name for name in devices):
        raise ValueError("E039 requires the fixed two-T4 environment")
    csv_path = WORK / "control/control.csv"
    digest = reference.stability.file_sha256(csv_path)
    if manifest["mode"] == "smoke" and digest != baseline["csv_sha256"]:
        raise ValueError("E039 smoke control does not reproduce frozen cloud bytes")
    expected = graph_fingerprints(bundle / "cloud-control/control.csv", list(reference.SMOKE_NAMES))
    actual = graph_fingerprints(csv_path, list(reference.SMOKE_NAMES))
    if actual != expected:
        raise ValueError("E039 anchor graphs differ from the frozen cloud control")
    cloud.save(WORK / "frozen_control.json", {
        "csv_sha256": digest, "anchor_fingerprints": actual, "runtime_versions": run["runtime_versions"],
        "devices": devices, "historical_server_parity": json.loads((WORK / "control_parity.json").read_text()),
        "paired_control_frozen_before_candidate": True, "experiment": "E039"})


def peaks(bundle: Path, manifest: dict):
    frozen_control()
    _, names = cloud.cohort(manifest)
    args = SimpleNamespace(mode=manifest["mode"], device="cuda", shard=None,
                           data_dir=WORK / "input", output_dir=WORK / "v5",
                           reference_dir=cloud.mounted("hengck23", "hengck23-cell-point-detector-demo"),
                           runtime_dir=bundle / "kaggle", tracking_repo=WORK / "control/tracking_repo")
    capture.run_movies(args, names, manifest["git_commit"])
    frozen_control()


def bridges(bundle: Path, manifest: dict):
    frozen_control()
    cloud.bridges(bundle, manifest)
    frozen_control()


def score(bundle: Path, manifest: dict):
    frozen = frozen_control()
    _, names = cloud.cohort(manifest)
    candidate = WORK / "candidate"
    candidate_receipt = json.loads((candidate / "run_manifest.json").read_text())
    if candidate_receipt.get("all_candidate_predictions_completed_before_scoring") is not True:
        raise ValueError("All paired candidate graphs must be frozen before scoring")
    candidate_sha = reference.stability.file_sha256(candidate / "submission.csv")
    scorer = bundle / "official"
    os.environ["PYTHONPATH"] = os.pathsep.join([str(scorer / "src"), str(scorer / "scripts")])
    control_dir = WORK / "control/geffs"
    subprocess.run([sys.executable, str(scorer / "scripts/csv_to_geffs.py"), "--csv",
                    str(WORK / "control/control.csv"), "--out-dir", str(control_dir), "--no-overwrite"], check=True)
    if sorted(reference.stability.movie_paths(control_dir)) != names:
        raise ValueError("Paired control GEFF coverage mismatch")
    data = Path(json.loads((WORK / "run_manifest.json").read_text())["data_dir"])
    args = SimpleNamespace(runtime_dir=bundle / "kaggle", scorer_dir=scorer, control_dir=control_dir,
                           data_dir=data, output_dir=candidate, minimum_pooled_delta=0.001,
                           expected_control_score=None, experiment="E039 fixed bridges on frozen Kaggle control")
    result = reference.evaluate(args, names)
    official = reference.stability.load_official_scorer(bundle / "kaggle", scorer)
    anchor_score = reference.stability.official_summary(
        official, result["official_rows"]["control"], list(reference.SMOKE_NAMES))["score"]
    if not math.isclose(anchor_score, manifest["cloud_baseline"]["score"], rel_tol=0, abs_tol=1e-9):
        raise ValueError("Official scoring of fixed cloud anchors changed")
    frozen_control()
    if reference.stability.file_sha256(candidate / "submission.csv") != candidate_sha:
        raise ValueError("Paired candidate changed during scoring")
    receipt = {"experiment": "E039", "mode": manifest["mode"], "git_commit": manifest["git_commit"],
               "protocol_sha256": manifest["protocol_sha256"], "technical_check_passed": True,
               "cloud_baseline": manifest["cloud_baseline"], "control": frozen,
               "candidate_csv_sha256": candidate_sha, "anchor_control_score": anchor_score,
               "development_promotion_passed": manifest["mode"] == "full" and result["promotion_passed"],
               "groups": result["groups"], "gates": result["gates"], "paired": result["paired"],
               "component_counts": candidate_receipt["component_counts"],
               "evidence": "Kaggle paired development; training overlap unresolved; not historical server replay",
               "public_score": None, "formal_submission_created": False}
    cloud.save(WORK / "paired_development_receipt.json", receipt)
    (WORK / "run_summary.md").write_text("# E039 Kaggle paired development\n\n" +
        "Both arms use one frozen cloud control. Smoke does not select parameters.\n\n```json\n" +
        json.dumps(receipt, indent=2) + "\n```\n")
    print(json.dumps(receipt, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--phase", choices=("control", "peaks", "bridges", "score"), required=True)
    args = parser.parse_args()
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("E039 runs only on Kaggle compute")
    bundle = args.bundle.resolve()
    manifest = cloud.verified_bundle(bundle)
    if manifest.get("experiment") != "E039" or manifest.get("completed_control"):
        raise ValueError("E039 requires its own manifest and fresh paired control")
    baseline = verify_repeat_evidence(
        json.loads((bundle / "cloud-control/control_diagnostic.json").read_text()),
        json.loads((bundle / "cloud-control/run_manifest.json").read_text()),
        reference.stability.file_sha256(bundle / "cloud-control/control.csv"))
    if baseline != manifest["cloud_baseline"]:
        raise ValueError("Cloud baseline evidence differs from the frozen manifest")
    cloud.WORK = WORK
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        globals()[args.phase](bundle, manifest)
    except Exception as exc:
        (WORK / "run_summary.md").write_text(
            f"# E039 failed\n\nPhase: {args.phase}\n\n{type(exc).__name__}: {exc}\n\nNo promotion.\n")
        raise


if __name__ == "__main__":
    main()
