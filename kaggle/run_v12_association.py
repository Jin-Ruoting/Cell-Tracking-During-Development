#!/usr/bin/env python3
"""E040 fixed-node v12 association, complete movies on private Kaggle compute."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import check_v12_checkpoint as checkpoint
import check_v12_fixed_nodes as cpu_check
import run_kaggle_paired as paired
import v12_edge_swap as swap

reference = paired.reference
WORK = Path("/kaggle/working/logs/e040-association")
EXCLUDED = {"44b6_12dfb391", "44b6_267148e4", "44b6_2a2eff9f", "44b6_341df25f",
            "44b6_949adeb1", "6bba_09961292"}


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return reference.stability.file_sha256(Path(path))


def verify_control(config):
    expected = config["control"]
    owner, slug = expected["kernel"].split("/")
    roots = {p.resolve() for p in (Path("/kaggle/input") / slug,
                                  Path("/kaggle/input/notebooks") / owner / slug) if p.is_dir()}
    if len(roots) != 1:
        raise ValueError("Missing unique completed control Notebook input")
    directory = roots.pop() / expected["directory"]
    path = directory / "control/control.csv"
    run = json.loads((directory / "run_manifest.json").read_text())
    if (digest(path) != expected["csv_sha256"] or run["git_commit"] != expected["source_git_commit"]
            or run["datasets"] != config["datasets"] or run["control_prediction_complete"] is not True
            or run["control_predicted_in_this_run"] is not True or run["predictions_read_ground_truth"] is not False
            or run["reference_sha256"] != reference.REFERENCE_SHA256
            or run["frozen_overrides"] != reference.FROZEN_OVERRIDES):
        raise ValueError("Frozen original control or its provenance changed")
    if config["mode"] == "smoke":
        cpu_check.frozen_control()
    else:
        frozen = json.loads((directory / "frozen_control.json").read_text())
        receipt = json.loads((directory / "paired_development_receipt.json").read_text())
        cohort = json.loads((directory / "cohort.json").read_text())
        paired.cloud.corpus.validate_manifest(cohort, "full")
        if (frozen != expected["frozen_receipt"] or frozen["csv_sha256"] != digest(path)
                or receipt != expected["completion_receipt"] or receipt["experiment"] != "E039"
                or receipt["mode"] != "full" or receipt["technical_check_passed"] is not True
                or cohort["datasets"] != config["datasets"]
                or paired.graph_fingerprints(path, list(reference.SMOKE_NAMES)) != frozen["anchor_fingerprints"]):
            raise ValueError("Completed full control receipt/anchor mismatch")
    return path, run


def unchanged(config):
    path, _ = verify_control(config)
    if digest(WORK / "control.csv") != digest(path):
        raise ValueError("Working control copy changed")


def setup(bundle, config):
    WORK.mkdir(parents=True, exist_ok=False)
    control, original = verify_control(config)
    support = paired.cloud.mounted("pilkwang", "biohub-tracking-support-pack-50ep-v1")
    os.environ.setdefault("POLARS_PREFER_PKG", "32")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--no-deps",
                    "--find-links", str(support / "wheels"), "tracksdata", "zarr==3.2.1", "numcodecs==0.15.1",
                    "donfig==0.8.1.post1", "geff==1.2.0.1.1", "geff-spec==1.1.1", "pyscipopt==6.2.1",
                    "ilpy==0.6.0", "rustworkx==0.18.0", "polars==1.42.0", "polars-runtime-32==1.42.0",
                    "bidict==0.23.1", "imagecodecs==2026.6.26"], check=True)
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "numpy", "scipy", "pandas", "zarr", "tracksdata", "geff", "polars")}
    if any(original["runtime_versions"][name] != version for name, version in versions.items()):
        raise ValueError("Scoring/inference package versions differ from frozen cloud control")
    shutil.copyfile(control, WORK / "control.csv")
    for name in ("original_movies", "candidate_movies", "movie_receipts", "image_only", "candidate"):
        (WORK / name).mkdir()
    data = paired.cloud.mounted("", "biohub-cell-tracking-during-development", competition=True)
    handles = {}
    try:
        with control.open(newline="") as handle:
            reader = csv.DictReader(handle)
            writers = {}
            for name in config["datasets"]:
                target = data / "train" / (name + ".zarr")
                if not target.is_dir():
                    raise FileNotFoundError(target)
                (WORK / "image_only" / target.name).symlink_to(target, target_is_directory=True)
                handles[name] = (WORK / "original_movies" / (name + ".csv")).open("x", newline="")
                writers[name] = csv.DictWriter(handles[name], fieldnames=reader.fieldnames)
                writers[name].writeheader()
            previous = -1
            for row in reader:
                if int(row["id"]) != previous + 1 or row["dataset"] not in writers:
                    raise ValueError("Unexpected control row ordering or movie")
                previous += 1
                writers[row["dataset"]].writerow(row)
    finally:
        for handle in handles.values():
            handle.close()
    control_topology = reference.validate_submission(WORK / "control.csv", WORK / "image_only", config["datasets"])
    save(WORK / "control_topology.json", control_topology)
    save(WORK / "runtime.json", {"versions": versions, "python": sys.version,
                                  "source_control": config["control"], "data_dir": str(data)})
    unchanged(config)


def fixed_graph(path):
    import numpy as np
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    nodes = {int(r["node_id"]): [int(r[k]) for k in ("t", "z", "y", "x")] for r in rows if r["row_type"] == "node"}
    if len(nodes) != sum(r["row_type"] == "node" for r in rows):
        raise ValueError("Duplicate fixed node")
    ids = [sorted(n for n, point in nodes.items() if point[0] == t) for t in range(100)]
    if any(len(frame) > 1024 for frame in ids):
        raise ValueError("Fixed frame exceeds reviewed 1024-node range; no subsampling allowed")
    points = [np.asarray([nodes[n][1:] for n in frame], dtype=np.float32).reshape(-1, 3) for frame in ids]
    edge_rows = [[] for _ in range(99)]
    for i, row in enumerate(rows):
        if row["row_type"] == "edge":
            a, b = int(row["source_id"]), int(row["target_id"])
            if a not in nodes or b not in nodes or nodes[b][0] != nodes[a][0] + 1 or not 0 <= nodes[a][0] < 99:
                raise ValueError("Invalid original edge")
            edge_rows[nodes[a][0]].append(i)
    return fields, rows, ids, points, edge_rows


def validate_rewrite(original, rows, fields, expected_changes):
    if len(original) != len(rows):
        raise ValueError("Candidate changed row count")
    actual_changes = 0
    outdegree = Counter(r["source_id"] for r in original if r["row_type"] == "edge")
    for before, after in zip(original, rows):
        allowed = {"target_id"} if before["row_type"] == "edge" else set()
        if any(before[k] != after[k] for k in fields if k not in allowed):
            raise ValueError("Fixed graph field was modified")
        if before["row_type"] == "edge" and outdegree[before["source_id"]] == 2 and before != after:
            raise ValueError("Original division edge was changed")
        actual_changes += before != after
    if actual_changes != expected_changes:
        raise ValueError("Change-count mismatch")
    for key in ("source_id", "target_id"):
        if Counter(r[key] for r in original if r["row_type"] == "edge") != Counter(r[key] for r in rows if r["row_type"] == "edge"):
            raise ValueError("Whole-movie degree changed")
    edges = [(r["source_id"], r["target_id"]) for r in rows if r["row_type"] == "edge"]
    if len(set(edges)) != len(edges):
        raise ValueError("Duplicate candidate edge")


def predict_movie(name, model, source):
    import numpy as np
    import torch
    import zarr

    start = time.monotonic()
    fields, rows, ids, points, edge_rows = fixed_graph(WORK / "original_movies" / (name + ".csv"))
    original = [row.copy() for row in rows]
    group = zarr.open_group(str(WORK / "image_only" / (name + ".zarr")), mode="r")
    image, attrs = group["0"], dict(group.attrs)
    transform = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
    if (tuple(image.shape) != (100, 64, 256, 256) or image.dtype != np.uint16
            or transform["type"] != "scale" or tuple(transform["scale"][-3:]) != tuple(swap.SCALE)):
        raise ValueError("Unexpected image format or physical scale")
    low, high = (float(attrs["image_statistics"]["quantiles"][k]) for k in ("0.001", "0.999"))
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("Invalid image-only quantiles")
    counters, audit = Counter(changed_edges=0, accepted_swaps=0), []
    previous = None
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for t in range(100):
            current = None
            if ids[t]:
                volume = image[t, ::1, ::4, ::4].astype(np.float32)
                volume = np.clip((volume - low) / (high - low + 1e-6), 0.0, None).astype(np.float32)
                coord = points[t] / np.asarray([1, 4, 4], dtype=np.float32)
                if not np.isfinite(volume).all() or (coord < 0).any() or (coord >= 64).any():
                    raise ValueError("Invalid normalized image or frozen coordinate")
                coord = torch.from_numpy(coord)[None].to("cuda")
                mask = torch.ones((1, len(ids[t])), dtype=torch.bool, device="cuda")
                layers, _ = model.unet.make_feature(torch.from_numpy(np.ascontiguousarray(volume))[None, None].to("cuda"))
                sampled = source.sample_pyr_feature_at_zyx([layers[3], layers[4], layers[2]], coord, (64, 64, 64), mask)
                if any(not torch.isfinite(f).all() for f in sampled):
                    raise ValueError("Nonfinite fixed-node features")
                current = (sampled, coord, mask)
                del layers
            if t:
                indices = edge_rows[t - 1]
                old_pairs = np.asarray([[int(rows[i]["source_id"]), int(rows[i]["target_id"])] for i in indices], dtype=np.int64).reshape(-1, 2)
                if previous is None or current is None:
                    if len(old_pairs):
                        raise ValueError("Edges incident to an empty frame")
                    audit.append({"frame": t - 1, "empty_endpoint_frame": True, "changed_edges": 0})
                else:
                    links = model.linker(previous[0], current[0], previous[1], current[1], previous[2], current[2], (64, 64, 64))
                    logits = links.edge_logit[0].float().cpu().numpy()
                    changed, decisions, counts = swap.swap_frame(ids[t - 1], ids[t], points[t - 1], points[t], old_pairs, logits)
                    for i, (a, b) in zip(indices, changed):
                        if int(rows[i]["source_id"]) != a:
                            raise ValueError("Swap altered source order")
                        rows[i]["target_id"] = str(int(b))
                    audit.append({"frame": t - 1, "counts": counts, "decisions": decisions,
                                  "logit_shape": list(logits.shape), "logit_range": [float(logits.min()), float(logits.max())],
                                  "logit_sha256": hashlib.sha256(logits.astype("<f4").tobytes()).hexdigest()})
                    counters.update(counts)
                    del links, logits
            previous = current
            if (t + 1) % 20 == 0:
                print(f"E040 FRAMES {name} {t + 1}/100 swaps={counters['accepted_swaps']}", flush=True)
    validate_rewrite(original, rows, fields, counters["changed_edges"])
    if len(audit) != 99:
        raise ValueError("Incomplete frame-pair coverage")
    path = WORK / "candidate_movies" / (name + ".csv")
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    record = {"dataset": name, "frames_considered": 100, "frame_pairs": 99, "node_counts": list(map(len, ids)),
              "counts": dict(counters), "pairs": audit, "all_original_node_fields_preserved": True,
              "all_node_degrees_preserved": True, "csv_sha256": digest(path),
              "seconds": time.monotonic() - start, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
              "quantiles": [low, high], "predictions_used_ground_truth": False}
    save(WORK / "movie_receipts" / (name + ".json"), record)
    print(json.dumps({k: v for k, v in record.items() if k not in {"pairs", "node_counts"}}), flush=True)


def predict(config, shard):
    unchanged(config)
    report = {}
    author = paired.cloud.mounted("hengck23", "hengck23-cell-point-detector-demo")
    saved = checkpoint.load_checkpoint(author, report, cpu_only=False)
    import torch
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise ValueError("Each worker must use one fixed T4")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    source = checkpoint.reviewed.load_reviewed(author, "model_v12")
    cfg = source.DotDict(**{k: source.DotDict(**saved["CFG"][k]) for k in ("unet_cfg", "tx_cfg")})
    model = source.End2EndCellLinker(CFG=cfg).eval()
    model.load_state_dict(saved["model_state_dict"], strict=True)
    del saved
    model = model.to("cuda")
    report.update(device=torch.cuda.get_device_name(0), dtype="float32", tf32=False, strict_model_load=True)
    save(WORK / f"worker-{shard}.json", report)
    for name in config["datasets"][shard::2]:
        predict_movie(name, model, source)
    unchanged(config)


def merge(config):
    unchanged(config)
    names = config["datasets"]
    if sorted(p.stem for p in (WORK / "movie_receipts").glob("*.json")) != names:
        raise ValueError("Incomplete movie prediction coverage")
    totals = Counter()
    all_rows, fields = [], None
    for name in names:
        record = json.loads((WORK / "movie_receipts" / (name + ".json")).read_text())
        path = WORK / "candidate_movies" / (name + ".csv")
        if (record["csv_sha256"] != digest(path) or record["frames_considered"] != 100
                or record["frame_pairs"] != 99 or record["predictions_used_ground_truth"] is not False
                or record["all_original_node_fields_preserved"] is not True or record["all_node_degrees_preserved"] is not True):
            raise ValueError("Candidate movie receipt failed")
        totals.update(record["counts"])
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            all_rows.extend(reader)
    all_rows.sort(key=lambda row: int(row["id"]))
    target = WORK / "candidate/submission.csv"
    with target.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    audit = reference.validate_submission(target, WORK / "image_only", names)
    original = json.loads((WORK / "control_topology.json").read_text())
    if audit["datasets"] != original["datasets"] or audit["rows"] != original["rows"]:
        raise ValueError("Candidate altered counts/degrees/divisions")
    save(WORK / "candidate/topology_audit.json", audit)
    save(WORK / "candidate/run_manifest.json", {"all_candidate_predictions_completed_before_scoring": True,
                                               "config": swap.CONFIG, "counts": dict(totals), "csv_sha256": digest(target)})


def score(bundle, config):
    unchanged(config)
    manifest = json.loads((WORK / "candidate/run_manifest.json").read_text())
    if manifest["all_candidate_predictions_completed_before_scoring"] is not True:
        raise ValueError("Predictions must precede scoring")
    candidate_sha = manifest["csv_sha256"]
    if digest(WORK / "candidate/submission.csv") != candidate_sha:
        raise ValueError("Candidate changed before official scoring")
    scorer = bundle / "official"
    os.environ["PYTHONPATH"] = os.pathsep.join([str(bundle / "kaggle"), str(scorer / "src"), str(scorer / "scripts")])
    control_dir = WORK / "control_geffs"
    subprocess.run([sys.executable, str(scorer / "scripts/csv_to_geffs.py"), "--csv", str(WORK / "control.csv"),
                    "--out-dir", str(control_dir), "--no-overwrite"], check=True)
    data = Path(json.loads((WORK / "runtime.json").read_text())["data_dir"])
    args = SimpleNamespace(runtime_dir=bundle / "kaggle", scorer_dir=scorer, control_dir=control_dir,
                           data_dir=data, output_dir=WORK / "candidate", minimum_pooled_delta=0.001,
                           expected_control_score=config["control"]["score"], experiment="E040 fixed-node v12 reciprocal swaps")
    result = reference.evaluate(args, config["datasets"])
    rows = result["official_rows"]
    official = reference.stability.load_official_scorer(bundle / "kaggle", scorer)
    anchor_score = reference.stability.official_summary(official, rows["control"], list(reference.SMOKE_NAMES))["score"]
    if (not math.isclose(result["groups"]["all"]["control"]["score"], config["control"]["score"], rel_tol=0, abs_tol=1e-9)
            or not math.isclose(anchor_score, 0.9638391788805566, rel_tol=0, abs_tol=1e-9)):
        raise ValueError("Official fixed control/anchor score changed")
    division_equal = all(all(rows["control"][n][k] == rows["candidate"][n][k]
                             for k in ("division_tp", "division_fp", "division_fn", "num_pred_nodes", "node_recall", "total_node_ratio"))
                         for n in config["datasets"])
    if not division_equal:
        raise ValueError("Node/division official components changed despite fixed geometry and division edges")
    exclusion = None
    if config["mode"] == "full":
        names = sorted(set(config["datasets"]) - EXCLUDED)
        if len(names) != 58:
            raise ValueError("Preregistered author-exclusion cohort changed")
        groups = {}
        for key, selected in {"all": names, **{p: [n for n in names if n.startswith(p + "_")] for p in ("44b6", "6bba")}}.items():
            c = reference.stability.official_summary(official, rows["control"], selected)
            v = reference.stability.official_summary(official, rows["candidate"], selected)
            groups[key] = {"n": len(selected), "control": c, "candidate": v, "delta": reference.stability.summary_delta(c, v)}
        exclusion = {"excluded": sorted(EXCLUDED), "groups": groups,
                     "passed": all(g["delta"]["score"] > 0 for g in groups.values())}
    unchanged(config)
    if digest(WORK / "candidate/submission.csv") != candidate_sha:
        raise ValueError("Candidate changed during official scoring")
    receipt = {"experiment": "E040", "mode": config["mode"], "git_commit": config["git_commit"],
               "protocol_sha256": config["protocol_sha256"], "technical_check_passed": True,
               "control": config["control"], "candidate_csv_sha256": candidate_sha, "counts": manifest["counts"],
               "anchor_control_score": anchor_score,
               "official_node_and_division_components_unchanged": division_equal,
               "groups": result["groups"], "gates": result["gates"], "paired": result["paired"],
               "author_exclusion": exclusion,
               "development_promotion_passed": config["mode"] == "full" and result["promotion_passed"]
                   and manifest["counts"]["changed_edges"] > 0 and exclusion["passed"],
               "formal_submission_created": False, "public_score": None,
               "evidence": "Frozen-control paired development; training overlap unknown"}
    save(WORK / "paired_development_receipt.json", receipt)
    (WORK / "run_summary.md").write_text("# E040 fixed-node association\n\n```json\n" + json.dumps(receipt, indent=2) + "\n```\n")
    print(json.dumps(receipt, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--phase", choices=("setup", "predict", "merge", "score"), required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1))
    args = parser.parse_args()
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("E040 model/evaluation may execute only on Kaggle compute")
    bundle = args.bundle.resolve()
    config = json.loads((bundle / "bundle_manifest.json").read_text())
    for name, value in config["files"].items():
        path = (bundle / name).resolve()
        if bundle not in path.parents or digest(path) != value:
            raise ValueError("Bundled file mismatch: " + name)
    if config["experiment"] != "E040" or config["config"] != swap.CONFIG:
        raise ValueError("Frozen E040 policy changed")
    try:
        if args.phase == "setup":
            setup(bundle, config)
        elif args.phase == "predict":
            if args.shard is None:
                raise ValueError("Prediction requires one explicit shard")
            predict(config, args.shard)
        elif args.phase == "merge":
            merge(config)
        else:
            score(bundle, config)
    except Exception as exc:
        if WORK.is_dir():
            (WORK / "run_summary.md").write_text(f"# E040 failed\n\nPhase {args.phase}: {type(exc).__name__}: {exc}\n\nNo promotion.\n")
        raise


if __name__ == "__main__":
    main()
