#!/usr/bin/env python3
"""E042: change only HOCT weights after the frozen upstream scale fix."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd

import run_hoct_scaled_consensus as scaled

ROOT = scaled.ROOT
base, reference = scaled.base, scaled.reference
read, save, digest = scaled.read, scaled.save, scaled.digest
V0_PROTOCOL = "af0df0fd39358e4a266c720bdd406266c8826761fde52baa8937f4cf4422bcaa"
V0_RUNS = {"smoke": "s221_hoct_scaled_smoke_20260924v1", "full": "s222_hoct_scaled_full_20260924v1"}
V0_COMMITS = {"smoke": "eb489e69e82b4f784a3679c8ec5466cb568603bf", "full": "b2b6828c07e815dabf8744bb99063d3844301336"}
ASSET = ROOT / "logs/s224_hoct_v1_cpu_20260924v1"
V1_SHA = "5bd836dfcb15ad796ea79a9595841a3e73b650a71c4acba3fc66aac65d745b33"


def protocol():
    files = scaled.packaging.source_closure(ROOT, ["kaggle/run_hoct_v1_consensus.py", "tests/test_hoct_compatibility.py", "tests/test_hoct_v1_preservation.py"])
    return scaled.packaging.protocol_digest(files)


def require_asset():
    receipt = read(ASSET / "runtime_receipt.json")
    weight = ASSET / "general_v1.pt"
    if (receipt.get("passed") is not True or receipt["git_commit"] != "9ab6581ffbfe2529efb2de5f785a22b1e7d0e66b"
            or receipt["sha256"] != V1_SHA or receipt["loaded_on_cpu"] is not True
            or receipt["parameters"] != 6_252_593 or receipt["input_projection_shape"] != [288, 19]
            or weight.stat().st_size != 25_496_698 or digest(weight) != V1_SHA):
        raise ValueError("Verified server v1 asset differs")
    return weight


def audit_preservation(original, exported):
    """Independently audit the exported graph, including unscored division edges."""
    if set(original.dataset.unique()) != set(exported.dataset.unique()):
        raise ValueError("Output contains a new or missing dataset")
    columns = [column for column in original.columns if column != "id"]
    old_nodes = original.loc[original.row_type == "node", columns].reset_index(drop=True)
    new_nodes = exported.loc[exported.row_type == "node", columns].reset_index(drop=True)
    if not old_nodes.equals(new_nodes):
        raise ValueError("An original E029 node field changed")
    records = []
    for name in sorted(original.dataset.unique()):
        nodes = old_nodes.loc[old_nodes.dataset == name]
        duplicate = nodes.duplicated(["t", "z", "y", "x"], keep=False)
        ambiguous = set(nodes.loc[duplicate, "node_id"].astype(int))
        edge = original.loc[(original.dataset == name) & (original.row_type == "edge")]
        before = set(zip(edge.source_id.astype(int), edge.target_id.astype(int)))
        edge = exported.loc[(exported.dataset == name) & (exported.row_type == "edge")]
        after = set(zip(edge.source_id.astype(int), edge.target_id.astype(int)))
        if len(after) != len(edge) or not after <= before:
            raise ValueError("Output contains a new or duplicate edge")
        degree = Counter(source for source, _ in before)
        protected = {(a, b) for a, b in before if degree[a] == 2 or a in ambiguous or b in ambiguous}
        if not protected <= after:
            raise ValueError("Original division or coincident-identity edge was removed")
        records.append({"dataset": name, "protected_edges": len(protected), "removed_edges": len(before - after)})
    return {"passed": True, "all_node_fields_unchanged": True, "only_ordinary_edges_deleted": True, "movies": records}


def infer(args):
    runtime = scaled.fixed_runtime()
    sys.path.insert(0, str(runtime))
    import hoct._api as api
    import torch
    import zarr
    from hoct import load_model, predict
    from tracksdata.functional import TilingScheme

    if Path(api.__file__).resolve() != runtime / "hoct/_api.py":
        raise ValueError("Scale-fixed library was not imported")
    previous = read(scaled.LEGACY / "inference" / args.image.stem / "receipt.json")
    versions = {key: importlib.metadata.version(key) for key in previous["versions"]}
    if versions != previous["versions"]:
        raise ValueError("Inference dependency versions changed")
    weight = require_asset()
    if digest(args.input_nodes) != args.input_sha256:
        raise ValueError("Original node input changed")
    if any((args.output_dir / name).exists() for name in ("pairs.npz", "receipt.json")):
        raise FileExistsError("Never overwrite a v1 prediction")
    with np.load(args.input_nodes, allow_pickle=False) as data:
        ids, points = data["ids"], data["points"]
    original_count = len(ids)
    ids, points, ambiguous = base.unique_positions(ids, points)
    started = time.monotonic()
    images = np.asarray(zarr.open_group(str(args.image), mode="r")["0"][:])
    if images.ndim != 4 or images.shape[0] != 100 or images.dtype != np.uint16:
        raise ValueError("Full-movie raw image contract changed")
    labels = base.geometry.rasterize(points, images.shape)
    prepared = time.monotonic()
    model = load_model(weight, device="cuda")
    with torch.inference_mode():
        solution = predict(model, labels=labels, images=images,
                           scale=(1.0, *base.geometry.SCALE), max_delta_t=1,
                           tiling_scheme=TilingScheme(tile_shape=(5, 32, 128, 128), overlap_shape=(1, 8, 16, 16)))
    nodes = solution.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    edges = solution.edge_attrs(attr_keys=[])
    out_points = np.column_stack([nodes[key].to_numpy() for key in ("t", "z", "y", "x")])
    mapping, max_distance = base.geometry.snap_nodes(out_points, points)
    remap = {int(node): int(ids[index]) for node, index in zip(nodes["node_id"].to_list(), mapping)}
    pairs = [(remap[int(a)], remap[int(b)]) for a, b in zip(edges["source_id"].to_list(), edges["target_id"].to_list())]
    times = dict(zip(map(int, ids), points[:, 0].astype(int)))
    if (not pairs or len(pairs) != len(set(pairs)) or any(times[b] - times[a] != 1 for a, b in pairs)
            or max(Counter(a for a, _ in pairs).values(), default=0) > 2
            or max(Counter(b for _, b in pairs).values(), default=0) > 1):
        raise ValueError("Invalid HOCT temporal solution")
    np.savez_compressed(args.output_dir / "pairs.npz", pairs=np.asarray(pairs, dtype=np.int64))
    receipt = {"movie": args.image.stem, "frames": len(images), "input_nodes": original_count,
               "represented_positions": len(ids), "ambiguous_node_ids": sorted(ambiguous),
               "solution_nodes": len(mapping), "solution_node_fraction": len(mapping) / original_count,
               "represented_position_coverage": len(mapping) / len(ids), "edges": len(pairs),
               "max_snap_distance_um": max_distance, "prepare_seconds": prepared - started,
               "inference_audit_seconds": time.monotonic() - prepared,
               "input_sha256": args.input_sha256, "pairs_sha256": digest(args.output_dir / "pairs.npz"),
               "checkpoint_sha256": V1_SHA, "versions": versions,
               "upstream_fix_commit": scaled.FIX_COMMIT, "scale_source_sha256": scaled.FIXED_SHA,
               "ground_truth_read": False, "fallback_used": False, "passed": True}
    save(args.output_dir / "receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


def run(args):
    scaled.fixed_runtime()
    weight = require_asset()
    names = reference.selected_names(args.control_dir, args.mode)
    current_protocol = protocol()
    if args.mode == "full":
        if args.smoke_dir is None:
            raise ValueError("Complete same-protocol v1 smoke is required")
        smoke = read(args.smoke_dir / "paired_receipt.json")
        if (smoke.get("technical_check_passed") is not True or smoke["mode"] != "smoke"
                or smoke["protocol_sha256"] != current_protocol):
            raise ValueError("V1 full run requires its unchanged technical receipt")
    v0 = ROOT / "logs" / V0_RUNS[args.mode]
    old = read(v0 / "paired_receipt.json")
    if (Path(str(v0) + ".done").read_text().strip() != "0"
            or old.get("technical_check_passed") is not True or old["experiment"] != "E041"
            or old["git_commit"] != V0_COMMITS[args.mode] or old["mode"] != args.mode
            or old["datasets"] != names or old["protocol_sha256"] != V0_PROTOCOL
            or old["all_predictions_complete_before_scoring"] is not True
            or old["upstream_fix_commit"] != scaled.FIX_COMMIT
            or old["control_sha256"] != base.geometry.CONTROL_SHA256):
        raise ValueError("Completed fixed-scale v0 control is required")
    frozen = {**old["frozen_legacy_artifacts"], **old["frozen_candidate_artifacts"]}
    for name in ("paired_receipt.json", "source_manifest.json", "fixed/stability.json", "fixed/submission.csv", "fixed/run_manifest.json"):
        frozen[str(v0 / name)] = digest(v0 / name)
    for name in ("runtime_receipt.json", "general_v1.pt"):
        frozen[str(ASSET / name)] = digest(ASSET / name)
    if any(digest(Path(path)) != sha for path, sha in frozen.items()):
        raise ValueError("Frozen v0/control artifacts changed")
    if digest(v0 / "fixed/submission.csv") != read(v0 / "fixed/run_manifest.json")["csv_sha256"]:
        raise ValueError("Completed v0 output checksum changed")
    if digest(args.control_csv) != base.geometry.CONTROL_SHA256:
        raise ValueError("Original E029 CSV changed")
    base.flow.require_replay_parity(args.replayed_control_csv, args.control_csv)
    graphs = sorted(args.control_dir.glob("*.geff"))
    if base.flow.graph_tree_sha256(graphs) != base.CONTROL_GRAPH_SHA256:
        raise ValueError("Original E029 graphs changed")
    frame = pd.read_csv(args.control_csv)
    frame = frame.loc[frame.dataset.isin(names)].copy()
    for name in names:
        folder = scaled.LEGACY / "inference" / name
        previous = read(v0 / "inference" / name / "receipt.json")
        if (previous.get("passed") is not True or previous["frames"] != 100
                or previous["checkpoint_sha256"] != base.geometry.WEIGHT_SHA256
                or previous["upstream_fix_commit"] != scaled.FIX_COMMIT
                or previous["scale_source_sha256"] != scaled.FIXED_SHA
                or previous["fallback_used"] is not False or previous["ground_truth_read"] is not False
                or digest(folder / "input_nodes.npz") != previous["input_sha256"]):
            raise ValueError("V0 weight/scale/input contract changed")
        nodes = frame.loc[(frame.dataset == name) & (frame.row_type == "node")].sort_values("node_id")
        with np.load(folder / "input_nodes.npz", allow_pickle=False) as cached:
            if (not np.array_equal(cached["ids"], nodes.node_id.to_numpy(dtype=np.int64))
                    or not np.array_equal(cached["points"], nodes[["t", "z", "y", "x"]].to_numpy(dtype=float))):
                raise ValueError("Original E029 nodes differ from fixed input")
        (args.output_dir / "inference" / name).mkdir(parents=True)
    manifest = {"experiment": "E042", "mode": args.mode, "datasets": names,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "protocol_sha256": current_protocol, "v0_protocol_sha256": V0_PROTOCOL,
                "weight_sha256": V1_SHA, "upstream_fix_commit": scaled.FIX_COMMIT,
                "control_sha256": base.geometry.CONTROL_SHA256, "v0_cache_regenerated": False,
                "frozen_control_artifacts": frozen, "all_predictions_complete_before_scoring": False}
    save(args.output_dir / "source_manifest.json", manifest)
    stop = threading.Event()

    def shard(index):
        for name in names[index::2]:
            if stop.is_set():
                raise RuntimeError("Peer failed; no partial-corpus quality result")
            folder = args.output_dir / "inference" / name
            input_nodes = scaled.LEGACY / "inference" / name / "input_nodes.npz"
            command = ["timeout", "--foreground", "900", sys.executable, str(Path(__file__).resolve()), "infer",
                       "--input-nodes", str(input_nodes), "--input-sha256", digest(input_nodes),
                       "--image", str(args.data_dir / "train" / (name + ".zarr")), "--output-dir", str(folder)]
            try:
                with (folder / "inference.log").open("x") as handle:
                    subprocess.run(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(index)},
                                   stdout=handle, stderr=subprocess.STDOUT, check=True)
                receipt = read(folder / "receipt.json")
                if (receipt.get("passed") is not True or receipt["frames"] != 100
                        or receipt["checkpoint_sha256"] != V1_SHA or receipt["input_sha256"] != digest(input_nodes)
                        or digest(folder / "pairs.npz") != receipt["pairs_sha256"]):
                    raise ValueError("Incomplete v1 prediction")
                print(f"INFERRED {name} edges={receipt['edges']}", flush=True)
            except Exception:
                stop.set()
                raise

    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(shard, i) for i in (0, 1)]:
            future.result()
    candidate_frozen = {str(args.output_dir / "inference" / name / filename): digest(args.output_dir / "inference" / name / filename)
                        for name in names for filename in ("pairs.npz", "receipt.json")}
    manifest.update(frozen_candidate_artifacts=candidate_frozen, all_predictions_complete_before_scoring=True)
    save(args.output_dir / "source_manifest.json", manifest)
    results = {}
    for arm in ("v0", "v1"):
        current = copy.copy(args)
        current.output_dir = args.output_dir / arm
        image_dir = current.output_dir / "input/test"
        image_dir.mkdir(parents=True)
        for name in names:
            (image_dir / (name + ".zarr")).symlink_to(args.data_dir / "train" / (name + ".zarr"), target_is_directory=True)
        final_csv = current.output_dir / "submission.csv"
        counts = []
        if arm == "v0":
            shutil.copyfile(v0 / "fixed/submission.csv", final_csv)
        else:
            keep = frame.row_type.eq("node").copy()
            for name in names:
                edges = frame.loc[(frame.dataset == name) & (frame.row_type == "edge")]
                pairs = list(zip(edges.source_id.astype(int), edges.target_id.astype(int)))
                folder = args.output_dir / "inference" / name
                receipt = read(folder / "receipt.json")
                with np.load(folder / "pairs.npz", allow_pickle=False) as saved:
                    consensus = set(map(tuple, saved["pairs"].tolist()))
                mask = base.retain_edges(pairs, consensus, True, set(receipt["ambiguous_node_ids"]))
                keep.loc[edges.index] = mask
                counts.append({"dataset": name, "removed_edges": len(pairs) - sum(mask)})
            candidate = frame.loc[keep].copy()
            candidate["id"] = np.arange(len(candidate))
            raw = current.output_dir / "raw_submission.csv"
            candidate.to_csv(raw, index=False)
            export = reference.normalize_export_boundary(raw, final_csv, image_dir, names)
            save(current.output_dir / "export_boundary_audit.json", export)
        save(current.output_dir / "topology_audit.json", reference.validate_submission(final_csv, image_dir, names))
        exported = pd.read_csv(final_csv)
        save(current.output_dir / "preservation_audit.json", audit_preservation(frame, exported))
        csv_sha = digest(final_csv)
        save(current.output_dir / "run_manifest.json", {**manifest, "arm": arm, "counts": counts, "csv_sha256": csv_sha})
        current.experiment = "E042 " + arm + " scale-fixed HOCT; divisions protected"
        current.expected_control_score = base.flow.E029_SCORE
        current.minimum_pooled_delta = 0.001
        results[arm] = reference.evaluate(current, names)
        if digest(final_csv) != csv_sha:
            raise ValueError("Scored output changed")
    original = read(v0 / "fixed/stability.json")
    if (results["v0"]["graph_signatures"] != original["graph_signatures"]
            or results["v0"]["official_rows"] != original["official_rows"]
            or results["v1"]["official_rows"]["control"] != original["official_rows"]["control"]):
        raise ValueError("Cached v0 and original E029 scores/graphs did not reproduce")
    for name in names:
        for key in ("division_tp", "division_fp", "division_fn", "num_pred_nodes", "node_recall", "total_node_ratio"):
            if results["v1"]["official_rows"]["candidate"][name][key] != original["official_rows"]["control"][name][key]:
                raise ValueError("Protected node/division components changed")
    if not math.isclose(results["v0"]["groups"]["all"]["candidate"]["score"], old["groups"]["all"]["candidate"]["score"], abs_tol=1e-9, rel_tol=0):
        raise ValueError("Fixed-scale v0 pooled score changed")
    exclusion = None
    if args.mode == "full":
        included = sorted(set(names) - scaled.EXCLUDED)
        if len(included) != 59:
            raise ValueError("Author-exclusion cohort changed")
        official = reference.stability.load_official_scorer(args.runtime_dir, args.scorer_dir)
        groups = {}
        for key, subset in {"all": included, **{p: [n for n in included if n.startswith(p + "_")] for p in ("44b6", "6bba")}}.items():
            c = reference.stability.official_summary(official, results["v1"]["official_rows"]["control"], subset)
            v = reference.stability.official_summary(official, results["v1"]["official_rows"]["candidate"], subset)
            groups[key] = {"n": len(subset), "delta": reference.stability.summary_delta(c, v)}
        exclusion = {"groups": groups, "passed": all(g["delta"]["score"] > 0 for g in groups.values())}
    if (digest(args.control_csv) != base.geometry.CONTROL_SHA256
            or base.flow.graph_tree_sha256(graphs) != base.CONTROL_GRAPH_SHA256
            or digest(weight) != V1_SHA or protocol() != current_protocol
            or any(digest(Path(path)) != sha for path, sha in {**frozen, **candidate_frozen}.items())):
        raise ValueError("Frozen prediction, asset or control changed")
    gain = results["v1"]["groups"]["all"]["candidate"]["score"] - results["v0"]["groups"]["all"]["candidate"]["score"]
    receipt = {**manifest, "technical_check_passed": True, "gain_over_scaled_v0": gain,
               "groups": results["v1"]["groups"], "gates": results["v1"]["gates"], "paired": results["v1"]["paired"],
               "author_exclusion": exclusion, "formal_submission_created": False,
               "development_promotion_passed": args.mode == "full" and results["v1"]["promotion_passed"] and gain > 0 and exclusion["passed"]}
    save(args.output_dir / "paired_receipt.json", receipt)
    (args.output_dir / "run_summary.md").write_text("# E042 v1 weight-only comparison\n\n```json\n" + json.dumps(receipt, indent=2) + "\n```\n")


def main():
    if Path(__file__).resolve().parents[1] != ROOT or Path(sys.prefix).name != "Kaggle":
        raise RuntimeError("E042 executes only on Uestc-220 in conda Kaggle")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("infer")
    for key in ("input-nodes", "image", "output-dir"):
        worker.add_argument("--" + key, type=Path, required=True)
    worker.add_argument("--input-sha256", required=True)
    experiment = sub.add_parser("run")
    for key in ("data-dir", "control-dir", "control-csv", "replayed-control-csv", "runtime-dir", "scorer-dir", "output-dir"):
        experiment.add_argument("--" + key, type=Path, required=True)
    experiment.add_argument("--mode", choices=("smoke", "full"), required=True)
    experiment.add_argument("--smoke-dir", type=Path)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.command == "infer":
        infer(args)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        try:
            run(args)
        except Exception as exc:
            (args.output_dir / "run_summary.md").write_text(f"# E042 failed\n\n{type(exc).__name__}: {exc}\n\nNo promotion.\n")
            raise


if __name__ == "__main__":
    main()
