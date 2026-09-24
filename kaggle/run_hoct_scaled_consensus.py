#!/usr/bin/env python3
"""E041: upstream scale fix with frozen v0 weights and E036 edge policy."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pandas as pd

import build_kaggle_development as packaging
import run_hoct_consensus_experiment as base

ROOT = Path("/data/zqjinruoting/Kaggle/Cell Tracking During Development")
AUDIT = ROOT / "logs/s220_hoct_scale_cpu_20260924v1"
LEGACY = ROOT / "logs/s197_hoct_consensus_full_20260923v1"
FIX_COMMIT = "8709ee9d3c4d7aae1f022b259d48dc6584237b02"
FIXED_SHA = {"_api.py": "12ca5f39eb95e28012247d3c1d7b3bf3dbbe83dcebc62cad1fa67d269b0d1b20",
             "data/_transforms.py": "9b742b80919dea269955381b331163c1d19093a4fe9484901452a736a67b920f",
             "features/graph.py": "71b48f54dfc03811e40b82df6a899a07c459ff53bf2594c2272d1e33014ed4ce"}
EXCLUDED = {"44b6_12dfb391", "44b6_267148e4", "44b6_2a2eff9f", "44b6_341df25f", "6bba_09961292"}
reference = base.reference
digest = base.geometry.digest


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def protocol():
    files = packaging.source_closure(ROOT, ["kaggle/run_hoct_scaled_consensus.py", "tests/test_hoct_compatibility.py"])
    return packaging.protocol_digest(files)


def fixed_runtime():
    report = read(AUDIT / "runtime_receipt.json")
    # The independent v1 acquisition failed. Only the completed, explicitly
    # recorded v0-relevant synthetic cases are prerequisites for this experiment.
    if (report["git_commit"] != "946c718b10ec472b45862a31e135a648d626c1a5"
            or report["upstream_fix_commit"] != FIX_COMMIT
            or report["old"]["passed"] is not True or report["fixed"]["passed"] is not True
            or report["ground_truth_accessed"] is not False):
        raise ValueError("Completed scale subchecks/provenance missing")
    bundle = AUDIT / "reviewed_sources"
    for name, sha in report["upstream_files"].items():
        path = (bundle / name).resolve()
        if bundle not in path.parents or digest(path) != sha:
            raise ValueError("Reviewed dependency source changed")
    runtime = bundle / "upstream/fixed"
    for name, sha in FIXED_SHA.items():
        if digest(runtime / "hoct" / name) != sha:
            raise ValueError("Official scale-fix source differs")
    return runtime


def infer(args):
    runtime = fixed_runtime()
    sys.path.insert(0, str(runtime))
    import hoct._api as api
    if Path(api.__file__).resolve() != runtime / "hoct/_api.py":
        raise ValueError("Scale-fixed library was not imported")
    previous = read(LEGACY / "inference" / args.image.stem / "receipt.json")
    versions = {key: importlib.metadata.version(key) for key in previous["versions"]}
    if versions != previous["versions"]:
        raise ValueError("Non-source inference environment changed from completed v0 cache")
    base.infer_movie(args)
    receipt = read(args.output_dir / "receipt.json")
    receipt.update(upstream_fix_commit=FIX_COMMIT, scale_source_sha256=FIXED_SHA,
                   legacy_input_receipt_sha256=digest(LEGACY / "inference" / args.image.stem / "receipt.json"))
    save(args.output_dir / "receipt.json", receipt)


def run(args):
    fixed_runtime()
    current_protocol = protocol()
    if args.mode == "full":
        if args.smoke_dir is None:
            raise ValueError("Complete fixed-protocol smoke is required")
        smoke = read(args.smoke_dir / "paired_receipt.json")
        if smoke.get("technical_check_passed") is not True or smoke["mode"] != "smoke" or smoke["protocol_sha256"] != current_protocol:
            raise ValueError("Full run needs same-protocol complete-movie technical receipt")
    names = reference.selected_names(args.control_dir, args.mode)
    old = read(LEGACY / "run_manifest.json")
    if (Path(str(LEGACY) + ".done").read_text().strip() != "0"
            or old["git_commit"] != "d5942c809f355f4508a302a2559eaf68c79d99a9"
            or old["mode"] != "full" or len(old["datasets"]) != 64
            or reference.stability.movie_names_sha256(old["datasets"]) != reference.CORPUS_SHA256
            or old["control_sha256"] != base.geometry.CONTROL_SHA256
            or digest(args.control_csv) != base.geometry.CONTROL_SHA256):
        raise ValueError("Frozen original control/cache provenance failed")
    base.flow.require_replay_parity(args.replayed_control_csv, args.control_csv)
    graphs = sorted(args.control_dir.glob("*.geff"))
    if base.flow.graph_tree_sha256(graphs) != base.CONTROL_GRAPH_SHA256:
        raise ValueError("Frozen original control graphs changed")
    frame = pd.read_csv(args.control_csv)
    frame = frame.loc[frame.dataset.isin(names)].copy()
    frozen = {str(LEGACY / name): digest(LEGACY / name) for name in
              ("run_manifest.json", "e036/run_manifest.json", "e036/stability.json")}
    for name in names:
        folder = LEGACY / "inference" / name
        receipt = read(folder / "receipt.json")
        if (receipt.get("passed") is not True or receipt["frames"] != 100 or receipt["fallback_used"] is not False
                or receipt["ground_truth_read"] is not False or receipt["checkpoint_sha256"] != base.geometry.WEIGHT_SHA256
                or digest(folder / "input_nodes.npz") != receipt["input_sha256"]
                or digest(folder / "pairs.npz") != receipt["pairs_sha256"]):
            raise ValueError("Incomplete or modified old cache: " + name)
        nodes = frame.loc[(frame.dataset == name) & (frame.row_type == "node")].sort_values("node_id")
        with np.load(folder / "input_nodes.npz", allow_pickle=False) as cached:
            if not np.array_equal(cached["ids"], nodes.node_id.to_numpy(dtype=np.int64)) or not np.array_equal(cached["points"], nodes[["t", "z", "y", "x"]].to_numpy(dtype=float)):
                raise ValueError("Cached input differs from immutable E029 nodes")
        for filename in ("input_nodes.npz", "pairs.npz", "receipt.json"):
            frozen[str(folder / filename)] = digest(folder / filename)
        (args.output_dir / "inference" / name).mkdir(parents=True)
    manifest = {"experiment": "E041", "mode": args.mode, "datasets": names, "protocol_sha256": current_protocol,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "upstream_fix_commit": FIX_COMMIT, "fixed_source_sha256": FIXED_SHA,
                "control_sha256": base.geometry.CONTROL_SHA256, "old_cache_regenerated": False,
                "frozen_legacy_artifacts": frozen, "all_predictions_complete_before_scoring": False}
    save(args.output_dir / "source_manifest.json", manifest)
    stop = threading.Event()

    def shard(index):
        for name in names[index::2]:
            if stop.is_set():
                raise RuntimeError("Peer failed; no partial-corpus result")
            old_folder, folder = LEGACY / "inference" / name, args.output_dir / "inference" / name
            command = ["timeout", "--foreground", "900", sys.executable, str(Path(__file__).resolve()), "infer",
                       "--input-nodes", str(old_folder / "input_nodes.npz"), "--input-sha256", digest(old_folder / "input_nodes.npz"),
                       "--image", str(args.data_dir / "train" / (name + ".zarr")),
                       "--weight", str(args.data_dir / "hoct-general-v0/general_v0.pt"), "--output-dir", str(folder)]
            try:
                with (folder / "inference.log").open("x") as handle:
                    subprocess.run(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(index)},
                                   stdout=handle, stderr=subprocess.STDOUT, check=True)
                receipt = read(folder / "receipt.json")
                if receipt.get("passed") is not True or digest(folder / "pairs.npz") != receipt["pairs_sha256"] or receipt["upstream_fix_commit"] != FIX_COMMIT:
                    raise ValueError("Incomplete corrected prediction")
                print(f"INFERRED {name} edges={receipt['edges']}", flush=True)
            except Exception:
                stop.set()
                raise

    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(shard, index) for index in (0, 1)]:
            future.result()
    candidate_frozen = {str(args.output_dir / "inference" / name / filename): digest(args.output_dir / "inference" / name / filename)
                        for name in names for filename in ("pairs.npz", "receipt.json")}
    manifest["frozen_candidate_artifacts"] = candidate_frozen
    manifest["all_predictions_complete_before_scoring"] = True
    save(args.output_dir / "source_manifest.json", manifest)
    results = {}
    for arm, cache in (("legacy", LEGACY), ("fixed", args.output_dir)):
        current = copy.copy(args)
        current.output_dir = args.output_dir / arm
        image_dir = current.output_dir / "input/test"
        image_dir.mkdir(parents=True)
        keep = frame.row_type.eq("node").copy()
        counts = []
        for name in names:
            (image_dir / (name + ".zarr")).symlink_to(args.data_dir / "train" / (name + ".zarr"), target_is_directory=True)
            edges = frame.loc[(frame.dataset == name) & (frame.row_type == "edge")]
            pairs = list(zip(edges.source_id.astype(int), edges.target_id.astype(int)))
            receipt = read(cache / "inference" / name / "receipt.json")
            with np.load(cache / "inference" / name / "pairs.npz", allow_pickle=False) as saved:
                consensus = set(map(tuple, saved["pairs"].tolist()))
            mask = base.retain_edges(pairs, consensus, True, set(receipt["ambiguous_node_ids"]))
            keep.loc[edges.index] = mask
            counts.append({"dataset": name, "removed_edges": len(pairs) - sum(mask)})
        candidate = frame.loc[keep].copy()
        if not frame.loc[frame.row_type == "node"].equals(candidate.loc[candidate.row_type == "node"]):
            raise ValueError("Original nodes changed")
        candidate["id"] = np.arange(len(candidate))
        raw = current.output_dir / "raw_submission.csv"
        candidate.to_csv(raw, index=False)
        current.experiment = "E041 " + arm + " HOCT v0; division edges protected"
        current.expected_control_score = base.flow.E029_SCORE
        current.minimum_pooled_delta = 0.001
        final_csv = current.output_dir / "submission.csv"
        export = reference.normalize_export_boundary(raw, final_csv, image_dir, names)
        save(current.output_dir / "export_boundary_audit.json", export)
        topology = reference.validate_submission(final_csv, image_dir, names)
        save(current.output_dir / "topology_audit.json", topology)
        exported = pd.read_csv(final_csv)
        node_columns = [column for column in frame.columns if column != "id"]
        old_nodes = frame.loc[frame.row_type == "node", node_columns].reset_index(drop=True)
        new_nodes = exported.loc[exported.row_type == "node", node_columns].reset_index(drop=True)
        if not old_nodes.equals(new_nodes):
            raise ValueError("Export changed an original node field")
        csv_sha = digest(final_csv)
        save(current.output_dir / "run_manifest.json", {**manifest, "arm": arm, "counts": counts, "csv_sha256": csv_sha})
        results[arm] = reference.evaluate(current, names)
        if digest(final_csv) != csv_sha:
            raise ValueError("Candidate CSV changed during scoring")
    original_result = read(LEGACY / "e036/stability.json")
    official = reference.stability.load_official_scorer(args.runtime_dir, args.scorer_dir)
    for name in names:
        if results["legacy"]["graph_signatures"]["candidate"][name] != original_result["graph_signatures"]["candidate"][name]:
            raise ValueError("Reconstructed legacy graph differs from completed E036")
        if results["fixed"]["official_rows"]["control"][name] != results["legacy"]["official_rows"]["control"][name]:
            raise ValueError("Paired control rows changed")
        for key in ("division_tp", "division_fp", "division_fn", "num_pred_nodes", "node_recall", "total_node_ratio"):
            if results["fixed"]["official_rows"]["candidate"][name][key] != results["fixed"]["official_rows"]["control"][name][key]:
                raise ValueError("Protected node/division components changed")
    expected = reference.stability.official_summary(official, original_result["official_rows"]["candidate"], names)["score"]
    if not math.isclose(results["legacy"]["groups"]["all"]["candidate"]["score"], expected, abs_tol=1e-9, rel_tol=0):
        raise ValueError("Legacy cached control score changed")
    expected_control = reference.stability.official_summary(official, original_result["official_rows"]["control"], names)["score"]
    if not math.isclose(results["fixed"]["groups"]["all"]["control"]["score"], expected_control, abs_tol=1e-9, rel_tol=0):
        raise ValueError("Original E029 control score changed")
    if not math.isclose(original_result["groups"]["all"]["candidate"]["score"], 0.9100497869496292, abs_tol=1e-9, rel_tol=0):
        raise ValueError("Frozen full legacy score receipt changed")
    exclusion = None
    if args.mode == "full":
        included = sorted(set(names) - EXCLUDED)
        if len(included) != 59:
            raise ValueError("Fixed author-exclusion cohort changed")
        groups = {}
        for key, selected in {"all": included, **{p: [n for n in included if n.startswith(p + "_")] for p in ("44b6", "6bba")}}.items():
            c = reference.stability.official_summary(official, results["fixed"]["official_rows"]["control"], selected)
            v = reference.stability.official_summary(official, results["fixed"]["official_rows"]["candidate"], selected)
            groups[key] = {"n": len(selected), "delta": reference.stability.summary_delta(c, v)}
        exclusion = {"groups": groups, "passed": all(g["delta"]["score"] > 0 for g in groups.values())}
    if digest(args.control_csv) != base.geometry.CONTROL_SHA256 or base.flow.graph_tree_sha256(graphs) != base.CONTROL_GRAPH_SHA256:
        raise ValueError("Original controls changed during evaluation")
    if any(digest(Path(path)) != sha for path, sha in {**frozen, **candidate_frozen}.items()):
        raise ValueError("Frozen cached or candidate predictions changed")
    legacy_gain = results["fixed"]["groups"]["all"]["candidate"]["score"] - expected
    receipt = {**manifest, "technical_check_passed": True, "gain_over_legacy": legacy_gain,
               "groups": results["fixed"]["groups"], "gates": results["fixed"]["gates"], "paired": results["fixed"]["paired"],
               "author_exclusion": exclusion,
               "development_promotion_passed": args.mode == "full" and results["fixed"]["promotion_passed"] and legacy_gain > 0 and exclusion["passed"],
               "formal_submission_created": False}
    save(args.output_dir / "paired_receipt.json", receipt)
    (args.output_dir / "run_summary.md").write_text("# E041 scale-fixed HOCT v0\n\n```json\n" + json.dumps(receipt, indent=2) + "\n```\n")


def main():
    if Path(__file__).resolve().parents[1] != ROOT or Path(sys.prefix).name != "Kaggle":
        raise RuntimeError("E041 executes only on Uestc-220 in conda Kaggle")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("infer")
    for key in ("input-nodes", "image", "weight", "output-dir"):
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
            (args.output_dir / "run_summary.md").write_text(f"# E041 failed\n\n{type(exc).__name__}: {exc}\n\nNo promotion.\n")
            raise


if __name__ == "__main__":
    main()
