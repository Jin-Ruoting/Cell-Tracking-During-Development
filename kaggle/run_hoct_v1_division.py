#!/usr/bin/env python3
"""E043 CPU-only division veto from complete, immutable E042 predictions."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

import run_hoct_v1_consensus as v1

ROOT = v1.ROOT
base, reference, scaled = v1.base, v1.reference, v1.scaled
read, save, digest = v1.read, v1.save, v1.digest
E042_PROTOCOL = "c08b20c469a683f40b28dc68116a3b8dc55d7dfc3df3646addd44461da133bdf"


def protocol():
    files = scaled.packaging.source_closure(ROOT, ["kaggle/run_hoct_v1_division.py", "tests/test_hoct_v1_division.py"])
    return scaled.packaging.protocol_digest(files)


def audit_intersection(original, exported, consensus, ambiguous):
    columns = [name for name in original.columns if name != "id"]
    old_nodes = original.loc[original.row_type == "node", columns].reset_index(drop=True)
    new_nodes = exported.loc[exported.row_type == "node", columns].reset_index(drop=True)
    if not old_nodes.equals(new_nodes) or set(original.dataset.unique()) != set(exported.dataset.unique()):
        raise ValueError("Original nodes, fields or dataset coverage changed")
    records = []
    for name in sorted(original.dataset.unique()):
        edge = original.loc[(original.dataset == name) & (original.row_type == "edge")]
        before = set(zip(edge.source_id.astype(int), edge.target_id.astype(int)))
        edge = exported.loc[(exported.dataset == name) & (exported.row_type == "edge")]
        after = set(zip(edge.source_id.astype(int), edge.target_id.astype(int)))
        expected = {pair for pair in before if pair in consensus[name]
                    or pair[0] in ambiguous[name] or pair[1] in ambiguous[name]}
        if len(after) != len(edge) or after != expected:
            raise ValueError("Export differs from the complete prescribed edge intersection")
        degree = Counter(a for a, _ in before)
        removed = before - after
        division_edges = {pair for pair in removed if degree[pair[0]] == 2}
        records.append({"dataset": name, "nodes": len(old_nodes.loc[old_nodes.dataset == name]),
                        "removed_edges": len(removed), "removed_division_edges": len(division_edges),
                        "removed_ordinary_edges": len(removed - division_edges),
                        "affected_division_parents": len({a for a, _ in division_edges})})
    return {"passed": True, "original_node_fields_unchanged": True,
            "exact_frozen_v1_intersection": True, "coincident_identities_protected": True,
            "movies": records}


def run(args):
    names = reference.selected_names(args.control_dir, args.mode)
    current_protocol = protocol()
    if args.mode == "full":
        if args.smoke_dir is None:
            raise ValueError("Same-protocol E043 technical run is required")
        smoke = read(args.smoke_dir / "paired_receipt.json")
        if (smoke.get("technical_check_passed") is not True or smoke["mode"] != "smoke"
                or smoke["protocol_sha256"] != current_protocol):
            raise ValueError("E043 technical protocol changed")
    previous = read(args.e042_dir / "paired_receipt.json")
    if (Path(str(args.e042_dir) + ".done").read_text().strip() != "0"
            or previous.get("technical_check_passed") is not True or previous["experiment"] != "E042"
            or previous["mode"] != args.mode or previous["datasets"] != names
            or previous["protocol_sha256"] != E042_PROTOCOL or previous["weight_sha256"] != v1.V1_SHA
            or previous["all_predictions_complete_before_scoring"] is not True
            or previous["control_sha256"] != base.geometry.CONTROL_SHA256):
        raise ValueError("Complete frozen E042 input is required")
    frozen = {**previous["frozen_control_artifacts"], **previous["frozen_candidate_artifacts"]}
    for name in ("paired_receipt.json", "v1/stability.json", "v1/submission.csv", "v1/run_manifest.json"):
        frozen[str(args.e042_dir / name)] = digest(args.e042_dir / name)
    if any(digest(Path(path)) != sha for path, sha in frozen.items()):
        raise ValueError("Completed E042 artifact changed")
    if digest(args.e042_dir / "v1/submission.csv") != read(args.e042_dir / "v1/run_manifest.json")["csv_sha256"]:
        raise ValueError("Protected E042 CSV changed")
    if digest(args.control_csv) != base.geometry.CONTROL_SHA256:
        raise ValueError("Original E029 CSV changed")
    base.flow.require_replay_parity(args.replayed_control_csv, args.control_csv)
    control_graphs = sorted(args.control_dir.glob("*.geff"))
    if base.flow.graph_tree_sha256(control_graphs) != base.CONTROL_GRAPH_SHA256:
        raise ValueError("Original E029 GEFF bytes changed")
    frame = pd.read_csv(args.control_csv)
    frame = frame.loc[frame.dataset.isin(names)].copy()
    keep = frame.row_type.eq("node").copy()
    consensus, ambiguous = {}, {}
    for name in names:
        folder = args.e042_dir / "inference" / name
        receipt = read(folder / "receipt.json")
        if (receipt.get("passed") is not True or receipt["frames"] != 100
                or receipt["checkpoint_sha256"] != v1.V1_SHA or receipt["fallback_used"] is not False
                or receipt["ground_truth_read"] is not False or receipt["upstream_fix_commit"] != scaled.FIX_COMMIT
                or receipt["scale_source_sha256"] != scaled.FIXED_SHA
                or digest(folder / "pairs.npz") != receipt["pairs_sha256"]):
            raise ValueError("V1 prediction contract changed")
        nodes = frame.loc[(frame.dataset == name) & (frame.row_type == "node")]
        ambiguous[name] = set(nodes.loc[nodes.duplicated(["t", "z", "y", "x"], keep=False), "node_id"].astype(int))
        if sorted(ambiguous[name]) != receipt["ambiguous_node_ids"]:
            raise ValueError("V1 coincident identity receipt changed")
        with np.load(folder / "pairs.npz", allow_pickle=False) as saved:
            consensus[name] = set(map(tuple, saved["pairs"].tolist()))
        edges = frame.loc[(frame.dataset == name) & (frame.row_type == "edge")]
        pairs = list(zip(edges.source_id.astype(int), edges.target_id.astype(int)))
        keep.loc[edges.index] = base.retain_edges(pairs, consensus[name], False, ambiguous[name])
    candidate = frame.loc[keep].copy()
    candidate["id"] = np.arange(len(candidate))
    image_dir = args.output_dir / "input/test"
    image_dir.mkdir(parents=True)
    for name in names:
        (image_dir / (name + ".zarr")).symlink_to(args.data_dir / "train" / (name + ".zarr"), target_is_directory=True)
    raw = args.output_dir / "raw_submission.csv"
    candidate.to_csv(raw, index=False)
    final_csv = args.output_dir / "submission.csv"
    save(args.output_dir / "export_boundary_audit.json", reference.normalize_export_boundary(raw, final_csv, image_dir, names))
    save(args.output_dir / "topology_audit.json", reference.validate_submission(final_csv, image_dir, names))
    audit = audit_intersection(frame, pd.read_csv(final_csv), consensus, ambiguous)
    save(args.output_dir / "intersection_audit.json", audit)
    csv_sha = digest(final_csv)
    manifest = {"experiment": "E043", "mode": args.mode, "datasets": names,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "protocol_sha256": current_protocol, "e042_protocol_sha256": E042_PROTOCOL,
                "weight_sha256": v1.V1_SHA, "control_sha256": base.geometry.CONTROL_SHA256,
                "source_e042": str(args.e042_dir), "all_predictions_complete_before_scoring": True,
                "model_inference_executed": False, "frozen_dependencies": frozen, "csv_sha256": csv_sha,
                "per_movie_counts": audit["movies"]}
    save(args.output_dir / "source_manifest.json", manifest)
    current = copy.copy(args)
    current.experiment = "E043 v1 intersection with division veto"
    current.expected_control_score = base.flow.E029_SCORE
    current.minimum_pooled_delta = 0.001
    result = reference.evaluate(current, names)
    original = read(args.e042_dir / "v1/stability.json")
    if result["official_rows"]["control"] != original["official_rows"]["control"]:
        raise ValueError("Original E029 control score changed")
    official = reference.stability.load_official_scorer(args.runtime_dir, args.scorer_dir)
    protected_rows, protected_signatures = {}, {}
    for index, name in enumerate(names):
        protected_rows[name], protected_signatures[name] = reference.stability.score_one(
            official, name, args.e042_dir / "v1/geffs" / (name + ".geff"), args.data_dir / "train")
        for key in ("num_pred_nodes", "node_recall", "total_node_ratio"):
            if result["official_rows"]["candidate"][name][key] != original["official_rows"]["control"][name][key]:
                raise ValueError("Preserved-node official components changed")
        print(f"PROTECTED CONTROL {index + 1}/{len(names)} {name}", flush=True)
    if (protected_rows != original["official_rows"]["candidate"]
            or protected_signatures != original["graph_signatures"]["candidate"]):
        raise ValueError("Protected E042 graph/score no longer reproduces")
    protected = reference.stability.official_summary(official, protected_rows, names)
    comparison = reference.stability.summary_delta(protected, result["groups"]["all"]["candidate"])
    exclusion = None
    if args.mode == "full":
        included = sorted(set(names) - scaled.EXCLUDED)
        if len(included) != 59:
            raise ValueError("Author-exclusion cohort changed")
        groups = {}
        for key, subset in {"all": included, **{p: [n for n in included if n.startswith(p + "_")] for p in ("44b6", "6bba")}}.items():
            c = reference.stability.official_summary(official, result["official_rows"]["control"], subset)
            v = reference.stability.official_summary(official, result["official_rows"]["candidate"], subset)
            groups[key] = {"n": len(subset), "delta": reference.stability.summary_delta(c, v)}
        exclusion = {"groups": groups, "passed": all(g["delta"]["score"] > 0 for g in groups.values())}
    extra = {"division_jaccard_improved_over_e029": result["groups"]["all"]["delta"]["division_jaccard"] > 0,
             "division_jaccard_improved_over_e042": comparison["division_jaccard"] > 0,
             "total_score_improved_over_e042": comparison["score"] > 0}
    if (digest(final_csv) != csv_sha or digest(args.control_csv) != base.geometry.CONTROL_SHA256
            or base.flow.graph_tree_sha256(control_graphs) != base.CONTROL_GRAPH_SHA256
            or protocol() != current_protocol
            or any(digest(Path(path)) != sha for path, sha in frozen.items())):
        raise ValueError("Frozen artifacts changed during evaluation")
    receipt = {**manifest, "technical_check_passed": True, "groups": result["groups"],
               "gates": result["gates"], "division_gates": extra, "paired": result["paired"],
               "protected_e042_score": protected, "delta_over_protected_e042": comparison,
               "author_exclusion": exclusion, "formal_submission_created": False,
               "development_promotion_passed": args.mode == "full" and result["promotion_passed"]
               and all(extra.values()) and exclusion["passed"]}
    save(args.output_dir / "paired_receipt.json", receipt)
    (args.output_dir / "run_summary.md").write_text("# E043 independent division veto\n\n```json\n" + json.dumps(receipt, indent=2) + "\n```\n")


def main():
    if Path(__file__).resolve().parents[1] != ROOT or Path(sys.prefix).name != "Kaggle":
        raise RuntimeError("E043 executes only on Uestc-220 conda Kaggle")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data-dir", "control-dir", "control-csv", "replayed-control-csv", "runtime-dir", "scorer-dir", "output-dir", "e042-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--smoke-dir", type=Path)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except Exception as exc:
        (args.output_dir / "run_summary.md").write_text(f"# E043 failed\n\n{type(exc).__name__}: {exc}\n\nNo promotion.\n")
        raise


if __name__ == "__main__":
    main()
