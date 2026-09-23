#!/usr/bin/env python3
"""Describe matched E029 coordinate residuals without editing predictions.

Residuals are conditional on the official 7-um node assignment. They diagnose
the existing development predictions and are not a counterfactual score or a
held-out estimate of a coordinate-regression model.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import audit_e000_error_budget as graphs
import run_flow_relink_experiment as flow


def summary(values):
    import numpy as np
    matrix = np.concatenate(values, axis=0)
    distance = np.linalg.norm(matrix, axis=1)
    return {
        "matched_nodes": len(matrix),
        "gt_minus_prediction_mean_zyx_um": matrix.mean(axis=0).tolist(),
        "gt_minus_prediction_median_zyx_um": np.median(matrix, axis=0).tolist(),
        "distance_quantiles_um": dict(zip(("p50", "p90", "p95"), np.quantile(distance, (.5, .9, .95)).tolist())),
        "fraction_within_2um": float((distance <= 2).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "data-dir", "runtime-dir", "scorer-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if flow.reference.stability.file_sha256(args.run_dir / "submission.csv") != flow.E029_CSV_SHA256:
        raise ValueError("Frozen E029 prediction CSV changed")
    names = flow.reference.selected_names(args.run_dir / "geffs", "full")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                       "OPENBLAS_NUM_THREADS": "4", "POLARS_MAX_THREADS": "4"})
    graphs.prepare_imports(args.runtime_dir, args.scorer_dir)
    import numpy as np
    import tracksdata as td
    from tracking_cellmot.io import open_dataset

    residuals = {}
    results = {}
    keys = td.DEFAULT_ATTR_KEYS
    for index, name in enumerate(names, 1):
        pred = graphs.load_graph(args.run_dir / "geffs" / f"{name}.geff", td)
        gt = graphs.load_graph(args.data_dir / "train" / f"{name}.geff", td)
        scale = np.asarray(open_dataset(args.data_dir / "train" / name, load_image=False).scale, dtype=float)
        graphs.match_graph(pred, gt, tuple(scale), 7.0)
        truth = {int(row[keys.NODE_ID]): row for row in gt.node_attrs().iter_rows(named=True)}
        differences = []
        for row in pred.node_attrs().iter_rows(named=True):
            matched = row.get(keys.MATCHED_NODE_ID)
            if matched is None or int(matched) < 0:
                continue
            other = truth[int(matched)]
            if int(row["t"]) != int(other["t"]):
                raise ValueError("Official node match crossed frames")
            difference = np.array([float(other[axis]) - float(row[axis]) for axis in ("z", "y", "x")]) * scale
            if not np.isfinite(difference).all() or np.linalg.norm(difference) > 7.00001:
                raise ValueError("Matched coordinate residual violated the physical matching contract")
            differences.append(difference)
        if not differences:
            raise ValueError(f"No matched nodes: {name}")
        residuals[name] = np.stack(differences)
        results[name] = summary([residuals[name]])
        print(f"[{index:02}/{len(names)}] {name}: {len(differences)} matched nodes", flush=True)
    groups = {"all": summary(list(residuals.values()))}
    for embryo in ("44b6", "6bba"):
        groups[embryo] = summary([value for name, value in residuals.items() if name.startswith(embryo + "_")])
    report = {"experiment": "E029 coordinate-residual diagnostic", "groups": groups,
              "datasets": results, "control_csv_sha256": flow.E029_CSV_SHA256,
              "prediction_graph_tree_sha256": flow.graph_tree_sha256([args.run_dir / "geffs" / f"{name}.geff" for name in names]),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "evidence_boundary": "Residuals conditional on official matching of existing public-training predictions; no independent model-validation claim."}
    (args.output_dir / "residuals.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# E029 coordinate-residual diagnostic", "",
             "Distances are measured in micrometers using actual dataset scale.", "",
             "| Group | Matched nodes | Median distance | P90 distance | Within 2 um |", "|---|---:|---:|---:|---:|"]
    for group, value in groups.items():
        quantiles = value["distance_quantiles_um"]
        lines.append(f"| {group} | {value['matched_nodes']} | {quantiles['p50']:.4f} | {quantiles['p90']:.4f} | {value['fraction_within_2um']:.2%} |")
    lines += ["", "## Signed residuals (ground truth minus prediction)", "", "```json", json.dumps(groups, indent=2), "```", "", report["evidence_boundary"], ""]
    (args.output_dir / "run_summary.md").write_text("\n".join(lines))
    print(json.dumps(groups, indent=2), flush=True)


if __name__ == "__main__":
    main()
