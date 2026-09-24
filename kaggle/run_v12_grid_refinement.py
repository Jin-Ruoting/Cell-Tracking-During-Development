#!/usr/bin/env python3
"""E044 paired integer-anchor refinement on frozen E029 graphs, Kaggle only."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import run_v12_association as infrastructure
import v12_grid_refinement as policy

reference = infrastructure.reference
WORK = Path("/kaggle/working/logs/e044-grid-refinement")
ARMS = ("anchor", "candidate")
infrastructure.WORK = WORK  # Process-local control/setup helper; no E040 source edits.


def array_sha(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def predict_movie(name, model, source):
    import numpy as np
    import torch
    import zarr

    started = time.monotonic()
    fields, original, ids, points, _ = infrastructure.fixed_graph(WORK / "original_movies" / (name + ".csv"))
    outputs = {arm: [r.copy() for r in original] for arm in ARMS}
    row_index = {int(r["node_id"]): i for i, r in enumerate(original) if r["row_type"] == "node"}
    group = zarr.open_group(str(WORK / "image_only" / (name + ".zarr")), mode="r")
    image, attrs = group["0"], dict(group.attrs)
    transform = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
    if (tuple(image.shape) != (100, *policy.SHAPE) or image.dtype != np.uint16
            or transform["type"] != "scale" or tuple(transform["scale"][-3:]) != policy.SCALE):
        raise ValueError("Image geometry changed")
    low, high = [float(attrs["image_statistics"]["quantiles"][k]) for k in ("0.001", "0.999")]
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("Invalid image quantiles")
    frames, totals = [], {arm: Counter() for arm in ARMS}
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for t in range(100):
            raw = policy.original_positions(points[t])
            anchors = policy.native_anchors(raw)
            proposals = {"anchor": anchors.astype(np.float64) * policy.DOWNSAMPLE}
            frame = {"frame": t, "nodes": len(raw), "original_sha256": array_sha(raw),
                     "anchor_sha256": array_sha(anchors)}
            if len(raw):
                volume = image[t, ::1, ::4, ::4].astype(np.float32)
                volume = np.clip((volume - low) / (high - low + 1e-6), 0, None).astype(np.float32)
                if not np.isfinite(volume).all():
                    raise ValueError("Nonfinite normalized image")
                frame["image_sha256"] = array_sha(volume)
                tensor = torch.from_numpy(np.ascontiguousarray(volume))[None, None].to("cuda")
                layers, feature = model.unet.make_feature(tensor)
                field = model.unet.refine_head(feature)
                coord = torch.from_numpy(anchors.astype(np.float32))[None].to("cuda")
                refined, logit = source.refine_node_peak(coord, torch.zeros((1, len(raw)), device="cuda"), field)
                if (tuple(field.shape) != (1, 4, 64, 64, 64) or refined.shape != coord.shape
                        or not torch.isfinite(field).all() or not torch.isfinite(refined).all()
                        or not torch.isfinite(logit).all()):
                    raise ValueError("Nonfinite or incomplete refinement output")
                residual_result = refined[0].float().cpu().numpy()
                proposals["candidate"] = residual_result.astype(np.float64) * policy.DOWNSAMPLE
                frame["refined_grid_sha256"] = array_sha(residual_result)
                del tensor, layers, feature, field, coord, refined, logit
            else:
                proposals["candidate"] = np.empty((0, 3), dtype=np.float64)
            frame["arms"] = {}
            for arm in ARMS:
                positions, audit = policy.safe_proposal(raw, proposals[arm], anchors)
                frame["arms"][arm] = {**audit, "output_sha256": array_sha(positions)}
                totals[arm].update({k: v for k, v in audit.items() if isinstance(v, int)})
                for node, position in zip(ids[t], positions):
                    outputs[arm][row_index[node]].update({k: str(int(v)) for k, v in zip(("z", "y", "x"), position)})
            frames.append(frame)
            if (t + 1) % 20 == 0:
                print(f"E044 FRAMES {name} {t + 1}/100", flush=True)
    receipt = {"dataset": name, "frames_considered": 100, "frames": frames,
               "predictions_used_ground_truth": False, "all_edges_ids_times_preserved": True,
               "policy": policy.POLICY, "arms": {}}
    for arm in ARMS:
        policy.validate_rewrite(original, outputs[arm], fields, totals[arm]["changed_nodes"])
        path = WORK / (arm + "_movies") / (name + ".csv")
        with path.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(outputs[arm])
        receipt["arms"][arm] = {"counts": dict(totals[arm]), "csv_sha256": infrastructure.digest(path)}
    receipt.update(seconds=time.monotonic() - started, peak_gpu_bytes=torch.cuda.max_memory_allocated())
    infrastructure.save(WORK / "movie_receipts" / (name + ".json"), receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "frames"}), flush=True)


def merge(config):
    infrastructure.unchanged(config)
    names = config["datasets"]
    if sorted(p.stem for p in (WORK / "movie_receipts").glob("*.json")) != names:
        raise ValueError("Incomplete movie coverage")
    original_topology = json.loads((WORK / "control_topology.json").read_text())
    for arm in ARMS:
        merged, totals, fields = [], Counter(), None
        for name in names:
            receipt = json.loads((WORK / "movie_receipts" / (name + ".json")).read_text())
            path = WORK / (arm + "_movies") / (name + ".csv")
            if (receipt["frames_considered"] != 100 or [f["frame"] for f in receipt["frames"]] != list(range(100))
                    or receipt["predictions_used_ground_truth"] is not False
                    or receipt["all_edges_ids_times_preserved"] is not True or receipt["policy"] != policy.POLICY
                    or receipt["arms"][arm]["csv_sha256"] != infrastructure.digest(path)):
                raise ValueError("Prediction receipt mismatch")
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle); fields = reader.fieldnames; rows = list(reader)
            with (WORK / "original_movies" / (name + ".csv")).open(newline="") as handle:
                original = list(csv.DictReader(handle))
            policy.validate_rewrite(original, rows, fields, receipt["arms"][arm]["counts"]["changed_nodes"])
            totals.update(receipt["arms"][arm]["counts"]); merged.extend(rows)
        merged.sort(key=lambda r: int(r["id"]))
        target = WORK / arm / "submission.csv"
        with target.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(merged)
        audit = reference.validate_submission(target, WORK / "image_only", names)
        if any(audit[k] != original_topology[k] for k in ("rows", "datasets")):
            raise ValueError("Graph topology changed")
        infrastructure.save(WORK / arm / "topology_audit.json", audit)
        infrastructure.save(WORK / arm / "run_manifest.json", {
            "all_candidate_predictions_completed_before_scoring": True, "policy": policy.POLICY,
            "counts": dict(totals), "csv_sha256": infrastructure.digest(target)})
    infrastructure.unchanged(config)


def score(bundle, config):
    infrastructure.unchanged(config)
    manifests = {arm: json.loads((WORK / arm / "run_manifest.json").read_text()) for arm in ARMS}
    for arm, manifest in manifests.items():
        if (manifest["all_candidate_predictions_completed_before_scoring"] is not True
                or manifest["csv_sha256"] != infrastructure.digest(WORK / arm / "submission.csv")):
            raise ValueError("Both outputs must be frozen before scoring")
    scorer = bundle / "official"
    os.environ["PYTHONPATH"] = os.pathsep.join([str(bundle / "kaggle"), str(scorer / "src"), str(scorer / "scripts")])
    control_dir = WORK / "control_geffs"
    subprocess.run([sys.executable, str(scorer / "scripts/csv_to_geffs.py"), "--csv", str(WORK / "control.csv"),
                    "--out-dir", str(control_dir), "--no-overwrite"], check=True)
    data = Path(json.loads((WORK / "runtime.json").read_text())["data_dir"])
    reports = {}
    for arm in ARMS:
        args = SimpleNamespace(runtime_dir=bundle / "kaggle", scorer_dir=scorer, control_dir=control_dir,
                               data_dir=data, output_dir=WORK / arm, minimum_pooled_delta=0.001,
                               expected_control_score=config["control"]["score"], experiment="E044 " + arm)
        reports[arm] = reference.evaluate(args, config["datasets"])
    result = reports["candidate"]
    if reports["anchor"]["official_rows"]["control"] != result["official_rows"]["control"]:
        raise ValueError("Repeated original official scores differ")
    official = reference.stability.load_official_scorer(bundle / "kaggle", scorer)
    anchor_control_score = reference.stability.official_summary(official, result["official_rows"]["control"], list(reference.SMOKE_NAMES))["score"]
    if (not math.isclose(anchor_control_score, 0.9638391788805566, abs_tol=1e-9, rel_tol=0)
            or not math.isclose(result["groups"]["all"]["control"]["score"], config["control"]["score"], abs_tol=1e-9, rel_tol=0)):
        raise ValueError("Frozen original control did not reproduce")
    gain_over_anchor = result["groups"]["all"]["candidate"]["score"] - reports["anchor"]["groups"]["all"]["candidate"]["score"]
    delta = result["groups"]["all"]["delta"]
    extra = {"node_recall_not_regressed": delta["node_recall"] >= 0,
             "division_jaccard_not_regressed": delta["division_jaccard"] >= 0,
             "gain_over_anchor_only": gain_over_anchor > 0,
             "actual_coordinate_changes": manifests["candidate"]["counts"]["changed_nodes"] > 0}
    exclusion = None
    if config["mode"] == "full":
        names = sorted(set(config["datasets"]) - infrastructure.EXCLUDED)
        if len(names) != 58:raise ValueError("Exclusion cohort changed")
        groups = {}
        for key, selected in {"all": names, **{p: [n for n in names if n.startswith(p + "_")] for p in ("44b6", "6bba")}}.items():
            old = reference.stability.official_summary(official, result["official_rows"]["control"], selected)
            new = reference.stability.official_summary(official, result["official_rows"]["candidate"], selected)
            groups[key] = {"n": len(selected), "control": old, "candidate": new, "delta": reference.stability.summary_delta(old, new)}
        exclusion = {"groups": groups, "excluded": sorted(infrastructure.EXCLUDED),
                     "passed": all(g["delta"]["score"] > 0 for g in groups.values())}
    infrastructure.unchanged(config)
    for arm in ARMS:
        if manifests[arm]["csv_sha256"] != infrastructure.digest(WORK / arm / "submission.csv"):
            raise ValueError("Output changed during scoring")
    receipt = {"experiment": "E044", "mode": config["mode"], "git_commit": config["git_commit"],
               "protocol_sha256": config["protocol_sha256"], "technical_check_passed": True,
               "control": config["control"], "policy": policy.POLICY, "all_edges_ids_times_preserved": True,
               "groups": result["groups"], "anchor_only_groups": reports["anchor"]["groups"],
               "gain_over_anchor_only": gain_over_anchor, "gates": result["gates"], "extra_gates": extra,
               "paired": result["paired"], "author_exclusion": exclusion, "output_manifests": manifests,
               "development_promotion_passed": config["mode"] == "full" and result["promotion_passed"]
                   and all(extra.values()) and exclusion["passed"], "formal_submission_created": False,
               "public_score": None, "evidence": "Paired development; training overlap unknown"}
    infrastructure.save(WORK / "paired_development_receipt.json", receipt)
    (WORK / "run_summary.md").write_text("# E044 grid refinement\n\n```json\n" + json.dumps(receipt, indent=2) + "\n```\n")
    print(json.dumps(receipt, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--phase", choices=("setup", "predict", "merge", "score"), required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1))
    args = parser.parse_args()
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Image inference/scoring execute only on Kaggle")
    bundle = args.bundle.resolve(); config = json.loads((bundle / "bundle_manifest.json").read_text())
    for name, digest in config["files"].items():
        path = (bundle / name).resolve()
        if bundle not in path.parents or infrastructure.digest(path) != digest:raise ValueError("Bundled source changed")
    if config["experiment"] != "E044" or config["config"] != policy.POLICY:raise ValueError("Frozen policy changed")
    try:
        if args.phase == "setup":
            infrastructure.setup(bundle, config)
            for name in ("anchor_movies", "anchor"):(WORK / name).mkdir()
        elif args.phase == "predict":
            if args.shard is None:raise ValueError("Prediction requires explicit shard")
            infrastructure.predict_movie = predict_movie
            infrastructure.predict(config, args.shard)
        elif args.phase == "merge":merge(config)
        else:score(bundle, config)
    except Exception as exc:
        if WORK.is_dir():
            (WORK / "run_summary.md").write_text(f"# E044 failed\n\n{args.phase}: {type(exc).__name__}: {exc}\n\nNo promotion.\n")
        raise


if __name__ == "__main__":
    main()
