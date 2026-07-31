#!/usr/bin/env python3
"""Evaluate a paired Biohub candidate with the pinned official scorer.

The script fails closed on provenance, movie coverage, E025 topology reports,
and the preregistered two-embryo/alternating-half stability partitions. It
uses the scorer's structured per-sample API rather than rounded stdout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from types import ModuleType


EXPECTED_EVALUATOR_SHA256 = (
    "03ad4049530d3682c77435194e5d921981f331df3462abcde4a7156d1a57b7d3"
)
EXPECTED_E025_NOTEBOOK_SHA256 = (
    "9bee9329ca19d03db77139255d7f4d2d38394628b5c7da82073ee34815952a2f"
)
EXPECTED_DEEPCENTER_SHA256 = (
    "8164d1ffa07f87e0506027a0392edeab7939a32bd5e3f756377c0d72885cf127"
)
EXPECTED_REFERENCE_NOTEBOOK_SHA256 = (
    "70e0c300ceae3cd7ee2cf1650c4a5f74463543e3aae1b486ba5f729a76281656"
)
EXPECTED_SUPPORT_PREDICTOR_SHA256 = (
    "c44e771ba5980b820f93091e03a303c25dfe8f3232e501f54dc9565731c234b9"
)
EXPECTED_PRIMARY_WEIGHT_SHA256 = (
    "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
)
EXPECTED_SECONDARY_WEIGHT_SHA256 = (
    "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"
)
SUMMARY_METRICS = (
    "score",
    "adj_edge_jaccard",
    "edge_jaccard",
    "division_jaccard",
    "node_recall",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument(
        "--control-validation-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--candidate-validation-report",
        type=Path,
        required=True,
    )
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--scorer-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=64)
    parser.add_argument(
        "--expected-names-sha256",
        required=True,
    )
    parser.add_argument("--minimum-pooled-delta", type=float, default=0.002)
    parser.add_argument("--tie-epsilon", type=float, default=1e-12)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--fail-on-rejection", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def movie_paths(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    paths = sorted(directory.glob("*.geff"))
    result = {path.stem: path for path in paths}
    if len(result) != len(paths):
        raise RuntimeError(f"Duplicate GEFF stems under {directory}")
    if not result:
        raise FileNotFoundError(f"No GEFF predictions under {directory}")
    return result


def movie_names_sha256(names: list[str]) -> str:
    payload = "".join(f"{name}.geff\n" for name in sorted(names)).encode()
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def require_close(
    actual: object,
    expected: float,
    label: str,
    tolerance: float = 1e-12,
) -> None:
    value = float(actual)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError(
            f"{label} mismatch: expected {expected}, got {value}"
        )


def validate_inference_manifest(
    manifest: dict[str, object],
    names: list[str],
) -> tuple[Path, Path]:
    expected_scalars = {
        "reference_notebook_sha256": EXPECTED_REFERENCE_NOTEBOOK_SHA256,
        "support_predictor_sha256": EXPECTED_SUPPORT_PREDICTOR_SHA256,
        "primary_weight_sha256": EXPECTED_PRIMARY_WEIGHT_SHA256,
        "secondary_weight_sha256": EXPECTED_SECONDARY_WEIGHT_SHA256,
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"Inference manifest {key} mismatch: "
                f"expected {expected}, got {manifest.get(key)}"
            )
    if manifest.get("datasets") != sorted(names):
        raise RuntimeError("Inference manifest dataset order changed")
    require_close(manifest.get("det_threshold"), 0.96875, "det_threshold")
    require_close(
        manifest.get("ilp_disappearance_weight"),
        1.5,
        "ilp_disappearance_weight",
    )

    blend = manifest.get("blend")
    if not isinstance(blend, dict):
        raise RuntimeError("Inference manifest has no blend configuration")
    require_close(blend.get("edge_weight"), 0.15, "blend.edge_weight")
    require_close(
        blend.get("detection_weight"),
        0.475,
        "blend.detection_weight",
    )
    require_close(
        blend.get("low_margin_max"),
        0.35,
        "blend.low_margin_max",
    )
    require_close(
        blend.get("edge_threshold"),
        0.48,
        "blend.edge_threshold",
    )
    require_close(
        blend.get("mix_temperature"),
        1.0,
        "blend.mix_temperature",
    )
    if blend.get("link_mode") != "low_margin_consensus":
        raise RuntimeError("Inference manifest link mode changed")

    edge_tta = manifest.get("edge_tta")
    if not isinstance(edge_tta, dict) or edge_tta.get("mode") != "original":
        raise RuntimeError("E026 requires original edge-TTA mode")

    guard = manifest.get("retention_guard")
    if not isinstance(guard, dict):
        raise RuntimeError("Inference manifest has no retention guard")
    required_guard = {
        "implementation_version": "dual_seed_frame_retention_guard_v1",
        "activation_variant": "blend_guard",
        "fallback_scope": "individual_frame",
        "reference_field": "primary_detection_logits",
        "candidate_extractor": "_detect_cells_pooled",
        "label_free": True,
    }
    for key, expected in required_guard.items():
        if guard.get(key) != expected:
            raise RuntimeError(
                f"Retention guard {key} mismatch: "
                f"expected {expected}, got {guard.get(key)}"
            )
    require_close(
        guard.get("minimum_candidate_retention"),
        0.90,
        "retention minimum",
    )
    ab_check = guard.get("ab_check")
    if not isinstance(ab_check, dict):
        raise RuntimeError("Inference manifest has no retention A/B check")
    required_ab = {
        "candidate_counts_identical": True,
        "control_fallback_frames": 0,
        "guard_activated": True,
    }
    for key, expected in required_ab.items():
        if ab_check.get(key) != expected:
            raise RuntimeError(
                f"Retention A/B check {key} mismatch: "
                f"expected {expected}, got {ab_check.get(key)}"
            )
    if int(ab_check.get("candidate_fallback_frames", 0)) <= 0:
        raise RuntimeError("Retention guard did not activate")
    paired_frames = int(ab_check.get("paired_frames", 0))
    if paired_frames <= 0:
        raise RuntimeError("Retention A/B check has no paired frames")

    results = manifest.get("results")
    if not isinstance(results, dict):
        raise RuntimeError("Inference manifest has no results")
    control = results.get("blend")
    candidate = results.get("blend_guard")
    if not isinstance(control, dict) or not isinstance(candidate, dict):
        raise RuntimeError("Inference manifest is missing an A/B arm")
    control_retention = control.get("retention_guard")
    candidate_retention = candidate.get("retention_guard")
    if (
        not isinstance(control_retention, dict)
        or not isinstance(candidate_retention, dict)
        or int(control_retention.get("evaluated_frames", 0)) != paired_frames
        or int(candidate_retention.get("evaluated_frames", 0)) != paired_frames
    ):
        raise RuntimeError("Retention A/B paired-frame coverage changed")
    control_predictor = control.get("patched_predictor_sha256")
    candidate_predictor = candidate.get("patched_predictor_sha256")
    if (
        not isinstance(control_predictor, str)
        or not control_predictor
        or candidate_predictor != control_predictor
    ):
        raise RuntimeError("Retention A/B patched predictor SHA256 changed")
    return Path(str(control["output_dir"])), Path(
        str(candidate["output_dir"])
    )


def validate_e025_report(
    report: dict[str, object],
    expected_baseline_dir: Path,
    expected_geff_dir: Path,
    names: list[str],
    label: str,
) -> None:
    if report.get("validation_mode") != "e025_exact":
        raise RuntimeError(f"{label}: validation mode is not e025_exact")
    if report.get("notebook_sha256") != EXPECTED_E025_NOTEBOOK_SHA256:
        raise RuntimeError(f"{label}: E025 notebook pin changed")
    if (
        report.get("deepcenter_checkpoint_sha256")
        != EXPECTED_DEEPCENTER_SHA256
    ):
        raise RuntimeError(f"{label}: DeepCenter pin changed")
    if Path(str(report.get("baseline_dir"))).resolve() != (
        expected_baseline_dir.resolve()
    ):
        raise RuntimeError(f"{label}: raw baseline directory changed")
    if report.get("datasets") != sorted(names):
        raise RuntimeError(f"{label}: validation dataset order changed")
    if int(report.get("max_outdegree", -1)) != 2:
        raise RuntimeError(f"{label}: strict maximum outdegree changed")

    output_dir = Path(str(report.get("output_dir"))).resolve()
    if output_dir / "score_e025_exact" / "geffs" != (
        expected_geff_dir.resolve()
    ):
        raise RuntimeError(f"{label}: scored GEFF directory mismatch")

    variants = report.get("variants")
    if not isinstance(variants, dict):
        raise RuntimeError(f"{label}: validation variants missing")
    per_movie = variants.get("e025_exact")
    if not isinstance(per_movie, dict) or set(per_movie) != set(names):
        raise RuntimeError(f"{label}: per-movie topology coverage mismatch")
    for dataset, audit in per_movie.items():
        if not isinstance(audit, dict):
            raise TypeError(f"{label}: invalid audit for {dataset}")
        if (
            int(audit.get("max_indegree", -1)) > 1
            or int(audit.get("max_outdegree", -1)) > 2
            or int(audit.get("nonbinary_sources", -1)) != 0
        ):
            raise RuntimeError(f"{label}: topology gate failed for {dataset}")


def load_official_scorer(
    runtime_dir: Path,
    scorer_dir: Path,
) -> ModuleType:
    evaluator_path = scorer_dir / "scripts" / "evaluate.py"
    if file_sha256(evaluator_path) != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError("Pinned official evaluator SHA256 changed")
    sys.path[:0] = [
        str(runtime_dir.resolve()),
        str((scorer_dir / "src").resolve()),
        str((scorer_dir / "scripts").resolve()),
    ]
    spec = importlib.util.spec_from_file_location(
        "biohub_official_evaluate",
        evaluator_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load official evaluator {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def graph_signature(graph) -> str:
    nodes = sorted(
        (
            int(row["node_id"]),
            int(row["t"]),
            float(row["z"]),
            float(row["y"]),
            float(row["x"]),
        )
        for row in graph.node_attrs().iter_rows(named=True)
    )
    edges = sorted(
        (int(row["source_id"]), int(row["target_id"]))
        for row in graph.edge_attrs().iter_rows(named=True)
    )
    payload = json.dumps(
        {"nodes": nodes, "edges": edges},
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def score_one(
    official: ModuleType,
    name: str,
    pred_path: Path,
    gt_dir: Path,
) -> tuple[dict[str, object], str]:
    gt_path = gt_dir / f"{name}.geff"
    pred = official._load_graph(pred_path)
    signature = graph_signature(pred)
    gt = official._load_graph(gt_path)
    scale = official._read_scale(gt_dir, name)
    evaluation = official.compute_metric(
        pred,
        gt,
        scale=scale,
        max_distance=7.0,
    )
    recall = (
        official.node_recall(pred, gt)
        if pred.num_edges() > 0 and pred.num_nodes() > 0
        else 0.0
    )
    n_total = official._read_estimated_n_total(gt_path)
    row = official.per_sample_metrics(evaluation, n_total, recall)
    return to_builtin(row), signature


def official_summary(
    official: ModuleType,
    rows: dict[str, dict[str, object]],
    names: list[str],
) -> dict[str, object]:
    summary = to_builtin(official.summarise([rows[name] for name in names]))
    if int(summary["n"]) != len(names) or int(summary["n_adj"]) != len(
        names
    ):
        raise RuntimeError(
            "Official summary skipped a dataset: "
            f"expected {len(names)}, got n={summary['n']}, "
            f"n_adj={summary['n_adj']}"
        )
    return summary


def build_partitions(names: list[str]) -> dict[str, list[str]]:
    embryos = {
        prefix: sorted(name for name in names if name.startswith(prefix))
        for prefix in ("44b6_", "6bba_")
    }
    covered = set().union(*(set(group) for group in embryos.values()))
    if covered != set(names):
        raise RuntimeError(
            f"Unexpected embryo IDs: {sorted(set(names) - covered)}"
        )
    if len(embryos["44b6_"]) != len(embryos["6bba_"]):
        raise RuntimeError("E026 embryo groups are not balanced")
    half0 = []
    half1 = []
    for group in embryos.values():
        half0.extend(group[0::2])
        half1.extend(group[1::2])
    return {
        "all": sorted(names),
        "44b6": embryos["44b6_"],
        "6bba": embryos["6bba_"],
        "half0": sorted(half0),
        "half1": sorted(half1),
    }


def summary_delta(
    control: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, float]:
    result = {}
    for metric in SUMMARY_METRICS:
        control_value = finite_float(control[metric], f"control {metric}")
        candidate_value = finite_float(
            candidate[metric],
            f"candidate {metric}",
        )
        result[metric] = candidate_value - control_value
    return result


def finite_float(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{label} is not finite: {number}")
    return number


def per_movie_diagnostic(
    row: dict[str, object],
) -> dict[str, object]:
    division_tp = int(row["division_tp"])
    division_fp = int(row["division_fp"])
    division_fn = int(row["division_fn"])
    division_denominator = division_tp + division_fp + division_fn
    division_jaccard = (
        division_tp / division_denominator
        if division_denominator
        else 0.0
    )
    adjusted_edge = finite_float(
        row["adj_edge_jaccard"],
        "per-movie adjusted edge Jaccard",
    )
    return {
        "score": adjusted_edge + 0.1 * division_jaccard,
        "edge_jaccard": finite_float(
            row["edge_jaccard"],
            "per-movie edge Jaccard",
        ),
        "adj_edge_jaccard": adjusted_edge,
        "division_jaccard": division_jaccard,
        "node_recall": finite_float(
            row["node_recall"],
            "per-movie node recall",
        ),
        "edge_tp": int(row["edge_tp"]),
        "edge_fp": int(row["edge_fp"]),
        "edge_fn": int(row["edge_fn"]),
        "division_tp": division_tp,
        "division_fp": division_fp,
        "division_fn": division_fn,
        "num_pred_nodes": int(row["num_pred_nodes"]),
        "total_node_ratio": finite_float(
            row["total_node_ratio"],
            "per-movie total node ratio",
        ),
    }


def outcome(delta: float, epsilon: float) -> str:
    if delta > epsilon:
        return "win"
    if delta < -epsilon:
        return "loss"
    return "tie"


def paired_stats(
    records: list[dict[str, object]],
) -> dict[str, object]:
    deltas = [float(record["delta_score"]) for record in records]
    return {
        "movies": len(records),
        "wins": sum(record["outcome"] == "win" for record in records),
        "ties": sum(record["outcome"] == "tie" for record in records),
        "losses": sum(record["outcome"] == "loss" for record in records),
        "median_delta": statistics.median(deltas) if deltas else None,
        "minimum_delta": min(deltas) if deltas else None,
        "maximum_delta": max(deltas) if deltas else None,
    }


def main() -> None:
    args = parse_args()
    if args.expected_count <= 0:
        raise ValueError("--expected-count must be positive")
    if (
        not math.isfinite(args.minimum_pooled_delta)
        or args.minimum_pooled_delta <= 0.0
    ):
        raise ValueError("--minimum-pooled-delta must be finite and positive")
    if not math.isfinite(args.tie_epsilon) or args.tie_epsilon < 0.0:
        raise ValueError("--tie-epsilon must be finite and non-negative")
    for output in (args.output_json, args.output_csv):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")

    control_paths = movie_paths(args.control_dir)
    candidate_paths = movie_paths(args.candidate_dir)
    if set(control_paths) != set(candidate_paths):
        raise RuntimeError("Control and candidate movie sets differ")
    names = sorted(control_paths)
    if len(names) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} movies, found {len(names)}"
        )
    names_sha256 = movie_names_sha256(names)
    if names_sha256 != args.expected_names_sha256:
        raise RuntimeError(
            "Movie-name SHA256 mismatch: "
            f"expected {args.expected_names_sha256}, got {names_sha256}"
        )

    inference_manifest = load_json(args.inference_manifest)
    raw_control, raw_candidate = validate_inference_manifest(
        inference_manifest,
        names,
    )
    control_report = load_json(args.control_validation_report)
    candidate_report = load_json(args.candidate_validation_report)
    validate_e025_report(
        control_report,
        raw_control,
        args.control_dir,
        names,
        "control",
    )
    validate_e025_report(
        candidate_report,
        raw_candidate,
        args.candidate_dir,
        names,
        "candidate",
    )

    official = load_official_scorer(args.runtime_dir, args.scorer_dir)
    control_rows = {}
    candidate_rows = {}
    control_signatures = {}
    candidate_signatures = {}
    for index, name in enumerate(names, start=1):
        control_rows[name], control_signatures[name] = score_one(
            official,
            name,
            control_paths[name],
            args.gt_dir,
        )
        candidate_rows[name], candidate_signatures[name] = score_one(
            official,
            name,
            candidate_paths[name],
            args.gt_dir,
        )
        print(f"[{index:02d}/{len(names):02d}] {name}", flush=True)

    partitions = build_partitions(names)
    group_results = {}
    for group_name, group_names in partitions.items():
        control_summary = official_summary(
            official,
            control_rows,
            group_names,
        )
        candidate_summary = official_summary(
            official,
            candidate_rows,
            group_names,
        )
        group_results[group_name] = {
            "datasets": group_names,
            "control": control_summary,
            "candidate": candidate_summary,
            "delta": summary_delta(control_summary, candidate_summary),
        }

    per_movie = []
    for name in names:
        control_metrics = per_movie_diagnostic(control_rows[name])
        candidate_metrics = per_movie_diagnostic(candidate_rows[name])
        delta = summary_delta(control_metrics, candidate_metrics)
        embryo = "44b6" if name.startswith("44b6_") else "6bba"
        embryo_names = partitions[embryo]
        half = embryo_names.index(name) % 2
        record = {
            "dataset": name,
            "embryo": embryo,
            "half": half,
            "affected": (
                control_signatures[name] != candidate_signatures[name]
            ),
            "outcome": outcome(delta["score"], args.tie_epsilon),
        }
        for metric in (
            "score",
            "edge_jaccard",
            "adj_edge_jaccard",
            "division_jaccard",
            "node_recall",
            "total_node_ratio",
        ):
            record[f"control_{metric}"] = control_metrics[metric]
            record[f"candidate_{metric}"] = candidate_metrics[metric]
            record[f"delta_{metric}"] = (
                float(candidate_metrics[metric])
                - float(control_metrics[metric])
            )
        for metric in (
            "edge_tp",
            "edge_fp",
            "edge_fn",
            "division_tp",
            "division_fp",
            "division_fn",
            "num_pred_nodes",
        ):
            record[f"control_{metric}"] = control_metrics[metric]
            record[f"candidate_{metric}"] = candidate_metrics[metric]
            record[f"delta_{metric}"] = (
                int(candidate_metrics[metric])
                - int(control_metrics[metric])
            )
        per_movie.append(record)

    affected = [record for record in per_movie if record["affected"]]
    per_embryo = {
        embryo: paired_stats(
            [record for record in per_movie if record["embryo"] == embryo]
        )
        for embryo in ("44b6", "6bba")
    }
    paired = {
        "all": paired_stats(per_movie),
        "affected": paired_stats(affected),
        "by_embryo": per_embryo,
    }
    all_delta = group_results["all"]["delta"]
    affected_stats = paired["affected"]
    gates = {
        "pooled_delta_at_least_minimum": (
            float(all_delta["score"]) >= args.minimum_pooled_delta
        ),
        "adjusted_edge_not_regressed": (
            float(all_delta["adj_edge_jaccard"]) >= 0.0
        ),
        "both_embryos_positive": all(
            float(group_results[group]["delta"]["score"]) > 0.0
            for group in ("44b6", "6bba")
        ),
        "both_alternating_halves_positive": all(
            float(group_results[group]["delta"]["score"]) > 0.0
            for group in ("half0", "half1")
        ),
        "guard_changed_at_least_one_movie": len(affected) > 0,
        "affected_wins_exceed_losses": (
            int(affected_stats["wins"]) > int(affected_stats["losses"])
        ),
        "affected_paired_median_positive": (
            affected_stats["median_delta"] is not None
            and float(affected_stats["median_delta"]) > 0.0
        ),
    }
    promotion_passed = all(gates.values())

    report = {
        "schema_version": 1,
        "evaluation": "E026 frozen E025 frame-retention-guard A/B",
        "evidence_boundary": (
            "two-embryo-stratified; frozen checkpoints are not "
            "leave-one-embryo-out training-disjoint models"
        ),
        "paired_metric_definition": (
            "per-movie adjusted_edge_jaccard + 0.1 * division_jaccard; "
            "division_jaccard is 0 when TP+FP+FN is 0"
        ),
        "clean_only": True,
        "metric_exploit_used": False,
        "expected_count": args.expected_count,
        "movie_names_sha256": names_sha256,
        "official_evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
        "minimum_pooled_delta": args.minimum_pooled_delta,
        "tie_epsilon": args.tie_epsilon,
        "provenance": {
            "inference_manifest": str(
                args.inference_manifest.resolve()
            ),
            "control_validation_report": str(
                args.control_validation_report.resolve()
            ),
            "candidate_validation_report": str(
                args.candidate_validation_report.resolve()
            ),
            "control_dir": str(args.control_dir.resolve()),
            "candidate_dir": str(args.candidate_dir.resolve()),
            "scorer_dir": str(args.scorer_dir.resolve()),
        },
        "groups": group_results,
        "paired": paired,
        "gates": gates,
        "promotion_passed": promotion_passed,
        "leaderboard_requirement": (
            "A promoted private Kernel must complete, pass downloaded-output "
            "audit, and score strictly above E025 public 0.912."
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_movie[0]))
        writer.writeheader()
        writer.writerows(per_movie)

    print(json.dumps({"gates": gates, "promotion_passed": promotion_passed}))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")
    if args.fail_on_rejection and not promotion_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
