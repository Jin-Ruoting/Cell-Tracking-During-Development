#!/usr/bin/env python3
"""Evaluate the frozen E028 cross-embryo PU appearance filter."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import evaluate_ab_stability as common
import evaluate_edge_tta_stability as paired
import run_pu_appearance_filter as pu


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
        "--control-inference-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--filter-manifest", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--scorer-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=64)
    parser.add_argument("--expected-names-sha256", required=True)
    parser.add_argument("--minimum-pooled-delta", type=float, default=0.002)
    parser.add_argument("--tie-epsilon", type=float, default=1e-12)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--fail-on-rejection", action="store_true")
    return parser.parse_args()


def require_exact_mapping(
    actual: object,
    expected: dict[str, object],
    label: str,
) -> dict[str, object]:
    if not isinstance(actual, dict):
        raise RuntimeError(f"{label} is missing")
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            raise RuntimeError(
                f"{label}.{key} mismatch: "
                f"expected {expected_value}, got {actual.get(key)}"
            )
    return actual


def validate_filter_manifest(
    manifest: dict[str, object],
    control_dir: Path,
    candidate_dir: Path,
    names: list[str],
    names_sha256: str,
) -> dict[str, dict[str, object]]:
    required_root = {
        "schema_version": pu.SCHEMA_VERSION,
        "experiment": "E028 cross-embryo PU appearance filter",
        "clean_only": True,
        "metric_exploit_used": False,
        "notebook_sha256": common.EXPECTED_E025_NOTEBOOK_SHA256,
        "movie_names_sha256": names_sha256,
    }
    require_exact_mapping(manifest, required_root, "filter manifest")
    if manifest.get("datasets") != sorted(names):
        raise RuntimeError("PU filter dataset order changed")
    if Path(str(manifest.get("baseline_dir"))).resolve() != (
        control_dir.resolve()
    ):
        raise RuntimeError("PU filter control directory changed")
    if Path(str(manifest.get("candidate_dir"))).resolve() != (
        candidate_dir.resolve()
    ):
        raise RuntimeError("PU filter candidate directory changed")
    if manifest.get("feature_names") != list(pu.FEATURE_NAMES):
        raise RuntimeError("PU appearance feature schema changed")

    require_exact_mapping(
        manifest.get("matching"),
        {
            "method": "maximum_cardinality_radius_matching",
            "radius_um": pu.MATCH_RADIUS_UM,
            "matched_prediction_label": "positive",
            "other_prediction_label": "unlabeled",
            "declared_negative_samples": 0,
        },
        "matching",
    )
    require_exact_mapping(
        manifest.get("training"),
        {
            "algorithm": "non_negative_positive_unlabeled_linear",
            "positive_prior": pu.POSITIVE_PRIOR,
            "oof_folds": pu.OOF_FOLDS,
            "epochs": pu.EPOCHS,
            "positive_batch_size": pu.POSITIVE_BATCH_SIZE,
            "unlabeled_batch_size": pu.UNLABELED_BATCH_SIZE,
            "max_unlabeled_per_positive": (
                pu.MAX_UNLABELED_PER_POSITIVE
            ),
            "learning_rate": pu.LEARNING_RATE,
            "weight_decay": pu.WEIGHT_DECAY,
            "base_seed": pu.BASE_SEED,
        },
        "training",
    )
    require_exact_mapping(
        manifest.get("filter_policy"),
        {
            "component_min_nodes": pu.COMPONENT_MIN_NODES,
            "component_max_nodes": pu.COMPONENT_MAX_NODES,
            "division_components_allowed": False,
            "positive_probability_quantile": pu.POSITIVE_QUANTILE,
            "median_below_threshold": True,
            "minimum_below_threshold_fraction": (
                pu.MIN_BELOW_THRESHOLD_FRACTION
            ),
            "maximum_removed_fraction": pu.MAX_REMOVED_FRACTION,
            "maximum_removed_nodes": pu.MAX_REMOVED_NODES,
            "whole_components_only": True,
            "nodes_added": 0,
            "edges_added": 0,
        },
        "filter policy",
    )

    models = manifest.get("models")
    if not isinstance(models, dict) or set(models) != {"44b6", "6bba"}:
        raise RuntimeError("PU target model coverage changed")
    for target, source in (("44b6", "6bba"), ("6bba", "44b6")):
        model = models[target]
        if not isinstance(model, dict):
            raise RuntimeError(f"PU model missing for {target}")
        required_model = {
            "target_embryo": target,
            "source_embryo": source,
            "target_labels_used": False,
        }
        require_exact_mapping(model, required_model, f"model {target}")
        threshold = float(model.get("threshold", float("nan")))
        if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise RuntimeError(f"PU threshold is invalid for {target}")
        model_path = Path(str(model.get("model_path")))
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if common.file_sha256(model_path) != model.get("model_sha256"):
            raise RuntimeError(f"PU model checksum changed for {target}")
        calibration = model.get("calibration")
        if not isinstance(calibration, dict):
            raise RuntimeError(f"PU calibration missing for {target}")
        if (
            calibration.get("source_embryo") != source
            or int(calibration.get("oof_positive_scores", 0)) <= 0
            or len(calibration.get("folds", [])) != pu.OOF_FOLDS
        ):
            raise RuntimeError(f"PU OOF calibration changed for {target}")
        final_model = model.get("final_model")
        if not isinstance(final_model, dict):
            raise RuntimeError(f"PU final model manifest missing for {target}")
        training_movies = set(final_model.get("training_movies", []))
        expected_training = {
            name for name in names if name.startswith(f"{source}_")
        }
        if training_movies != expected_training:
            raise RuntimeError(
                f"PU source-only training coverage changed for {target}"
            )

    per_movie = manifest.get("per_movie")
    if not isinstance(per_movie, dict) or set(per_movie) != set(names):
        raise RuntimeError("PU per-movie filter coverage changed")
    total_removed = 0
    for name, row in per_movie.items():
        if not isinstance(row, dict):
            raise TypeError(f"Invalid PU filter row for {name}")
        target = "44b6" if name.startswith("44b6_") else "6bba"
        source = "6bba" if target == "44b6" else "44b6"
        if (
            row.get("target_embryo") != target
            or row.get("source_embryo") != source
        ):
            raise RuntimeError(f"PU embryo isolation changed for {name}")
        raw_nodes = int(row.get("raw_nodes", -1))
        removed = int(row.get("removed_nodes", -1))
        cap = min(
            pu.MAX_REMOVED_NODES,
            int(math.floor(raw_nodes * pu.MAX_REMOVED_FRACTION)),
        )
        if removed < 0 or removed > cap or int(row.get("node_cap", -1)) != cap:
            raise RuntimeError(f"PU removal cap failed for {name}")
        if int(row.get("selected_components", -1)) < 0:
            raise RuntimeError(f"PU component count is invalid for {name}")
        topology = row.get("topology")
        if not isinstance(topology, dict):
            raise RuntimeError(f"PU topology audit missing for {name}")
        if int(topology.get("nodes", -1)) != raw_nodes - removed:
            raise RuntimeError(f"PU node accounting failed for {name}")
        if (
            int(topology.get("duplicate_edges", -1)) != 0
            or int(topology.get("dangling_edges", -1)) != 0
            or int(topology.get("nonconsecutive_edges", -1)) != 0
            or int(topology.get("max_indegree", -1)) > 1
            or int(topology.get("max_outdegree", -1)) > 2
            or int(topology.get("nonbinary_sources", -1)) != 0
        ):
            raise RuntimeError(f"PU topology gate failed for {name}")
        total_removed += removed
    if total_removed <= 0:
        raise RuntimeError("PU appearance filter did not remove any node")
    if int(manifest.get("total_removed_nodes", -1)) != total_removed:
        raise RuntimeError("PU total removal accounting changed")
    return per_movie


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

    control_paths = common.movie_paths(args.control_dir)
    candidate_paths = common.movie_paths(args.candidate_dir)
    if set(control_paths) != set(candidate_paths):
        raise RuntimeError("PU control and candidate movie sets differ")
    names = sorted(control_paths)
    if len(names) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} movies, found {len(names)}"
        )
    names_sha256 = common.movie_names_sha256(names)
    if names_sha256 != args.expected_names_sha256:
        raise RuntimeError("PU movie-name SHA256 mismatch")

    inference_manifest = common.load_json(
        args.control_inference_manifest
    )
    raw_control, _ = common.validate_inference_manifest(
        inference_manifest,
        names,
    )
    control_report = common.load_json(args.control_validation_report)
    common.validate_e025_report(
        control_report,
        raw_control,
        args.control_dir,
        names,
        "control",
    )
    filter_manifest = common.load_json(args.filter_manifest)
    filter_rows = validate_filter_manifest(
        filter_manifest,
        args.control_dir,
        args.candidate_dir,
        names,
        names_sha256,
    )

    official = common.load_official_scorer(
        args.runtime_dir,
        args.scorer_dir,
    )
    control_rows = {}
    candidate_rows = {}
    control_signatures = {}
    candidate_signatures = {}
    for index, name in enumerate(names, start=1):
        control_rows[name], control_signatures[name] = common.score_one(
            official,
            name,
            control_paths[name],
            args.gt_dir,
        )
        candidate_rows[name], candidate_signatures[name] = common.score_one(
            official,
            name,
            candidate_paths[name],
            args.gt_dir,
        )
        expected_nodes = int(filter_rows[name]["topology"]["nodes"])
        if int(candidate_rows[name]["num_pred_nodes"]) != expected_nodes:
            raise RuntimeError(f"PU candidate node count changed for {name}")
        print(f"[{index:02d}/{len(names):02d}] {name}", flush=True)

    args.experiment = "e028"
    report, per_movie = paired.build_report(
        args,
        names,
        names_sha256,
        control_rows,
        candidate_rows,
        control_signatures,
        candidate_signatures,
        official,
    )
    report["provenance"] = {
        "control_inference_manifest": str(
            args.control_inference_manifest.resolve()
        ),
        "control_validation_report": str(
            args.control_validation_report.resolve()
        ),
        "filter_manifest": str(args.filter_manifest.resolve()),
        "control_dir": str(args.control_dir.resolve()),
        "candidate_dir": str(args.candidate_dir.resolve()),
        "scorer_dir": str(args.scorer_dir.resolve()),
    }
    report["pu_filter"] = {
        "total_removed_nodes": filter_manifest["total_removed_nodes"],
        "models": filter_manifest["models"],
        "filter_policy": filter_manifest["filter_policy"],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_movie[0]))
        writer.writeheader()
        writer.writerows(per_movie)
    print(
        json.dumps(
            {
                "gates": report["gates"],
                "promotion_passed": report["promotion_passed"],
            }
        )
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")
    if args.fail_on_rejection and not report["promotion_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
