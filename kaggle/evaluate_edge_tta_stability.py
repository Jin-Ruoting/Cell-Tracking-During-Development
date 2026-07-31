#!/usr/bin/env python3
"""Evaluate frozen E023/E027 edge-TTA candidates with stability gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import evaluate_ab_stability as common


EXPECTED_EDGE_TTA_NOTEBOOK_SHA256 = (
    "fd4d166ef72afc8db2e191df6e7dad661b18151f6faf9fa303e97531b6de892c"
)
EXPERIMENTS = ("e023", "e027")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
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
    parser.add_argument(
        "--control-inference-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--candidate-inference-manifest",
        type=Path,
        required=True,
    )
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


def validate_common_manifest(
    manifest: dict[str, object],
    names: list[str],
    disappearance_weight: float,
    label: str,
) -> Path:
    expected_pins = {
        "reference_notebook_sha256": common.EXPECTED_REFERENCE_NOTEBOOK_SHA256,
        "support_predictor_sha256": common.EXPECTED_SUPPORT_PREDICTOR_SHA256,
        "primary_weight_sha256": common.EXPECTED_PRIMARY_WEIGHT_SHA256,
        "secondary_weight_sha256": common.EXPECTED_SECONDARY_WEIGHT_SHA256,
    }
    for key, expected in expected_pins.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"{label}: {key} mismatch: "
                f"expected {expected}, got {manifest.get(key)}"
            )
    if manifest.get("datasets") != sorted(names):
        raise RuntimeError(f"{label}: inference dataset order changed")
    common.require_close(
        manifest.get("det_threshold"),
        0.96875,
        f"{label} det_threshold",
    )
    common.require_close(
        manifest.get("ilp_disappearance_weight"),
        disappearance_weight,
        f"{label} ilp_disappearance_weight",
    )
    blend = manifest.get("blend")
    if not isinstance(blend, dict):
        raise RuntimeError(f"{label}: blend configuration missing")
    expected_blend = {
        "edge_weight": 0.15,
        "detection_weight": 0.475,
        "mix_temperature": 1.0,
        "low_margin_max": 0.35,
        "edge_threshold": 0.48,
    }
    for key, expected in expected_blend.items():
        common.require_close(
            blend.get(key),
            expected,
            f"{label} blend.{key}",
        )
    if blend.get("link_mode") != "low_margin_consensus":
        raise RuntimeError(f"{label}: blend link mode changed")
    results = manifest.get("results")
    if not isinstance(results, dict):
        raise RuntimeError(f"{label}: results missing")
    blend_result = results.get("blend")
    if not isinstance(blend_result, dict):
        raise RuntimeError(f"{label}: blend result missing")
    output_dir = Path(str(blend_result.get("output_dir"))).resolve()
    if int(blend_result.get("datasets", -1)) != len(names):
        raise RuntimeError(f"{label}: blend result coverage changed")
    return output_dir


def validate_edge_tta_contract(
    experiment: str,
    control: dict[str, object],
    candidate: dict[str, object],
    names: list[str],
) -> None:
    control_tta = control.get("edge_tta")
    if experiment == "e027":
        if not isinstance(control_tta, dict):
            raise RuntimeError("E027 control edge-TTA manifest missing")
        required_control = {
            "mode": "original",
            "requested_view_count": 1,
            "unique_spatial_view_count": 1,
            "view_names": ["identity"],
            "view_weights": {"identity": 1.0},
            "detection_tta_changed": False,
        }
        if any(
            control_tta.get(key) != expected
            for key, expected in required_control.items()
        ):
            raise RuntimeError("E027 control is not identity edge TTA")
    elif control_tta is not None:
        if not isinstance(control_tta, dict):
            raise RuntimeError("E023 control edge-TTA manifest is invalid")
        if control_tta.get("mode") != "original":
            raise RuntimeError("E023 control is not identity edge TTA")

    edge_tta = candidate.get("edge_tta")
    if not isinstance(edge_tta, dict):
        raise RuntimeError(f"{experiment.upper()} candidate edge TTA missing")
    if (
        edge_tta.get("reference_notebook_sha256")
        != EXPECTED_EDGE_TTA_NOTEBOOK_SHA256
    ):
        raise RuntimeError("Edge-TTA reference notebook pin changed")
    if edge_tta.get("implementation_version") != "dual_seed_edge_tta_v2":
        raise RuntimeError("Edge-TTA implementation version changed")
    required_common = {
        "models": "all_loaded",
        "aggregation_domain": "raw_edge_logits",
        "aggregation_stage": "per_seed_before_seed_calibration",
        "node_policy": "shared_canonical_detection_nodes",
        "feature_alignment": "inverse_map_to_canonical_zyx",
        "detection_tta_changed": False,
    }
    for key, expected in required_common.items():
        if edge_tta.get(key) != expected:
            raise RuntimeError(
                f"Edge-TTA {key} mismatch: "
                f"expected {expected}, got {edge_tta.get(key)}"
            )

    if experiment == "e023":
        expected_views = (
            "identity",
            "flip_x",
            "flip_y",
            "flip_xy",
            "rot90",
            "rot270",
            "transpose",
            "legacy_anti_transpose",
        )
        required = {
            "mode": "pilkwang_legacy_d4",
            "application": "global",
            "requested_view_count": 8,
            "unique_spatial_view_count": 7,
            "legacy_anti_transpose_duplicates_flip_x": True,
        }
        for key, expected in required.items():
            if edge_tta.get(key) != expected:
                raise RuntimeError(
                    f"E023 edge-TTA {key} mismatch: "
                    f"expected {expected}, got {edge_tta.get(key)}"
                )
        weights = edge_tta.get("view_weights")
        if (
            edge_tta.get("view_names") != list(expected_views)
            or not isinstance(weights, dict)
            or set(weights) != set(expected_views)
        ):
            raise RuntimeError("E023 edge-TTA view weights missing")
        for view, weight in weights.items():
            common.require_close(weight, 0.125, f"E023 weight {view}")
        return

    expected_views = (
        "identity",
        "flip_x",
        "flip_y",
        "flip_xy",
        "rot90",
        "rot270",
        "transpose",
        "anti_transpose",
    )
    required = {
        "mode": "corrected_d4",
        "application": "ambiguous_parent_consensus",
        "requested_view_count": 8,
        "unique_spatial_view_count": 8,
        "legacy_anti_transpose_duplicates_flip_x": False,
    }
    for key, expected in required.items():
        if edge_tta.get(key) != expected:
            raise RuntimeError(
                f"E027 edge-TTA {key} mismatch: "
                f"expected {expected}, got {edge_tta.get(key)}"
            )
    common.require_close(
        edge_tta.get("ambiguous_parent_margin_max"),
        0.35,
        "E027 ambiguous-parent margin",
    )
    expected_policy = {
        "replacement_scope": "selected_target_logit_columns",
        "requires_identity_seed_disagreement": True,
        "requires_multiview_seed_consensus": True,
        "requires_primary_identity_low_margin": True,
        "requires_primary_parent_change": True,
        "unselected_policy": "identity_logits_exact",
    }
    if edge_tta.get("ambiguous_parent_policy") != expected_policy:
        raise RuntimeError("E027 ambiguous-parent policy changed")
    weights = edge_tta.get("view_weights")
    if (
        edge_tta.get("view_names") != list(expected_views)
        or not isinstance(weights, dict)
        or set(weights) != set(expected_views)
    ):
        raise RuntimeError("E027 edge-TTA view weights changed")
    common.require_close(weights.get("identity"), 0.5, "E027 identity weight")
    for view, weight in weights.items():
        if view != "identity":
            common.require_close(weight, 1.0 / 14.0, f"E027 weight {view}")
    results = candidate.get("results")
    blend_result = results.get("blend") if isinstance(results, dict) else None
    rerank = (
        blend_result.get("parent_rerank")
        if isinstance(blend_result, dict)
        else None
    )
    if not isinstance(rerank, dict):
        raise RuntimeError("E027 parent-rerank diagnostics missing")
    if int(rerank.get("selected_targets", 0)) <= 0:
        raise RuntimeError("E027 parent rerank did not select any target")
    datasets = rerank.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(names):
        raise RuntimeError("E027 parent-rerank dataset coverage changed")


def validate_e000_report(
    report: dict[str, object],
    expected_baseline_dir: Path,
    expected_geff_dir: Path,
    names: list[str],
    label: str,
) -> None:
    if report.get("e000_only") is not True:
        raise RuntimeError(f"{label}: report is not E000-only")
    if report.get("e025_exact") is not False:
        raise RuntimeError(f"{label}: E025 flag unexpectedly enabled")
    if report.get("notebook_sha256") != common.EXPECTED_E025_NOTEBOOK_SHA256:
        raise RuntimeError(f"{label}: postprocess notebook pin changed")
    if report.get("deepcenter_checkpoint") is not None:
        raise RuntimeError(f"{label}: E000 unexpectedly used DeepCenter")
    if Path(str(report.get("baseline_dir"))).resolve() != (
        expected_baseline_dir.resolve()
    ):
        raise RuntimeError(f"{label}: raw baseline directory changed")
    if report.get("datasets") != sorted(names):
        raise RuntimeError(f"{label}: validation dataset order changed")
    if int(report.get("max_outdegree", -1)) != 2:
        raise RuntimeError(f"{label}: strict maximum outdegree changed")
    output_dir = Path(str(report.get("output_dir"))).resolve()
    if output_dir / "score_e000_safe" / "geffs" != (
        expected_geff_dir.resolve()
    ):
        raise RuntimeError(f"{label}: scored GEFF directory mismatch")
    validate_topology_variant(report, names, "e000_safe", label)


def validate_topology_variant(
    report: dict[str, object],
    names: list[str],
    variant: str,
    label: str,
) -> None:
    variants = report.get("variants")
    per_movie = variants.get(variant) if isinstance(variants, dict) else None
    if not isinstance(per_movie, dict) or set(per_movie) != set(names):
        raise RuntimeError(f"{label}: per-movie topology coverage mismatch")
    for dataset, audit in per_movie.items():
        if not isinstance(audit, dict):
            raise TypeError(f"{label}: invalid audit for {dataset}")
        optional_zero_counts = (
            "duplicate_edges",
            "dangling_edges",
            "nonconsecutive_edges",
        )
        required_counts = {
            "max_indegree",
            "max_outdegree",
            "nonbinary_sources",
        }
        if not required_counts <= set(audit):
            raise RuntimeError(
                f"{label}: topology fields missing for {dataset}"
            )
        if (
            any(
                key in audit and int(audit[key]) != 0
                for key in optional_zero_counts
            )
            or int(audit.get("max_indegree", -1)) > 1
            or int(audit.get("max_outdegree", -1)) > 2
            or int(audit.get("nonbinary_sources", -1)) != 0
        ):
            raise RuntimeError(f"{label}: topology gate failed for {dataset}")


def build_report(
    args: argparse.Namespace,
    names: list[str],
    names_sha256: str,
    control_rows: dict[str, dict[str, object]],
    candidate_rows: dict[str, dict[str, object]],
    control_signatures: dict[str, str],
    candidate_signatures: dict[str, str],
    official,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    partitions = common.build_partitions(names)
    group_results = {}
    for group_name, group_names in partitions.items():
        control_summary = common.official_summary(
            official,
            control_rows,
            group_names,
        )
        candidate_summary = common.official_summary(
            official,
            candidate_rows,
            group_names,
        )
        group_results[group_name] = {
            "datasets": group_names,
            "control": control_summary,
            "candidate": candidate_summary,
            "delta": common.summary_delta(
                control_summary,
                candidate_summary,
            ),
        }

    per_movie = []
    for name in names:
        control_metrics = common.per_movie_diagnostic(control_rows[name])
        candidate_metrics = common.per_movie_diagnostic(candidate_rows[name])
        delta = common.summary_delta(control_metrics, candidate_metrics)
        embryo = "44b6" if name.startswith("44b6_") else "6bba"
        embryo_names = partitions[embryo]
        record: dict[str, object] = {
            "dataset": name,
            "embryo": embryo,
            "half": embryo_names.index(name) % 2,
            "affected": (
                control_signatures[name] != candidate_signatures[name]
            ),
            "outcome": common.outcome(
                delta["score"],
                args.tie_epsilon,
            ),
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
    affected_stats = common.paired_stats(affected)
    paired = {
        "all": common.paired_stats(per_movie),
        "affected": affected_stats,
        "by_embryo": {
            embryo: common.paired_stats(
                [
                    record
                    for record in per_movie
                    if record["embryo"] == embryo
                ]
            )
            for embryo in ("44b6", "6bba")
        },
    }
    all_delta = group_results["all"]["delta"]
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
        "candidate_changed_at_least_one_movie": len(affected) > 0,
        "affected_wins_exceed_losses": (
            int(affected_stats["wins"]) > int(affected_stats["losses"])
        ),
        "affected_paired_median_positive": (
            affected_stats["median_delta"] is not None
            and float(affected_stats["median_delta"]) > 0.0
        ),
    }
    report = {
        "schema_version": 1,
        "evaluation": (
            f"{args.experiment.upper()} frozen edge-TTA paired evaluation"
        ),
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
        "official_evaluator_sha256": common.EXPECTED_EVALUATOR_SHA256,
        "minimum_pooled_delta": args.minimum_pooled_delta,
        "tie_epsilon": args.tie_epsilon,
        "groups": group_results,
        "paired": paired,
        "gates": gates,
        "promotion_passed": all(gates.values()),
        "leaderboard_requirement": (
            "A promoted private Kernel must complete, pass downloaded-output "
            "audit, and score strictly above E025 public 0.912."
        ),
    }
    return report, per_movie


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
        raise RuntimeError("Control and candidate movie sets differ")
    names = sorted(control_paths)
    if len(names) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} movies, found {len(names)}"
        )
    names_sha256 = common.movie_names_sha256(names)
    if names_sha256 != args.expected_names_sha256:
        raise RuntimeError(
            "Movie-name SHA256 mismatch: "
            f"expected {args.expected_names_sha256}, got {names_sha256}"
        )

    control_manifest = common.load_json(args.control_inference_manifest)
    candidate_manifest = common.load_json(args.candidate_inference_manifest)
    disappearance_weight = 1.575 if args.experiment == "e023" else 1.5
    raw_control = validate_common_manifest(
        control_manifest,
        names,
        disappearance_weight,
        "control",
    )
    raw_candidate = validate_common_manifest(
        candidate_manifest,
        names,
        disappearance_weight,
        "candidate",
    )
    validate_edge_tta_contract(
        args.experiment,
        control_manifest,
        candidate_manifest,
        names,
    )
    control_report = common.load_json(args.control_validation_report)
    candidate_report = common.load_json(args.candidate_validation_report)
    if args.experiment == "e023":
        validate_e000_report(
            control_report,
            raw_control,
            args.control_dir,
            names,
            "control",
        )
        validate_e000_report(
            candidate_report,
            raw_candidate,
            args.candidate_dir,
            names,
            "candidate",
        )
    else:
        common.validate_e025_report(
            control_report,
            raw_control,
            args.control_dir,
            names,
            "control",
        )
        common.validate_e025_report(
            candidate_report,
            raw_candidate,
            args.candidate_dir,
            names,
            "candidate",
        )
        validate_topology_variant(
            control_report,
            names,
            "e025_exact",
            "control",
        )
        validate_topology_variant(
            candidate_report,
            names,
            "e025_exact",
            "candidate",
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
        print(f"[{index:02d}/{len(names):02d}] {name}", flush=True)

    report, per_movie = build_report(
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
        "candidate_inference_manifest": str(
            args.candidate_inference_manifest.resolve()
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
