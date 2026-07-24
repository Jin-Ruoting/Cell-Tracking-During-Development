#!/usr/bin/env python3
"""Resolve saved pre-ILP graphs across a sweep of division costs."""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path


def weight_slug(weight: float) -> str:
    rendered = f"{weight:.6g}".replace("-", "m").replace(".", "p")
    return rendered.replace("+", "")


def prediction_paths(root: Path) -> list[Path]:
    paths = sorted(root.rglob("*.geff"), key=lambda path: path.name)
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise ValueError("duplicate GEFF dataset names under input root")
    return paths


@contextlib.contextmanager
def suppress_solver_output():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(
            devnull
        ):
            yield


def solve_sweep(
    input_root: Path,
    output_root: Path,
    division_weights: list[float],
    edge_weight: float,
    appearance_weight: float,
    disappearance_weight: float,
) -> None:
    import tracksdata as td

    from biohub_tracking.io import save_graph

    inputs = prediction_paths(input_root)
    if not inputs:
        raise FileNotFoundError(f"no GEFF predictions found under {input_root}")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)

    for division_weight in division_weights:
        label = f"division_{weight_slug(division_weight)}"
        candidate_dir = output_root / label
        candidate_dir.mkdir()
        print(
            f"candidate={label} division_weight={division_weight} "
            f"datasets={len(inputs)}",
            flush=True,
        )
        for index, input_path in enumerate(inputs, start=1):
            loaded = td.graph.IndexedRXGraph.from_geff(input_path)
            graph = loaded[0] if isinstance(loaded, tuple) else loaded
            if graph.num_edges() > 0:
                solver = td.solvers.ILPSolver(
                    edge_weight=edge_weight * td.EdgeAttr("edge_prob"),
                    appearance_weight=appearance_weight,
                    disappearance_weight=disappearance_weight,
                    division_weight=division_weight,
                )
                with suppress_solver_output():
                    graph = solver.solve(graph)
            save_graph(graph, candidate_dir / input_path.name)
            print(
                f"  [{index:03d}/{len(inputs):03d}] {input_path.stem}",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--division-weight",
        type=float,
        action="append",
        required=True,
    )
    parser.add_argument("--edge-weight", type=float, default=-1.0)
    parser.add_argument("--appearance-weight", type=float, default=0.0)
    parser.add_argument("--disappearance-weight", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    solve_sweep(
        args.input_root,
        args.output_root,
        args.division_weight,
        args.edge_weight,
        args.appearance_weight,
        args.disappearance_weight,
    )


if __name__ == "__main__":
    main()
