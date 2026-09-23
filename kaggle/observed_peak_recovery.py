"""Isolated, fail-closed x138 observed-peak components on E029 postprocessing."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import run_flow_relink_experiment as flow

CONFIG = {
    "READMIT_RADIUS_UM": 4.0, "READMIT_MIN_SCORE": 0.965,
    "GAPFILL_MAX_GAP": 3, "GAPFILL_MIN_SCORE": 0.5,
    "GAPFILL_STEP_UM": 5.0, "GAPFILL_PEAK_RADIUS_UM": 3.5,
    "GAPFILL_EXCLUDE_UM": 2.0, "GAPFILL_ALLOW_SYNTHETIC": 0,
    "GAPFILL_CONTEXT": True, "GAPFILL_MAX_ADDED_FRAC": 0.03,
}
FUNCTIONS = ("_gapfill_bump", "build_low_detection_pool", "load_low_detections",
             "readmit_discarded_detections", "fill_gaps_from_low_detections")


def selected_functions(path: Path) -> str:
    if flow.reference.stability.file_sha256(path) != flow.FLOW_REFERENCE_SHA256:
        raise ValueError("Pinned x138 source changed")
    cells = json.loads(path.read_text())["cells"]
    if len(cells) != 12 or cells[5]["cell_type"] != "code":
        raise ValueError("x138 source layout changed")
    source = "".join(cells[5]["source"])
    functions = [flow.function_node(source, name) for name in FUNCTIONS]

    class RejectMissingCache(ast.NodeTransformer):
        count = 0

        def visit_Return(self, node):
            if isinstance(node.value, ast.Constant) and node.value.value is None:
                self.count += 1
                return ast.parse("raise RuntimeError('Required observed-peak cache is unavailable')").body[0]
            return node

    # Preserve the published algorithms, but never accept a missing cache or
    # a partially mutated graph after an exception as a successful experiment.
    for function in functions:
        if function.name in {"load_low_detections", "readmit_discarded_detections"}:
            for node in ast.walk(function):
                if isinstance(node, ast.ExceptHandler):
                    node.body = [ast.Raise()]
        if function.name == "load_low_detections":
            strict = RejectMissingCache()
            strict.visit(function)
            if strict.count != 3:
                raise ValueError("Cache fallback anchors changed")
            function.body.insert(1, ast.parse("_gapfill_bump(stats, 'peak_cache_loads')").body[0])
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])))


def patch_filter(source: str, arm: str) -> str:
    """Insert exactly one component into the original filter, preserving peers."""
    if arm not in {"readmit", "gapfill"}:
        raise ValueError("Unknown observed-peak arm")
    function = flow.function_node(source, "filter_output_graph")

    class Insert(ast.NodeTransformer):
        count = 0

        def visit_Assign(self, node):
            node = self.generic_visit(node)
            if not isinstance(node.value, ast.Call) or not isinstance(node.value.func, ast.Name):
                return node
            anchor = "motion_relink_edges" if arm == "readmit" else "recover_strict_gap2"
            if node.value.func.id != anchor:
                return node
            self.count += 1
            addition = (
                "_gapfill_bump(stats, 'peak_component_calls')\n"
                "if motion_edges:\n"
                "    _before_readmit = len(nodes_by_id)\n"
                "    nodes_by_id = readmit_discarded_detections(nodes_by_id, motion_edges, stats, dataset=dataset)\n"
                "    if len(nodes_by_id) > _before_readmit:\n"
                "        motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs) or motion_edges\n"
            ) if arm == "readmit" else (
                "_gapfill_bump(stats, 'peak_component_calls')\n"
                "nodes_by_id, edges = fill_gaps_from_low_detections(\n"
                "    nodes_by_id, edges, stats, dataset=dataset, frame_cache=repair_frame_cache)\n"
            )
            return [node, *ast.parse(addition).body]

    inserter = Insert()
    inserter.visit(function)
    if inserter.count != 1:
        raise ValueError("Observed-peak insertion anchor changed")
    if not isinstance(function.body[-1], ast.Return):
        raise ValueError("Filter return anchor changed")
    function.body[-1:-1] = ast.parse(
        "stats['readmitted_retained_nodes'] = sum(bool(n.get('readmitted')) for n in nodes_by_id.values())\n"
        "stats['gapfill_retained_nodes'] = sum(bool(n.get('gapfill_peak')) for n in nodes_by_id.values())\n"
    ).body
    return ast.unparse(ast.fix_missing_locations(function))


def patch_postprocessing(source: str, helpers: str, arm: str) -> str:
    tree = ast.parse(source)
    replacement = flow.function_node(patch_filter(source, arm), "filter_output_graph")
    tree.body = [replacement if isinstance(n, ast.FunctionDef) and n.name == "filter_output_graph" else n
                 for n in tree.body]
    additions = ast.parse("\n".join(f"{key} = {value!r}" for key, value in CONFIG.items()) + "\n" + helpers)
    # Existing source imports numpy before these annotated helper definitions.
    index = next(i for i, n in enumerate(tree.body)
                 if isinstance(n, ast.FunctionDef) and n.name == "filter_output_graph")
    tree.body[index:index] = additions.body
    return ast.unparse(ast.fix_missing_locations(tree))
