#!/usr/bin/env python3
"""Prepare the attributed, output-free Kaggle baseline notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


UPSTREAM_URL = (
    "https://www.kaggle.com/code/yusuketogashi/"
    "clean-approach-lightweight-local-cv-no-hack"
)

ATTRIBUTION = f"""# Biohub Clean Baseline | No Metric Exploit

This notebook is a reproducible fork of Yusuke Togashi's public
[Clean Approach + Lightweight Local CV | No Hack]({UPSTREAM_URL}) notebook,
observed on 2026-07-23 after the patched-metric rescore.

Changes in this repository are deliberately narrow: attribution and title text are
made explicit, execution outputs are removed, and notebook metadata is normalized.
The upstream tracking and post-processing logic is otherwise preserved.

No artificial hubs, fake forks, negative-time nodes, out-of-volume nodes, or
cross-dataset edges are introduced.
"""


def source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def prepare_notebook(upstream: Path, output: Path) -> None:
    notebook = json.loads(upstream.read_text(encoding="utf-8"))
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("upstream file is not a non-empty Jupyter notebook")

    cells[0]["cell_type"] = "markdown"
    cells[0]["source"] = ATTRIBUTION
    cells[0].pop("outputs", None)
    cells[0].pop("execution_count", None)

    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

        text = source_text(cell)
        if "BIOHUB_PRESET = 'biohub_132_clean_short_track_rescue_lightcv_nohack'" in text:
            cell["source"] = text.replace(
                "BIOHUB_PRESET = 'biohub_132_clean_short_track_rescue_lightcv_nohack'",
                "BIOHUB_PRESET = 'buaaauto_clean_baseline_nohack'",
            )

    notebook["metadata"] = {
        key: value
        for key, value in notebook.get("metadata", {}).items()
        if key in {"kernelspec", "language_info"}
    }
    notebook["nbformat"] = 4
    notebook["nbformat_minor"] = int(notebook.get("nbformat_minor", 5))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_notebook(args.upstream, args.output)
    print(f"Prepared {args.output}")


if __name__ == "__main__":
    main()
