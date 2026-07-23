#!/usr/bin/env python3
"""Prepare the attributed, output-free E001 baseline-control notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


UPSTREAM_URL = (
    "https://www.kaggle.com/code/yusuketogashi/"
    "clean-approach-lightweight-local-cv-no-hack"
)
UPSTREAM_SHA256 = "b754eaffca194e1b1ebbf5aa6471016996313eea1f18af4ff94316df749a2684"
E001_TAG = "buaaauto_e001_clean_baseline_no_rescue"

ATTRIBUTION = f"""# Biohub E001 | Clean Baseline Control | No Metric Exploit

This notebook is a reproducible fork of Yusuke Togashi's public
[Clean Approach + Lightweight Local CV | No Hack]({UPSTREAM_URL}) notebook,
observed on 2026-07-23 after the patched-metric rescore. The pinned upstream
notebook SHA256 is `{UPSTREAM_SHA256}`.

The upstream revision tests an adaptive five-node short-track rescue against its
stated `0.908` clean baseline. E001 selects that baseline control arm by disabling
only the rescue switch. Detection, learned edge scoring, ILP settings, and all
other graph post-processing parameters are preserved.

No artificial hubs, fake forks, negative-time nodes, out-of-volume nodes, or
cross-dataset edges are introduced.
"""

CONTROLLED_EXPERIMENT = """## Controlled experiment

E001 is the control arm documented by the pinned public notebook revision.

| Setting | E000 rescue candidate | E001 control |
|---|---:|---:|
| Motion relaxed gate | `9.5 µm` | `9.5 µm` |
| Minimum retained track length | `6` | `6` |
| Adaptive five-node rescue | enabled | **disabled** |
| Detection / ILP / other post-processing | unchanged | **unchanged** |

E000's fixed-8 official-spec score was `0.87892959136423`, versus the public
notebook's control reference `0.879219745066878`. E001 therefore removes the
single changed branch before another leaderboard submission.
"""

SELECTED_SETTINGS = """## Selected settings

| Setting | Value |
|---|---:|
| Public clean leaderboard reference | `0.908` |
| Detection threshold | `0.96875` |
| ILP appearance/disappearance cost | `0.0 / 1.575` |
| Motion tight/relaxed gate | `6.0 / 9.5 µm` |
| Minimum track length | `6` |
| Adaptive short-track rescue | **disabled** |
| Safe-division parent/sister radii | `4.66 / 8.5 µm` |
"""

PROMOTION_RULE = """## Promotion rule

- Treat E001 as the clean A/B control, not as a claimed improvement.
- Submit only after the notebook completes and preserves `submission.csv` through
  the fixed-8 diagnostic.
- Compare its verified leaderboard score directly with E000 submission `54923913`.
"""

CV_DESCRIPTION = """## Fixed-8 Official-Spec Lite CV

The hidden-test `submission.csv` is written and protected before diagnostics.
The fixed public-train split verifies the no-rescue control with the same scorer,
split, detector, and ILP settings used for E000.

Inspect edge TP/FP/FN, node recall, graph counts, and the paired score together.
"""

SUBMISSION_DESCRIPTION = """## Submission output

After the version completes, verify the fixed-8 summary, clean graph audit, row
count, and preserved submission hash before submitting `submission.csv`.
"""


def source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def prepare_notebook(upstream: Path, output: Path) -> None:
    raw = upstream.read_bytes()
    source_sha256 = hashlib.sha256(raw).hexdigest()
    if source_sha256 != UPSTREAM_SHA256:
        raise ValueError(
            "upstream notebook drifted: "
            f"expected {UPSTREAM_SHA256}, got {source_sha256}"
        )

    notebook = json.loads(raw)
    cells = notebook.get("cells")
    if not isinstance(cells, list) or len(cells) <= 19:
        raise ValueError("upstream file does not match the pinned notebook layout")

    cells[0]["cell_type"] = "markdown"
    cells[0]["source"] = ATTRIBUTION
    cells[0].pop("outputs", None)
    cells[0].pop("execution_count", None)
    cells[3]["source"] = CONTROLLED_EXPERIMENT
    cells[5]["source"] = SELECTED_SETTINGS
    cells[14]["source"] = PROMOTION_RULE
    cells[17]["source"] = CV_DESCRIPTION
    cells[19]["source"] = SUBMISSION_DESCRIPTION

    replacements = {
        "biohub_132_clean_short_track_rescue_lightcv_nohack": E001_TAG,
        (
            "Clean recall test: conservatively rescue high-confidence five-node "
            "track components from the legitimate 0.908 baseline"
        ): "A/B control: disable the E000 five-node short-track rescue",
        (
            "# Keep this disabled so the experiment stays focused on ILP cost "
            "calibration."
        ): "# E001 changes only this A/B switch relative to E000.",
        'os.environ["BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE"] = "1"': (
            'os.environ["BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE"] = "0"'
        ),
        (
            "Public title: 🧬 Biohub 132 | Clean Short-Track Rescue + Light Local "
            "CV | No Hack"
        ): "Public title: Biohub E001 | Clean Baseline Control | No Metric Exploit",
        (
            "Strategy: rescue only high-confidence five-node components removed "
            "by the minimum-length-six filter"
        ): "Strategy: disable the E000 five-node rescue as a controlled baseline",
        "Adaptive short-track rescue: True": "Adaptive short-track rescue: False",
        (
            "Submit recommendation: PROMOTE ONLY IF FIXED-8 LITE CV IMPROVES "
            "WITH CONTROLLED FP"
        ): "Submit recommendation: VERIFY CONTROL ARM AND COMPARE WITH E000",
    }
    replacement_counts = {old: 0 for old in replacements}

    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

        text = source_text(cell)
        for old, new in replacements.items():
            count = text.count(old)
            replacement_counts[old] += count
            text = text.replace(old, new)
        cell["source"] = text

    missing = [old for old, count in replacement_counts.items() if count == 0]
    if missing:
        raise ValueError(f"pinned notebook transform did not match: {missing}")

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
