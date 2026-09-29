#!/usr/bin/env python3
"""Package pinned public inference and two declared final-day config transfers.

The public Notebook is downloaded separately and is never committed here.
This builder only reads source; it performs no model inference or API mutation.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import re
import subprocess

import run_geometric_reference as reference

SOURCE = "anvithpothula/biohub-0-953-lb-original"
SOURCE_SHA256 = "6b655e39bbfd2d3d6c762badea69847d3f00f5b548f385cb01b07ee2600fde6d"
HEAD_SHA256 = "625a0d9340f48193f2ec294fc2d81c5bb3c03087eab78ef0ae998a9c4c7da00c"
PROFILES = {
    "E045": {},
    "E046": {"BIOHUB_ILP_DIVISION_WEIGHT": "0.4"},
    "E047": {"BIOHUB_ILP_DIVISION_WEIGHT": "0.4", "BIOHUB_READMIT_MIN_SCORE": "0.94"},
}
DATASETS = {
    "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
    "pilkwang/biohub-temporal-unet3d-seed314159-v1",
    "pilkwang/biohub-tracking-support-pack-50ep-v1",
    "anvithpothula/biohub-v1284-head-s075",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_sources(path):
    if sha(path.read_bytes()) != SOURCE_SHA256:
        raise ValueError("Public source changed; review it before repackaging")
    notebook = json.loads(path.read_text())
    if len(notebook["cells"]) != 12:
        raise ValueError("Unexpected source layout")
    sources = []
    for cell in notebook["cells"][:6]:
        if cell["cell_type"] != "code":
            raise ValueError("Unexpected inference cell")
        source = cell["source"]
        source = "".join(source) if isinstance(source, list) else source
        compile(source, "public-source", "exec")
        sources.append(source)
    if 'os.environ["BIOHUB_VALIDATOR_ENABLE"] = "0"' not in sources[0]:
        raise ValueError("Author's validator is no longer disabled")
    return sources


def audit_source(experiment, proof):
    validator = inspect.getsource(reference.validate_submission).replace(
        "stability.file_sha256(csv_path)", "hashlib.sha256(csv_path.read_bytes()).hexdigest()")
    boundary = inspect.getsource(reference.normalize_export_boundary)
    for name in ("source_path", "output_path"):
        boundary = boundary.replace(f"stability.file_sha256({name})",
                                    f"hashlib.sha256({name}.read_bytes()).hexdigest()")
    return "import hashlib\n" + boundary + "\n" + validator + f'''
audit_root = WORKING_DIR / "logs" / "{experiment.lower()}-public953"
audit_root.mkdir(parents=True, exist_ok=False)
raw_csv = audit_root / "raw_submission.csv"
SUBMISSION_PATH.rename(raw_csv)
export = normalize_export_boundary(raw_csv, SUBMISSION_PATH, TEST_DIR, sorted(test_stems))
audit = validate_submission(SUBMISSION_PATH, TEST_DIR, sorted(test_stems))
stats = pd.read_csv(RUN_STATS_PATH)
if sorted(stats.dataset.tolist()) != sorted(test_stems):
    raise ValueError("Run statistics do not cover the actual test movies")
for name in ("repair_fallback", "deadline_degraded"):
    if name not in stats or stats[name].isna().any():
        raise ValueError("Missing runtime fallback diagnostics")
cache = Path(os.environ["BIOHUB_CACHE_DIR"])
predictor = REPO_DIR / "scripts" / "predict_unet_transformer.py"
predictor_text = predictor.read_text()
if "_v1284_refine(ds_path, t, arr, unet_out[:, f_idx])" not in predictor_text:
    raise ValueError("Coordinate refinement patch missing")
if "UNetNodeTransformer._index_features = _v1284_index" not in predictor_text:
    raise ValueError("Fractional feature sampling patch missing")
coverage = {{}}
for name in sorted(test_stems):
    path = cache / (name + ".npz")
    if not path.is_file():
        raise ValueError("Missing real-peak cache: " + name)
    with np.load(path, allow_pickle=False) as peak_file:
        fields = sorted(peak_file.files)
        if not {{"coords", "low_coords", "low_score"}}.issubset(fields):
            raise ValueError("Incomplete real-peak cache: " + name)
    coverage[name] = {{"bytes": path.stat().st_size, "fields": fields}}
if os.environ.get("V1284_MODE") != "candidate":
    raise ValueError("The published coordinate head was not enabled")
audit.update(experiment={experiment!r}, source_proof={proof!r},
             public_score=None, selection_policy="final_day_public_reference_transfer",
             actual_environment={{k: v for k, v in os.environ.items() if k.startswith(("BIOHUB_", "V1284_"))}},
             coordinate_head_sha256=hashlib.sha256(Path(os.environ["V1284_HEAD"]).read_bytes()).hexdigest(),
             predictor_sha256=hashlib.sha256(predictor.read_bytes()).hexdigest(),
             peak_cache_files=coverage,
             repair_fallback_movies=int((stats.repair_fallback != 0).sum()),
             deadline_degraded_movies=int((stats.deadline_degraded != 0).sum()),
             export=export)
if audit["coordinate_head_sha256"] != {HEAD_SHA256!r}:
    raise ValueError("Coordinate head changed during inference")
(audit_root / "output_audit.json").write_text(json.dumps(audit, indent=2) + "\\n")
stats.to_csv(audit_root / "run_stats.csv", index=False)
(audit_root / "run_summary.md").write_text("# {experiment} inference output\\n\\n```json\\n" + json.dumps(audit, indent=2) + "\\n```\\n")
print(json.dumps(audit, indent=2))
'''


def build(args):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+", args.kernel):
        raise ValueError("Invalid Kernel identifier")
    root = Path(__file__).resolve().parents[1]
    subprocess.run(["git", "diff", "--exit-code", "HEAD"], cwd=root, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    upstream = subprocess.check_output(["git", "rev-parse", "@{upstream}"], cwd=root, text=True).strip()
    if revision != upstream:
        raise ValueError("Commit and push before packaging")
    sources = read_sources(args.reference_notebook)
    metadata = json.loads(args.reference_notebook.with_name("kernel-metadata.json").read_text())
    if (metadata["id"] != SOURCE or set(metadata["dataset_sources"]) != DATASETS
            or metadata["competition_sources"] != ["biohub-cell-tracking-during-development"]
            or metadata["kernel_sources"] or metadata["model_sources"] or metadata["enable_internet"]):
        raise ValueError("Public input contract changed")
    proof = {"source_kernel": SOURCE, "source_notebook_sha256": SOURCE_SHA256,
             "git_commit": revision, "experiment": args.experiment,
             "config_overrides": PROFILES[args.experiment],
             "original_inference_cells_sha256": [sha(s.encode()) for s in sources],
             "head_sha256": HEAD_SHA256, "training_validator_packaged": False,
             "full64_promotion_claimed": False}
    cells = [{"cell_type": "markdown", "metadata": {}, "source":
              f"# {args.experiment}: fixed public 0.953 method\n\n"
              f"Source: https://www.kaggle.com/code/{SOURCE}\n\n"
              "Original models: Pilkwang Kim; harmonic fusion: Igor Zharov; "
              "coordinate head and public method: Anvith Pothula. "
              "Public source released under Apache 2.0; coordinate head CC0. "
              "External public score is not this account's score. "
              "Declared overrides: " + json.dumps(PROFILES[args.experiment])}]
    def code(text):
        compile(text, "packaged-cell", "exec")
        cells.append({"cell_type": "code", "metadata": {}, "source": text,
                      "outputs": [], "execution_count": None})
    code(sources[0])
    code(f"os.environ.update({PROFILES[args.experiment]!r})\n"
         "assert os.environ['BIOHUB_VALIDATOR_ENABLE'] == '0'\n"
         f"assert os.environ['BIOHUB_ILP_DIVISION_WEIGHT'] == {PROFILES[args.experiment].get('BIOHUB_ILP_DIVISION_WEIGHT', '1.2')!r}\n"
         f"assert os.environ['BIOHUB_READMIT_MIN_SCORE'] == {PROFILES[args.experiment].get('BIOHUB_READMIT_MIN_SCORE', '0.965')!r}\n")
    for source in sources[1:4]:
        code(source)
    code("import hashlib\nfrom pathlib import Path\n"
         "_head_roots = [Path('/kaggle/input/datasets/anvithpothula/biohub-v1284-head-s075/v1284_head.pt'),\n"
         "               Path('/kaggle/input/biohub-v1284-head-s075/v1284_head.pt')]\n"
         "_heads = sorted({p.resolve() for p in _head_roots if p.is_file()})\n"
         "if len(_heads) != 1:\n    raise ValueError('Expected one public coordinate head')\n"
         f"if hashlib.sha256(_heads[0].read_bytes()).hexdigest() != {HEAD_SHA256!r}:\n"
         "    raise ValueError('Public coordinate head checksum mismatch')\n")
    for source in sources[4:]:
        code(source)
    code(audit_source(args.experiment, proof))
    metadata.pop("id_no", None)
    metadata.update(id=args.kernel, title=args.kernel.split("/")[1].replace("-", " "),
                    code_file="submission.ipynb", is_private=True, enable_internet=False)
    notebook = {"nbformat": 4, "nbformat_minor": 4, "cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (("submission.ipynb", notebook), ("kernel-metadata.json", metadata), ("source_proof.json", proof)):
        (args.output_dir / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"kernel": args.kernel, "experiment": args.experiment,
                      "source_commit": revision, "notebook_sha256": sha((args.output_dir / "submission.ipynb").read_bytes())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-notebook", type=Path, required=True)
    parser.add_argument("--experiment", choices=PROFILES, required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    build(parser.parse_args())
