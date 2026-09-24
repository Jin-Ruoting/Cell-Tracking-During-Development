#!/usr/bin/env python3
"""Retrieve a verified own-Kaggle HOCT v1 artifact directly on Uestc-220.

Local mode passes a short-lived download URL over SSH stdin, never model bytes
or Kaggle credentials. Server mode downloads into logs and checks CPU loading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import urlopen

ROOT = Path("/data/zqjinruoting/Kaggle/Cell Tracking During Development")
SIZE = 25_496_698
SHA = "5bd836dfcb15ad796ea79a9595841a3e73b650a71c4acba3fc66aac65d745b33"
SOURCE = "b2b6828c07e815dabf8744bb99063d3844301336"
KERNEL = "buaaauto/biohub-hoct-scale-cpu-audit"
ARTIFACT = "logs/hoct-scale-cpu/general_v1.pt"


def validate_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in {"www.kaggleusercontent.com", "storage.googleapis.com"}
            or parsed.username or parsed.password or parsed.port not in (None, 443)):
        raise ValueError("Expected an official Kaggle output download host")


def relay(args):
    if Path(__file__).resolve().parents[1] == ROOT:
        raise RuntimeError("Relay runs locally with the existing Kaggle authentication")
    receipt = json.loads(args.receipt.read_text())
    if (receipt.get("passed") is not True
            or receipt["source_manifest"]["git_commit"] != SOURCE
            or receipt["v1_asset"]["sha256"] != SHA
            or receipt["v1_asset"]["bytes"] != SIZE
            or receipt["v1_asset"]["loaded_on_cpu"] is not True
            or receipt["model_forward_executed"] is not False):
        raise ValueError("Completed S223 asset receipt is required")
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest

    api = KaggleApi()
    api.authenticate()
    urls = []
    token = None
    with api.build_kaggle_client() as client:
        for _ in range(10):
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = KERNEL.split("/")
            api._set_paging(request, 200, token)
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            urls.extend(item.url for item in response.files or [] if item.file_name == ARTIFACT)
            token = response.next_page_token
            if not token:
                break
        else:
            raise ValueError("Unexpected output pagination")
    if len(urls) != 1:
        raise ValueError("Expected exactly one completed HOCT v1 output")
    validate_url(urls[0])
    payload = {"url": urls[0], "kernel": KERNEL, "artifact": ARTIFACT,
               "cpu_receipt_sha256": hashlib.sha256(args.receipt.read_bytes()).hexdigest(),
               "cpu_source_commit": SOURCE, "official_weight_sha256": SHA}
    command = ("cd " + shlex.quote(str(ROOT))
               + " && source /home/zqjinruoting/anaconda3/etc/profile.d/conda.sh"
               + " && conda activate Kaggle && python kaggle/fetch_hoct_v1_asset.py server --run-name "
               + shlex.quote(args.run_name))
    subprocess.run(["ssh", "Uestc-220", command], input=json.dumps(payload), text=True,
                   check=True, timeout=300)


def server(args):
    if Path(__file__).resolve().parents[1] != ROOT or Path(sys.prefix).name != "Kaggle":
        raise RuntimeError("Asset check requires Uestc-220 conda Kaggle")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Server tree must be clean")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if revision != subprocess.check_output(["git", "rev-parse", "@{upstream}"], text=True).strip():
        raise ValueError("Pull --ff-only before running")
    payload = json.loads(sys.stdin.read(100_000))
    if (payload["kernel"] != KERNEL or payload["artifact"] != ARTIFACT
            or payload["cpu_source_commit"] != SOURCE or payload["official_weight_sha256"] != SHA):
        raise ValueError("Unexpected artifact provenance")
    validate_url(payload["url"])
    folder = ROOT / "logs" / args.run_name
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {k: v for k, v in payload.items() if k != "url"}
    report.update(passed=False, git_commit=revision, device="cpu", model_forward_executed=False,
                  competition_input_read=False, quality_score=None, base_environment_modified=False)
    try:
        weight = folder / "general_v1.pt"
        digest = hashlib.sha256()
        size = 0
        with urlopen(payload["url"], timeout=20) as source, weight.open("xb") as output:
            validate_url(source.url)
            while True:
                if time.monotonic() - started > 180:
                    raise TimeoutError("Asset download budget exceeded")
                chunk = source.read(min(1024 * 1024, SIZE + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > SIZE:
                    raise ValueError("Official asset size exceeded")
                output.write(chunk)
                digest.update(chunk)
        if size != SIZE or digest.hexdigest() != SHA:
            raise ValueError("Official weight checksum/size mismatch")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        model = torch.jit.load(str(weight), map_location="cpu").eval()
        if any(p.device.type != "cpu" or not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Checkpoint parameter contract failed")
        shape = list(model.input_proj.weight.shape)
        parameters = sum(p.numel() for p in model.parameters())
        if shape != [288, 19] or parameters != 6_252_593:
            raise ValueError("Kaggle CPU checkpoint interface did not reproduce")
        report.update(passed=True, bytes=size, sha256=SHA, parameters=parameters,
                      input_projection_shape=shape, forward_schema=str(model.forward.schema),
                      torch_version=torch.__version__, loaded_on_cpu=True)
    except Exception as exc:
        # Signed URLs must never appear in public text, logs or process arguments.
        report["error_type"] = type(exc).__name__
        raise RuntimeError("HOCT v1 server acquisition/load failed; see sanitized receipt") from None
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        (folder / "runtime_receipt.json").write_text(text)
        (folder / "run_summary.md").write_text("# HOCT v1 server asset check\n\n```json\n" + text + "```\n")
        print(text, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("relay", "server"):
        current = sub.add_parser(name)
        current.add_argument("--run-name", required=True)
        if name == "relay":
            current.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_name):
        raise ValueError("Invalid run name")
    (relay if args.action == "relay" else server)(args)


if __name__ == "__main__":
    main()
