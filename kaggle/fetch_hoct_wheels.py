#!/usr/bin/env python3
"""Fetch fixed, platform-compatible HOCT wheels from official PyPI metadata."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from urllib.parse import urlparse

from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

PACKAGES = {"hoct": "0.2.0", "spatial-graph": "0.1.1", "pooch": "1.9.0", "gurobipy": "12.0.3"}


def fetch(url, size_limit, output=None):
    """Bound the entire request, including resolution, using verified IPv4 transport."""
    if urlparse(url).scheme != "https":
        raise ValueError("Official downloads must use HTTPS")
    # Native wheels are much larger than metadata. S191 transferred 1.53 MB
    # in 15 seconds; budget at least 64 KiB/s, with a five-minute file cap.
    seconds = 15 if output is None else max(30, min(300, math.ceil(size_limit / 65536) + 15))
    try:
        result = subprocess.run(
            ["curl", "-4", "--proto", "=https", "--fail", "--silent", "--show-error",
             "--connect-timeout", "5", "--max-time", str(seconds), "--speed-limit", "16384",
             "--speed-time", "30", "--max-filesize", str(size_limit), url],
            check=True, stdout=output if output is not None else subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=seconds + 5,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Official download failed: {exc.stderr.decode(errors='replace')}") from exc
    if output is None and len(result.stdout) > size_limit:
        raise ValueError("Metadata exceeds the bounded download size")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    supported = {tag: index for index, tag in enumerate(sys_tags())}
    for package, version in PACKAGES.items():
        url = f"https://pypi.org/pypi/{package}/{version}/json"
        metadata = json.loads(fetch(url, 2 * 1024 * 1024))
        if metadata["info"]["version"] != version:
            raise ValueError("PyPI version mismatch")
        choices = []
        for item in metadata["urls"]:
            if item["packagetype"] != "bdist_wheel" or item["yanked"]:
                continue
            name, wheel_version, _, tags = parse_wheel_filename(item["filename"])
            compatible = tags.intersection(supported)
            if (canonicalize_name(name) == canonicalize_name(package)
                    and str(wheel_version) == version and compatible):
                choices.append((min(supported[tag] for tag in compatible), item["filename"], item))
        if not choices:
            raise ValueError(f"No compatible official wheel for {package} {version}")
        wheel = min(choices, key=lambda value: value[:2])[2]
        if urlparse(wheel["url"]).hostname != "files.pythonhosted.org" or Path(wheel["filename"]).name != wheel["filename"]:
            raise ValueError("Unexpected official wheel host or filename")
        target = args.output_dir / wheel["filename"]
        partial = target.with_suffix(target.suffix + ".part")
        if target.exists() or partial.exists():
            raise FileExistsError(target)
        with partial.open("xb") as handle:
            fetch(wheel["url"], wheel["size"], output=handle)
        digest = hashlib.sha256(partial.read_bytes()).hexdigest()
        if partial.stat().st_size != wheel["size"] or digest != wheel["digests"]["sha256"]:
            raise ValueError("Official wheel size/hash mismatch")
        partial.rename(target)
        records[package] = {"version": version, "url": wheel["url"], "filename": wheel["filename"],
                            "bytes": wheel["size"], "sha256": digest}
        (args.output_dir.parent / "download_receipt.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"Verified {package} {version}: {wheel['size']} bytes", flush=True)


if __name__ == "__main__":
    main()
