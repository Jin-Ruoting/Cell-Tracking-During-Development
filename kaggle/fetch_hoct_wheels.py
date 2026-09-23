#!/usr/bin/env python3
"""Fetch fixed pure-Python HOCT reference wheels from official PyPI metadata."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

PACKAGES = {"hoct": "0.2.0", "spatial-graph": "0.1.1", "pooch": "1.9.0"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for package, version in PACKAGES.items():
        url = f"https://pypi.org/pypi/{package}/{version}/json"
        with urlopen(url, timeout=10) as response:
            metadata = json.load(response)
        if metadata["info"]["version"] != version:
            raise ValueError("PyPI version mismatch")
        choices = [item for item in metadata["urls"] if item["filename"].endswith("-py3-none-any.whl")
                   and item["packagetype"] == "bdist_wheel" and not item["yanked"]]
        if len(choices) != 1:
            raise ValueError(f"Expected one pure-Python wheel for {package}")
        wheel = choices[0]
        if urlparse(wheel["url"]).hostname != "files.pythonhosted.org" or Path(wheel["filename"]).name != wheel["filename"]:
            raise ValueError("Unexpected official wheel host or filename")
        target = args.output_dir / wheel["filename"]
        partial = target.with_suffix(target.suffix + ".part")
        if target.exists() or partial.exists():
            raise FileExistsError(target)
        with urlopen(wheel["url"], timeout=15) as response, partial.open("xb") as handle:
            while block := response.read(65536):
                handle.write(block)
                if handle.tell() > wheel["size"]:
                    raise ValueError("Download exceeds official size")
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
