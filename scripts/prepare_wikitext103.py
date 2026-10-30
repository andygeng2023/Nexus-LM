#!/usr/bin/env python3
"""Download and safely extract the WikiText-103 raw train/validation/test splits."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import shutil
import urllib.request
import zipfile

URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
SHA256 = "91c00ae287f0d699e18605c84afc9e45c192bc6b7797ff8837e5474655a33794"
SPLITS = {
    "wikitext-103-raw/wiki.train.raw": "train.txt",
    "wikitext-103-raw/wiki.valid.raw": "validation.txt",
    "wikitext-103-raw/wiki.test.raw": "test.txt",
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def download(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(URL, headers={"User-Agent": "NEXUS-LM-dataset-fetch/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as target:
            expected = int(response.headers.get("Content-Length", "0"))
            received = 0
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
                received += len(chunk)
                if expected:
                    print(f"\rDownload: {received / 1e6:.1f}/{expected / 1e6:.1f} MB", end="", flush=True)
        print()
        actual = sha256(temporary)
        if actual != SHA256:
            raise RuntimeError(f"archive checksum mismatch (expected {SHA256}, got {actual})")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

def safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            target = (destination / item.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe path in archive: {item.filename}")
        source.extractall(destination)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/wikitext-103"))
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs = {name: data_dir / filename for name, filename in SPLITS.items()}
    if all(path.exists() and path.stat().st_size > 100_000 for path in outputs.values()):
        print(f"WikiText-103 already prepared at {data_dir}")
        return 0
    archive = data_dir / "wikitext-103-raw-v1.zip"
    if not archive.exists() or sha256(archive) != SHA256:
        print(f"Fetching {URL}")
        download(archive)
    extracted = data_dir / "extracted"
    if extracted.exists():
        shutil.rmtree(extracted)
    extracted.mkdir(parents=True)
    safe_extract(archive, extracted)
    for member, filename in SPLITS.items():
        source = extracted / member
        if not source.is_file():
            raise RuntimeError(f"archive is missing expected split: {member}")
        target = outputs[member]
        shutil.copyfile(source, target)
        print(f"{filename}: {target.stat().st_size / 1e6:.1f} MB")
    (data_dir / "DATASET-ATTRIBUTION.txt").write_text(
        "WikiText-103 raw v1\n"
        "Original dataset: Salesforce AI Research WikiText\n"
        "Homepage: https://www.salesforce.com/blog/the-wikitext-long-term-dependency-language-modeling-dataset/\n"
        f"Archive: {URL}\nSHA-256: {SHA256}\n"
        "Dataset card licence metadata includes CC BY-SA 3.0 and GFDL; versioned\n"
        "dataset metadata has also stated CC BY-SA 4.0. Review the exact upstream\n"
        "terms and attribution requirements before redistributing derivatives.\n"
        "Paper: Merity et al. (2016), Pointer Sentinel Mixture Models, arXiv:1609.07843.\n",
        encoding="utf-8",
    )
    print(f"Prepared separate train/validation/test splits in {data_dir}")
    print("The training CLI does not consume the test split.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
