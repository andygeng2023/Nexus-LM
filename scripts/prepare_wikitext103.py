#!/usr/bin/env python3
"""Download and prepare the official WikiText-103 raw splits from Hugging Face."""
from __future__ import annotations

import argparse
from pathlib import Path

DATASET_ID = "Salesforce/wikitext"
CONFIG = "wikitext-103-raw-v1"
EXPECTED_ROWS = {"train": 1_801_350, "validation": 3_760, "test": 4_358}
OUTPUTS = {"train": "train.txt", "validation": "validation.txt", "test": "test.txt"}


def write_split(split, destination: Path, expected_rows: int) -> None:
    """Write the text column in dataset order, retaining blank rows."""
    if len(split) != expected_rows:
        raise RuntimeError(
            f"unexpected {destination.stem} row count: "
            f"expected {expected_rows}, got {len(split)}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for row in split:
                text = row.get("text")
                if not isinstance(text, str):
                    raise RuntimeError(f"dataset row in {destination.name} has no text string")
                output.write(text)
                output.write("\n")
        if temporary.stat().st_size < 100_000:
            raise RuntimeError(f"prepared split is unexpectedly small: {destination.name}")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/wikitext-103"))
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install it with 'python -m pip install datasets'"
        ) from exc

    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs = {split: data_dir / filename for split, filename in OUTPUTS.items()}
    if all(path.exists() and path.stat().st_size > 100_000 for path in outputs.values()):
        print(f"WikiText-103 already prepared at {data_dir}")
        return 0

    print(f"Loading {DATASET_ID} ({CONFIG}) from the Hugging Face dataset repository")
    dataset = load_dataset(DATASET_ID, CONFIG, trust_remote_code=False)
    for split_name, destination in outputs.items():
        if split_name not in dataset:
            raise RuntimeError(f"dataset repository is missing the {split_name!r} split")
        print(f"Preparing {split_name} ({len(dataset[split_name]):,} rows)")
        write_split(dataset[split_name], destination, EXPECTED_ROWS[split_name])
        print(f"{destination.name}: {destination.stat().st_size / 1e6:.1f} MB")

    (data_dir / "DATASET-ATTRIBUTION.txt").write_text(
        "WikiText-103 raw v1\n"
        "Dataset: Salesforce/wikitext on Hugging Face\n"
        "Dataset URL: https://huggingface.co/datasets/Salesforce/wikitext\n"
        "Configuration: wikitext-103-raw-v1\n"
        "The dataset card identifies CC BY-SA 4.0. Review upstream terms and "
        "attribution requirements before redistributing derivatives.\n"
        "Paper: Merity et al. (2016), Pointer Sentinel Mixture Models, "
        "arXiv:1609.07843.\n",
        encoding="utf-8",
    )
    print(f"Prepared separate train/validation/test splits in {data_dir}")
    print("The training CLI does not consume the test split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
