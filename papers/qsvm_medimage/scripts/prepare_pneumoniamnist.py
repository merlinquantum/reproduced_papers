"""Download PneumoniaMNIST and convert it to the project input format."""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

DATASET_URL = "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1"
DATASET_MD5 = "28209eda62fecd6e6a2d98b1501bb15f"
LABEL_NAMES = {0: "normal", 1: "pneumonia"}


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and file_md5(destination) == DATASET_MD5:
        print(f"Dataset already present: {destination}")
        return

    print(f"Downloading PneumoniaMNIST to {destination}...")
    with urlopen(DATASET_URL) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)

    actual_md5 = file_md5(destination)
    if actual_md5 != DATASET_MD5:
        destination.unlink()
        raise RuntimeError(
            f"Invalid download checksum: expected {DATASET_MD5}, got {actual_md5}"
        )


def build_dataframe(images: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    flattened_images = images.reshape(len(images), -1).astype(np.float32) / 255.0
    flat_labels = labels.reshape(-1)
    targets = [LABEL_NAMES[int(label)] for label in flat_labels]
    return pd.DataFrame(
        {
            "target": targets,
            "embedding": list(flattened_images),
        }
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the public PneumoniaMNIST chest X-ray dataset."
    )
    parser.add_argument(
        "--download_path",
        type=Path,
        default=Path("../../data/qsvm_medimage/pneumoniamnist.npz"),
        help="Location of the official NPZ archive",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("../../data/qsvm_medimage/pneumoniamnist_train.pkl"),
        help="Output pickle understood by the training scripts",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="train",
        help="Official split to convert (default: train)",
    )
    args = parser.parse_args(argv)

    download_dataset(args.download_path)
    with np.load(args.download_path) as dataset:
        splits = ["train", "val", "test"] if args.split == "all" else [args.split]
        images = np.concatenate([dataset[f"{split}_images"] for split in splits])
        labels = np.concatenate([dataset[f"{split}_labels"] for split in splits])

    dataframe = build_dataframe(images, labels)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_pickle(args.output_path)

    print(f"Saved {len(dataframe)} samples to {args.output_path}")
    print(f"Features: {len(dataframe.iloc[0]['embedding'])}")
    print(dataframe["target"].value_counts().to_string())


if __name__ == "__main__":
    main()
