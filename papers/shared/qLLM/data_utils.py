"""
Data loading and preprocessing utilities for qLLM experiments.
"""

import json
import os
from typing import Any

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset


def load_data(args):
    """Load and prepare the dataset according to paper: 256 samples per label, 85% train, 15% val"""
    if args.verbose:
        print(f"\nLoading dataset '{args.dataset}' with paper sampling strategy...")

    dataset = load_dataset("SetFit/sst2")

    # Combine all splits (train, validation, test) for maximum data
    full_train_dataset = concatenate_datasets(
        [dataset["train"], dataset["validation"], dataset["test"]]
    )

    # Paper approach: 256 samples from each label {+, -}

    # Separate by label
    label_0_samples = [
        example for example in full_train_dataset if example["label"] == 0
    ]
    label_1_samples = [
        example for example in full_train_dataset if example["label"] == 1
    ]

    # Sample 256 from each label
    np.random.seed(args.seed)
    samples_per_label = 256

    if len(label_0_samples) < samples_per_label:
        raise ValueError(
            f"Not enough samples for label 0: {len(label_0_samples)} < {samples_per_label}"
        )
    if len(label_1_samples) < samples_per_label:
        raise ValueError(
            f"Not enough samples for label 1: {len(label_1_samples)} < {samples_per_label}"
        )

    selected_label_0 = np.random.choice(
        len(label_0_samples), samples_per_label, replace=False
    )
    selected_label_1 = np.random.choice(
        len(label_1_samples), samples_per_label, replace=False
    )

    sampled_data = []
    sampled_data.extend([label_0_samples[i] for i in selected_label_0])
    sampled_data.extend([label_1_samples[i] for i in selected_label_1])

    # Shuffle the combined data
    np.random.shuffle(sampled_data)

    # Split into 85% training and 15% validation
    total_samples = len(sampled_data)  # 512 samples
    train_size = int(0.85 * total_samples)  # 435 samples

    train_data = sampled_data[:train_size]
    val_data = sampled_data[train_size:]

    # Create datasets in the expected format
    train_dataset = {
        "sentence": [ex["text"] for ex in train_data],
        "label": [ex["label"] for ex in train_data],
    }
    eval_dataset = {
        "sentence": [ex["text"] for ex in val_data],
        "label": [ex["label"] for ex in val_data],
    }

    # Use remaining original training data as test set (N - 2×256 samples)
    # Track which original indices were used for sampling
    used_label_0_indices = set()
    used_label_1_indices = set()

    # Map selected indices back to original dataset indices
    label_0_idx = 0
    label_1_idx = 0
    for i, example in enumerate(full_train_dataset):
        if example["label"] == 0:
            if label_0_idx in selected_label_0:
                used_label_0_indices.add(i)
            label_0_idx += 1
        else:  # label == 1
            if label_1_idx in selected_label_1:
                used_label_1_indices.add(i)
            label_1_idx += 1

    used_indices = used_label_0_indices | used_label_1_indices
    remaining_indices = [
        i for i in range(len(full_train_dataset)) if i not in used_indices
    ]
    test_dataset = {
        "sentence": [full_train_dataset[i]["text"] for i in remaining_indices],
        "label": [full_train_dataset[i]["label"] for i in remaining_indices],
    }

    # Extract texts and labels for contrastive learning
    texts = train_dataset["sentence"]
    features = [texts]
    labels = torch.tensor(train_dataset["label"])

    if args.verbose:
        print("Paper sampling strategy applied:")
        print(f"- Sampled {samples_per_label} samples from each label")
        print(
            f"- Training: {len(train_dataset['sentence'])} samples ({len(train_dataset['sentence']) / total_samples * 100:.1f}%)"
        )
        print(
            f"- Validation: {len(eval_dataset['sentence'])} samples ({len(eval_dataset['sentence']) / total_samples * 100:.1f}%)"
        )
        print(f"- Test: {len(test_dataset['sentence'])} samples")
        print(f"- Train label distribution: {np.bincount(train_dataset['label'])}")
        print(f"- Val label distribution: {np.bincount(eval_dataset['label'])}")

    return train_dataset, eval_dataset, test_dataset, features, labels


def prepare_data(x_train, y_train, x_val, y_val, x_test, y_test):
    if not isinstance(x_train, torch.Tensor):
        x_train = torch.FloatTensor(x_train)
        y_train = torch.LongTensor(y_train)
        x_val = torch.FloatTensor(x_val)
        y_val = torch.LongTensor(y_val)
        x_test = torch.FloatTensor(x_test)
        y_test = torch.LongTensor(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_embeddings_from_json(file_path: str) -> dict[str, Any]:
    """
    Load embeddings from a JSON file.

    Args:
        file_path: path to the JSON file

    Returns:
        dictionary containing embeddings, labels, sentences, and metadata
    """
    with open(file_path) as f:
        data = json.load(f)

    # Convert embeddings back to numpy array
    data["embeddings"] = np.array(data["embeddings"])

    return data


def create_dataset_from_embeddings(embeddings_dir: str, split_name: str) -> Dataset:
    """
    Create a HuggingFace Dataset from saved embeddings.

    Args:
        embeddings_dir: directory containing the JSON files
        split_name: name of the split to load (train, eval, test)

    Returns:
        HuggingFace Dataset object
    """
    file_path = os.path.join(embeddings_dir, f"{split_name}_embeddings.json")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embeddings file not found: {file_path}")

    data = load_embeddings_from_json(file_path)

    dataset = Dataset.from_dict(
        {
            "sentence": data["sentences"],
            "label": data["labels"],
            "embedding": data[
                "embeddings"
            ].tolist(),  # Convert to list for Dataset compatibility
        }
    )

    return dataset
