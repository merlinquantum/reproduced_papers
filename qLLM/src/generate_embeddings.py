#!/usr/bin/env python3
"""
Script to generate sentence transformer embeddings and save them as JSON files.
Uses the same data loading approach as main.py.
"""

import argparse
import json
import os

import numpy as np
from data_utils import load_data
from setfit_utils import load_model


def save_embeddings_to_json(
    embeddings: np.ndarray,
    labels: list[int],
    sentences: list[str],
    output_path: str,
    split_name: str,
):
    """
    Save embeddings, labels, and sentences to a JSON file.

    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        labels: list of integer labels
        sentences: list of sentence strings
        output_path: directory to save the file
        split_name: name of the split (train, eval, test)
    """
    data = {
        "embeddings": embeddings.tolist(),
        "labels": labels,
        "sentences": sentences,
        "embedding_dim": embeddings.shape[1],
        "num_samples": embeddings.shape[0],
    }

    os.makedirs(output_path, exist_ok=True)
    filename = os.path.join(output_path, f"{split_name}_embeddings.json")

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(labels)} {split_name} samples to {filename}")
    print(f"  - Embedding dimension: {embeddings.shape[1]}")
    print(f"  - Label distribution: {np.bincount(labels)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and save sentence transformer embeddings"
    )

    # Data parameters (same as main.py)
    parser.add_argument("--dataset", type=str, default="sst2", help="Dataset name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    # Model parameters
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-mpnet-base-v2",
        help="Pre-trained sentence transformer model name",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)"
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./embeddings",
        help="Directory to save embedding JSON files",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set up device
    import torch

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.verbose:
        print(f"Using device: {device}")
        print(f"Model: {args.model_name}")
        print(f"Output directory: {args.output_dir}")

    # Load data using the same function as main.py
    print("Loading data...")
    train_dataset, eval_dataset, test_dataset, features, labels = load_data(args)

    # Load the sentence transformer model
    print("Loading sentence transformer model...")
    model, sentence_transformer = load_model(args, device)
    sentence_transformer.eval()

    # Generate embeddings for each split
    print("Generating embeddings...")

    # Train embeddings
    train_sentences = train_dataset["sentence"]
    train_labels = train_dataset["label"]
    train_embeddings = sentence_transformer.encode(train_sentences)
    save_embeddings_to_json(
        train_embeddings, train_labels, train_sentences, args.output_dir, "train"
    )

    # Eval embeddings
    eval_sentences = eval_dataset["sentence"]
    eval_labels = eval_dataset["label"]
    eval_embeddings = sentence_transformer.encode(eval_sentences)
    save_embeddings_to_json(
        eval_embeddings, eval_labels, eval_sentences, args.output_dir, "eval"
    )

    # Test embeddings
    test_sentences = test_dataset["sentence"]
    test_labels = test_dataset["label"]
    test_embeddings = sentence_transformer.encode(test_sentences)
    save_embeddings_to_json(
        test_embeddings, test_labels, test_sentences, args.output_dir, "test"
    )

    print(f"\nAll embeddings saved to {args.output_dir}")
    print("\nTo load the embeddings back as datasets, use:")
    print("  from generate_embeddings import create_dataset_from_embeddings")
    print(
        f"  train_dataset = create_dataset_from_embeddings('{args.output_dir}', 'train')"
    )


if __name__ == "__main__":
    main()
