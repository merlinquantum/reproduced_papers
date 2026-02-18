import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

try:
    from papers.shared.qSSL.cifar import (
        denormalize_tensor,
        load_finetuning_data,
        load_transformed_data,
    )
    from runtime_lib.data_paths import paper_data_dir
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from papers.shared.qSSL.cifar import (
        denormalize_tensor,
        load_finetuning_data,
        load_transformed_data,
    )
    from runtime_lib.data_paths import paper_data_dir

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Display examples of SSL and finetuning datasets"
    )
    parser.add_argument("--datadir", default=None, help="Data directory")
    parser.add_argument(
        "--classes", type=int, default=10, help="Number of classes to use"
    )
    args = parser.parse_args()
    args.datadir = paper_data_dir("qSSL", args.datadir)

    # CIFAR-10 class names
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # Normalization values for denormalization
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    print("Loading SSL training dataset...")
    ssl_dataset = load_transformed_data(args)

    print("Loading finetuning datasets...")
    finetune_train, finetune_val = load_finetuning_data(args)

    # Display SSL training examples (augmented pairs)
    print("\n=== SSL Training Examples (Augmented Pairs) ===")
    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    fig.suptitle(
        "SSL Training (Augmented Pairs) And Linear Evaluation data", fontsize=14
    )

    for i in range(3):
        # Get an SSL sample (returns [query, key] pair and label)
        ssl_sample, label = ssl_dataset[i]

        # Display query image
        query_img = ssl_sample[0].clone()
        query_img = denormalize_tensor(query_img, mean, std)
        query_img = torch.clamp(query_img, 0, 1)
        axes[0, i * 2].imshow(query_img.permute(1, 2, 0))
        axes[0, i * 2].set_title(f"Query {i + 1}\n{class_names[label]}")
        axes[0, i * 2].axis("off")

        # Display key image (augmented version)
        key_img = ssl_sample[1].clone()
        key_img = denormalize_tensor(key_img, mean, std)
        key_img = torch.clamp(key_img, 0, 1)
        axes[0, i * 2 + 1].imshow(key_img.permute(1, 2, 0))
        axes[0, i * 2 + 1].set_title(f"Key {i + 1}\n{class_names[label]}")
        axes[0, i * 2 + 1].axis("off")

    # Display finetuning training examples
    print("\n=== Finetuning Training Examples ===")
    for i in range(3):
        # Get a finetuning sample
        finetune_img, label = finetune_train[i]

        # Display finetuning image
        ft_img = finetune_img.clone()
        ft_img = denormalize_tensor(ft_img, mean, std)
        ft_img = torch.clamp(ft_img, 0, 1)
        axes[1, i * 2].imshow(ft_img.permute(1, 2, 0))
        axes[1, i * 2].set_title(f"Train {i + 1}\n{class_names[label]}")
        axes[1, i * 2].axis("off")

        # Display validation image
        val_img, val_label = finetune_val[i]
        val_img = val_img.clone()
        val_img = denormalize_tensor(val_img, mean, std)
        val_img = torch.clamp(val_img, 0, 1)
        axes[1, i * 2 + 1].imshow(val_img.permute(1, 2, 0))
        axes[1, i * 2 + 1].set_title(f"Val {i + 1}\n{class_names[val_label]}")
        axes[1, i * 2 + 1].axis("off")

    plt.tight_layout()
    plt.show()

    # Print dataset statistics
    print("\n=== Dataset Statistics ===")
    print(f"SSL dataset size: {len(ssl_dataset)}")
    print(f"Finetuning train size: {len(finetune_train)}")
    print(f"Finetuning val size: {len(finetune_val)}")
    print(f"Using {args.classes} classes")

    # Show a few more individual examples with detailed info
    print("\n=== Detailed SSL Example ===")
    sample_idx = 0
    ssl_sample, label = ssl_dataset[sample_idx]
    print(f"Label: {label} ({class_names[label]})")
    print(f"Query shape: {ssl_sample[0].shape}")
    print(f"Key shape: {ssl_sample[1].shape}")
    print(
        f"Query tensor stats - min: {ssl_sample[0].min():.3f}, max: {ssl_sample[0].max():.3f}"
    )
    print(
        f"Key tensor stats - min: {ssl_sample[1].min():.3f}, max: {ssl_sample[1].max():.3f}"
    )

    print("\n=== Detailed Finetuning Example ===")
    ft_sample, ft_label = finetune_train[sample_idx]
    print(f"Label: {ft_label} ({class_names[ft_label]})")
    print(f"Image shape: {ft_sample.shape}")
    print(f"Tensor stats - min: {ft_sample.min():.3f}, max: {ft_sample.max():.3f}")
