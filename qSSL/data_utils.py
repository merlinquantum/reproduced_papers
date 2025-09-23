import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import ImageFilter
from torch.utils.data import Subset
from torchvision import transforms


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [0.1, 2.0]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_first_n_classes(dataset, n_classes):
    """Get indices of samples belonging to first n classes"""
    targets = np.array(dataset.targets)
    indices = np.where(targets < n_classes)[0]
    return Subset(dataset, indices)


# load data for the SSL training
# apply transformation to generate pairs of augmented images


def load_transformed_data(args):
    # Normalization for CIFAR
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )

    augmentation = [
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomApply(
            [
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
    ]

    full_dataset = torchvision.datasets.CIFAR10(
        root=args.datadir,
        train=True,
        download=True,
        transform=TwoCropsTransform(transforms.Compose(augmentation)),
    )
    train_dataset = get_first_n_classes(full_dataset, args.classes)
    # Print actual labels used after filtering
    unique_labels = {train_dataset[i][1] for i in range(len(train_dataset))}
    print(f"SSL training dataset - Actual labels used: {sorted(unique_labels)}")
    return train_dataset


# load training + validation data for the fine-tuning of the model
def load_finetuning_data(args):
    # Normalization for CIFAR (same as pre-training)
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )

    # Simple augmentations for fine-tuning
    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),  # Keep this basic augmentation
            transforms.RandomCrop(32, padding=4),  # Gentle spatial augmentation
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Minimal or no augmentation for validation
    val_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = torchvision.datasets.CIFAR10(
        root=args.datadir,
        train=True,
        download=False,
        transform=train_transforms,  # Note: Single transform, not TwoCropsTransform
    )
    train_dataset = get_first_n_classes(train_dataset, args.classes)
    # Print actual labels used after filtering
    unique_labels = {train_dataset[i][1] for i in range(len(train_dataset))}
    print(f"Fine-tuning training dataset - Actual labels used: {sorted(unique_labels)}")

    val_dataset = torchvision.datasets.CIFAR10(
        root=args.datadir, train=False, download=False, transform=val_transforms
    )
    val_dataset = get_first_n_classes(val_dataset, args.classes)
    # Print actual labels used after filtering
    unique_labels = {val_dataset[i][1] for i in range(len(val_dataset))}
    print(
        f"Fine-tuning validation dataset - Actual labels used: {sorted(unique_labels)}"
    )

    return train_dataset, val_dataset


def denormalize_tensor(tensor, mean, std):
    """Denormalize a tensor for visualization"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Display examples of SSL and finetuning datasets"
    )
    parser.add_argument("--datadir", default="./data", help="Data directory")
    parser.add_argument(
        "--classes", type=int, default=10, help="Number of classes to use"
    )
    args = parser.parse_args()

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
