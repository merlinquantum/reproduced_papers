"""CIFAR-10 SSL utilities shared by qSSL and other projects."""

from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np
import torchvision
from PIL import ImageFilter
from torch.utils.data import Subset
from torchvision import transforms


class TwoCropsTransform:
    """Generate two augmented views for SimCLR-style training."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur:
    """Gaussian blur augmentation from SimCLR (https://arxiv.org/abs/2002.05709)."""

    def __init__(self, sigma: Iterable[float] | None = None):
        if sigma is None:
            sigma = [0.1, 2.0]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


def get_first_n_classes(dataset, n_classes: int):
    """Return a subset restricted to the first n CIFAR-10 labels."""

    targets = np.array(dataset.targets)
    indices = np.where(targets < n_classes)[0]
    return Subset(dataset, indices)


def _cifar_normalize():
    return transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )


def load_transformed_data(args):
    """SSL training dataset with strong augmentations and paired views."""

    normalize = _cifar_normalize()

    augmentation = [
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
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

    unique_labels = {train_dataset[i][1] for i in range(len(train_dataset))}
    print(f"SSL training dataset - Actual labels used: {sorted(unique_labels)}")
    return train_dataset


def load_finetuning_data(args):
    """Train/val splits for linear evaluation with light augments."""

    normalize = _cifar_normalize()

    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = torchvision.datasets.CIFAR10(
        root=args.datadir,
        train=True,
        download=False,
        transform=train_transforms,
    )
    train_dataset = get_first_n_classes(train_dataset, args.classes)
    unique_labels = {train_dataset[i][1] for i in range(len(train_dataset))}
    print(f"Fine-tuning training dataset - Actual labels used: {sorted(unique_labels)}")

    val_dataset = torchvision.datasets.CIFAR10(
        root=args.datadir, train=False, download=False, transform=val_transforms
    )
    val_dataset = get_first_n_classes(val_dataset, args.classes)
    unique_labels = {val_dataset[i][1] for i in range(len(val_dataset))}
    print(
        f"Fine-tuning validation dataset - Actual labels used: {sorted(unique_labels)}"
    )

    return train_dataset, val_dataset


def denormalize_tensor(tensor, mean, std):
    """Denormalize a tensor for visualization."""

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
