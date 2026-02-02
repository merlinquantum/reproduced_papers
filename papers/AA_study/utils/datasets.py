import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from torch.utils.data import Dataset
import pandas as pd
import regex as re
from typing import Tuple
from torch.utils.data import DataLoader


def generate_fig_1_dataset(
    num_samples_per_class: int = 2000,
    shuffle: bool = True,
    seed: int | None = None,
) -> TensorDataset:
    class_1 = np.random.uniform(4, 5, num_samples_per_class * 2).reshape(
        num_samples_per_class, 2
    )
    class_2 = np.random.uniform(5.5, 6.5, num_samples_per_class * 2).reshape(
        num_samples_per_class, 2
    )

    features = np.vstack([class_1, class_2])
    labels = np.concatenate(
        [np.zeros(num_samples_per_class), np.ones(num_samples_per_class)]
    )

    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(features))
        features = features[indices]
        labels = labels[indices]

    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return TensorDataset(features_tensor, labels_tensor)


def generate_fig_2_dataset(
    num_samples_per_class: int = 2000,
    shuffle: bool = True,
    seed: int | None = None,
) -> TensorDataset:
    class_1 = np.random.uniform(-1, 1, num_samples_per_class * 2).reshape(
        num_samples_per_class, 2
    )
    class_2 = np.random.uniform(3, 5, num_samples_per_class * 2)
    sign_generator = [
        -1 if i < 0.5 else 1 for i in np.random.random(num_samples_per_class * 2)
    ]
    class_2 *= sign_generator
    class_2 = class_2.reshape(num_samples_per_class, 2)

    features = np.vstack([class_1, class_2])
    labels = np.concatenate(
        [np.zeros(num_samples_per_class), np.ones(num_samples_per_class)]
    )

    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(features))
        features = features[indices]
        labels = labels[indices]

    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return TensorDataset(features_tensor, labels_tensor)


def generate_fig_3_dataset(
    num_samples_per_class: int = 2000,
    shuffle: bool = True,
    seed: int | None = None,
) -> TensorDataset:
    class_1_x_1 = np.random.uniform(-3, -1, num_samples_per_class)
    class_1_x_2 = np.random.normal(-2, 1, num_samples_per_class)
    class_1 = [[i, j] for i, j in zip(class_1_x_1, class_1_x_2)]

    class_2_x_1 = np.random.uniform(1, 3, num_samples_per_class)
    class_2_x_2 = np.random.normal(10, 11, num_samples_per_class)
    class_2 = [[i, j] for i, j in zip(class_2_x_1, class_2_x_2)]

    features = np.vstack([class_1, class_2])
    labels = np.concatenate(
        [np.zeros(num_samples_per_class), np.ones(num_samples_per_class)]
    )

    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(features))
        features = features[indices]
        labels = labels[indices]

    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return TensorDataset(features_tensor, labels_tensor)


def get_data_loader(
    dataset: TensorDataset, batch_size: int = 50, shuffle: bool = True
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class HFImageDataset(Dataset):
    """
    Torch Dataset wrapper for HF datasets with image/label pairs.
    """

    def __init__(self, dataset, transform: callable = None):
        self.dataset = pd.DataFrame(dataset)
        self.transform = transform

    def __len__(self) -> int:
        l = len(self.dataset["image"])
        return l

    def __getitem__(self, idx: int):
        """
        Return one sample from the dataset.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple[torch.Tensor, int]
            Image tensor and its label.
        """
        img = self.dataset["image"].iloc[idx]
        label = self.dataset["label"].iloc[idx]
        # string to list
        img_list = re.split(r",", img)
        # remove '[' and ']'
        img_list[0] = img_list[0][1:]
        img_list[-1] = img_list[-1][:-1]
        # convert to float
        img_float = [float(el) for el in img_list]
        # convert to image
        img_square = torch.unflatten(torch.tensor(img_float), 0, (1, 28, 28))
        if self.transform is not None:
            img_square = self.transform(img_square)
        return img_square, label


def create_known_datasets(
    batch_size: int = 600,
) -> Tuple[Dataset, Dataset, DataLoader, DataLoader]:
    """
    Create MNIST train/validation datasets and data loaders.
    Parameters
    ----------
    batch_size: int
        The number of elements per batches. Default is 128

    Returns
    -------
    Tuple[Dataset, Dataset, DataLoader, DataLoader]
        Train dataset, validation dataset, train loader and validation loader.
    """
    train_dataset = HFImageDataset(
        load_dataset("Quandela/PercevalQuest-MNIST", split="train")
    )
    val_dataset = HFImageDataset(
        load_dataset("Quandela/PercevalQuest-MNIST", split="validation")
    )
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    return train_dataset, val_dataset, train_loader, val_loader
