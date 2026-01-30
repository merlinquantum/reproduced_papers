import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


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
    class_2_x_2 = np.random.normal(2, 1, num_samples_per_class)
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
