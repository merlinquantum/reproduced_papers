import numpy as np
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from torch.utils.data import Dataset
import pandas as pd
import regex as re
from typing import Tuple
from torch.utils.data import DataLoader
import pennylane as qml
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor
import medmnist
from medmnist import INFO


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


def get_bas():
    try:
        [ds] = qml.data.load("other", name="bars-and-stripes")
        x_train = np.array(ds.train["4"]["inputs"])
        y_train = np.array(ds.train["4"]["labels"])
        x_test = np.array(ds.test["4"]["inputs"])
        y_test = np.array(ds.test["4"]["labels"])

        x_train, x_test = (
            x_train[:400].reshape(400, 4, 4),
            x_test[:200].reshape(200, 4, 4),
        )
        y_train, y_test = y_train[:400], y_test[:200]

        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0

        return TensorDataset(
            torch.tensor(x_train).unsqueeze(dim=1),
            torch.tensor(y_train),
        ), TensorDataset(torch.tensor(x_test).unsqueeze(dim=1), torch.tensor(y_test))

    except Exception as exc:
        print(f"Error loading PennyLane BAS dataset: {exc}")
        raise


def get_data_loader(
    dataset: TensorDataset, batch_size: int = None, shuffle: bool = True
) -> DataLoader:
    if batch_size is None:
        return DataLoader(dataset, shuffle=shuffle)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class HFImageDataset(Dataset):
    """
    Torch Dataset wrapper for HF datasets with image/label pairs.
    """

    def __init__(self, dataset, transform: callable = None):
        self.dataset = pd.DataFrame(dataset)[:500]
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


def get_perceval_challenge_MNIST(
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


def _to_int_label(y):
    if torch.is_tensor(y):
        y = y.reshape(-1)[0].item()
    try:
        y = y.item()
    except Exception:
        pass
    return int(y)


class BinaryBalancedSubset(Dataset):
    """
    Keep exactly n_per_class samples for two labels,
    remap them to {0,1}, and optionally shuffle order.
    """

    def __init__(self, base_dataset, keep_labels, n_per_class, seed=0, shuffle=True):
        assert len(keep_labels) == 2, "keep_labels must have exactly 2 labels"

        self.ds = base_dataset
        self.keep = list(keep_labels)
        self.map = {self.keep[0]: 0, self.keep[1]: 1}

        per = {self.keep[0]: [], self.keep[1]: []}

        # collect indices per class
        for i in range(len(self.ds)):
            _, y = self.ds[i]
            y = _to_int_label(y)
            if y in per:
                per[y].append(i)

        a, b = self.keep
        if len(per[a]) < n_per_class or len(per[b]) < n_per_class:
            raise ValueError(
                f"Not enough samples for requested n_per_class={n_per_class}. "
                f"Available: {a}->{len(per[a])}, {b}->{len(per[b])}."
            )

        rng = random.Random(seed)
        rng.shuffle(per[a])
        rng.shuffle(per[b])

        # pick exactly n_per_class from each
        chosen_a = per[a][:n_per_class]
        chosen_b = per[b][:n_per_class]

        # ordering depends on shuffle flag
        if shuffle:
            chosen = chosen_a + chosen_b
            rng.shuffle(chosen)
        else:
            # first all class0, then all class1
            chosen = chosen_a + chosen_b

        self.indices = chosen

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.ds[self.indices[idx]]
        y = _to_int_label(y)
        return x, self.map[y]


def dataset_to_tensordataset(dataset):
    X_list, Y_list = [], []
    for x, y in dataset:
        X_list.append(x)
        Y_list.append(int(y))

    X = torch.stack(X_list)
    Y = torch.tensor(Y_list).long()
    return TensorDataset(X, Y)


def get_binary_dataset(
    name: str = "MNIST",
    num_samples_per_class: int = 2000,
    eval_samples_per_class: int = 50,
    root: str = "./data/AA_study/",
    seed: int = 0,
    shuffle: bool = True,
):
    """
    Returns (train_tensor_ds, eval_tensor_ds) as TensorDataset objects.

    shuffle:
      - True  -> mixed order of both classes
      - False -> first all class0 samples, then all class1 samples

    Notes:
      - num_samples_per_class controls train size only.
      - eval_samples_per_class controls eval/test size only.
    """

    name_l = name.strip().lower()
    transform_32 = Compose([Resize((32, 32)), ToTensor()])

    # ---- MNIST ----
    if name_l == "mnist":
        train_base = datasets.MNIST(
            root=root, train=True, download=True, transform=transform_32
        )
        eval_base = datasets.MNIST(
            root=root, train=False, download=True, transform=transform_32
        )
        keep = [0, 1]

        train_bin = BinaryBalancedSubset(
            train_base, keep, num_samples_per_class, seed=seed, shuffle=shuffle
        )
        eval_bin = BinaryBalancedSubset(
            eval_base, keep, eval_samples_per_class, seed=seed + 1, shuffle=shuffle
        )

        return dataset_to_tensordataset(train_bin), dataset_to_tensordataset(eval_bin)

    # ---- CIFAR-10 ----
    if name_l in ["cifar10", "cifar-10"]:
        train_base = datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform_32
        )
        eval_base = datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform_32
        )
        keep = [0, 2]  # airplane vs bird

        train_bin = BinaryBalancedSubset(
            train_base, keep, num_samples_per_class, seed=seed, shuffle=shuffle
        )
        eval_bin = BinaryBalancedSubset(
            eval_base, keep, eval_samples_per_class, seed=seed + 1, shuffle=shuffle
        )

        return dataset_to_tensordataset(train_bin), dataset_to_tensordataset(eval_bin)

    # ---- EuroSAT ----
    if name_l in ["eurosat", "euro_sat", "euro-sat"]:
        base = datasets.EuroSAT(root=root, download=True, transform=transform_32)

        forest_idx = base.class_to_idx["Forest"]
        sealake_idx = base.class_to_idx["SeaLake"]
        keep = [forest_idx, sealake_idx]

        train_bin = BinaryBalancedSubset(
            base, keep, num_samples_per_class, seed=seed, shuffle=shuffle
        )
        eval_bin = BinaryBalancedSubset(
            base, keep, eval_samples_per_class, seed=seed + 1, shuffle=shuffle
        )

        return dataset_to_tensordataset(train_bin), dataset_to_tensordataset(eval_bin)

    # ---- PathMNIST ----
    if name_l in ["pathmnist", "path_mnist", "path-mnist"]:
        info = INFO["pathmnist"]
        DataClass = getattr(medmnist, info["python_class"])

        train_base = DataClass(
            split="train", download=True, root=root, transform=transform_32
        )
        eval_base = DataClass(
            split="test", download=True, root=root, transform=transform_32
        )

        # find adipose/background indices
        label_map = info["label"]
        inv = {str(v).lower(): int(k) for k, v in label_map.items()}

        adipose_idx = inv["adipose"]
        background_idx = inv["background"]

        keep = [adipose_idx, background_idx]

        train_bin = BinaryBalancedSubset(
            train_base, keep, num_samples_per_class, seed=seed, shuffle=shuffle
        )
        eval_bin = BinaryBalancedSubset(
            eval_base, keep, eval_samples_per_class, seed=seed + 1, shuffle=shuffle
        )

        return dataset_to_tensordataset(train_bin), dataset_to_tensordataset(eval_bin)

    raise ValueError("Unknown dataset name. Use MNIST, CIFAR10, EuroSAT, or PathMNIST.")
