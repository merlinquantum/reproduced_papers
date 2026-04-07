"""
File from the original repo
"""

from os import listdir
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def _load_openml_dataset(dataset: str):
    if dataset == "mnist":
        openml_name = "mnist_784"
    elif dataset == "fashion":
        openml_name = "Fashion-MNIST"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    data, targets = fetch_openml(
        openml_name,
        version=1,
        return_X_y=True,
        as_frame=False,
    )

    data = data.astype(np.float32).reshape(-1, 28, 28)
    targets = targets.astype(np.int64)

    return train_test_split(
        data,
        targets,
        test_size=10000,
        random_state=42,
        stratify=targets,
    )


def _normalize_pca_features(features: np.ndarray) -> np.ndarray:
    normalized = []
    for x in features:
        x_min = x.min()
        x_max = x.max()
        if x_max == x_min:
            normalized.append(np.zeros_like(x))
        else:
            normalized.append((x - x_min) * (np.pi / (x_max - x_min)))
    return np.asarray(normalized, dtype=np.float32)


def _limit_samples_per_class(
    x: np.ndarray,
    y: np.ndarray,
    samples_per_class: int | None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if samples_per_class is None:
        return x, y

    rng = np.random.default_rng(random_state)
    selected_indices = []

    for class_label in np.unique(y):
        class_indices = np.flatnonzero(y == class_label)
        if len(class_indices) < samples_per_class:
            raise ValueError(
                f"Requested {samples_per_class} samples for class {class_label}, "
                f"but only {len(class_indices)} are available."
            )
        chosen = rng.choice(class_indices, size=samples_per_class, replace=False)
        selected_indices.append(chosen)

    selected_indices = np.sort(np.concatenate(selected_indices))
    return x[selected_indices], y[selected_indices]


def data_load_and_process(
    dataset,
    feature_reduction="resize256",
    classes=[0, 1],
    samples_per_class: int | None = None,
    shuffle: bool = True,
    shuffle_seed: int = 42,
):
    """
    This part of the code was originally written to use Brain signal dataset.
    This implementation is currently out of interest; hence commented out.
    Will include this back later when needed.

    if dataset == 'signal':
        dataset_signal = pd.read_csv('/data/ROI_' +str(ROI)+ '_df_length256_zero_padding.csv')

        dataset_value = dataset_signal.iloc[:,:-1]
        dataset_label = dataset_signal.iloc[:,-1]

        x_train, x_test, y_train, y_test = train_test_split(dataset_value, dataset_label, test_size=0.2, shuffle=True,
                                                            stratify=dataset_label, random_state=10)

        x_train, x_test, y_train, y_test =\
            x_train.values.tolist(), x_test.values.tolist(), y_train.values.tolist(), y_test.values.tolist()
        y_train = [1 if y == 1 else -1 for y in y_train]
        y_test = [1 if y ==1 else -1 for y in y_test]
    """

    if dataset in {"mnist", "fashion"}:
        x_train, x_test, y_train, y_test = _load_openml_dataset(dataset)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0

    if len(classes) == 2:
        train_filter_tf = np.where((y_train == classes[0]) | (y_train == classes[1]))
        test_filter_tf = np.where((y_test == classes[0]) | (y_test == classes[1]))

    elif len(classes) == 3:  # For multicalss classification
        train_filter_tf = np.where(
            (y_train == classes[0]) | (y_train == classes[1]) | (y_train == classes[2])
        )
        test_filter_tf = np.where(
            (y_test == classes[0]) | (y_test == classes[1]) | (y_test == classes[2])
        )

    x_train, y_train = x_train[train_filter_tf], y_train[train_filter_tf]
    x_test, y_test = x_test[test_filter_tf], y_test[test_filter_tf]
    x_train, y_train = _limit_samples_per_class(x_train, y_train, samples_per_class)
    x_test, y_test = _limit_samples_per_class(
        x_test, y_test, samples_per_class, random_state=43
    )

    if feature_reduction == False:
        if shuffle:
            rng = np.random.default_rng(shuffle_seed)
            train_perm = rng.permutation(len(x_train))
            test_perm = rng.permutation(len(x_test))
            x_train, y_train = x_train[train_perm], y_train[train_perm]
            x_test, y_test = x_test[test_perm], y_test[test_perm]
        return (
            torch.as_tensor(x_train, dtype=torch.float32),
            torch.as_tensor(x_test, dtype=torch.float32),
            torch.as_tensor(y_train),
            torch.as_tensor(y_test),
        )

    if isinstance(feature_reduction, int):
        x_train_flat = x_train.reshape(len(x_train), -1)
        x_test_flat = x_test.reshape(len(x_test), -1)

        pca = PCA(feature_reduction)
        X_train = pca.fit_transform(x_train_flat)
        X_test = pca.transform(x_test_flat)

        x_train = _normalize_pca_features(X_train)
        x_test = _normalize_pca_features(X_test)
        if shuffle:
            rng = np.random.default_rng(shuffle_seed)
            train_perm = rng.permutation(len(x_train))
            test_perm = rng.permutation(len(x_test))
            x_train, y_train = x_train[train_perm], y_train[train_perm]
            x_test, y_test = x_test[test_perm], y_test[test_perm]
        return (
            torch.as_tensor(x_train, dtype=torch.float32),
            torch.as_tensor(x_test, dtype=torch.float32),
            torch.as_tensor(y_train),
            torch.as_tensor(y_test),
        )
