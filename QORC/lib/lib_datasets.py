#!/usr/bin/python3
# -*- coding: utf-8 -*-

import random
import numpy as np

import torch
from torch.utils.data import Dataset


class tensor_dataset(Dataset):
    def __init__(self, np_x, np_y, device, dtype, transform=None, n_side_pixels=None):
        if isinstance(np_x, torch.Tensor):
            self.np_x = np_x.detach().clone().to(device=device, dtype=dtype)
        else:
            self.np_x = torch.tensor(np_x, device=device, dtype=dtype)

        if isinstance(np_y, torch.Tensor):
            self.np_y = np_y.detach().clone().to(device=device, dtype=torch.long)
        else:
            self.np_y = torch.tensor(np_y, device=device, dtype=torch.long)

        self.n_items = self.np_x.shape[0]

        assert self.n_items == self.np_y.shape[0], (
            f"tensor_dataset: x and y do not have the same number of rows. "
            f"self.np_x.shape: {self.np_x.shape}, self.np_y.shape: {self.np_y.shape}"
        )

        self.transform = transform
        self.n_side_pixels = n_side_pixels

    def __getitem__(self, index):
        image = self.np_x[index]
        label = self.np_y[index]
        if self.transform:
            if self.n_side_pixels:
                n_pixels = self.n_side_pixels * self.n_side_pixels
                image = self.transform(
                    image.view(self.n_side_pixels, self.n_side_pixels)
                ).view(n_pixels)
            else:
                image = self.transform(image)
        return image, label

    def __len__(self):
        return self.n_items


def seed_worker(worker_id, seed=42):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory, seed=42):
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=lambda id: seed_worker(id, seed),
        generator=torch.Generator().manual_seed(seed),
    )


def split_fold_numpy(label, data, n_fold, fold_index, split_seed=-1):
    if split_seed >= 0:
        np.random.seed(split_seed)
        shuffled_indices = np.random.permutation(len(label))
        label = label[shuffled_indices]
        data = data[shuffled_indices]
    fold_size = len(label) // n_fold
    val_start = fold_index * fold_size
    val_end = (fold_index + 1) * fold_size if fold_index < n_fold - 1 else len(label)
    val_indices = np.arange(val_start, val_end)
    train_indices = np.array([i for i in range(len(label)) if i not in val_indices])
    return (
        label[val_indices],
        data[val_indices],
        label[train_indices],
        data[train_indices],
    )
