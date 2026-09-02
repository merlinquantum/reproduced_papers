"""Training loop with the paper's validation-plateau convergence criterion.

Ported from ``utils/trainer.py``.  Adam + MSE, mini-batches over a shuffled
training set, tracking the best-validation model.  The original convergence
rule requires at least 400 epochs and then compares the mean validation loss of
the two most recent 200-epoch windows.  Because full convergence typically takes
thousands of epochs (see the paper's "Epochs to Convergence" statistics), a
hard ``max_epochs`` cap is provided so that reduced-compute reproductions run a
fixed, fair epoch budget for every model.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils import shuffle


class Trainer:
    def __init__(
        self,
        model,
        random_id=42,
        learning_rate=0.001,
        batch_size=64,
        max_epochs=None,
        min_epochs=400,
        window=200,
        use_convergence=True,
    ):
        self.model = model
        self.random_id = random_id
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.window = window
        self.use_convergence = use_convergence
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.cost = nn.MSELoss()

    def check_convergence(self, cost_validation: list[float]) -> bool:
        min_len = 2 * self.window
        if len(cost_validation) < min_len:
            return False
        mean1 = np.mean(cost_validation[-2 * self.window : -self.window])
        mean2 = np.mean(cost_validation[-self.window :])
        std2 = np.std(cost_validation[-self.window :])
        return np.abs(mean1 - mean2) < std2 / 2

    def train(self, xtr, ytr, xval, yval, xte, yte):
        t0 = time.time()
        cost_tr, cost_val, cost_te = [], [], []
        min_val = np.inf
        best_state = None
        epoch = 0
        while True:
            epoch += 1
            xtr, ytr = shuffle(xtr, ytr, random_state=self.random_id)
            total, nb = 0.0, 0
            for j in range(0, len(xtr), self.batch_size):
                xb, yb = xtr[j : j + self.batch_size], ytr[j : j + self.batch_size]
                self.optimizer.zero_grad()
                loss = self.cost(self.model(xb), yb)
                loss.backward()
                self.optimizer.step()
                total += loss.item()
                nb += 1
            cost_tr.append(total / nb)
            with torch.no_grad():
                lval = self.cost(self.model(xval), yval).item()
                lte = self.cost(self.model(xte), yte).item()
            cost_val.append(lval)
            cost_te.append(lte)
            if lval < min_val:
                min_val = lval
                best_state = copy.deepcopy(self.model.state_dict())

            converged = False
            if self.use_convergence and epoch >= self.min_epochs:
                converged = self.check_convergence(cost_val)
            if self.max_epochs is not None and epoch >= self.max_epochs:
                converged = True
            if converged:
                break

        total_time = round(time.time() - t0, 2)
        return {
            "cost_training": cost_tr,
            "cost_validation": cost_val,
            "cost_testing": cost_te,
            "model_end": self.model.state_dict(),
            "model_best_validation": best_state,
            "total_time": total_time,
            "epochs": epoch,
        }
