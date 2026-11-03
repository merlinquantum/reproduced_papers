"""Checkpointable upstream InfiniteSampler order shared by pool priors."""

import numpy as np
import torch


class PoolStream:
    """Replay a shuffled pool with exact permutation, cursor, and RNG state."""

    def sample(self, batch_size, device="cpu"):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        indices = np.empty(batch_size, dtype=np.int64)
        # Yield before swapping, as upstream; applying the swap now preserves
        # the next yielded value while making the checkpoint boundary explicit.
        offsets = (
            self.rng.randint(self.window, size=batch_size) if self.window >= 2 else None
        )
        for index in range(batch_size):
            i = self.position % len(self.order)
            indices[index] = self.order[i]
            if offsets is not None:
                j = (i - int(offsets[index])) % len(self.order)
                self.order[i], self.order[j] = self.order[j], self.order[i]
            self.position += 1
        return self.values[torch.from_numpy(indices)].to(device)

    def state_dict(self):
        name, keys, pos, has_gauss, cached = self.rng.get_state()
        return {
            "metadata": self.metadata,
            "order": torch.from_numpy(self.order.copy()),
            "position": self.position,
            "rng_name": name,
            "rng_keys": torch.from_numpy(keys.astype(np.int64)),
            "rng_position": pos,
            "rng_has_gauss": has_gauss,
            "rng_cached_gaussian": cached,
        }

    def load_state_dict(self, state):
        if state["metadata"] != self.metadata:
            raise ValueError("Received pool or normalization changed since checkpoint")
        order = state["order"].cpu().numpy().copy()
        if order.shape != self.order.shape or not np.array_equal(
            np.sort(order), np.arange(len(order))
        ):
            raise ValueError("Invalid saved sample permutation")
        self.order = order
        self.position = state["position"]
        self.rng.set_state(
            (
                state["rng_name"],
                state["rng_keys"].cpu().numpy().astype(np.uint32),
                state["rng_position"],
                state["rng_has_gauss"],
                state["rng_cached_gaussian"],
            )
        )
