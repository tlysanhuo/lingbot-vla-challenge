"""Checkpoint random transforms inside StatefulDataLoader workers."""
import random

import numpy as np
import torch
from torch.utils.data import get_worker_info


class StatefulAugmentationRNG:
    def state_dict(self):
        # With num_workers=0 the trainer owns the shared RNG, including diffusion
        # draws after fetching a batch. Its checkpoint restores that later state.
        if get_worker_info() is None:
            return {}
        numpy_state = np.random.get_state()
        return {"augmentation_rng": {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
            # Torchdata's incremental snapshot comparison understands tensors.
            "numpy": (numpy_state[0], torch.from_numpy(numpy_state[1].copy()),
                      *numpy_state[2:]),
        }}

    def load_state_dict(self, state):
        rng = state.get("augmentation_rng")
        if rng is None:
            return
        torch.set_rng_state(rng["torch"])
        random.setstate(rng["python"])
        numpy_state = rng["numpy"]
        np.random.set_state((numpy_state[0], numpy_state[1].numpy(), *numpy_state[2:]))
