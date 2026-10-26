"""Isolated, checkpointable flow draws generated outside the compiled model.

Keep the upstream sampling formula exactly, including sample_beta's two
uniform-power draws. This change separates streams; it does not change the
timestep distribution or touch initialization, DataLoader or inference RNGs.
"""
import torch


class FlowMatchingRNG:
    VERSION = 1

    def __init__(self, seed, dp_rank, dp_size, device):
        if not 0 <= dp_rank < dp_size:
            raise ValueError("Invalid data-parallel rank/size")
        self.seed = int(seed)
        self.dp_rank = int(dp_rank)
        self.dp_size = int(dp_size)
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed((self.seed + 1000003 * (self.dp_rank + 1)) % (2**63 - 1))
        self.calls = 0

    def sample(self, actions, state):
        if actions.device != state.device or state.device != self.device:
            raise ValueError("Flow RNG and input devices must match")
        noise = torch.randn(actions.shape, device=self.device, dtype=state.dtype,
                            generator=self.generator)
        gamma1 = torch.rand((actions.size(0),), device=self.device,
                            generator=self.generator).pow(1 / 1.5)
        gamma2 = torch.rand((actions.size(0),), device=self.device,
                            generator=self.generator).pow(1 / 1.0)
        time = (gamma1 / (gamma1 + gamma2) * 0.999 + 0.001).to(torch.float32).to(state.dtype)
        self.calls += 1
        return noise, time

    def state_dict(self):
        return {"version": self.VERSION, "seed": self.seed, "dp_rank": self.dp_rank,
                "dp_size": self.dp_size, "device_type": self.device.type,
                "calls": self.calls, "generator_state": self.generator.get_state().clone()}

    @classmethod
    def validate_state(cls, saved, *, seed, dp_rank, dp_size):
        if not isinstance(saved, dict):
            raise ValueError("Checkpoint is missing isolated flow RNG state")
        for key, expected in (("version", cls.VERSION), ("seed", seed),
                              ("dp_rank", dp_rank), ("dp_size", dp_size)):
            if saved.get(key) != expected:
                raise ValueError(f"Flow RNG {key} mismatch")
        rng = saved.get("generator_state")
        if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.ndim != 1 or not rng.numel():
            raise ValueError("Invalid flow generator state")
        if not isinstance(saved.get("calls"), int) or saved["calls"] < 0:
            raise ValueError("Invalid flow RNG call count")

    def load_state_dict(self, saved):
        self.validate_state(saved, seed=self.seed, dp_rank=self.dp_rank, dp_size=self.dp_size)
        if saved.get("device_type") != self.device.type:
            raise ValueError("Cannot resume flow RNG on a different device type")
        self.generator.set_state(saved["generator_state"].cpu())
        self.calls = saved["calls"]
