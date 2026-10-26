"""Distributed, resumable frame/episode-mixture sampling from clean metadata."""
import hashlib
import json
from pathlib import Path
import torch
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler


def load_episode_weights(path, dataset_root, dataset_len):
    raw = Path(path).read_bytes()
    m = json.loads(raw)
    root = Path(dataset_root).resolve()
    if m.get("schema_version") != 1 or root != Path(m["dataset_root"]).resolve():
        raise ValueError("Sampling manifest schema/root mismatch")
    if m["total_frames"] != dataset_len:
        raise ValueError("Sampling frame count mismatch")
    for rel, digest in m["metadata_sha256"].items():
        p = (root / rel).resolve()
        if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError("Sampling metadata changed: " + rel)
    episodes = m["episodes"]
    if not episodes or [e["episode_index"] for e in episodes] != list(range(len(episodes))):
        raise ValueError("Episode indices must be contiguous")
    lengths = torch.tensor([e["length"] for e in episodes], dtype=torch.long)
    if (lengths <= 0).any() or int(lengths.sum()) != dataset_len:
        raise ValueError("Invalid episode lengths")
    alpha = float(m["episode_uniform_fraction"])
    if not 0 <= alpha <= 1:
        raise ValueError("Invalid mixture fraction")
    per_frame = (1-alpha)/dataset_len + alpha/(len(episodes)*lengths.double())
    return torch.repeat_interleave(per_frame, lengths), hashlib.sha256(raw).hexdigest()


class EpisodeBalancedDistributedSampler(StatefulDistributedSampler):
    """Replacement draws from one global stream, strided across ranks."""
    def __init__(self, dataset, *, manifest_path, dataset_root, num_replicas,
                 rank, shuffle=True, seed=0, drop_last=False):
        if not shuffle:
            raise ValueError("Balanced sampling requires shuffle=True")
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                         shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.weights, self.manifest_sha256 = load_episode_weights(
            manifest_path, dataset_root, len(dataset))

    def __iter__(self):
        self.yielded = 0
        if self.next_yielded is not None:
            self.yielded = self.next_yielded
            self.next_yielded = None
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, True, generator=g)
        local = indices[self.rank:self.total_size:self.num_replicas]
        for idx in local[self.yielded:].tolist():
            self.yielded += 1
            yield idx

    def state_dict(self):
        return {**super().state_dict(), "epoch": self.epoch,
                "sampling_sha256": self.manifest_sha256, "world_size": self.num_replicas,
                "rank": self.rank, "seed": self.seed, "drop_last": self.drop_last}

    def load_state_dict(self, state):
        expected = {"sampling_sha256": self.manifest_sha256, "world_size": self.num_replicas,
                    "rank": self.rank, "seed": self.seed, "drop_last": self.drop_last}
        if any(state.get(k) != v for k, v in expected.items()):
            raise ValueError("Sampler checkpoint recipe/rank mismatch")
        if not isinstance(state.get("epoch"), int) or state["epoch"] < 0:
            raise ValueError("Invalid sampler epoch")
        if not isinstance(state.get("yielded"), int) or not 0 <= state["yielded"] <= self.num_samples:
            raise ValueError("Invalid sampler progress")
        self.epoch = state["epoch"]
        super().load_state_dict(state)
