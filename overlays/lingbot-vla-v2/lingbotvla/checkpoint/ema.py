"""CPU-offloaded EMA of rank-local state, with ordinary DCP model layout."""
import torch
from torch.distributed.tensor import DTensor
from .checkpointer import ModelState


class ShardedEMA:
    _checkpoint_stateful = True

    def __init__(self, model, decay):
        if not 0 < decay < 1:
            raise ValueError('EMA decay must be between zero and one')
        self.decay = float(decay)
        self.num_updates = 0
        self.model_state = ModelState(model)
        self.shadow = {}
        with torch.no_grad():
            for name, value in self.model_state.state_dict()['model'].items():
                if not torch.is_tensor(value):
                    raise TypeError(f'EMA requires a tensor state: {name}')
                dtype = torch.float32 if value.is_floating_point() else value.dtype
                self.shadow[name] = value.detach().to(device='cpu', dtype=dtype).clone()

    @staticmethod
    def local(value):
        return value.to_local() if isinstance(value, DTensor) else value

    @torch.no_grad()
    def update(self):
        current = self.model_state.state_dict()['model']
        if current.keys() != self.shadow.keys():
            raise ValueError('model state changed after EMA initialization')
        for name, value in current.items():
            target = self.local(self.shadow[name])
            source = self.local(value.detach()).to(device='cpu', dtype=target.dtype)
            if target.shape != source.shape:
                raise ValueError(f'EMA local shard shape changed: {name}')
            if target.is_floating_point():
                target.lerp_(source, 1 - self.decay)
            else:
                target.copy_(source)
        self.num_updates += 1

    def state_dict(self):
        # Same nested keys as ModelState: existing raw/EMA HF exports work.
        return {'model': self.shadow}

    def load_state_dict(self, state):
        incoming = state['model']
        if incoming.keys() != self.shadow.keys():
            raise ValueError('EMA checkpoint keys do not match')
        for name, value in incoming.items():
            if value.shape != self.shadow[name].shape or value.dtype != self.shadow[name].dtype:
                raise ValueError(f'EMA checkpoint metadata mismatch: {name}')
            self.local(self.shadow[name]).copy_(self.local(value))

    def metadata(self):
        return {'decay': self.decay, 'num_updates': self.num_updates}

    def restore_metadata(self, metadata, global_step):
        if metadata is None or metadata.get('decay') != self.decay:
            raise ValueError('EMA decay/state missing or incompatible with this run')
        if metadata.get('num_updates') != global_step:
            raise ValueError('EMA update count must match optimizer steps')
        self.num_updates = int(metadata['num_updates'])
