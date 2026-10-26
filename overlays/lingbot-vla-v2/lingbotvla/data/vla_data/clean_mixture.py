"""One train-only decision shared by all views, frames and state augmentation."""
import math

import torch


def sample_complete_clean(probability, *, policy_eval=False):
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("complete_clean_probability must be finite and in [0, 1]")
    # Preserve the exact legacy random stream when disabled or used at inference.
    if policy_eval or probability == 0.0:
        return False
    if probability == 1.0:
        return True
    return bool(torch.rand(()).item() < probability)
