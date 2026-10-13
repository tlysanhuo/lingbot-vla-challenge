"""Explicit weights-only stage initialization and strict within-stage resume."""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist


def build_stage_contract(args):
    if not args.train.training_stage:
        return None
    if args.train.load_checkpoint_path:
        output = Path(args.train.output_dir).resolve()
        checkpoint = Path(args.train.load_checkpoint_path).resolve()
        if not checkpoint.is_relative_to(output):
            raise ValueError('new stage: initialize model.model_path from v2; never resume its optimizer/scheduler')
    parent = Path(args.model.model_path).resolve()
    output = Path(args.train.output_dir).resolve()
    if parent.is_relative_to(output):
        raise ValueError('stage parent weights must be outside the new output directory')
    keys = ('max_steps', 'lr', 'lr_min', 'lr_decay_style', 'lr_warmup_ratio',
            'lr_decay_ratio', 'lr_start', 'optimizer', 'global_batch_size',
            'micro_batch_size', 'gradient_accumulation_steps', 'world_size',
            'enable_fp32', 'ema_decay', 'use_moe_expert_lr', 'loss_type',
            'weight_decay', 'max_grad_norm', 'seed', 'align_params')
    contract = {'schema_version': 1, 'stage': args.train.training_stage,
                'parent_weights': str(parent), 'model': dict(vars(args.model)),
                'data': dict(vars(args.data)),
                'training': {k: getattr(args.train, k, None) for k in keys}}
    # False preserves contracts saved before this opt-in data flag existed.
    # True must remain in the contract so incompatible resumes are rejected.
    if not contract['data'].get('teacher_clean_rgb', False):
        contract['data'].pop('teacher_clean_rgb', None)
    path = getattr(args.data, 'seen_bindings_file', None)
    if path:
        contract['seen_bindings_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    # Make the representation stable across serialization of tuple-like fields.
    return json.loads(json.dumps(contract, sort_keys=True, default=str))


def _validate_local_stage_checkpoint(checkpoint_path, contract):
    rank = dist.get_rank() if dist.is_initialized() else 0
    path = Path(checkpoint_path) / 'extra_state' / f'extra_state_rank_{rank}.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('stage_contract') != contract:
        raise ValueError('checkpoint belongs to another stage or a changed training recipe')
    step = int(saved['global_step'])
    if step < 0 or step > contract['training']['max_steps']:
        raise ValueError('checkpoint step is outside the stage budget')
    if not (Path(checkpoint_path) / 'optimizer/.metadata').is_file():
        raise ValueError('stage checkpoint has no complete optimizer metadata')
    if contract['training']['ema_decay']:
        if not (Path(checkpoint_path) / 'ema/.metadata').is_file():
            raise ValueError('stage checkpoint has no complete EMA metadata')
        meta = saved.get('ema_metadata')
        if not meta or meta.get('num_updates') != step or meta.get('decay') != contract['training']['ema_decay']:
            raise ValueError('EMA metadata does not match the stage checkpoint')


def validate_stage_checkpoint(checkpoint_path, contract):
    """All ranks agree to enter DCP load, including when one extra-state file is bad."""
    if contract is None:
        return
    error = None
    try:
        _validate_local_stage_checkpoint(checkpoint_path, contract)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [error]
    if dist.is_initialized() and dist.get_world_size() > 1:
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    failures = [f"rank {rank}: {message}" for rank, message in enumerate(errors) if message]
    if failures:
        raise ValueError("Stage checkpoint rejected before DCP load: " + "; ".join(failures))


def extra_stage_state(contract, ema):
    if contract is None and ema is None:
        return {}
    rng = {'python': random.getstate(), 'numpy': np.random.get_state()}
    if torch.cuda.is_initialized():
        rng['cuda'] = torch.cuda.get_rng_state(torch.cuda.current_device())
    return {'stage_contract': contract, 'ema_metadata': ema.metadata() if ema else None,
            'stage_rng_state': rng}


def restore_stage_state(extra, contract, ema):
    if contract is not None and extra.get('stage_contract') != contract:
        raise ValueError('training stage mismatch')
    if ema is not None:
        ema.restore_metadata(extra.get('ema_metadata'), extra['global_step'])
    if contract is not None:
        rng = extra['stage_rng_state']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        if 'cuda' in rng:
            torch.cuda.set_rng_state(rng['cuda'], torch.cuda.current_device())
