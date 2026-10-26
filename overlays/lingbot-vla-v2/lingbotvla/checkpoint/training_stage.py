"""Explicit weights-only stage initialization and strict within-stage resume."""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from lingbotvla.utils.flow_rng import FlowMatchingRNG


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
    if not contract['data'].get('complete_clean_probability', 0.0):
        contract['data'].pop('complete_clean_probability', None)
    if getattr(args.train, 'flow_rank_rng', False):
        contract['training']['flow_rank_rng'] = True
    # Keep v3/v4 contracts unchanged when the new training-only feature is off.
    if not contract['data'].get('local_erasing_probability', 0.0):
        for name in ('local_erasing_probability', 'local_erasing_area_min',
                     'local_erasing_area_max', 'local_erasing_camera'):
            contract['data'].pop(name, None)
    sampling = getattr(args.data, 'sampling_manifest', None)
    if sampling:
        contract['sampling_manifest_sha256'] = hashlib.sha256(Path(sampling).read_bytes()).hexdigest()
    else:
        contract['data'].pop('sampling_manifest', None)
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
    if contract['training'].get('flow_rank_rng', False):
        FlowMatchingRNG.validate_state(saved.get('flow_rng_state'),
            seed=contract['training']['seed'], dp_rank=rank,
            dp_size=contract['training']['world_size'])
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


def extra_stage_state(contract, ema, flow_rng=None):
    if contract is None and ema is None:
        return {}
    rng = {'python': random.getstate(), 'numpy': np.random.get_state()}
    if torch.cuda.is_initialized():
        rng['cuda'] = torch.cuda.get_rng_state(torch.cuda.current_device())
    extra = {'stage_contract': contract, 'ema_metadata': ema.metadata() if ema else None,
             'stage_rng_state': rng}
    if flow_rng is not None:
        extra['flow_rng_state'] = flow_rng.state_dict()
    elif contract is not None and contract['training'].get('flow_rank_rng', False):
        raise ValueError('Enabled flow RNG was not provided for checkpoint save')
    return extra


def restore_stage_state(extra, contract, ema, flow_rng=None):
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
    if flow_rng is not None:
        flow_rng.load_state_dict(extra.get('flow_rng_state'))
    elif contract is not None and contract['training'].get('flow_rank_rng', False):
        raise ValueError('Enabled flow RNG was not provided for checkpoint restore')
