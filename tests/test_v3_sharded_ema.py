#!/usr/bin/env python3
"""Tiny DTensor/DCP round trip; CPU multi-rank or CUDA single-rank."""
import argparse
import os
from pathlib import Path
import sys

ANT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANT / 'src/lingbot-vla-v2'))
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from lingbotvla.checkpoint.ema import ShardedEMA
from lingbotvla.checkpoint.checkpointer import DistributedCheckpointer
from lingbotvla.checkpoint.format_utils import dcp_to_torch_state_dict
from lingbotvla.checkpoint.training_stage import validate_stage_checkpoint

parser = argparse.ArgumentParser();parser.add_argument('--device', choices=['cpu','cuda'], default='cpu');args=parser.parse_args()
torch.set_num_threads(2)
if args.device == 'cuda':torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('nccl' if args.device == 'cuda' else 'gloo')
rank=dist.get_rank();world=dist.get_world_size()
mesh=init_device_mesh(args.device,(world,))
root=ANT/'results/v3_preparation'/('sharded_'+args.device+'_'+os.environ['MASTER_PORT'])
class Model(torch.nn.Module):
 def __init__(self,value):
  super().__init__();self.weight=torch.nn.Parameter(distribute_tensor(torch.full((8,4),value,device=args.device),mesh,[Shard(0)]))
model=Model(1.0);ema=ShardedEMA(model,.75)
assert ema.local(ema.shadow['weight']).device.type == 'cpu'
assert ema.local(ema.shadow['weight']).numel() == 32//world
with torch.no_grad():model.weight.add_(2)
ema.update()
assert torch.equal(ema.local(ema.shadow['weight']),torch.full((8//world,4),1.5))
contract={'stage':'tiny_sharded_resume', 'training':{'max_steps':2,'ema_decay':.75}}
optimizer=torch.optim.AdamW(model.parameters(), lr=.01)
state={'model':model,'ema':ema,'optimizer':optimizer,'extra_state':{'global_step':1,'ema_metadata':ema.metadata(),'stage_contract':contract}}
DistributedCheckpointer.save(str(root),state)
dist.barrier()
validate_stage_checkpoint(root,contract)
if world > 1:
 missing=root/'extra_state'/f'extra_state_rank_{world-1}.pt'
 held=missing.with_suffix('.held')
 if rank==0:missing.rename(held)
 dist.barrier()
 try:
  try:validate_stage_checkpoint(root,contract)
  except ValueError as exc:assert f'rank {world-1}:' in str(exc)
  else:raise AssertionError('one missing rank state must reject on every rank')
 finally:
  dist.barrier()
  if rank==0:held.rename(missing)
  dist.barrier()
if rank==0:
 dense=dcp_to_torch_state_dict(str(root),ema=True)
 assert torch.equal(dense['weight'],torch.full((8,4),1.5))
dist.barrier()
restored=Model(0.0);ema2=ShardedEMA(restored,.75)
state2={'model':restored,'ema':ema2,'optimizer':torch.optim.AdamW(restored.parameters(),lr=.01),'extra_state':{},'strict_optimizer':True}
DistributedCheckpointer.load(str(root),state2)
ema2.restore_metadata(state2['extra_state']['ema_metadata'],1)
with torch.no_grad():restored.weight.add_(4)
ema2.update()
assert torch.equal(ema2.local(ema2.shadow['weight']),torch.full((8//world,4),2.875))
dist.barrier()
if rank==0:print('PASS',args.device,'world_size',world,'CPU-sharded EMA arithmetic, DCP round trip, dense export, continuation and collective rejection',flush=True)
dist.destroy_process_group()
