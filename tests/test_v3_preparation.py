#!/usr/bin/env python3
"""Small CPU checks of binding safety, EMA arithmetic, and stage recovery."""
import copy
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest

ANT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANT / 'src/lingbot-vla-v2'))
import numpy as np
import torch
import torch.distributed as dist
from lingbotvla.data.vla_data.seen_bindings import SeenBindingBuilder, SeenInstructionBindings, episode_digest
from lingbotvla.checkpoint.ema import ShardedEMA
from lingbotvla.checkpoint.checkpointer import DistributedCheckpointer
from lingbotvla.checkpoint.format_utils import dcp_to_torch_state_dict
from lingbotvla.checkpoint.training_stage import build_stage_contract, validate_stage_checkpoint, extra_stage_state, restore_stage_state

ROOT = ANT / 'results/v3_preparation' / ('tests_' + str(os.getpid()))
ROOT.mkdir(parents=True)


def args_for(path):
    return SimpleNamespace(
        model=SimpleNamespace(model_path=str(ROOT / 'parent/hf_ckpt')),
        data=SimpleNamespace(seen_bindings_file=None, train_path='clean2500'),
        train=SimpleNamespace(training_stage='v3_test', load_checkpoint_path=None,
                              output_dir=str(path), max_steps=6, lr=.01, lr_min=.001,
                              lr_decay_style='cosine', ema_decay=.9))


class BindingTests(unittest.TestCase):
    def test_entities_and_arm_preserved(self):
        builder = SeenBindingBuilder({'pick': ['Use {a} to pick {A}.', 'Pick {A} with {a}.',
                                               'Use the right arm to pick {A}.', 'Pick {A}.']})
        record = builder.bind('Use the left arm to pick the red cup.')
        self.assertEqual(record['status'], 'bound')
        self.assertEqual(record['candidates'], ['Use the left arm to pick the red cup.', 'Pick the red cup with the left arm.'])

    def test_rejects_false_arm_and_wrong_object_type(self):
        b = SeenBindingBuilder({'place_bread_skillet': ['Put {B} on {A} with {a}.']},
                               {'the shoe':['shoe'], 'the box':['shoe-box']})
        self.assertEqual(b.bind('Put the shoe on the box with the left arm.')['status'], 'unmatched')
        b = SeenBindingBuilder({'pick':['Use {a} to pick {A}.']})
        self.assertEqual(b.bind('Use ridges with the right arm to pick the bottle.')['status'], 'unmatched')

    def test_does_not_add_spatial_goal(self):
        b = SeenBindingBuilder({'place_a2b_left':['Place {A} on {B}.','Set {A} onto {B}.','Place {A} inside {B}.','Place {A} on the left of {B}.']})
        r=b.bind('Place the cup on the plate.')
        self.assertEqual(r['candidates'],['Place the cup on the plate.','Set the cup onto the plate.'])

    def test_literal_bimanual_roles_keep_original(self):
        b=SeenBindingBuilder({'pick_dual_bottles':['Pick {A} with the left arm and {B} with the right arm.']})
        text='Pick the red bottle with the left arm and the green bottle with the right arm.'
        self.assertEqual(b.bind(text)['candidates'],[text])

    def test_entity_order_is_preserved(self):
        b=SeenBindingBuilder({'place_a2b_left':['Place {A} on {B}.','On {B}, place {A}.']})
        self.assertEqual(b.bind('Place the cup on the plate.')['candidates'],['Place the cup on the plate.'])

    def test_ambiguous_task_keeps_original(self):
        record = SeenBindingBuilder({'a': ['Pick {A}.'], 'b': ['Pick {A}.']}).bind('Pick the cup.')
        self.assertEqual(record['status'], 'ambiguous')
        self.assertEqual(record['candidates'], ['Pick the cup.'])

    def test_manifest_binds_episode_not_just_task(self):
        episodes = [{'episode_index': 0, 'tasks': ['Original'], 'length': 100}]
        p = ROOT / 'binding.json'
        p.write_text(json.dumps({'schema_version': 1, 'source_pool': 'release_seen_only',
                                'episode_metadata_sha256': episode_digest(episodes),
                                'episodes': {'0': {'original': 'Original', 'candidates': ['Original', 'Alternative']}}}))
        binding = SeenInstructionBindings(p)
        with self.assertRaises(RuntimeError): binding.sample(0, 'Original')
        binding.validate_episodes(episodes)
        with self.assertRaises(ValueError): binding.sample(1, 'Original')
        with self.assertRaises(ValueError): binding.sample(0, 'Another object')
        generator = torch.Generator().manual_seed(7)
        self.assertEqual({binding.sample(0, 'Original', generator) for _ in range(30)}, {'Original', 'Alternative'})
        with self.assertRaises(ValueError): binding.validate_episodes([{'episode_index': 0, 'tasks': ['Other'], 'length': 100}])


class StateNoiseTests(unittest.TestCase):
    def test_only_valid_arm_inputs_change(self):
        from lingbotvla.data.vla_data.transform import perturb_arm_state
        state = torch.ones(55)
        mask = torch.zeros(55, dtype=torch.bool); mask[:12] = True
        noise = perturb_arm_state(state, mask, .01, 1.0, torch.Generator().manual_seed(5))
        self.assertTrue(torch.equal(noise[12:], state[12:]))
        self.assertFalse(torch.equal(noise[:12], state[:12]))
        before = torch.get_rng_state().clone()
        self.assertIs(perturb_arm_state(state, mask, .01, 0), state)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


from lingbotvla.data.vla_data.base_dataset import VLADataset


class TinyRandomVLADataset(VLADataset):
    """Exercise production dataset checkpoint hooks with tiny random samples."""
    def __init__(self):
        self.bindings = SeenInstructionBindings(ROOT / 'worker_bindings.json')
        self.bindings.validate_episodes([{'episode_index': 0, 'tasks': ['Original'], 'length': 48}])

    def __len__(self):
        return 48

    def __getitem__(self, index):
        from lingbotvla.data.vla_data.transform import perturb_arm_state
        return {'index': index, 'instruction': self.bindings.sample(0, 'Original'),
                'state': perturb_arm_state(torch.ones(4), torch.tensor([True, True, False, False]), .01, .5),
                'python_random': random.random(), 'numpy_random': float(np.random.random())}


class DataloaderResumeTests(unittest.TestCase):
    def test_prefetched_worker_rng_round_trip(self):
        from lingbotvla.data.data_loader import DistributedDataloader
        from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
        episodes = [{'episode_index': 0, 'tasks': ['Original'], 'length': 48}]
        (ROOT / 'worker_bindings.json').write_text(json.dumps({
            'schema_version': 1, 'source_pool': 'release_seen_only',
            'episode_metadata_sha256': episode_digest(episodes),
            'episodes': {'0': {'original': 'Original', 'candidates': ['Original', 'Alternative']}}}))
        def make_loader():
            dataset = TinyRandomVLADataset()
            sampler = StatefulDistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=42)
            return DistributedDataloader(dataset, batch_size=4, sampler=sampler, num_workers=2,
                                         prefetch_factor=2, pin_memory=False)
        torch.manual_seed(19)
        loader = make_loader()
        iterator = iter(loader)
        try:
            for _ in range(3): next(iterator)
            saved = copy.deepcopy(loader.state_dict())
            expected = list(iterator)
        finally:
            iterator._shutdown_workers()
        torch.manual_seed(987)
        resumed = make_loader()
        resumed.load_state_dict(saved)
        iterator2 = iter(resumed)
        try:
            actual = list(iterator2)
        finally:
            iterator2._shutdown_workers()
        self.assertEqual(len(actual), len(expected))
        for before, after in zip(expected, actual):
            self.assertEqual(before['instruction'], after['instruction'])
            for key in ('index', 'state', 'python_random', 'numpy_random'):
                self.assertTrue(torch.equal(before[key], after[key]), key)


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dist.init_process_group('gloo', init_method='file://' + str(ROOT / 'rendezvous'), rank=0, world_size=1)

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()

    def test_ema_arithmetic_and_integer_buffers(self):
        model = torch.nn.Linear(2, 1, bias=False)
        model.register_buffer('count', torch.tensor(0))
        with torch.no_grad(): model.weight.fill_(2)
        ema = ShardedEMA(model, .9)
        with torch.no_grad(): model.weight.fill_(4); model.count.fill_(3)
        ema.update()
        self.assertTrue(torch.allclose(ema.shadow['weight'], torch.full_like(model.weight, 2.2)))
        self.assertEqual(ema.shadow['count'].item(), 3)
        self.assertEqual(ema.num_updates, 1)
        self.assertEqual(ema.shadow['weight'].device.type, 'cpu')

    def test_complete_resume_matches_uninterrupted(self):
        args = args_for(ROOT / 'stage')
        contract = build_stage_contract(args)
        torch.manual_seed(123); random.seed(123); np.random.seed(123)
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=.001)
        ema = ShardedEMA(model, .9)
        def step(m, o, s, e):
            x = torch.randn(3, 4); y = torch.randn(3, 2)
            ((m(x) - y) ** 2).mean().backward();o.step();e.update();s.step();o.zero_grad()
        for _ in range(3): step(model, optimizer, scheduler, ema)
        state = {'model': model, 'ema': ema, 'optimizer': optimizer,
                 'extra_state': {'global_step': 3, 'lr_scheduler': scheduler.state_dict(),
                                 'torch_rng_state': torch.get_rng_state(), **extra_stage_state(contract, ema)}}
        checkpoint = ROOT / 'stage/checkpoints/global_step_3'
        DistributedCheckpointer.save(str(checkpoint), state)
        validate_stage_checkpoint(checkpoint, contract)
        exported = dcp_to_torch_state_dict(str(checkpoint), ema=True)
        for k, v in ema.shadow.items(): self.assertTrue(torch.equal(exported[k], v))
        from lingbotvla.utils.async_hf_checkpoint import AsyncHFCheckpointSaver
        from transformers import PretrainedConfig
        from safetensors.torch import load_file
        saver = AsyncHFCheckpointSaver(enabled=False)
        saver.submit(global_step=3, save_checkpoint_path=str(checkpoint), output_dir=str(ROOT / 'stage'),
                     ckpt_manager='dcp', save_ema=True, enable_fp32=True, model_assets=[PretrainedConfig()])
        saver.wait_all_best_effort()
        for folder, expected in [('hf_ckpt', model.state_dict()), ('ema_hf_ckpt', ema.shadow)]:
            values = {}
            for shard in (checkpoint / folder).glob('*.safetensors'): values.update(load_file(str(shard)))
            self.assertEqual(set(values), set(expected))
            for k, v in expected.items(): self.assertTrue(torch.equal(values[k], v), folder + ':' + k)

        for _ in range(3): step(model, optimizer, scheduler, ema)
        expected_raw = copy.deepcopy(model.state_dict());expected_ema = copy.deepcopy(ema.shadow)
        expected_lr = scheduler.get_last_lr()
        resumed = torch.nn.Linear(4, 2)
        opt2 = torch.optim.AdamW(resumed.parameters(), lr=.01)
        sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=6, eta_min=.001)
        ema2 = ShardedEMA(resumed, .9)
        loaded = {'model': resumed, 'ema': ema2, 'optimizer': opt2, 'extra_state': {}, 'strict_optimizer': True}
        DistributedCheckpointer.load(str(checkpoint), loaded)
        sch2.load_state_dict(loaded['extra_state']['lr_scheduler'])
        torch.set_rng_state(loaded['extra_state']['torch_rng_state'])
        restore_stage_state(loaded['extra_state'], contract, ema2)
        for _ in range(3): step(resumed, opt2, sch2, ema2)
        for k, v in expected_raw.items(): self.assertTrue(torch.equal(resumed.state_dict()[k], v), k)
        for k, v in expected_ema.items(): self.assertTrue(torch.equal(ema2.shadow[k], v), k)
        self.assertEqual(ema2.num_updates, 6);self.assertEqual(sch2.get_last_lr(), expected_lr)
        changed = copy.deepcopy(contract); changed['training']['lr'] = .2
        with self.assertRaises(ValueError): validate_stage_checkpoint(checkpoint, changed)
        metadata = checkpoint / 'ema/.metadata'; metadata.rename(metadata.with_suffix('.held'))
        try:
            with self.assertRaises(ValueError): validate_stage_checkpoint(checkpoint, contract)
        finally:metadata.with_suffix('.held').rename(metadata)

    def test_compiled_model_ema_uses_canonical_names(self):
        base=torch.nn.Linear(2,1)
        model=torch.compile(base,backend='eager')
        ema=ShardedEMA(model,.9)
        self.assertEqual(set(ema.shadow),set(base.state_dict()))
        ema.update()
        self.assertEqual(ema.num_updates,1)

    def test_strict_resume_rejects_missing_optimizer_moment(self):
        from unittest.mock import patch
        import torch.distributed.checkpoint as dcp
        from lingbotvla.checkpoint.checkpointer import OptimizerState
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        checkpoint = ROOT / 'partial_optimizer'
        DistributedCheckpointer.save(str(checkpoint), {'model': model, 'optimizer': optimizer})
        partial = copy.deepcopy(OptimizerState(model, optimizer).state_dict())
        first = next(iter(partial['optim']['state'].values()))
        del first['exp_avg']
        with patch.object(OptimizerState, 'state_dict', return_value=partial):
            dcp.save({'state': OptimizerState(model, optimizer)}, checkpoint_id=checkpoint / 'optimizer')
        restored = torch.nn.Linear(2, 1)
        opt2 = torch.optim.AdamW(restored.parameters(), lr=.01)
        with self.assertRaises(dcp.CheckpointException):
            DistributedCheckpointer.load(str(checkpoint), {'model': restored, 'optimizer': opt2,
                                                           'strict_optimizer': True})

    def test_completed_stage_performs_no_more_updates(self):
        # Execute the real epoch loop at its resume boundary without loading 6B.
        import ast
        from unittest.mock import Mock
        path = ANT / 'src/lingbot-vla-v2/tasks/vla/train_lingbotvla.py'
        main = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'main')
        loop = next(n for n in main.body if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Name) and n.target.id == 'epoch')
        namespace = {'args': SimpleNamespace(train=SimpleNamespace(max_steps=4000, num_train_epochs=100)),
                     'start_epoch': 7, 'global_step': 4000, 'logger': Mock()}
        exec(compile(ast.Module(body=[loop], type_ignores=[]), str(path), 'exec'), namespace)
        self.assertEqual(namespace['global_step'], 4000)
        self.assertNotIn('data_iterator', namespace)

    def test_cannot_resume_v2_optimizer_in_new_stage(self):
        args = args_for(ROOT / 'fresh_stage')
        args.train.load_checkpoint_path = str(ROOT / 'v2/checkpoints/global_step_19000')
        with self.assertRaises(ValueError): build_stage_contract(args)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
