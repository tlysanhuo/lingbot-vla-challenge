"""Launch one historical training stage with explicit GPU selection."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import yaml
from common import ROOT, base_env, gpu_gate, render_config

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['v1', 'v2', 'v3'], required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = render_config(ROOT / 'configs' / (args.stage + '.yaml'))
    selected = args.gpus.split(',')
    if len(selected) != 4 or len(set(selected)) != 4:
        raise SystemExit('Published recipes use four distinct GPUs; change the recipe explicitly for another world size.')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=4', '--max-restarts=0', 'tasks/vla/train_lingbotvla.py', '<rendered-config>']
    if args.dry_run:
        print(yaml.safe_dump(config, sort_keys=False)); print(' '.join(command)); return
    gpu_gate(args.gpus)
    for location in [config['model']['model_path'], config['model']['tokenizer_path'], config['data']['train_path']]:
        if not Path(location).exists():
            raise SystemExit(f'Missing required asset: {location}')
    if args.stage == 'v3' and not Path(config['data']['seen_bindings_file']).is_file():
        raise SystemExit('Build the seen-only instruction bindings first.')
    source = ROOT / 'src/lingbot-vla-v2'
    if not (source / '.project-upstream.json').exists():
        raise SystemExit('Run scripts/setup_sources.py first.')
    run = Path(config['train']['output_dir'])
    run.mkdir(parents=True, exist_ok=True)
    rendered = run / 'launch_config.yaml'
    rendered.write_text(yaml.safe_dump(config, sort_keys=False))
    env = base_env(); env.update(CUDA_VISIBLE_DEVICES=args.gpus, WANDB_MODE='online', NCCL_SOCKET_IFNAME='lo', NCCL_IB_DISABLE='1')
    env['LD_LIBRARY_PATH'] = str(Path(sys.executable).resolve().parents[1] / 'lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    command[-1] = str(rendered)
    subprocess.run(command, cwd=source, env=env, check=True)

if __name__ == '__main__':
    main()
