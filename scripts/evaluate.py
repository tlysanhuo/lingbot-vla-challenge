"""Prepare/run the pinned evaluation client: randomized first, then clean."""
import argparse
from pathlib import Path
import subprocess
import json
import os
from common import ROOT, base_env, gpu_gate, paths

def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Unexpected launcher revision: ' + old)
    return text.replace(old, new, 1)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--gpus', required=True)
    p.add_argument('--episodes', type=int, choices=[10, 100], default=100)
    p.add_argument('--conda-sh', type=Path, required=True)
    p.add_argument('--inference-env', default='lingbotvla')
    p.add_argument('--sim-env', default='RoboTwin')
    p.add_argument('--port', type=int, default=9880)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/evaluation')
    p.add_argument('--dedicated-node', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    gpus = args.gpus.split(',')
    if len(gpus) == 8 and not args.dedicated_node:
        p.error('Eight GPUs require an independent node and --dedicated-node')
    if len(gpus) not in (1,2,3,4,8) or len(set(gpus)) != len(gpus) or not all(x.isdigit() for x in gpus):
        raise SystemExit('Choose 1–4 distinct GPUs')
    source = ROOT / 'src/lingbot-vla-v2'
    launcher = source / 'experiment/robotwin/start_robotwin_infer_and_eval.sh'
    if not launcher.is_file():
        raise SystemExit('Run scripts/setup_sources.py first.')
    original = launcher.read_text()
    args.output.mkdir(parents=True, exist_ok=True)
    env = base_env()
    env.update(QWEN3VL_PATH=paths()['MODEL_ROOT'] + '/Qwen3-VL-4B-Instruct', TEST_NUM=str(args.episodes), ANT_CUROBO_SAFE='1')
    env['ANT_WARP_CACHE_ROOT'] = str(args.output.resolve() / 'cache/warp')
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONPATH'] = str(ROOT / 'runtime_patches') + ':' + env.get('PYTHONPATH', '')
    for setting, instruction in [('demo_randomized', 'unseen'), ('demo_clean', 'seen')]:
        text = once(original, 'eval_client_dst="${eval_workdir}/script/eval_policy_client_lingbotvla.py"', "eval_client_dst='" + str(ROOT / 'src/RoboTwin/script/eval_policy_client_lingbotvla.py') + "'")
        text = once(text, '        --task_config ${task_config} ' + chr(92) + '\n',
                    '        --task_config ${task_config} ' + chr(92) + '\n' +
                    f'        --instruction_type {instruction} --test_num {args.episodes} ' + chr(92) + '\n')
        norm = ROOT / 'configs/robotwin_historical_norm_stats.json'
        inference_anchor = "        --model_path '${model_path}' " + chr(92) + '\n'
        text = once(text, inference_anchor, inference_anchor + "        --robot_norm_path '" + str(norm.resolve()) + "' " + chr(92) + '\n')
        text = once(text, 'max_retries=3', 'max_retries=1')
        text = text.replace('Done(100)', f'Done({args.episodes})').replace('"$total_ep" = "100"', f'"$total_ep" = "{args.episodes}"')
        text = text.replace('NO(${total_ep}/100)', f'NO(${{total_ep}}/{args.episodes})').replace('NO(0/100)', f'NO(0/{args.episodes})')
        generated = args.output / (setting + '.sh'); generated.write_text(text)
        subprocess.run(['bash', '-n', str(generated)], check=True)
        command = ['bash', str(generated), '--model_path', str(args.checkpoint.resolve()), '--inference_workdir', str(source) + '/',
                   '--eval_workdir', str(ROOT / 'src/RoboTwin'), '--conda_sh', str(args.conda_sh.resolve()),
                   '--inference_env', args.inference_env, '--sim_env', args.sim_env, '--output_base', str(args.output.resolve()),
                   '--gpu_ids', args.gpus, '--num_gpus', str(len(gpus)), '--num_per_gpu', '1', '--num_tasks', '50',
                   '--task_config', setting, '--start_port', str(args.port), '--execute_steps', '30', '--use_length', '50',
                   '--use_bf16', 'False', '--use_fp32', 'True', '--use_compile', 'True', '--run_tag', f'confirmation_{args.episodes}', '--no_video']
        print(json.dumps({'setting': setting, 'instruction': instruction, 'command': command}, ensure_ascii=False))
        if not args.dry_run:
            gpu_gate(args.gpus, max_cards=len(gpus))
            cfg = json.loads((args.checkpoint / 'config.json').read_text())
            if cfg.get('chunk_size') != 50 or cfg.get('num_steps') != 10:
                raise SystemExit('Expected chunk50 and 10 denoising steps')
            subprocess.run(command, cwd=source, env=env, check=True)

if __name__ == '__main__':
    main()
