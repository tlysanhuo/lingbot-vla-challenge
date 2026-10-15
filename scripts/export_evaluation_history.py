"""Archive every available evaluation directory by model version (CPU/read-only)."""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from export_evaluation_snapshot import ROOT, OUT, now, write_json

ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
COUNTS = re.compile(r'Success rate:\s*(\d+)/(\d+).*?current seed:\s*(\d+)')


def parse_task(directory, task):
    log = directory / 'eval_logs' / (task + '.log')
    result = directory / 'eval_results' / task / '_result.txt'
    if not log.exists():
        assert not result.exists(), f'Missing evidence: {directory.name}/{task}'
        return None
    payload = log.read_bytes()
    text = ANSI.sub('', payload.decode(errors='replace').replace('\r\n', '\n').replace('\r', '\n'))
    events = [tuple(map(int, match)) for match in COUNTS.findall(text)]
    if not events:
        assert not result.exists(), f'Missing counts: {directory.name}/{task}'
        return None
    # Retries in a single historical log are not additional independent samples.
    sequence = []
    for event in events:
        if sequence and event[1] <= sequence[-1][1]:
            sequence = []
        sequence.append(event)
    success, attempts, _ = sequence[-1]
    contiguous = [e[1] for e in sequence] == list(range(1, attempts + 1))
    assert 0 <= success <= attempts
    instruction = 'unknown'
    if result.exists():
        result_text = result.read_text()
        match = re.search(r'Instruction Type:\s*(\w+)', result_text)
        instruction = match.group(1) if match else 'unknown'
        assert abs(float(result_text.strip().splitlines()[-1]) - success / attempts) < 1e-6
    execution = re.findall(r'execute first (\d+) steps', text)
    full_chunk = re.findall(r'step:\s*(\d+)\s*/\s*\d+\s*\ninfer time', text)
    return {'successes': success, 'attempts': attempts, 'instruction_type': instruction,
            'execute_steps': int(execution[-1]) if execution else (50 if '50' in full_chunk else None),
            'execution_evidence': 'explicit' if execution else 'step50_before_infer' if '50' in full_chunk else 'unknown',
            'result_and_log_verified': result.exists(), 'contiguous_sequence': contiguous,
            'seeds': [e[2] for e in sequence], 'cumulative_successes': [e[0] for e in sequence],
            'source_log_sha256': hashlib.sha256(payload).hexdigest(),
            'source_result_sha256': hashlib.sha256(result.read_bytes()).hexdigest() if result.exists() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-root', type=Path, required=True)
    args = parser.parse_args()
    snapshot = json.loads((OUT / 'latest_snapshot.json').read_text())
    modern = {s['source_run']: (run, setting, s) for run in snapshot['runs'].values()
              for setting, s in run['settings'].items() if s['source_run']}
    archive = []
    directories = sorted((d for d in args.eval_root.iterdir() if d.is_dir()),
                         key=lambda d: (re.search(r'20\d{6}_\d{6}', d.name).group(0)
                                        if re.search(r'20\d{6}_\d{6}', d.name) else '', d.name))
    ids = collections.Counter()
    for directory in directories:
        models = set()
        for file in sorted((directory / 'inference_logs').glob('*.log')):
            models.update(re.findall(r'loading model from:\s*(\S+)', file.read_text(errors='replace')))
        versions = {re.search(r'sft_(v\d+)', model).group(1) for model in models if re.search(r'sft_(v\d+)', model)}
        fallback = re.search(r'sft_(v\d+)', directory.name)
        version = next(iter(versions)) if len(versions) == 1 else fallback.group(1) if fallback else 'environment'
        assert len(versions) <= 1, directory.name
        ids[version] += 1
        run_id = f'{version}-{ids[version]:02d}'
        setting = ('randomized' if '_demo_randomized_' in directory.name else
                   'clean' if '_demo_clean_' in directory.name else 'unknown')
        per_task, partial = {}, {}
        if directory.name in modern:
            run, setting, current = modern[directory.name]
            for task, row in current['per_task'].items():
                if row['attempts']:
                    exported = {**row, 'instruction_type': current['instruction_type'],
                                'execute_steps': run['protocol']['execute_steps'],
                                'execution_evidence': 'captured_protocol', 'contiguous_sequence': True}
                    (per_task if row['status'] == 'complete' else partial)[task] = exported
            models = {run['checkpoint']}
            status = current['status']
            precision = 'FP32'
        else:
            names = {p.stem for p in (directory / 'eval_logs').glob('*.log')}
            names.update(p.parent.name for p in (directory / 'eval_results').glob('*/_result.txt'))
            for task in sorted(names):
                row = parse_task(directory, task)
                if row:
                    (per_task if row['result_and_log_verified'] else partial)[task] = row
            status = 'complete_50_tasks' if len(per_task) == 50 else 'subset_or_interrupted' if per_task else 'no_completed_task'
            precision = 'unknown: not independently captured in historical per-run logs'
        steps = sorted({int(m) for model in models for m in re.findall(r'global_step_(\d+)', model)})
        variant = 'average_16k_17k_18k_19k' if 'sft_v2_avg' in directory.name else (
            'EMA' if any('ema_hf_ckpt' in m for m in models) else 'raw' if models else 'unknown')
        date = re.search(r'20\d{6}_\d{6}', directory.name)
        successes = sum(r['successes'] for r in per_task.values())
        attempts = sum(r['attempts'] for r in per_task.values())
        archive.append({'id': run_id, 'version': version, 'source_run': directory.name,
                        'recorded_timestamp_tag': date.group(0) if date else None,
                        'checkpoint_paths': sorted('/'.join(Path(m).parts[-4:]) for m in models),
                        'checkpoint_steps': steps, 'variant': variant,
                        'version_evidence': 'loaded_model_path' if versions else 'directory_name' if fallback else 'environment_record',
                        'setting': setting, 'status': status, 'precision': precision,
                        'completed_tasks': len(per_task), 'successes': successes, 'attempts': attempts,
                        'success_rate': 100 * successes / attempts if attempts else None,
                        'instruction_types': sorted({r['instruction_type'] for r in per_task.values()}),
                        'execute_steps': sorted({r['execute_steps'] for r in per_task.values()}, key=str),
                        'episodes_per_completed_task': dict(collections.Counter(r['attempts'] for r in per_task.values())),
                        'per_task': per_task, 'partial_per_task': partial})
    dest = OUT / 'history'
    dest.mkdir(exist_ok=True)
    index = {'schema_version': 1, 'snapshot_at': snapshot['snapshot_at'], 'archive_exported_at': now(),
             'scope': 'Every directory in the supplied eval-root; no invented results for absent runs. Modern active/stopped runs use the same frozen snapshot as latest_snapshot.json.',
             'directory_count': len(directories), 'runs_with_completed_tasks': sum(bool(r['attempts']) for r in archive),
             'notes': ['Setting rates count only tasks with both result-file and log evidence.',
                       'Subset/panel/interrupted entries are not full-50-task scores.',
                       'Runs remain separate; retries and repeated seeds are not combined into a model score.',
                       'No archived v0 policy score was found in this evaluation root; v0 checkpoint selection was cancelled.'],
             'versions': {}}
    for version in ['v0', 'v1', 'v2', 'v3', 'v4', 'environment']:
        runs = [r for r in archive if r['version'] == version]
        filename = 'results/evaluation/history/' + version + '.json'
        write_json(ROOT / filename, {'version': version, 'snapshot_at': snapshot['snapshot_at'], 'runs': runs})
        index['versions'][version] = {'file': filename, 'run_count': len(runs),
                                      'runs_with_results': sum(bool(r['attempts']) for r in runs)}
    assert sum(v['run_count'] for v in index['versions'].values()) == len(directories)
    write_json(dest / 'index.json', index)
    print(json.dumps(index['versions'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
