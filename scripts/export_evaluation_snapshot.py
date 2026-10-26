"""Export read-only evaluation snapshots; never launch or alter a rollout.

Example: python scripts/export_evaluation_snapshot.py --state v3_full100=/path/status.json
Supply all three named states when refreshing the published report.
"""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/evaluation'
BJT = timezone(timedelta(hours=8))
FIELDS = ('task', 'setting', 'instruction_type', 'episode', 'seed', 'instruction', 'success', 'action_steps')
LABELS = {'v3_full100': 'v3 4k raw · 100集全量',
          'v4_3k10': 'v4 3k raw · 10集筛选',
          'v4_6k10': 'v4 6k raw · 10集筛选',
          'v5_full100': 'v5 3k raw · 100集全量',
          'v7_full100': 'v7 2k raw · 100集全量'}


def now():
    return datetime.now(BJT).isoformat(timespec='seconds')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def export_run(run_id, state_file, tasks):
    state = json.loads(state_file.read_text())
    target = state['episodes_per_task']
    assert target == (100 if run_id.endswith('full100') else 10)
    records = []
    settings = {}
    evidence = {}
    for setting, instruction in [('randomized', 'unseen'), ('clean', 'seen')]:
        name = 'demo_' + setting
        done = next((r for r in state.get('results', []) if r['setting'] == name), None)
        run_dir = Path(done['run_dir']) if done else (
            Path(state['run_dir']) if state.get('setting') == name and state.get('run_dir') else None)
        rows = {}
        for task in tasks:
            finished = []
            verified = False
            journal = run_dir / 'eval_results' / task / 'episodes.jsonl' if run_dir else None
            if journal and journal.exists():
                # A writer may be mid-record. Only complete newline-terminated records count.
                captured = b''.join(line for line in journal.read_bytes().splitlines(keepends=True)
                                    if line.endswith(b'\n'))
                events = [json.loads(line) for line in captured.splitlines()]
                assert all(e['task'] == task and e['setting'] == name and
                           e['instruction_type'] == instruction for e in events), task
                protocols = [e for e in events if e['event'] == 'protocol']
                assert len(protocols) == 1, task
                protocol = protocols[0]
                assert (protocol['episodes'], protocol['description_count'], protocol['start_seed'],
                        protocol['execute_steps'], protocol['expert_check']) == (target, 100, 100000, 30, True)
                accepted = {e['seed'] for e in events if e['event'] == 'expert' and e['accepted']}
                finished = [e for e in events if e['event'] == 'episode_result']
                assert [e['episode'] for e in finished] == list(range(1, len(finished) + 1)), task
                assert len(finished) <= target and len({e['seed'] for e in finished}) == len(finished)
                assert all(e['seed'] in accepted and type(e['success']) is bool for e in finished)
                result_file = journal.with_name('_result.txt')
                if len(finished) == target and result_file.exists():
                    successes = sum(e['success'] for e in finished)
                    result_text = result_file.read_text()
                    assert 'Instruction Type: ' + instruction in result_text, task
                    assert abs(float(result_text.strip().splitlines()[-1]) - successes / target) < 1e-7
                    log = re.sub(r'\x1b\[[0-9;]*m', '',
                                 (run_dir / 'eval_logs' / (task + '.log')).read_text(errors='replace'))
                    counts = re.findall(r'Success rate: (\d+)/(\d+)', log)
                    assert counts and tuple(map(int, counts[-1])) == (successes, target), task
                    verified = True
                evidence[setting + '/' + task] = hashlib.sha256(captured).hexdigest()
                records.extend({k: e[k] for k in FIELDS} for e in finished)
            attempts = len(finished)
            status = ('complete' if verified else 'stopped' if attempts and state['status'] == 'stopped_by_user'
                      else 'in_progress' if attempts else 'not_started')
            rows[task] = {'successes': sum(e['success'] for e in finished), 'attempts': attempts,
                          'target': target, 'status': status, 'result_and_log_verified': verified}
        completed = [r for r in rows.values() if r['status'] == 'complete']
        attempts = sum(r['attempts'] for r in rows.values())
        settings[setting] = {
            'instruction_type': instruction,
            'status': ('complete' if len(completed) == len(tasks) else 'not_started' if not attempts
                       else 'stopped' if state['status'] == 'stopped_by_user' else 'in_progress'),
            'source_run': run_dir.name if run_dir else None,
            'tasks_complete': len(completed), 'episodes_recorded': attempts,
            'successes_recorded': sum(r['successes'] for r in rows.values()),
            'complete_task_successes': sum(r['successes'] for r in completed),
            'complete_task_attempts': sum(r['attempts'] for r in completed),
            'per_task': rows,
        }
    payload = ''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in records)
    filename = run_id + '_episodes.jsonl'
    (OUT / filename).write_text(payload)
    norm = state.get('inference_normalization_override')
    return {'label': LABELS[run_id], 'source_id': state_file.parent.name,
            'checkpoint': '/'.join(Path(state['model']).parts[-4:]),
            'source_status': state['status'], 'episodes_per_task': target,
            'episode_file': 'results/evaluation/' + filename,
            'episode_file_sha256': hashlib.sha256(payload.encode()).hexdigest(),
            'settings': settings,
            'protocol': {'description_count': state['description_count'],
                         'execute_steps': state['execute_steps'], 'fp32': state['fp32'],
                         'compile': state['compile'], 'chunk_size': 50, 'denoising_steps': 10,
                         'expert_check': True, 'seed_start': 100000,
                         'normalization_sha256': norm['sha256'] if norm else None},
            'captured_journal_prefix_sha256': evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', action='append', required=True, metavar='ID=STATUS_JSON')
    args = parser.parse_args()
    sources = dict(value.split('=', 1) for value in args.state)
    assert set(sources) == set(LABELS), 'Provide v3_full100, v4_3k10 and v4_6k10 states'
    tasks = sorted(json.loads((OUT / 'v3_confirmation.json').read_text())['results']['clean'])
    started = now()
    runs = {run_id: export_run(run_id, Path(sources[run_id]), tasks) for run_id in LABELS}
    report = {'schema_version': 1, 'capture_started_at': started, 'snapshot_at': now(),
              'results_type': 'internal evaluation snapshot; not an official score or live dashboard',
              'tasks': tasks, 'runs': runs,
              'aggregation': 'Published setting rates use complete tasks only. Partial episodes remain in per-task records. Unstarted tasks have no rate. No overall mean for incomplete settings.',
              'media': {'evaluation_video_available': False, 'sample_evidence': 'episode journals'}}
    write_json(OUT / 'latest_snapshot.json', report)
    print(json.dumps({key: {s: {'tasks_complete': v['tasks_complete'], 'episodes_recorded': v['episodes_recorded']}
                            for s, v in run['settings'].items()} for key, run in runs.items()}, indent=2))


if __name__ == '__main__':
    main()
