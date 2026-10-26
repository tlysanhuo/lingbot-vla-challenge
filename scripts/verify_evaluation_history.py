"""Recompute archive/snapshot tables from their committed evidence; CPU only."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text())


def main():
    snapshot = load('results/evaluation/latest_snapshot.json')
    tasks = set(snapshot['tasks'])
    assert len(tasks) == 50
    modern_sources = {}
    for run in snapshot['runs'].values():
        payload = (ROOT / run['episode_file']).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == run['episode_file_sha256']
        events = [json.loads(line) for line in payload.splitlines()]
        grouped = defaultdict(list)
        for e in events:
            assert set(e) == {'task', 'setting', 'instruction_type', 'episode', 'seed', 'instruction', 'success', 'action_steps'}
            assert e['task'] in tasks and type(e['success']) is bool and e['action_steps'] > 0
            setting = e['setting'].removeprefix('demo_')
            assert e['instruction_type'] == {'clean': 'seen', 'randomized': 'unseen'}[setting]
            grouped[setting, e['task']].append(e)
        for setting, half in run['settings'].items():
            assert set(half['per_task']) == tasks
            done = []
            for task, row in half['per_task'].items():
                records = grouped[setting, task]
                assert len(records) == row['attempts'] <= row['target'] == run['episodes_per_task']
                assert sum(r['success'] for r in records) == row['successes']
                assert [r['episode'] for r in records] == list(range(1, len(records) + 1))
                assert len({r['seed'] for r in records}) == len(records)
                if row['status'] == 'complete':
                    assert row['attempts'] == row['target'] and row['result_and_log_verified']
                    done.append(row)
                if row['status'] == 'not_started':
                    assert row['attempts'] == row['successes'] == 0
            assert half['tasks_complete'] == len(done)
            assert half['complete_task_successes'] == sum(r['successes'] for r in done)
            assert half['complete_task_attempts'] == sum(r['attempts'] for r in done)
            assert half['episodes_recorded'] == sum(r['attempts'] for r in half['per_task'].values())
            assert half['successes_recorded'] == sum(r['successes'] for r in half['per_task'].values())
            if half['source_run']:
                modern_sources[half['source_run']] = half
    if (ROOT / 'results/evaluation/v3_full100.json').exists():
        full = load('results/evaluation/v3_full100.json')
        current = snapshot['runs']['v3_full100']
        assert full['episode_file_sha256'] == current['episode_file_sha256']
        assert all(h['status'] == 'complete' for h in current['settings'].values())
        rates = []
        for setting, half in current['settings'].items():
            expected = {task: {key: row[key] for key in ('successes', 'attempts')}
                        for task, row in half['per_task'].items()}
            assert full['results'][setting] == expected
            successes = sum(row['successes'] for row in expected.values())
            attempts = sum(row['attempts'] for row in expected.values())
            rate = 100 * successes / attempts
            assert full['summary'][setting] == {'successes': successes, 'attempts': attempts, 'rate': rate}
            rates.append(rate)
        assert full['equal_setting_mean'] == sum(rates) / len(rates)
        comparison = load('results/evaluation/v3_10eps_vs_100eps.json')
        old = load('results/evaluation/v3_confirmation.json')['results']
        for setting in ('clean', 'randomized'):
            previous = 100 * sum(r['successes'] for r in old[setting].values()) / sum(r['attempts'] for r in old[setting].values())
            assert comparison['old_10eps_percent'][setting] == previous
            assert comparison['new_100eps_percent'][setting] == full['summary'][setting]['rate']
            assert abs(comparison['delta_percentage_points'][setting] - (full['summary'][setting]['rate'] - previous)) < 1e-9
    index = load('results/evaluation/history/index.json')
    # Legacy archive capture time stays historical when final snapshots are added.
    assert index['snapshot_at'] <= snapshot['snapshot_at']
    count = with_results = 0
    expected_csv = {}
    for version, meta in index['versions'].items():
        archive = load(meta['file'])
        assert archive['version'] == version and archive['snapshot_at'] == index['snapshot_at']
        runs = archive['runs']
        assert len(runs) == meta['run_count']
        assert sum(bool(r['attempts']) for r in runs) == meta['runs_with_results']
        count += len(runs)
        with_results += meta['runs_with_results']
        for run in runs:
            assert run['version'] == version and set(run['per_task']) <= tasks
            assert not set(run['per_task']) & set(run['partial_per_task'])
            assert len(run['per_task']) == run['completed_tasks']
            assert sum(r['successes'] for r in run['per_task'].values()) == run['successes']
            assert sum(r['attempts'] for r in run['per_task'].values()) == run['attempts']
            assert run['success_rate'] == (100 * run['successes'] / run['attempts'] if run['attempts'] else None)
            for partial, rows in [(False, run['per_task']), (True, run['partial_per_task'])]:
                for task, row in rows.items():
                    assert 0 <= row['successes'] <= row['attempts'] and row['attempts'] > 0
                    assert row['result_and_log_verified'] is not partial
                    if row.get('contiguous_sequence') and 'seeds' in row:
                        assert len(row['seeds']) == len(row['cumulative_successes']) == row['attempts']
                        previous = 0
                        for total in row['cumulative_successes']:
                            assert total - previous in (0, 1)
                            previous = total
                        assert previous == row['successes']
                    expected_csv[(run['id'], task, str(partial))] = (row['successes'], row['attempts'])
            if run['source_run'] in modern_sources:
                half = modern_sources[run['source_run']]
                assert run['successes'] == half['complete_task_successes']
                assert run['attempts'] == half['complete_task_attempts']
    assert count == index['directory_count'] and with_results == index['runs_with_completed_tasks']
    csv_rows = list(csv.DictReader((ROOT / 'results/evaluation/all_task_results.csv').open()))
    actual_csv = {(r['run_id'], r['task'], r['partial_task']): (int(r['successes']), int(r['attempts'])) for r in csv_rows}
    assert len(actual_csv) == len(csv_rows) and actual_csv == expected_csv
    cases = load('results/evaluation/cases.json')['cases']
    assert len(cases) == 2 and {c['episode']['success'] for c in cases} == {True, False}
    for case in cases:
        lines = (ROOT / case['episode_file']).read_text().splitlines()
        assert json.loads(lines[case['line'] - 1]) == case['episode']
        assert case['video'] is None
    print(f'PASS: {count} run directories, {with_results} with results, {len(csv_rows)} task rows, frozen episode snapshots and 2 cases')


if __name__ == '__main__':
    main()
