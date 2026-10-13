"""Export selected historical metrics from TensorBoard without starting training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
if not hasattr(np, 'string_'): np.string_ = np.bytes_
if not hasattr(np, 'unicode_'): np.unicode_ = np.str_
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader

KEYS = ['training_loss','training_vla_loss','training_depth_loss','training_future_depth_loss',
        'training_future_video_loss','training_grad_norm','training_lr','training_expert_lr','steptime']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events-dir', type=Path, required=True)
    parser.add_argument('--max-step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.max_step < 1: parser.error('--max-step must be positive')
    files = sorted(args.events_dir.glob('events.out.tfevents.*'))
    if not files: raise SystemExit('No event files found')
    tags = {k.replace('training_', 'training/', 1) if k.startswith('training_') else k: k for k in KEYS}
    values = {}; duplicates = 0; beyond = 0
    for path in files:
        for event in LegacyEventFileLoader(str(path)).Load():
            for value in event.summary.value:
                if value.tag not in tags or not value.HasField('simple_value'): continue
                step = int(event.step)
                if step < 1: continue
                if step > args.max_step: beyond += 1; continue
                key = (step, tags[value.tag])
                if key in values: duplicates += 1
                if key not in values or event.wall_time >= values[key][0]:
                    values[key] = (float(event.wall_time), float(value.simple_value))
    steps = sorted({step for step, key in values if key == 'training_vla_loss'})
    if not steps: raise SystemExit('No action-loss records found')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['step'] + KEYS); writer.writeheader()
        for step in steps:
            writer.writerow({'step': step, **{k: values.get((step, k), (0, ''))[1] for k in KEYS}})
    report = {'source': 'original TensorBoard scalar events', 'rows': len(steps), 'first_step': steps[0],
              'last_step': steps[-1], 'retained_through_checkpoint': args.max_step,
              'missing_steps': sorted(set(range(1, args.max_step + 1)) - set(steps)),
              'duplicate_scalar_records': duplicates, 'scalar_records_after_checkpoint_excluded': beyond,
              'deduplication': 'latest wall_time per (optimizer step, metric); no interpolation or imputation',
              'source_event_files': [{'index': i, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for i, p in enumerate(files)]}
    args.output.with_suffix('.provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['rows', 'first_step', 'last_step', 'duplicate_scalar_records']}))

if __name__ == '__main__': main()
