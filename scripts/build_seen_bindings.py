#!/usr/bin/env python3
"""Build auditable seen-only bindings from small LeRobot episode metadata."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import os

import pyarrow.parquet as pq

ANT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('seen_bindings', ANT / 'src/lingbot-vla-v2/lingbotvla/data/vla_data/seen_bindings.py')
bindings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bindings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path(os.environ.get('DATA_ROOT', str(ANT / 'data/RoboTwin_lerobot_v30'))))
    parser.add_argument('--output', type=Path, default=Path(os.environ.get('OUTPUT_ROOT', str(ANT / 'outputs'))) / 'prepared/seen_bindings.json')
    args = parser.parse_args()
    episodes = []
    for path in sorted((args.dataset / 'meta/episodes').rglob('*.parquet')):
        episodes.extend(pq.read_table(path, columns=['episode_index', 'tasks', 'length']).to_pylist())
    episodes.sort(key=lambda e: e['episode_index'])
    if len(episodes) != 2500 or any(len(e['tasks']) != 1 for e in episodes):
        raise ValueError('expected the released clean 2500 single-instruction episodes')
    catalog = bindings.seen_catalog(ANT / 'src/RoboTwin/description/task_instruction')
    entities = bindings.seen_entities(ANT / "src/RoboTwin/description/objects_description")
    builder = bindings.SeenBindingBuilder(catalog, entities)
    records = {str(e['episode_index']): builder.bind(e['tasks'][0]) for e in episodes}
    counts = Counter(r['status'] for r in records.values())
    varied = sum(len(r['candidates']) > 1 for r in records.values())
    summary = {'episodes': len(records), 'status_counts': dict(counts),
               'episodes_with_variation': varied, 'coverage': varied / len(records),
               'max_candidates': max(len(r['candidates']) for r in records.values())}
    manifest = {'schema_version': 1, 'source_pool': 'release_seen_only',
                'dataset_root': str(args.dataset.resolve()),
                'episode_metadata_sha256': bindings.episode_digest(episodes),
                'seen_catalog_sha256': bindings.catalog_digest(catalog),
                'seen_entities_sha256': bindings.catalog_digest(entities),
                'binding_policy': 'typed_entities_exact_slots_spatial_order_v2',
                'summary': summary, 'episodes': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))
    print(args.output)


if __name__ == '__main__':
    main()
