"""Recompute released results and check their episode-level evidence (CPU only)."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main():
    results=json.loads((ROOT/'results/evaluation/v3_confirmation.json').read_text())['results']
    protocol=json.loads((ROOT/'results/evaluation/protocol.json').read_text())
    assert set(results)=={'clean','randomized'}
    assert set(results['clean'])==set(results['randomized']) and len(results['clean'])==50
    episodes=[json.loads(s) for s in (ROOT/'results/evaluation/v3_episodes.jsonl').read_text().splitlines()]
    assert len(episodes)==1000
    grouped=defaultdict(list)
    for e in episodes:
        setting=e['setting'].removeprefix('demo_')
        assert e['instruction_type']=={'clean':'seen','randomized':'unseen'}[setting]
        assert isinstance(e['success'],bool) and e['instruction'] and e['instruction']!=e['task']
        grouped[(setting,e['task'])].append(e)
    summary={}
    for setting,tasks in results.items():
        success=attempts=0
        for task,row in tasks.items():
            records=grouped[(setting,task)]
            assert len(records)==row['attempts']==10
            assert sum(e['success'] for e in records)==row['successes']
            assert len({e['seed'] for e in records})==10
            assert {e['episode'] for e in records}==set(range(1,11))
            success+=row['successes'];attempts+=row['attempts']
        summary[setting]={'successes':success,'attempts':attempts,'rate':100*success/attempts}
    assert summary['clean']['successes']==404 and summary['randomized']['successes']==252
    rows=list(csv.DictReader((ROOT/'results/training/v3_wandb.csv').open()))
    provenance=json.loads((ROOT/'results/training/v3_wandb_provenance.json').read_text())
    assert len(rows)==provenance['returned_rows']==3998
    steps=[int(float(r['step'])) for r in rows]
    assert steps==sorted(set(steps)) and steps[0]==1 and steps[-1]==4000
    assert all(math.isfinite(float(v)) for r in rows for v in r.values())
    assert protocol['description_count']==100 and protocol['execute_steps']==30
    summary['equal_setting_mean']=sum(summary[s]['rate'] for s in ['clean','randomized'])/2
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('PASS: 100 task-setting cells, 1,000 episodes, and 3,998 W&B records')

if __name__=='__main__':main()
