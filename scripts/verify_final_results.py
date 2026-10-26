"""Verify all completed final model scores from committed episode evidence."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1]
expected={'v3':(2369,3902),'v5':(2398,3942),'v7':(2366,3974)}
for v,(randomized,clean) in expected.items():
 data=json.loads((R/'results/evaluation'/f'{v}_full100.json').read_text())
 lines=[json.loads(x) for x in (R/'results/evaluation'/f'{v}_full100_episodes.jsonl').read_text().splitlines()]
 assert len(lines)==10000
 for setting,wins in [('randomized',randomized),('clean',clean)]:
  tasks=data['results'][setting];assert len(tasks)==50
  assert sum(x['attempts'] for x in tasks.values())==5000
  assert sum(x['successes'] for x in tasks.values())==wins
  for task,row in tasks.items():
   rr=[x for x in lines if x['task']==task and x['setting']=='demo_'+setting]
   assert len(rr)==100 and [x['episode'] for x in rr]==list(range(1,101))
   assert len({x['seed'] for x in rr})==100
   assert all(x['instruction_type']=={'clean':'seen','randomized':'unseen'}[setting] for x in rr)
   assert sum(x['success'] for x in rr)==row['successes']
 print(v,'PASS',randomized,clean,(randomized+clean)/100)
