from collections import Counter
from common import *

queue=read(OUT/'protocol/execution_queue.json')
workers=sorted({j['worker'] for j in queue if j.get('worker')})
active={w:read(OUT/f'status/{w}.json') for w in workers if (OUT/f'status/{w}.json').exists()}
value=dict(exports=len(list((OUT/'predictions').glob('*.jsonl'))),expected_exports=72,gradients=len(list((OUT/'gradients').glob('*.json'))),queue=dict(Counter(j['state'] for j in queue)),active=active,failed=[j for j in queue if j['state']=='failed'],finalizer=read(OUT/'status/finalizer.json'))
for j in value['failed']:value.setdefault('errors',{})[j['id']]=(OUT/f'logs/{j["id"]}_{j["worker"]}.log').read_text()[-1600:]
print(json.dumps(value),flush=True)
