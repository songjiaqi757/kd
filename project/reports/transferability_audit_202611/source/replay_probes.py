"""Bounded CPU replay of frozen Probe checkpoints; no fitting/calibration."""
from collections import defaultdict
from importlib import import_module
import sys
import numpy as np
import torch
from common import *
sys.path[:0]=[str(ROOT/'project/src'),str(ROOT/'project/sims_native/src')]

def main():
    torch.set_num_threads(2);plan=read(OUT/'protocol/plan.json');results=[]
    for d,p in plan['protocols'].items():
        a=read(p['assets']);directory=Path(a['teacher_features']).parent
        index=rows(directory/'index.jsonl');features=np.load(directory/'features.npy',mmap_mode='r');completed=np.load(directory/'completed.npy',mmap_mode='r')
        assert completed.all()
        selected=set()
        for split in ['train','valid']:
            selected.update(sorted({r['parent_sample_id'] for r in index if r['split']==split})[:16])
        subset=[r for r in index if r['parent_sample_id'] in selected]
        package='rdid_sims' if d.startswith('sims') else 'rdid_mosei'
        cls=import_module(package+'.probe').TeacherProbe
        for probe in p['probes']:
            root=Path(probe['directory']);state=torch.load(root/'teacher_probe.pt',map_location='cpu',weights_only=False)
            model=cls(**{k:state[k] for k in ['input_size','hidden_size','classes','dropout']});model.load_state_dict(state['model_state_dict']);model.eval()
            accum=defaultdict(lambda:dict(weight=0.,score=0.,logits=np.zeros(state['classes'])))
            with torch.inference_mode():
                for start in range(0,len(subset),128):
                    chosen=subset[start:start+128];input=torch.from_numpy(np.array(features[[r['job_index'] for r in chosen]],dtype=np.float32));out=model(input)
                    for r,score,logits in zip(chosen,out['regression'].numpy(),out['classification_logits'].numpy()):
                        v=accum[(r['parent_sample_id'],r['subset'])];w=r['aggregation_weight'];v['weight']+=w;v['score']+=float(score)*w;v['logits']+=logits.astype(float)*w
            cached={(r['parent_sample_id'],r['subset']):r for r in rows(root/'predictions.jsonl')}
            scores=[abs(v['score']/v['weight']-cached[key]['probe_score']) for key,v in accum.items()]
            logits=[float(np.max(np.abs(v['logits']/v['weight']-cached[key]['classification_logits']))) for key,v in accum.items()]
            record=dict(dataset=d,probe_seed=probe['seed'],sampled_train_parents=16,sampled_valid_parents=16,subset_pairs=len(accum),cpu_replay_max_score_error=max(scores),cpu_replay_max_logit_error=max(logits),tolerance=1e-4,passed=max(scores)<1e-4 and max(logits)<1e-4,scope='32 deterministic train/valid parents x 7 subsets, all windows per parent',checkpoint_sha256=probe['checkpoint_sha256'])
            results.append(record);print(json.dumps(record),flush=True)
            assert record['passed'],record
            del state,model
    table('probe_checkpoint_replay.csv',results)
    save(OUT/'status/probe_replay.json',dict(status='complete',passed=all(r['passed'] for r in results),checks=results))

if __name__=='__main__':main()
