"""Report every frozen run and policy without selecting by official test."""
from pathlib import Path
import fcntl,json,os,time
BASE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def save(p,d):
    q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');os.replace(q,p)
def main():
    with (BASE/'.summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=read(BASE/'queue.json');results=[]
        for job in data:
            if job['kind']!='test' or job['state']!='complete':continue
            output=BASE/'tests'/job['id'];r=read(output/'report.json')
            results.append(dict(id=job['spec']['id'],method=job['spec']['name'],seed=job['spec']['seed'],
                                policy=job['policy'],epoch=r['selected_epoch'],metrics=r['metrics'],report=str(output/'report.json')))
        counts={s:sum(j['state']==s for j in data) for s in ['pending','running','complete','failed','skipped']}
        state='complete' if all(j['state']=='complete' for j in data) else ('finished_with_failures' if not counts['pending'] and not counts['running'] else 'running')
        save(BASE/'summary.json',dict(status=state,counts=counts,results=results,seed13_only=True,test_used_for_selection=False))
        lines=['# MOSI First — seed 13','',f'Status: {state}. Jobs: {counts}.',
               'Frozen protocol: up to 30 epochs, validation-MAE early stopping patience 7, batch size 8, LR 1e-4, weight decay 0.01.',
               'Only seed 13. Three fixed teacher prediction caches. MAE and Acc2 best are selected using validation only; last is retained for resume.',
               'Official-test results are exploratory reporting and do not select methods, matrices, hyperparameters or checkpoints. No cross-seed conclusions are available.','',
               '| Method | Seed | Checkpoint policy | Epoch | MAE | Acc2 nonzero | F1 nonzero | Pearson |',
               '|---|---:|---|---:|---:|---:|---:|---:|']
        for r in sorted(results,key=lambda r:(r['id'],r['policy'])):
            m=r['metrics'];lines.append(f"| {r['method']} | {r['seed']} | {r['policy']} | {r['epoch']} | {m['mae']:.6f} | {m['acc2_nonzero']:.6f} | {m['f1_weighted_nonzero']:.6f} | {m['pearson']:.6f} |")
        lines += ['','Missing original Probe checkpoints are documented in teacher_manifest.json. Cached predictions have been verified against all three prediction files.']
        (BASE/'summary.md').write_text('\n\n'.join(lines[:6])+'\n'+'\n'.join(lines[6:])+'\n')
        print(json.dumps(dict(status=state,counts=counts,results=len(results))))
if __name__=='__main__':main()
