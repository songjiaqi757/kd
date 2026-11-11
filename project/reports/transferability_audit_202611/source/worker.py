"""A bounded shared-filesystem queue of Phase-0 inference/gradient jobs only."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from common import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',required=True);parser.add_argument('--initialize',action='store_true');parser.add_argument('--export-cap-mib',type=int);arg=parser.parse_args()
    queue=OUT/'protocol/execution_queue.json'
    if arg.initialize:
        assert not queue.exists()
        tasks=[dict(id=f'{d}_gradients_seed13',dataset=d,mode='gradients',seed=13,state='pending') for d in ['mosei','sims','sims_v2','mosi']]
        tasks += [dict(id=f'{d}_export_seed{seed}',dataset=d,mode='export',seed=seed,state='pending') for d in ['mosei','sims_v2','sims','mosi'] for seed in [13,42,2026]]
        save(queue,tasks);print('Initialized 16 Phase 0 tasks');return
    while True:
        with (OUT/'protocol/queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            tasks=read(queue);job=next((j for j in tasks if j['state']=='pending'),None)
            if job is None:break
            job.update(state='running',worker=arg.worker,started_unix=time.time(),pid=os.getpid());save(queue,tasks)
        command=[sys.executable,'-u',str(OUT/'source/gpu.py'),'--dataset',job['dataset'],'--mode',job['mode'],'--seeds',str(job['seed']),'--worker',arg.worker,'--cap-mib','16384','--batch-size','4']
        if job['mode']=='export' and arg.export_cap_mib:command += ['--export-cap-mib',str(arg.export_cap_mib)]
        save(OUT/f'status/{arg.worker}.json',dict(status='running',phase='initializing',job=job['id'],pid=os.getpid()))
        print(json.dumps(dict(job=job['id'],command=command)),flush=True)
        with (OUT/f'logs/{job["id"]}_{arg.worker}.log').open('a') as log:
            process=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1'},stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            code=process.wait()
            if code:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        with (OUT/'protocol/queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);tasks=read(queue);entry=next(j for j in tasks if j['id']==job['id'])
            entry.update(state='complete' if code==0 else 'failed',exit_code=code,finished_unix=time.time());save(queue,tasks)
        if code:raise RuntimeError(f'Phase 0 job failed: {job["id"]}; inspect log; no later phase is authorized')
    save(OUT/f'status/{arg.worker}_queue.json',dict(status='queue_exhausted',pid=os.getpid(),time_unix=time.time()))

if __name__=='__main__':main()
