"""One bounded GPU process per host; resumes completed checkpoints, records failures."""
import argparse
import fcntl
import signal
import subprocess
import sys
import time
from common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',required=True);ap.add_argument('--mode',choices=['gradients','trajectory'],default='gradients');arg=ap.parse_args()
    suffix='' if arg.mode=='gradients' else '_trajectory'
    lock=(OUT/f'protocol/{arg.worker}.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    schedule=read(OUT/'protocol/schedule.json');errors=[]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',USE_TF='0',USE_FLAX='0')
    for d in schedule['shards'][arg.worker]:
        log=(OUT/f'logs/{arg.worker}_{d}_{arg.mode}.log').open('a')
        code=1
        for attempt in [1,2]:
            proc=subprocess.Popen(['/ai/sjq/miniconda3/envs/kd/bin/python','-u',str(OUT/'source/gpu.py'),'--dataset',d,'--mode',arg.mode,'--seeds','13','--cap-mib','12288','--worker',arg.worker],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(OUT/f'status/worker_{arg.worker}{suffix}.json',dict(status='running',dataset=d,pid=proc.pid,attempt=attempt,started_unix=time.time(),worker=arg.worker))
            try:code=proc.wait(timeout=24*3600)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM);code=-124
            # Terminate only descendants in this worker's process group, including OOM orphans.
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            if code==0:break
            time.sleep(10)
        if code:errors.append(dict(dataset=d,exit_code=code,log=str(OUT/f'logs/{arg.worker}_{d}_{arg.mode}.log')))
        log.close()
    save(OUT/f'status/worker_{arg.worker}{suffix}.json',dict(status='failed' if errors else 'complete',worker=arg.worker,errors=errors,finished_unix=time.time()))

if __name__=='__main__':main()
