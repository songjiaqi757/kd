import concurrent.futures
import fcntl
import subprocess
import sys
import time
from common import *

def run(d):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    status=OUT/f'status/analysis_{d}.json'
    if status.exists() and read(status)['status']=='complete':return 0
    with (OUT/f'logs/analysis_{d}.log').open('a') as log:
        save(status,dict(status='running',dataset=d,started_unix=time.time()))
        p=subprocess.run([sys.executable,str(OUT/'source/analyze.py'),'--dataset',d],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:save(status,dict(status='failed',dataset=d,exit_code=p.returncode,log=str(OUT/f'logs/analysis_{d}.log')))
    return p.returncode

def main():
    lock=(OUT/'protocol/cpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:codes=list(pool.map(run,['mosi','mosei','sims','sims_v2']))
    save(OUT/'status/cpu_worker.json',dict(status='complete' if not any(codes) else 'failed',exit_codes=codes))

if __name__=='__main__':main()
