"""Finish the current Phase-0 campaign, never enqueue training or later phases."""
import subprocess
import sys
import time
from common import *

def main():
    last=-1
    while True:
        tasks=read(OUT/'protocol/execution_queue.json');failed=[j for j in tasks if j['state']=='failed']
        if failed:
            save(OUT/'status/finalizer.json',dict(status='failed',failed_jobs=failed));return 1
        exports=len(list((OUT/'predictions').glob('*.jsonl')))
        done=all(j['state']=='complete' for j in tasks)
        if done or (exports-last>=6):
            with (OUT/'logs/summary.log').open('a') as log:
                code=subprocess.call([sys.executable,'-u',str(OUT/'source/summarize.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if code:
                save(OUT/'status/finalizer.json',dict(status='failed',stage='summarize',exit_code=code));return code
            last=exports
        if done:
            with (OUT/'logs/verification.log').open('a') as log:
                code=subprocess.call([sys.executable,'-u',str(OUT/'source/verify.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            save(OUT/'status/finalizer.json',dict(status='complete' if code==0 else 'failed',stage='verify',exit_code=code,phase=0,later_phases_scheduled=False))
            return code
        save(OUT/'status/finalizer.json',dict(status='waiting_for_exports',prediction_exports=exports,completed_jobs=sum(j['state']=='complete' for j in tasks),total_jobs=len(tasks),phase=0,later_phases_scheduled=False))
        time.sleep(20)

if __name__=='__main__':raise SystemExit(main())
