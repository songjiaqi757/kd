"""Flock-coordinated, memory-aware queue across the three shared-filesystem hosts."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

BASE=Path(__file__).resolve().parent
ROOT=Path('/ai/sjq/kd')
PYTHON='/ai/sjq/miniconda3/envs/kd/bin/python'
ENV=os.environ.copy()
ENV.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
           TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',
           USE_TF='0',USE_FLAX='0',HF_HUB_DISABLE_PROGRESS_BARS='1',TRANSFORMERS_VERBOSITY='error')

def read(p):return json.loads(p.read_text())
def save(p,value):
    temporary=p.with_name(p.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    os.replace(temporary,p)

@contextmanager
def queue():
    with (BASE/'.queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=read(BASE/'queue.json')
        yield data
        save(BASE/'queue.json',data)
        save(BASE/'campaign_status.json',dict(updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
             counts={s:sum(j['state']==s for j in data) for s in ['pending','running','complete','failed','skipped']},jobs=data))

def command(job):
    spec=job['spec']; run=BASE/'runs'/spec['id']; plan=read(BASE/'plan.json')
    if job['kind']=='test':
        return [PYTHON,'-u',str(BASE/'evaluate_mosi.py'),'--run',str(run),'--output',str(BASE/'tests'/job['id']),
                '--test-manifest',str(BASE/'assets/official_test_windowed.jsonl'),'--device','cuda:0',
                '--batch-size','8','--num-workers','2','--checkpoint-selection',job['policy'],'--allow-official-test']
    cmd=[PYTHON,'-u',str(BASE/'train_mosi.py'),'--method','full_kd' if spec['mechanism']=='none' else 'uniform_interaction',
         '--mechanism',spec['mechanism'],'--loss-kind',spec['loss_kind'],'--matrix-seed',str(spec['matrix_seed']),
         '--experiment-id',spec['id'],'--assets',str(BASE/'assets/protocol.json'),'--output',str(run),
         '--seed',str(spec['seed']),'--device','cuda:0','--batch-size','8','--epochs',str(plan['epochs']),'--patience',str(plan['patience']),
         '--retain-best-mae-and-acc2','--checkpoint-selection','valid_mae','--learning-rate','0.0001',
         '--weight-decay','0.01','--num-workers','2','--progress-every','20','--diagnostic-every','100',
         '--task-ce',str(spec['task_ce']),'--lambda-interaction',str(spec['lambda_interaction'])]
    if (run/'last.pt').exists():cmd.append('--resume')
    return cmd

def gpu_info():
    values=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    return values[0].strip(),int(values[1]),len([p for p in processes if p.strip().isdigit()])

def verify_artifacts(job):
    out=BASE/('runs' if job['kind']=='train' else 'tests')/job['id']
    assert read(out/'status.json')['status']=='complete'
    if job['kind']=='train':
        expected={'best_valid_mae.pt','best_valid_acc2_nonzero.pt','last.pt'}
        assert {p.name for p in out.glob('*.pt')}==expected
        assert not list((out/'checkpoints').glob('*.pt'))
        config=read(out/'run_config.json')
        assert config['seed']==13 and config['epochs']==30 and config['patience']==7
        assert read(out/'report.json')['epochs_run']<=30
        for row in [json.loads(line) for line in (out/'loss_diagnostics.jsonl').open()]:
            assert all(__import__('math').isfinite(float(row[k])) for k in ['base_loss','auxiliary','total'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);args=p.parse_args()
    assert read(BASE/'preflight/asset_audit.json')['passed']
    assert read(BASE/'preflight/objective_checks.json')['passed']
    assert read(BASE/'preflight/smoke_audit.json')['passed']
    frozen=read(BASE/'source_fingerprints.json')
    for path,digest in frozen.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    with (BASE/'workers'/(args.worker+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            job=None
            with queue() as data:
                gpu,free,count=gpu_info()
                running=[j for j in data if j['state']=='running' and j.get('gpu')==gpu]
                recent=sum(38000 for j in running if time.time()-j['started_unix']<90)
                terminal=all(j['state'] in ['complete','failed','skipped'] for j in data)
                if len(running)<2 and count<2 and free-recent>=38000 and shutil.disk_usage(BASE).free>8*1024**3:
                    for j in sorted(data,key=lambda j:j['kind']!='test'):
                        if j['state']!='pending':continue
                        if j.get('resume_gpu') and j['resume_gpu']!=gpu:continue
                        if j['kind']=='test':
                            dependency=next(t for t in data if t['id']==j['dependency'])
                            if dependency['state'] in ['failed','skipped']:
                                j.update(state='skipped',reason='training prerequisite failed');continue
                            if dependency['state']!='complete':continue
                        j.update(state='running',worker=args.worker,gpu=gpu,started_unix=time.time(),attempts=j.get('attempts',0)+1)
                        job=dict(j);break
            if job is None:
                save(BASE/'workers'/(args.worker+'.json'),dict(status='finished' if terminal else 'waiting_for_slot',pid=os.getpid(),gpu=gpu,free_mib=free,compute_processes=count))
                if terminal:return
                time.sleep(5);continue
            cmd=command(job)
            logfile=BASE/'logs'/(job['id']+'.log')
            with logfile.open('a') as log:
                log.write(json.dumps(dict(worker=args.worker,command=cmd))+'\n');log.flush()
                child=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(BASE/'workers'/(args.worker+'.json'),dict(status='running',pid=os.getpid(),child_pid=child.pid,job=job['id'],gpu=gpu))
                while child.poll() is None:time.sleep(5)
                code=child.returncode
            if code==0:
                try:verify_artifacts(job)
                except Exception as error:
                    code=99
                    with logfile.open('a') as log:log.write('Artifact verification failed: '+repr(error)+'\n')
            with queue() as data:
                current=next(j for j in data if j['id']==job['id'])
                current.update(state='complete' if code==0 else 'failed',exit_code=code,finished_unix=time.time(),log=str(logfile))
                if code and job['kind']=='train':
                    for j in data:
                        if j.get('dependency')==job['id']:j.update(state='skipped',reason='training failed; see log')
            subprocess.run([PYTHON,str(BASE/'summarize.py')],cwd=ROOT,env=ENV)
            print(json.dumps(dict(job=job['id'],exit_code=code)),flush=True)

if __name__=='__main__':main()
