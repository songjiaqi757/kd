"""Attach the verified extension while retaining all existing GPU children."""
from pathlib import Path
from datetime import datetime, timezone
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time

from teacher_test_common import ROOT, atomic
sys.path.insert(0, str(ROOT/'project/sims_native/scripts'))
from run_sims_gpu_queue import identity

Q=ROOT/'outputs/experiments/sims_native_v1/queue'
OUTPUT=ROOT/'outputs/experiments/teacher_test_v1'
SCHEDULER=ROOT/'project/sims_native/scripts/run_sims_gpu_queue.py'


def start():
    command=[sys.executable,'-u',str(SCHEDULER),'--plan',str(Q/'plan.json'),'--adopt-running']
    with (Q/'scheduler.log').open('a') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return process,command


def main():
    lock=(Q/'teacher_test_extension.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=json.loads((Q/'plan.json').read_text())
    new=json.loads((Q/'plan.teacher_test_candidate.json').read_text())
    assert len(old['jobs'])==145 and len(new['jobs'])==150
    assert new['jobs'][:145]==old['jobs']
    assert all(new['source_sha256'][p]==digest for p,digest in old['source_sha256'].items()), 'Existing frozen source identities changed'
    verification=json.loads((OUTPUT/'preflight_verification.json').read_text())
    assert len(verification)==4 and all(row['status']=='passed' for row in verification.values())
    subprocess.run([sys.executable,str(SCHEDULER),'--plan',str(Q/'plan.teacher_test_candidate.json'),'--dry-run'],check=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup=Q/'extensions'/('teacher_test_'+stamp);backup.mkdir(parents=True)
    for name in ['plan.json','queue_status.json','process.json']:
        shutil.copy2(Q/name,backup/name)
    state=json.loads((Q/'queue_status.json').read_text())
    assert state['status']=='running'
    parent=state['pid']
    cmd=Path(f'/proc/{parent}/cmdline').read_bytes().split(b'\0')
    assert str(SCHEDULER).encode() in cmd and str(Q/'plan.json').encode() in cmd
    previous_active={name:row for name,row in state['jobs'].items() if row['state']=='running'}
    for name,row in previous_active.items():
        assert identity(row['pid'])==row['start_ticks'], f'Original child changed before adoption: {name}'
    parent_identity=identity(parent)
    os.kill(parent,signal.SIGTERM)
    for _ in range(100):
        if identity(parent)!=parent_identity:break
        time.sleep(.1)
    assert identity(parent)!=parent_identity, 'Old scheduler did not release resource lock'
    os.replace(Q/'plan.teacher_test_candidate.json',Q/'plan.json')
    process=None
    try:
        process,command=start()
        deadline=time.time()+25
        while time.time()<deadline:
            assert process.poll() is None, 'Replacement scheduler exited'
            current=json.loads((Q/'queue_status.json').read_text())
            if current.get('pid')==process.pid and len(current['jobs'])==150:break
            time.sleep(.5)
        else:raise RuntimeError('New queue status was not published')
        jobs_by_id={j['id']:j for j in old['jobs']}
        children={}
        for name,row in previous_active.items():
            alive=identity(row['pid'])==row['start_ticks']
            marker=Path(jobs_by_id[name]['completion_marker'])
            complete=marker.exists() and json.loads(marker.read_text()).get('status')=='complete'
            assert alive or complete, f'Original child unexpectedly stopped: {name}'
            children[name]={'pid':row['pid'],'same_process_alive':alive,'completed':complete}
        added=[j['id'] for j in new['jobs'][145:]]
        if current['jobs']['sims_summary']['state']!='complete':
            assert all(current['jobs'][name]['state']=='pending' for name in added), 'Teacher test started before the student campaign finished'
        record={'status':'registered','registered_at':datetime.now(timezone.utc).isoformat(),
            'original_job_count':145,'new_job_count':150,'added_jobs':added,'old_scheduler_pid':parent,
            'new_scheduler_pid':process.pid,'retained_children':children,'backup':str(backup),
            'teacher_tests_currently':{name:current['jobs'][name]['state'] for name in added}}
        old_process=json.loads((backup/'process.json').read_text())
        atomic({**old_process,'pid':process.pid,'started_unix':time.time(),'command':command,
            'teacher_test_extension':str(OUTPUT/'plan.json')},Q/'process.json')
        atomic(record,OUTPUT/'registration.json')
        print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate();process.wait(timeout=10)
        shutil.copy2(backup/'plan.json',Q/'plan.json')
        restored,command=start()
        atomic({'pid':restored.pid,'started_unix':time.time(),'command':command,'restored_after_extension_failure':True},Q/'process.json')
        raise


if __name__=='__main__':main()
