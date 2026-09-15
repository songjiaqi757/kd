import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_tav_live_queue as queue


def limits():
    return {"max_jobs":5,"slots_per_gpu":{"0":3,"1":2},"gpu_headroom_gib":8,
            "method_budget_gib":{"frozen":16,"adapted":34}}


def occupants():
    return {"M0_seed13":{"gpu":0},"M1_seed42":{"gpu":0},
            "M6_seed13":{"gpu":1},"M1_seed2026":{"gpu":1}}


def test_fifth_task_fits_gpu0_and_sixth_is_rejected():
    active=occupants()
    state={"used_mib":22286,"total_mib":85651,"utilization":40}
    assert queue.admit(state,0,active,'M3_seed42',limits())
    assert not queue.admit({**state,'used_mib':40690},1,active,'M3_seed42',limits())
    active['M3_seed42']={'gpu':0}
    assert not queue.admit(state,0,active,'M4_seed42',limits())


def test_three_adapted_models_are_not_packed_into_one_gpu():
    active={'M3_seed42':{'gpu':0},'M6_seed13':{'gpu':0}}
    assert not queue.admit({'used_mib':10000,'total_mib':85651,'utilization':20},0,active,'M4_seed42',limits())


def test_completion_releases_slot_and_replication_waits_only_for_own_audit():
    active=occupants()
    state={'used_mib':22286,'total_mib':85651,'utilization':0}
    pending=[('M0',42),('M3',42),('M3',2026)]
    assert queue.candidate(pending,{'M1','M3'},0,active,state,limits())==('M3',42)
    active['M3_seed42']={'gpu':0}
    assert queue.candidate(pending,{'M1','M3'},0,active,state,limits()) is None
    del active['M3_seed42']
    pending.remove(('M3',42))
    assert queue.candidate(pending,{'M1','M3'},0,active,state,limits())==('M3',2026)


def test_unknown_gpu_or_actual_memory_pressure_blocks_admission():
    assert not queue.admit(None,0,occupants(),'M3_seed42',limits())
    assert not queue.admit({'used_mib':60000,'total_mib':85651,'utilization':0},0,occupants(),'M3_seed42',limits())


def test_identity_detects_pid_reuse():
    token=queue.identity(os.getpid())
    assert queue.alive({'pid':os.getpid(),'identity':token})
    assert not queue.alive({'pid':os.getpid(),'identity':{**token,'start_ticks':'wrong'}})


def test_limits_are_reloaded_and_invalid_expansion_rejected(tmp_path):
    path=tmp_path/'limits.json'
    path.write_text(json.dumps(limits()))
    assert queue.load_limits(path)['max_jobs']==5
    path.write_text(json.dumps({**limits(),'max_jobs':4}))
    assert queue.load_limits(path)['max_jobs']==4
    path.write_text(json.dumps({**limits(),'max_jobs':6}))
    with pytest.raises(ValueError):queue.load_limits(path)


def test_controller_stop_preserves_adopted_training_pid(tmp_path,monkeypatch):
    root=tmp_path/'root'
    scripts=root/'project/scripts';scripts.mkdir(parents=True)
    trainer=scripts/'train_tav_distillation.py'
    trainer.write_text('import time\nwhile True: time.sleep(.1)\n')
    base=root/'experiment';(base/'assets').mkdir(parents=True)
    output=base/'students/M0_seed13'
    args=[sys.executable,str(trainer),'--method','M0','--seed','13','--output',str(output)]
    child=subprocess.Popen(args,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'})
    try:
        time.sleep(.1)
        token=queue.identity(child.pid)
        item={'pid':child.pid,'identity':token,'gpu':0,'started':time.time()}
        (base/'live_processes.json').write_text(json.dumps({'running':{'M0_seed13':item}}))
        (base/'assets/protocol.json').write_text('{}')
        (base/'plan.json').write_text(json.dumps({'schedule_version':queue.VERSION}))
        (base/'live_limits.json').write_text(json.dumps(limits()))
        monkeypatch.setattr(queue,'ROOT',root)
        monkeypatch.setattr(queue,'verify',lambda *a,**k:None)
        monkeypatch.setattr(queue,'complete',lambda *a:False)
        monkeypatch.setattr(queue.signal,'signal',lambda *a:None)
        def interrupt(gpu):raise KeyboardInterrupt('controller handover')
        monkeypatch.setattr(queue,'gpu_state',interrupt)
        queue.run(base)
        assert child.poll() is None
        assert queue.identity(child.pid)==token
        state=json.loads((base/'status.json').read_text())
        assert state['training_processes_left_running'] is True
        assert state['running']['M0_seed13']['pid']==child.pid
    finally:
        child.terminate();child.wait(timeout=5)
