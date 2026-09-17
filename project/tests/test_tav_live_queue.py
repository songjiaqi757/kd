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


def test_calibrated_budgets_refill_m6_after_frozen_job_finishes(tmp_path):
    active = {'M3_seed42': {'gpu': 0}, 'M0_seed42': {'gpu': 0},
              'M0_seed2026': {'gpu': 1}, 'M3_seed2026': {'gpu': 1}}
    state = {'used_mib': 38768, 'total_mib': 85651, 'utilization': 30}
    pending = [('M4', 42), ('M6', 42), ('M6', 2026)]
    assert not queue.admit(state, 0, active, 'M6_seed42', limits())
    calibrated = {**limits(), 'method_budget_gib': {'frozen': 10, 'adapted': 32},
                  'pending_priority': ['M6_seed42', 'M6_seed2026']}
    path = tmp_path / 'limits.json'
    path.write_text(json.dumps(calibrated))
    calibrated = queue.load_limits(path)
    assert queue.candidate(pending, {'M4', 'M6'}, 0, active, state, calibrated) == ('M6', 42)
    assert not queue.admit(state, 1, active, 'M6_seed42', calibrated)  # GPU1 slots full
    assert not queue.admit({**state, 'used_mib': 46000}, 0, active, 'M6_seed42', calibrated)
    active['M6_seed42'] = {'gpu': 0}
    assert not queue.admit(state, 0, active, 'M6_seed2026', calibrated)  # five jobs
    assert not queue.admit(state, 0, {'M3_seed42': {'gpu': 0}, 'M6_seed42': {'gpu': 0}},
                           'M4_seed42', calibrated)  # three adapted jobs exceed capacity


@pytest.mark.parametrize('budgets', [{'frozen': 9, 'adapted': 32}, {'frozen': 10, 'adapted': 31}])
def test_budgets_below_calibrated_floor_rejected(tmp_path, budgets):
    path = tmp_path / 'limits.json'
    path.write_text(json.dumps({**limits(), 'method_budget_gib': budgets}))
    with pytest.raises(ValueError, match='invalid resource limits'):
        queue.load_limits(path)


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


def test_six_jobs_admit_m6_on_gpu1_with_existing_five_unchanged(tmp_path):
    active = {'M3_seed42': {'gpu': 0}, 'M0_seed42': {'gpu': 0}, 'M6_seed42': {'gpu': 0},
              'M0_seed2026': {'gpu': 1}, 'M3_seed2026': {'gpu': 1}}
    original = dict(active)
    settings = {**limits(), 'max_jobs': 6, 'slots_per_gpu': {'0': 3, '1': 3},
                'method_budget_gib': {'frozen': 10, 'adapted': 32},
                'pending_priority': ['M6_seed2026', 'M4_seed42', 'M4_seed2026']}
    path = tmp_path / 'limits.json'
    path.write_text(json.dumps(settings))
    settings = queue.load_limits(path)
    pending = [('M6', 2026), ('M4', 42), ('M4', 2026)]
    state = {'used_mib': 38896, 'total_mib': 85651, 'utilization': 17}
    assert queue.candidate(pending, {'M6', 'M4'}, 1, active, state, settings) == ('M6', 2026)
    assert queue.candidate(pending, {'M6', 'M4'}, 0, active, state, settings) is None
    assert active == original
    assert not queue.admit({**state, 'used_mib': 46000}, 1, active, 'M6_seed2026', settings)
    active['M6_seed2026'] = {'gpu': 1}
    assert not queue.admit(state, 1, active, 'M4_seed42', settings)
    assert not queue.admit(state, 0, active, 'M4_seed42', settings)
    path.write_text(json.dumps({**settings, 'max_jobs': 7}))
    with pytest.raises(ValueError, match='invalid resource limits'):
        queue.load_limits(path)


def test_m6_priority_after_completion_preserves_active_jobs():
    active = {'M1_seed42': {'gpu': 0}, 'M3_seed42': {'gpu': 0},
              'M0_seed42': {'gpu': 0}, 'M0_seed2026': {'gpu': 1}, 'M3_seed2026': {'gpu': 1}}
    original = dict(active)
    pending = [('M4', 42), ('M4', 2026), ('M6', 42), ('M6', 2026)]
    settings = {**limits(), 'pending_priority': ['M6_seed42', 'M6_seed2026', 'M4_seed42', 'M4_seed2026']}
    state = {'used_mib': 38000, 'total_mib': 85651, 'utilization': 30}
    assert queue.candidate(pending, {'M4', 'M6'}, 1, active, state, settings) is None
    assert active == original
    # A slot becomes available naturally; priority applies to future admission only.
    del active['M3_seed2026']
    assert queue.candidate(pending, {'M4', 'M6'}, 1, active, state, settings) == ('M6', 42)
    pending.remove(('M6', 42))
    assert queue.candidate(pending, {'M4', 'M6'}, 1, active, state, settings) == ('M6', 2026)
    pending.remove(('M6', 2026))
    assert queue.candidate(pending, {'M4', 'M6'}, 1, active, state, settings) == ('M4', 42)
    assert queue.candidate(pending, {'M4', 'M6'}, 1, active, {**state, 'used_mib': 60000}, settings) is None


def test_priority_hot_reload_and_no_reintroduction(tmp_path):
    path = tmp_path / 'limits.json'
    pending = [('M4', 42), ('M4', 2026), ('M6', 42), ('M6', 2026)]
    path.write_text(json.dumps(limits()))
    assert queue.ordered_pending(pending, queue.load_limits(path)) == pending
    path.write_text(json.dumps({**limits(), 'pending_priority': ['M6_seed42', 'M6_seed2026', 'M1_seed13']}))
    assert queue.ordered_pending(pending, queue.load_limits(path)) == [('M6', 42), ('M6', 2026), ('M4', 42), ('M4', 2026)]
    assert pending == [('M4', 42), ('M4', 2026), ('M6', 42), ('M6', 2026)]


@pytest.mark.parametrize('priority', ['M6_seed42', ['M6_seed42', 'M6_seed42'], ['M5_seed42'], [None]])
def test_invalid_priority_rejected(tmp_path, priority):
    path = tmp_path / 'limits.json'
    path.write_text(json.dumps({**limits(), 'pending_priority': priority}))
    with pytest.raises(ValueError, match='invalid pending priority'):
        queue.load_limits(path)


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
