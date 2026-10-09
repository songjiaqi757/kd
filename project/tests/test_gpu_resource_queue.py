from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_gpu_resource_queue import can_launch, ready_jobs, validate_plan, marker_complete
import run_gpu_resource_queue as queue


def job(name='train', kind='train', memory=34000, dependencies=None):
    return dict(id=name, kind=kind, memory_mib=memory, disk_mib=2000,
                dependencies=dependencies or [], command=['python', 'trainer', '--retain-best-mae-and-acc2'])


def test_initializing_jobs_reserve_memory_before_cuda_allocation():
    active = [job('initializing')]
    assert not can_launch(job(), active, 70000, {}, 6000, 100000)
    assert can_launch(job(), active, 80000, {}, 6000, 100000)


def test_reservations_do_not_double_count_observed_gpu_memory():
    active = [job('existing')]
    assert can_launch(job(), active, 55167, {'existing': 30460}, 6000, 100000)


def test_two_trainers_can_overlap_one_evaluation_if_budget_fits():
    active = [job('a'), job('b')]
    evaluation = job('evaluation', 'evaluate', 8000)
    assert can_launch(evaluation, active, 25651, {'a': 30000, 'b': 30000}, 6000, 100000)
    assert not can_launch(job('c'), active, 85000, {'a': 34000, 'b': 34000}, 6000, 100000)
    assert not can_launch(evaluation, active, 15000, {}, 6000, 100000)


def test_disk_headroom_and_evaluation_priority_and_dependencies():
    train = job('t')
    evaluation = job('e', 'evaluate', 8000, ['t'])
    other = job('u')
    states = {'t': {'state': 'running'}, 'e': {'state': 'pending'}, 'u': {'state': 'pending'}}
    assert [r['id'] for r in ready_jobs([train, evaluation, other], states)] == ['u']
    states['t']['state'] = 'complete'
    assert [r['id'] for r in ready_jobs([train, evaluation, other], states)] == ['e', 'u']
    assert not can_launch(train, [], 85000, {}, 6000, 6000)


def test_plan_enforces_retention_and_test_authorization():
    train = job()
    evaluation = job('e', 'evaluate', 8000, ['train'])
    evaluation['command'] = ['python', 'evaluate', '--allow-official-test']
    plan = dict(dataset='mosei', jobs=[train, evaluation])
    with pytest.raises(ValueError, match='explicitly enabled'):
        validate_plan(plan)
    plan['allow_official_test'] = True
    validate_plan(plan)
    train['command'].append('--save-every-epoch')
    with pytest.raises(ValueError, match='Epoch'):
        validate_plan(plan)


def test_failed_status_is_not_completion(tmp_path):
    marker = tmp_path / 'status.json'
    task = {'completion_marker': str(marker)}
    assert not marker_complete(task)
    marker.write_text('{"status":"failed"}')
    assert not marker_complete(task)
    marker.write_text('{"status":"complete"}')
    assert marker_complete(task)


def test_adoption_tracks_process_identity_without_restarting(monkeypatch):
    monkeypatch.setattr(queue, 'identity', lambda pid: 'original')
    process = queue.AdoptedProcess(123, 'original')
    assert process.poll() is None
    monkeypatch.setattr(queue, 'identity', lambda pid: 'reused_pid')
    assert process.poll() == 0
    monkeypatch.setattr(queue, 'identity', lambda pid: None)
    assert process.poll() == 0


def test_observed_routing_peak_prevents_two_large_trainers():
    routed = job('routing', memory=50000)
    assert not can_launch(job('second_routing', memory=50000), [routed], 39000,
                          {'routing': 46000}, 6000, 100000)
    assert can_launch(job('eval', 'evaluate', 20000), [routed], 39000,
                      {'routing': 46000}, 6000, 100000)
