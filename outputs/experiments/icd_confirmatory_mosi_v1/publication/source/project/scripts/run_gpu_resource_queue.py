#!/usr/bin/env python3
"""Schedule independent training/evaluation DAG jobs with peak-memory reservations."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def atomic_json(value, path):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def identity(pid):
    try:
        # Fields after comm start at field 3; starttime is field 22.
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def is_descendant(pid, ancestor):
    for _ in range(30):
        if pid == ancestor:
            return True
        if pid <= 1:
            return False
        try:
            pid = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
        except FileNotFoundError:
            return False
    return False


def gpu_snapshot(device):
    data = subprocess.check_output(['nvidia-smi', '-i', str(device),
        '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    uuid, free = [item.strip() for item in data.strip().split(',')]
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                        '--format=csv,noheader,nounits'], text=True)
    used = {}
    for line in processes.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) == 3 and parts[0] == uuid:
            used[int(parts[1])] = int(parts[2])
    return int(free), used


def can_launch(job, active, free_mib, used_by_reservation, safety_mib, disk_free_mib):
    # Account for jobs that have not allocated their peak (or even loaded CUDA) yet.
    pending_allocation = sum(max(0, row['memory_mib'] - used_by_reservation.get(row['id'], 0))
                             for row in active)
    if free_mib - pending_allocation < job['memory_mib'] + safety_mib:
        return False
    if sum(row['kind'] == job['kind'] for row in active) >= 2:
        return False
    disk_reservation = sum(row.get('disk_mib', 0) for row in active)
    return disk_free_mib >= 5120 + disk_reservation + job.get('disk_mib', 0)


def ready_jobs(jobs, states):
    return sorted((job for job in jobs if states[job['id']]['state'] == 'pending'
                   and all(states[dep]['state'] == 'complete' for dep in job['dependencies'])),
                  key=lambda job: job['kind'] != 'evaluate')


def validate_plan(plan):
    if plan['dataset'] not in {'mosi', 'mosei'}:
        raise ValueError('Unsupported dataset')
    ids = [job['id'] for job in plan['jobs']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate job IDs')
    seen = set()
    for job in plan['jobs']:
        if not set(job['dependencies']) <= seen:
            raise ValueError('Dependencies must precede their job')
        seen.add(job['id'])
        if job['kind'] not in {'train', 'evaluate'} or job['memory_mib'] <= 0:
            raise ValueError('Invalid resource request')
        command = job['command']
        if '--save-every-epoch' in command:
            raise ValueError('Epoch checkpoint saving is disabled')
        if '--allow-official-test' in command and (
                not plan.get('allow_official_test') or job['kind'] != 'evaluate'):
            raise ValueError('Official test must be explicitly enabled by the locked plan')
        if job['kind'] == 'train' and '--retain-best-mae-and-acc2' not in command:
            raise ValueError('Dual-best checkpoint retention is required')


def marker_complete(job):
    path = Path(job['completion_marker'])
    return path.exists() and json.loads(path.read_text()).get('status') == 'complete'


class AdoptedProcess:
    """Track a surviving child of the previous scheduler without sending it signals.

    A non-parent cannot recover its exit status. Completion therefore also requires
    the job's explicit success marker, checked by the ordinary completion path.
    """
    def __init__(self, pid, start_ticks):
        self.pid = pid
        self.start_ticks = start_ticks

    def poll(self):
        return None if identity(self.pid) == self.start_ticks else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--adopt-running', action='store_true',
                        help='Adopt verified surviving children after replacing a scheduler')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    if args.dry_run:
        print(json.dumps({'jobs': len(plan['jobs']), 'dataset': plan['dataset'],
                          'device': plan['device'], 'allow_official_test': plan.get('allow_official_test', False)}))
        return
    directory = args.plan.resolve().parent
    lock = (ROOT / f'outputs/.resource_queue_gpu{plan["device"]}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_file = directory / 'queue_status.json'
    active = {}
    states = {job['id']: {'state': 'complete' if marker_complete(job) else 'pending'} for job in plan['jobs']}
    if status_file.exists():
        previous = json.loads(status_file.read_text())
        for name, item in previous['jobs'].items():
            if item.get('state') == 'running' and identity(item['pid']) == item.get('start_ticks'):
                if not args.adopt_running:
                    raise RuntimeError(f'Previous queue child still alive: {name}; do not duplicate')
                job = next(job for job in plan['jobs'] if job['id'] == name)
                command = Path(f'/proc/{item["pid"]}/cmdline').read_bytes().split(b'\0')
                command = [value.decode() for value in command if value]
                if command not in (job['command'], job['command'] + ['--resume']):
                    raise RuntimeError(f'Surviving child command differs from plan: {name}')
                active[name] = {'process': AdoptedProcess(item['pid'], item['start_ticks']), 'job': job}
                states[name] = item
                print(json.dumps({'event': 'adopted', 'job': name, 'pid': item['pid']}), flush=True)
        for name, item in previous['jobs'].items():
            if name in states and item['state'] == 'failed':
                states[name] = item
    next_launch = 0.0
    while True:
        for name, entry in list(active.items()):
            result = entry['process'].poll()
            if result is None:
                continue
            success = result == 0 and marker_complete(entry['job'])
            states[name].update(state='complete' if success else 'failed', returncode=result,
                                finished_at=time.time())
            del active[name]
            print(json.dumps({'event': 'finished', 'job': name, **states[name]}), flush=True)
        for job in plan['jobs']:
            if states[job['id']]['state'] == 'pending' and any(
                    states[dep]['state'] in {'failed', 'blocked'} for dep in job['dependencies']):
                states[job['id']]['state'] = 'blocked'
        reservations = [dict(entry['job'], pid=entry['process'].pid) for entry in active.values()]
        reservations += [entry for entry in plan.get('external_reservations', [])
                         if identity(entry['pid']) == entry['start_ticks']]
        free, observed = gpu_snapshot(plan['device'])
        allocations = {row['id']: sum(mem for pid, mem in observed.items() if is_descendant(pid, row['pid']))
                       for row in reservations}
        paused = (directory / 'STOP_AFTER_CURRENT').exists()
        # A failure requires inspection; existing children finish, no new jobs launch.
        failed = any(item['state'] == 'failed' for item in states.values())
        if not paused and not failed and time.monotonic() >= next_launch:
            for job in ready_jobs(plan['jobs'], states):
                if not can_launch(job, reservations, free, allocations, plan['safety_mib'],
                                  shutil.disk_usage(directory).free // 1024**2):
                    continue
                command = list(job['command'])
                run = Path(job['run'])
                if job['kind'] == 'train' and (run / 'run_config.json').exists():
                    if not (run / 'last.pt').exists():
                        raise RuntimeError(f'Partial training without last checkpoint: {run}')
                    command.append('--resume')
                logs = directory / '.logs'
                logs.mkdir(exist_ok=True)
                with (logs / f'{job["id"]}.log').open('a') as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                active[job['id']] = {'process': process, 'job': job}
                states[job['id']] = {'state': 'running', 'pid': process.pid,
                                    'start_ticks': identity(process.pid), 'started_at': time.time()}
                print(json.dumps({'event': 'launched', 'job': job['id'], 'pid': process.pid}), flush=True)
                next_launch = time.monotonic() + plan.get('launch_settle_seconds', 30)
                break
        pending = any(item['state'] == 'pending' for item in states.values())
        finished = not active and (failed or paused or not pending)
        state = 'failed' if failed else 'paused' if paused else 'complete' if finished else 'running'
        atomic_json({'status': state, 'pid': os.getpid(), 'updated_at': time.time(),
                     'dataset': plan['dataset'], 'device': plan['device'], 'gpu_free_mib': free,
                     'external_active': [r['id'] for r in reservations if r['id'].startswith('external_')],
                     'allow_official_test': plan.get('allow_official_test', False), 'jobs': states}, status_file)
        if finished:
            return
        time.sleep(10)


if __name__ == '__main__':
    main()
