#!/usr/bin/env python3
"""Print the saved live experiment status without touching GPUs."""
import json
from datetime import datetime
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/experiments/video_source_v2'


def main():
    path=BASE/'status.json'
    if not path.exists():
        print('实验队列尚未启动。');return
    s=json.loads(path.read_text())
    print('Video source v2：教师时间修正 + 三模态对等 LoRA')
    print(f"状态：{s['status']}    阶段：{s['stage']}")
    print(f"更新：{datetime.fromtimestamp(s['updated_at']).astimezone().isoformat(timespec='seconds')}（{int(time.time()-s['updated_at'])} 秒前）")
    if 'error' in s:print('错误：',s['error'])
    if 'progress' in s:print('进度：',json.dumps(s['progress'],ensure_ascii=False))
    if 'gpus' in s:print('GPU：',json.dumps(s['gpus'],ensure_ascii=False))
    for job in s.get('active',[]):
        print(f"GPU{job['gpu']} {job['run']}：{json.dumps(job['progress'],ensure_ascii=False)}")
        print('  日志：',job['log'])
    if 'current_log' in s:print('当前详细日志：',s['current_log'])
    plan_path=BASE/'plan.json'
    plan=json.loads(plan_path.read_text()) if plan_path.exists() else {}
    schedule=plan.get('student_schedule',{})
    modes=plan.get('student_modes',['frozen_video','video_lora','ta_only'])
    seeds=plan.get('student_seeds',[13,42,2026])
    if schedule and not schedule.get('controls_enabled',True):
        modes=schedule['priority_modes']
        print('本轮仅运行：',', '.join(modes))
        print('暂缓（不自动启动）：',', '.join(schedule['deferred_modes']))
    completed=[]
    for seed in seeds:
        for mode in modes:
            p=BASE/'students'/f'{mode}_seed{seed}'/'status.json'
            if p.exists() and json.loads(p.read_text()).get('status')=='complete':completed.append(f'{mode}_seed{seed}')
    print(f'本轮学生实验完成：{len(completed)}/{len(seeds)*len(modes)}')
    if completed:print('已完成：',', '.join(completed))
    print('C2：已按用户要求停止，不重跑。')
    if s['status']=='complete' or s['stage']=='controls_deferred':
        print('结果：',ROOT/'docs/video_source_v2_results.md')


if __name__=='__main__':
    main()
