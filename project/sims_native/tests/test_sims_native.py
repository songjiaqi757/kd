import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

NATIVE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(NATIVE / 'scripts'), str(NATIVE / 'src')]
from rdid_sims.metrics import sentiment_metrics
from rdid_sims.probe import TeacherProbe
from rdid_sims.student import SubsetFusion
from sims_native_common import five_class_index
from run_sims_gpu_queue import can_launch, update_measured_reservations
from run_sims_stage import teacher_timeout_seconds


def test_mae_preserves_native_units_and_accepts_intermediate_values():
    metric = sentiment_metrics([0.6, -.2, 0], [.2, .2, 0])
    assert metric['mae'] == pytest.approx(.8 / 3)
    assert 'acc7' not in metric
    assert metric['acc2'] == pytest.approx(2 / 3)


def test_official_zero_and_closed_right_class_boundaries():
    assert [five_class_index(v) for v in [-1, -.7, -.1, 0, .1, .7, 1]] == [0, 0, 1, 2, 2, 3, 4]
    result = sentiment_metrics([-1, -.7, -.1, 0, .1, .7, 1], [-1, -.7, -.1, 0, .1, .7, 1])
    assert result['acc2'] == result['acc3'] == result['acc5'] == 1
    assert sentiment_metrics([0, .2], [-.2, 0])['acc2'] == .5
    with pytest.raises(ValueError):
        five_class_index(1.2)


def test_metrics_match_independent_official_interval_reference():
    rng = np.random.default_rng(819)
    target = rng.choice(np.arange(-1., 1.01, .2), size=100)
    prediction = rng.uniform(-1.5, 1.5, size=100)
    target, prediction = np.clip(target, -1, 1), np.clip(prediction, -1, 1)
    result = sentiment_metrics(target, prediction)
    assert result['mae'] == pytest.approx(np.abs(target-prediction).mean())
    assert result['acc2'] == np.mean((target > 0) == (prediction > 0))
    for count, boundaries in [(3, [-1.01,-.1,.1,1.01]), (5, [-1.01,-.7,-.1,.1,.7,1.01])]:
        values = []
        for a in (target, prediction):
            classes = a.copy()
            for index in range(count):
                classes[(a > boundaries[index]) & (a <= boundaries[index+1])] = index
            values.append(classes)
        assert result[f'acc{count}'] == np.mean(values[0] == values[1])


def test_native_probe_and_fusion_heads_have_five_classes_and_unit_range():
    probe = TeacherProbe(input_size=8, hidden_size=16)
    fusion = SubsetFusion(hidden_size=16, tokens_per_modality=2, layers=1, heads=2, ffn_size=32)
    for output in [probe(torch.randn(4, 8)), fusion({m: torch.randn(4, 2, 16) for m in ('t','a','v')}, 'tav')]:
        assert output['classification_logits'].shape == (4, 5)
        assert output['regression'].abs().max() <= 1


def test_reserves_memory_before_cuda_allocation_and_releases_only_measured_headroom():
    active = [{'id': 'first', 'kind': 'train', 'memory_mib': 34000}]
    job = {'id': 'second', 'kind': 'train', 'memory_mib': 34000}
    assert can_launch(job, active, 85000, {}, 6000, 100000)
    assert not can_launch(job, active, 60000, {}, 6000, 100000)
    active *= 2
    assert not can_launch(job, active, 85000, {}, 6000, 100000)


def test_reservation_learning_requires_completed_full_run_and_matching_family(tmp_path):
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({'epochs_run': 20, 'peak_gpu_reserved_gib': 10}))
    jobs = [{'id': 'done', 'run': str(tmp_path), 'resource_family': 'sims_subset', 'memory_mib': 34000},
            {'id': 'pending', 'run': 'absent', 'resource_family': 'sims_subset', 'memory_mib': 34000},
            {'id': 'other', 'run': 'absent', 'resource_family': 'simsv2_subset', 'memory_mib': 34000}]
    states = {j['id']: {'state': 'complete' if j['id'] == 'done' else 'pending'} for j in jobs}
    update_measured_reservations(jobs, states)
    assert jobs[1]['memory_mib'] == 16000
    assert jobs[2]['memory_mib'] == 34000


def test_dual_checkpoint_selection_uses_native_official_acc2_and_earliest_tie(tmp_path):
    from evaluate_main_table_test import selected_checkpoint
    rows = [{'epoch': 1, 'valid_metrics': {'mae': .3, 'acc2': .8, 'acc2_nonzero': .9}},
            {'epoch': 2, 'valid_metrics': {'mae': .2, 'acc2': .9, 'acc2_nonzero': .7}},
            {'epoch': 3, 'valid_metrics': {'mae': .25, 'acc2': .9, 'acc2_nonzero': 1.0}}]
    (tmp_path / 'history.json').write_text(json.dumps(rows))
    for policy in ['valid_mae', 'valid_acc2']:
        torch.save({'epoch': 2, 'valid_metrics': rows[1]['valid_metrics'], 'checkpoint_selection': policy},
                   tmp_path / f'best_{policy}.pt')
        assert selected_checkpoint(tmp_path, policy)[1] == 2


def test_teacher_watchdog_bounds_a_stuck_decoder_but_allows_cold_loading():
    assert teacher_timeout_seconds(None, 7, 0, 100) == 1100
    assert teacher_timeout_seconds({'pid': 6, 'phase': 'reading_media', 'updated_at': 0}, 7, 100, 110) == 1190
    assert teacher_timeout_seconds({'pid': 7, 'phase': 'loading_model', 'updated_at': 0}, 7, 0, 1100) == 100
    assert teacher_timeout_seconds({'pid': 7, 'phase': 'reading_media', 'updated_at': 10}, 7, 0, 191) == 0
    assert teacher_timeout_seconds({'pid': 7, 'phase': 'model_inference', 'updated_at': 100}, 7, 0, 120) == 160
