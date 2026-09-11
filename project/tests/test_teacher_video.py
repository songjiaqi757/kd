from types import SimpleNamespace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from rdid_mosei.teacher_video import (
    assert_timing_compatible, read_teacher_media, sampled_video_kwargs, video_preprocessing_audit,
)


def test_actual_sampled_fps_is_scalar_and_disables_resampling():
    assert sampled_video_kwargs([object()], {'fps':[1.9626168224]}) == {
        'fps':1.9626168224, 'do_sample_frames':False}
    assert sampled_video_kwargs(None, {'fps':[]}) == {}


@pytest.mark.parametrize('fps', [0, -1, float('nan'), float('inf')])
def test_invalid_fps_rejected(fps):
    with pytest.raises(ValueError, match='invalid sampled'):
        sampled_video_kwargs([object()], {'fps':[fps]})


def test_ambiguous_multiple_videos_rejected():
    with pytest.raises(ValueError, match='exactly one'):
        sampled_video_kwargs([object(),object()], {'fps':[2.,4.]})


def test_missing_policy_means_legacy_and_cannot_mix():
    assert_timing_compatible({}, 'legacy_v1')
    assert_timing_compatible({'video_timing_policy':'sampled_fps_v2'}, 'sampled_fps_v2')
    with pytest.raises(ValueError, match='use a new output'):
        assert_timing_compatible({}, 'sampled_fps_v2')


def test_time_grid_consistency_checked():
    p = SimpleNamespace(video_processor=SimpleNamespace(temporal_patch_size=2))
    inputs = {'video_second_per_grid':torch.tensor([1.]), 'video_grid_thw':torch.tensor([[7,22,30]])}
    assert video_preprocessing_audit(inputs, p, {'fps':2.})['sampled_fps'] == 2.
    with pytest.raises(ValueError, match='time spacing'):
        video_preprocessing_audit(inputs, p, {'fps':4.})
    assert video_preprocessing_audit({}, p, {}) == {}


def test_media_wrapper_retains_fps_and_legacy_remains_reproducible(monkeypatch):
    import qwen_omni_utils
    def fake(conversation, **kwargs):
        result = (None, None, ['video'])
        return (*result, {'fps':[4.]}) if kwargs.get('return_video_kwargs') else result
    monkeypatch.setattr(qwen_omni_utils, 'process_mm_info', fake)
    assert read_teacher_media([], 'sampled_fps_v2')[-1] == {'fps':4.,'do_sample_frames':False}
    assert read_teacher_media([], 'legacy_v1')[-1] == {}
