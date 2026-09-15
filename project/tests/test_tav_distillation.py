import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from rdid_mosei.tav_distillation import TAVDistillationStudent, interaction_loss, interaction_weights
from rdid_mosei.interaction import inverse_mobius
from rdid_mosei.student import SUBSETS
from prepare_tav_protocol import utility_from_train
from train_tav_distillation import compute_loss
from test_video_adaptation import model, inputs
from run_tav_distillation import admission, command
from summarize_tav_distillation import clustered_mae, paired_errors


def student(method):
    reference = model()
    # Start from raw model objects, before adapter injection.
    from transformers import Qwen3Model, WavLMModel, VideoMAEModel
    torch.manual_seed(13)
    t = Qwen3Model(reference.text_encoder.config)
    a = WavLMModel(reference.audio_encoder.config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(113)
        v = VideoMAEModel(reference.video_encoder.config)
    return TAVDistillationStudent(t, a, v, method=method, hidden_size=16,
        video_hidden_size=16, video_layers=2, video_dropout=0, video_checkpointing=False)


@pytest.mark.parametrize("method", [f"M{i}" for i in range(7)])
def test_adaptation_and_loss_reach_only_authorized_parameters(method):
    m = student(method).train()
    out = m(**inputs(), all_subsets=method in ("M4", "M5", "M6"))
    full = out["tav"] if method in ("M4", "M5", "M6") else out
    loss = full["classification_logits"].square().mean()
    if method in ("M4", "M5", "M6"):
        loss = loss + interaction_loss(out, torch.ones(2,7), torch.ones(2,7), torch.ones(7),
                                       torch.ones(2), 0.25, method)[0]
    loss.backward()
    for modality, encoder in (("t", m.text_encoder), ("a", m.audio_encoder), ("v", m.video_encoder)):
        adapted = (method not in ("M0", "M1")) if modality != "v" else method in ("M3", "M4", "M5", "M6")
        assert any(p.requires_grad for p in encoder.parameters()) == adapted
        for name, p in encoder.named_parameters():
            if not p.requires_grad:
                assert p.grad is None
            if adapted and "lora_B" in name:
                assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
    assert m.pools["v"].input_projection.weight.grad.norm() > 0


def test_m3_eval_matches_original_v2_forward():
    m = student("M3").eval()
    from rdid_mosei.video_adaptation import VideoAdaptationStudent
    with torch.no_grad():
        expected = VideoAdaptationStudent.forward(m, **inputs())
        actual = m(**inputs())
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


def test_shared_initialization_and_frozen_backbone_modes():
    m0, m1, m2, m3 = [student(method).train() for method in ("M0", "M1", "M2", "M3")]
    for name, p in m0.pools.named_parameters():
        for m in (m1, m2, m3):
            assert torch.equal(p, dict(m.pools.named_parameters())[name])
    assert not m0.text_encoder.training and not m1.audio_encoder.training
    assert all(not x.training for x in m0.text_encoder.modules())


def test_exact_coordinate_targets_and_weight_scale():
    mean = torch.tensor([[1.,2.,3.,-.2,.4,.5,.1], [0.,0.,0.,0.,0.,0.,0.]])
    values = inverse_mobius(mean, 0.3)
    outputs = {s:{"regression":values[:,i]} for i,s in enumerate(SUBSETS)}
    for method in ("M4", "M5", "M6"):
        loss, weights = interaction_loss(outputs, mean, torch.zeros_like(mean),
            torch.arange(1,8).float(), torch.tensor([1.,.3]), .3, method)
        assert loss.item() < 1e-12
        assert torch.isfinite(weights).all()
        torch.testing.assert_close(weights.mean(-1), torch.ones(2))


def test_m0_is_independent_of_teacher_values():
    out = {"regression":torch.tensor([.1,.2]), "classification_logits":torch.randn(2,7)}
    batch = {"sentiment":torch.tensor([1.,-1.]), "classes":torch.tensor([4,2]), "weights":torch.ones(2),
             "teacher_scores":torch.full((2,),float("nan")), "teacher_logits":torch.full((2,7),float("nan"))}
    assert torch.isfinite(compute_loss(out, batch, torch.device("cpu"), SimpleNamespace(method="M0"), {}))


def test_utility_floor_handles_constant_coordinates():
    x = np.array([[0.,1.],[0.,2.],[0.,3.]])
    raw, weights = utility_from_train(x, np.array([1.,2.,3.]))
    assert raw.tolist() == pytest.approx([0.,1.])
    assert np.isfinite(weights).all() and np.all(weights > 0) and weights.mean() == pytest.approx(1.)


def test_unknown_occupied_or_full_gpu_is_not_admitted():
    assert not admission(None, 0, 2)
    assert not admission({"used_mib": 3000, "total_mib":85651, "utilization":0}, 0, 2)
    state={"used_mib":28000, "total_mib":85651, "utilization":90}
    assert admission(state, 1, 2)
    assert not admission(state, 2, 2)
    assert not admission({**state,"used_mib":60000}, 1, 2)


def test_queue_preserves_interrupted_run_and_resumes_completed_epoch(tmp_path):
    directory=tmp_path/'students/M6_seed13'
    directory.mkdir(parents=True)
    (directory/'run_config.json').write_text('{}')
    assert '--resume' not in command(tmp_path, 'M6', 13)
    assert not directory.exists()
    assert len(list(directory.parent.glob('M6_seed13_interrupted_before_epoch1_*')))==1
    directory.mkdir()
    (directory/'run_config.json').write_text('{}')
    (directory/'last.pt').write_bytes(b'checkpoint')
    assert '--resume' in command(tmp_path, 'M6', 13)


def test_cluster_resampling_keeps_video_members_together():
    # Every cluster has identical mean error, despite unequal cluster sizes;
    # video-cluster bootstrap must therefore have a degenerate interval.
    result=clustered_mae(['a','a','b'], np.array([-2.,0.,-1.]), 10000)
    assert result['ci95']==[-1.,-1.]
    assert result['candidate_minus_baseline_mae']==-1.
    assert result['video_clusters']==2


@pytest.mark.parametrize('audit_passes', [True, False])
def test_rolling_replication_audits_each_method_and_preserves_gpu_limits(tmp_path, monkeypatch, audit_passes):
    import json
    import run_tav_distillation as queue
    base=tmp_path/'experiment'
    (base/'assets').mkdir(parents=True)
    (base/'assets/protocol.json').write_text('{}')
    launched, finished, early_replications = [], set(), []
    active, audited = {}, set()
    maximum = {0:0, 1:0}
    monkeypatch.setattr(queue, 'ROOT', tmp_path)
    monkeypatch.setattr(queue, 'verify', lambda *a, **k: None)
    monkeypatch.setattr(queue, 'gpu_state', lambda gpu: {'used_mib':14,'total_mib':85651,'utilization':0})
    monkeypatch.setattr(queue, 'complete', lambda directory: directory.name in finished)
    monkeypatch.setattr(queue.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(queue.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(queue.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0))

    def technical_audit(*a, methods=queue.CORE_METHODS):
        assert all(f'{m}_seed13' in finished for m in methods)
        if not audit_passes and 'M1' in methods:
            raise ValueError('temperature mismatch')
        audited.update(methods)
        return {'status':'pass','observed_mae':{'M3':.4,'M4':.6,'M6':.5}}

    class Child:
        def __init__(self, cmd, **kwargs):
            method=cmd[cmd.index('--method')+1]
            seed=int(cmd[cmd.index('--seed')+1])
            self.name=f'{method}_seed{seed}'
            assert '--no-diagnostics' in cmd
            if seed!=13:
                assert method in audited and f'{method}_seed13' in finished
                if not all(f'{m}_seed13' in finished for m in queue.CORE_METHODS):
                    early_replications.append(self.name)
            self.gpu=int(kwargs['env']['CUDA_VISIBLE_DEVICES'])
            active[self.name]=self.gpu
            count=sum(gpu==self.gpu for gpu in active.values())
            assert count<=2
            maximum[self.gpu]=max(maximum[self.gpu],count)
            self.remaining=12 if method in ('M0','M6') and seed==13 else 3
            self.stopped=False
            launched.append(self.name)
            self.pid=len(launched)
        def poll(self):
            if self.stopped:
                return -15
            self.remaining-=1
            if self.remaining>0:
                return None
            finished.add(self.name)
            active.pop(self.name, None)
            return 0
        def terminate(self):
            self.stopped=True
            active.pop(self.name, None)
        def wait(self, **kwargs):
            return -15 if self.stopped else 0

    monkeypatch.setattr(queue, 'audit', technical_audit)
    monkeypatch.setattr(queue.subprocess, 'Popen', Child)
    if audit_passes:
        queue.run(base, 2)
        assert len(launched)==15
        assert set(launched)=={f'{m}_seed{s}' for m in queue.CORE_METHODS for s in (13,42,2026)}
        assert early_replications
        assert json.loads((base/'status.json').read_text())['stage']=='P0_complete'
    else:
        with pytest.raises(ValueError, match='temperature mismatch'):
            queue.run(base, 2)
        assert 'M1_seed42' not in launched and 'M1_seed2026' not in launched
        assert json.loads((base/'status.json').read_text())['status']=='failed'
    assert maximum=={0:2, 1:2}
    assert not active
