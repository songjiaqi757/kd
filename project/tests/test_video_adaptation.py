import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3Model, VideoMAEConfig, VideoMAEModel, WavLMConfig, WavLMModel

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from rdid_mosei.video_adaptation import VideoAdaptationStudent, sample_video_frames
from train_video_adaptation import load_rows, load_trainable, rng_state, restore_rng, shuffled_video_donors, trainable_state


def model(mode="video_lora", checkpointing=True):
    torch.manual_seed(13)
    t = Qwen3Model(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                              num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8))
    a = WavLMModel(WavLMConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                               intermediate_size=32, conv_dim=(8,8,8), conv_stride=(4,4,4),
                               conv_kernel=(8,4,4), num_conv_pos_embeddings=8,
                               num_conv_pos_embedding_groups=2, num_buckets=8, max_bucket_distance=16))
    v = None
    if mode != "ta_only":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(113)
            c = VideoMAEConfig(image_size=8, patch_size=4, num_frames=4, tubelet_size=2,
                               hidden_size=16, intermediate_size=32, num_attention_heads=2,
                               num_hidden_layers=3, use_mean_pooling=False)
            c._attn_implementation = "sdpa"
            v = VideoMAEModel(c)
    return VideoAdaptationStudent(t, a, v, mode=mode, hidden_size=16, video_hidden_size=16,
                                   video_layers=2, video_dropout=0,
                                   video_checkpointing=checkpointing)


def inputs():
    g = torch.Generator().manual_seed(2026)
    return dict(input_ids=torch.randint(0,32,(2,5),generator=g), text_attention_mask=torch.ones(2,5,dtype=torch.long),
                input_values=torch.randn(2,512,generator=g), audio_attention_mask=torch.ones(2,512,dtype=torch.long),
                pixel_values=torch.randn(2,4,3,8,8,generator=g))


def test_zero_visual_adapter_and_common_initialization_match_frozen_control():
    left, right = model("frozen_video").eval(), model().eval()
    for n,p in left.named_parameters():
        if not n.startswith("video_encoder."):
            assert torch.equal(p, dict(right.named_parameters())[n]), n
    with torch.no_grad():
        x = inputs()
        expected = left.video_encoder(x["pixel_values"]).last_hidden_state
        torch.testing.assert_close(left.encode_video(x["pixel_values"]), expected)
        torch.testing.assert_close(right.encode_video(x["pixel_values"]), expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(left(**x)["regression"], right(**x)["regression"], atol=1e-6, rtol=1e-5)


def test_actual_qkv_adapters_update_and_frozen_backbone_does_not():
    m = model().train()
    before = {n:p.detach().clone() for n,p in m.video_encoder.named_parameters()}
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    out = m(**inputs())
    loss = torch.nn.functional.cross_entropy(out["classification_logits"], torch.tensor([1,5]))
    loss.backward()
    for n,p in m.video_encoder.named_parameters():
        if "lora_B" in n:
            assert p.grad is not None and float(p.grad.norm()) > 0, n
        if not p.requires_grad:
            assert p.grad is None, n
    opt.step()
    for n,p in m.video_encoder.named_parameters():
        if "lora_B" in n:
            assert not torch.equal(p, before[n]), n
        if not p.requires_grad:
            assert torch.equal(p, before[n]), n
    assert len(m.video_lora_modules) == 8
    assert all(not p.requires_grad for p in m.video_encoder.encoder.layer[0].parameters())


def test_bfloat16_external_bias_preserves_zero_adapter_output():
    left,right=model("frozen_video").eval(),model().eval()
    with torch.no_grad():
        for index in range(3):
            for name in ("q_bias","v_bias"):
                bias=torch.linspace(-.123,.234,16)
                getattr(left.video_encoder.encoder.layer[index].attention.attention,name).copy_(bias)
                getattr(right.video_encoder.encoder.layer[index].attention.attention,name).copy_(bias)
        left.video_encoder.to(torch.bfloat16)
        right.video_encoder.to(torch.bfloat16)
        pixels=inputs()["pixel_values"].bfloat16()
        torch.testing.assert_close(left.encode_video(pixels),right.encode_video(pixels),atol=0,rtol=0)


def test_visual_checkpoint_recomputes_and_preserves_gradients():
    left, right = model(checkpointing=False).train(), model(checkpointing=True).train()
    x = inputs()["pixel_values"]
    calls = []
    handle = right.video_encoder.encoder.layer[-1].register_forward_pre_hook(lambda *a: calls.append(1))
    left.encode_video(x).square().mean().backward()
    right.encode_video(x).square().mean().backward()
    handle.remove()
    assert len(calls) >= 2
    for (n,p), (nn,q) in zip(left.video_encoder.named_parameters(), right.video_encoder.named_parameters()):
        assert n == nn
        if p.requires_grad:
            torch.testing.assert_close(p.grad,q.grad)


def test_ta_only_has_no_visual_encoder_or_visual_input():
    m = model("ta_only").train()
    assert m.video_encoder is None
    x = inputs()
    with pytest.raises(ValueError, match="must not receive"):
        m(**x)
    x["pixel_values"] = None
    out = m(**x)
    out["classification_logits"].sum().backward()
    assert all(p.grad is None for p in m.pools["v"].parameters())
    assert m.fusion.missing_tokens["v"].grad is not None


def test_checkpoint_roundtrip_and_missing_adapter_rejected():
    m = model().eval()
    with torch.no_grad():
        for n,p in m.named_parameters():
            if "video_encoder" in n and "lora_B" in n:
                p.add_(0.02)
        expected = m(**inputs())["regression"]
    state = trainable_state(m)
    restored = model().eval()
    load_trainable(restored,state)
    with torch.no_grad():
        torch.testing.assert_close(restored(**inputs())["regression"],expected)
    state.pop(next(n for n in state if "video_encoder" in n))
    with pytest.raises(ValueError,match="checkpoint keys"):
        load_trainable(restored,state)


def test_optimizer_and_rng_resume_matches_continuous_next_step():
    m = model().train()
    optimizer = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4)
    generator = torch.Generator().manual_seed(42)
    def step(mm, oo):
        oo.zero_grad()
        out = mm(**inputs())
        loss = torch.nn.functional.cross_entropy(out["classification_logits"],torch.tensor([2,4]))
        loss.backward(); oo.step()
        return loss.detach().clone()
    step(m,optimizer)
    saved = dict(model=trainable_state(m), optimizer=copy.deepcopy(optimizer.state_dict()), rng=rng_state(generator))
    expected_loss = step(m,optimizer)
    expected = trainable_state(m)
    restored = model().train()
    opt = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],lr=1e-4)
    load_trainable(restored,saved["model"]); opt.load_state_dict(saved["optimizer"])
    restore_rng(saved["rng"],generator)
    torch.testing.assert_close(step(restored,opt),expected_loss,atol=0,rtol=0)
    for n,p in trainable_state(restored).items():
        torch.testing.assert_close(p,expected[n],atol=0,rtol=0)


def test_cross_video_shuffle_is_bijective_and_avoids_same_source():
    rows = [{"sample_id":str(i),"video_id":str(i//3)} for i in range(18)]
    donors = shuffled_video_donors(rows,13)
    assert {r["sample_id"] for r in donors} == {r["sample_id"] for r in rows}
    assert all(a["video_id"] != b["video_id"] for a,b in zip(rows,donors))
    assert donors == shuffled_video_donors(rows,13)


def test_ta_teacher_selection_and_duplicate_targets(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    targets = tmp_path / "targets.jsonl"
    rows = [dict(sample_id=s,parent_sample_id=s,video_id=s,split=split,sentiment=1.,class_7_index=4,
                 aggregation_weight=2.) for s,split in [("a","train"),("b","valid")]]
    teachers = [dict(parent_sample_id=r["sample_id"],split=r["split"],target_sentiment=1.,class_7_index=4,
                      subset=subset,probe_score=score,classification_logits=[0.]*7)
                for r in rows for subset,score in [("ta",0.2),("tav",0.8)]]
    manifest.write_text("".join(json.dumps(r)+"\n" for r in rows))
    targets.write_text("".join(json.dumps(r)+"\n" for r in teachers))
    args=SimpleNamespace(manifest=manifest,teacher_targets=targets,mode="ta_only",limit_per_split=None)
    tr,va = load_rows(args)
    assert tr[0]["teacher_score"] == va[0]["teacher_score"] == 0.2
    with targets.open("a") as f:f.write(json.dumps(teachers[0])+"\n")
    with pytest.raises(ValueError,match="duplicate teacher"):
        load_rows(args)


def test_streaming_sampler_retains_legacy_indices(monkeypatch):
    import rdid_mosei.video_adaptation as module
    class Frame:
        def __init__(self,index): self.index=index
        def to_ndarray(self,format):
            assert format=="rgb24"
            return np.full((2,2,3),self.index,dtype=np.uint8)
    class Container:
        streams=SimpleNamespace(video=[SimpleNamespace(frames=7)])
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def decode(self,video):return iter(Frame(i) for i in range(7))
    monkeypatch.setattr(module.av,"open",lambda _:Container())
    sampled=sample_video_frames("unused",16)
    assert [int(x[0,0,0]) for x in sampled] == np.linspace(0,6,16).round().astype(int).tolist()


def test_promotion_requires_effect_size_and_ta_control_win():
    from run_video_adaptation_queue import pilot_gate
    def gate(a,b,c):return pilot_gate({"mean_valid_mae":{"frozen_video":a,"video_lora":b,"ta_only":c}},.005)
    assert gate(.51,.50,.505)
    assert not gate(.51,.508,.52)
    assert not gate(.51,.50,.49)


def test_resource_guard_fails_closed_on_systemd_error(monkeypatch):
    import run_video_adaptation_queue as queue
    monkeypatch.setattr(queue.subprocess,"run",lambda *a,**kw:SimpleNamespace(stdout="",stderr="bus unavailable",returncode=1))
    ready,info=queue.gpu_ready(1)
    assert not ready and "systemd_error" in info


def test_cluster_comparison_and_label_mismatch(tmp_path):
    from run_video_adaptation_queue import compare
    left=tmp_path/"left.jsonl";right=tmp_path/"right.jsonl"
    rows=[dict(parent_sample_id=str(i),video_id=str(i//2),split="valid",target_sentiment=1.,prediction=.5) for i in range(6)]
    left.write_text("".join(json.dumps(r)+"\n" for r in rows))
    for r in rows:r["prediction"]+=.2
    right.write_text("".join(json.dumps(r)+"\n" for r in rows))
    result=compare(left,right,repetitions=100)
    assert result["video_clusters"]==3
    assert result["delta_mae"]==pytest.approx(-.2)
    assert result["cluster_ci95"]==pytest.approx([-.2,-.2])
    rows[0]["target_sentiment"]=2.
    right.write_text("".join(json.dumps(r)+"\n" for r in rows))
    with pytest.raises(ValueError,match="labels/video"):
        compare(left,right,repetitions=100)
