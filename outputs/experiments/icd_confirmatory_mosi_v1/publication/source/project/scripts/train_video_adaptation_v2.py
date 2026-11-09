#!/usr/bin/env python3
"""Timing-v2 teacher targets and equal attention-layer LoRA coverage."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project/src"))
from rdid_mosei.metrics import sentiment_metrics
from rdid_mosei.video_adaptation import MODES, VideoAdaptationStudent, sample_video_frames
from train_student_baseline import (
    add_window_weights, aggregate_windows, seed_everything,
    weighted_full_kd_loss, weighted_task_loss,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=ROOT / "dataset/cmu_mosei/manifests/official_train_valid_windowed.jsonl")
    p.add_argument("--teacher-targets", type=Path, default=ROOT / "outputs/probe/official_train_valid_seed2026/predictions.jsonl")
    p.add_argument("--text-model", type=Path, default=ROOT / "model/Qwen3-0.6B-Base")
    p.add_argument("--audio-model", type=Path, default=ROOT / "model/WavLM-Base-Plus")
    p.add_argument("--video-model", type=Path, default=ROOT / "model/VideoMAE-Base")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--video-layers", type=int, default=12)
    p.add_argument("--video-rank", type=int, default=8)
    p.add_argument("--video-alpha", type=float, default=16)
    p.add_argument("--video-dropout", type=float, default=0.05)
    p.add_argument("--video-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--limit-per-split", type=int, help="Smoke only: retain this many whole parent utterances per split")
    p.add_argument("--teacher-probe-report", type=Path, required=True)
    p.add_argument("--teacher-subset", choices=("ta", "tav"), default="tav")
    p.add_argument("--diagnostic-repetitions", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after-epoch", type=int, help="Save a resumable epoch boundary and exit (recovery verification)")
    p.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--dry-run", action="store_true", help="Validate inputs and print protocol without loading models")
    args = p.parse_args()
    report = json.loads(args.teacher_probe_report.read_text())
    if args.teacher_targets.resolve().parent != args.teacher_probe_report.resolve().parent:
        p.error("teacher targets and calibration report must belong to the same Probe run")
    feature_config = json.loads((Path(report["features"]) / "run_config.json").read_text())
    if feature_config.get("video_timing_policy") != "sampled_fps_v2":
        p.error("timing-v2 teacher features required")
    args.teacher_calibration_temperature = float(report["calibration"]["after"]["temperature"])
    if args.diagnostic_repetitions < 1:
        p.error("diagnostic repetitions must be positive")
    if min(args.batch_size, args.epochs, args.patience, args.video_layers, args.video_rank, args.progress_every) <= 0:
        p.error("counts must be positive")
    if args.num_workers < 0 or args.teacher_calibration_temperature <= 0:
        p.error("invalid workers or temperature")
    if args.limit_per_split is not None and args.limit_per_split <= 0:
        p.error("limit must be positive")
    return args


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(payload, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    os.replace(temp, path)


def atomic_save(payload, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


@contextmanager
def run_lock(output, resume):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "run_config.json").exists() and not resume:
            raise FileExistsError(f"existing run: use --resume or a new output: {output}")
        yield


def load_rows(args):
    rows = [json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate manifest sample IDs")
    if {r["split"] for r in rows} != {"train", "valid"}:
        raise ValueError("exactly train/valid required; test is forbidden in this experiment")
    parents = {}
    for r in rows:
        key = r["parent_sample_id"]
        signature = (r["split"], float(r["sentiment"]), int(r["class_7_index"]), r["video_id"])
        if parents.setdefault(key, signature) != signature:
            raise ValueError(f"conflicting parent metadata: {key}")
        if not math.isfinite(float(r["sentiment"])) or not float(r["aggregation_weight"]) > 0:
            raise ValueError(f"invalid target/weight: {key}")
    train_videos = {r["video_id"] for r in rows if r["split"] == "train"}
    if train_videos & {r["video_id"] for r in rows if r["split"] == "valid"}:
        raise ValueError("train/valid video overlap")
    if args.limit_per_split:
        keep = set()
        for split in ("train", "valid"):
            keep.update(list(dict.fromkeys(r["parent_sample_id"] for r in rows if r["split"] == split))[:args.limit_per_split])
        rows = [r for r in rows if r["parent_sample_id"] in keep]
    subset = args.teacher_subset
    targets = {}
    for line in args.teacher_targets.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["subset"] != subset:
            continue
        key = r["parent_sample_id"]
        if key in targets:
            raise ValueError(f"duplicate teacher target: {key}/{subset}")
        targets[key] = r
    for r in rows:
        t = targets[r["parent_sample_id"]]
        if t["split"] != r["split"] or abs(float(t["target_sentiment"]) - float(r["sentiment"])) > 1e-6 or int(t["class_7_index"]) != int(r["class_7_index"]):
            raise ValueError(f"teacher/manifest mismatch: {r['sample_id']}")
        if len(t["classification_logits"]) != 7 or not np.isfinite([t["probe_score"], *t["classification_logits"]]).all():
            raise ValueError("invalid teacher values")
        r["teacher_score"], r["teacher_logits"] = t["probe_score"], t["classification_logits"]
    add_window_weights(rows)
    return [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "valid"]


def shuffled_video_donors(rows, seed):
    """A bijection with every donor from a different source video."""
    if len({r["video_id"] for r in rows}) < 2:
        raise ValueError("video shuffle requires at least two source videos")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(rows))
    for i in range(len(rows)):
        if rows[i]["video_id"] == rows[permutation[i]]["video_id"]:
            choices = [j for j in range(len(rows)) if
                       rows[i]["video_id"] != rows[permutation[j]]["video_id"] and
                       rows[j]["video_id"] != rows[permutation[i]]["video_id"]]
            if not choices:
                raise ValueError("cannot build cross-video permutation")
            j = int(rng.choice(choices))
            permutation[i], permutation[j] = permutation[j], permutation[i]
    return [rows[int(j)] for j in permutation]


class VideoDataset(Dataset):
    def __init__(self, rows, processor, *, mode, perturbation="normal", seed=2026):
        self.rows, self.processor, self.mode = rows, processor, mode
        self.perturbation, self.seed = perturbation, seed
        self.donors = shuffled_video_donors(rows, seed) if perturbation == "shuffle_video" else rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        wave, sr = sf.read(r["audio_segment_path"], dtype="float32", always_2d=False)
        if wave.ndim != 1 or sr != 16000:
            raise ValueError(f"invalid audio: {r['sample_id']}")
        pixels = None
        if self.mode != "ta_only":
            frames = sample_video_frames(self.donors[index]["silent_video_path"], 16)
            if self.perturbation == "repeat_frame":
                frames = [frames[len(frames) // 2]] * len(frames)
            elif self.perturbation == "shuffle_frames":
                permutation = np.random.default_rng(self.seed + index).permutation(len(frames))
                frames = [frames[int(j)] for j in permutation]
            pixels = self.processor(frames, return_tensors="pt")["pixel_values"][0]
        return {"row": r, "wave": wave, "pixels": pixels}


class Collator:
    def __init__(self, tokenizer, audio_processor):
        self.tokenizer, self.audio_processor = tokenizer, audio_processor

    def __call__(self, items):
        rows = [x["row"] for x in items]
        text = self.tokenizer([r["text"] for r in rows], padding=True, truncation=True,
                              max_length=256, return_tensors="pt")
        audio = self.audio_processor([x["wave"] for x in items], sampling_rate=16000,
                                     padding=True, return_attention_mask=True, return_tensors="pt")
        return {"inputs": {"input_ids": text["input_ids"], "text_attention_mask": text["attention_mask"],
                           "input_values": audio["input_values"], "audio_attention_mask": audio.get("attention_mask"),
                           "pixel_values": None if items[0]["pixels"] is None else torch.stack([x["pixels"] for x in items])},
                "sentiment": torch.tensor([r["sentiment"] for r in rows], dtype=torch.float32),
                "classes": torch.tensor([r["class_7_index"] for r in rows], dtype=torch.long),
                "weights": torch.tensor([r["sample_weight"] for r in rows], dtype=torch.float32),
                "teacher_scores": torch.tensor([r["teacher_score"] for r in rows], dtype=torch.float32),
                "teacher_logits": torch.tensor([r["teacher_logits"] for r in rows], dtype=torch.float32),
                "rows": rows}


def batch_inputs(batch, device):
    return {k: None if v is None else v.to(device, non_blocking=True) for k, v in batch["inputs"].items()}


def compute_loss(output, batch, device, temperature):
    w = batch["weights"].to(device)
    task, _, _ = weighted_task_loss(output, batch["sentiment"].to(device), batch["classes"].to(device), w, 0.5)
    kd, _, _ = weighted_full_kd_loss(output, batch["teacher_scores"].to(device),
                                    batch["teacher_logits"].to(device), w, 2.0, temperature, 1.0, 1.0)
    return task + kd


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    records = []
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(**batch_inputs(batch, device))
        for row, pred, logits in zip(batch["rows"], out["regression"].float().cpu().tolist(), out["classification_logits"].float().cpu().tolist()):
            records.append({"parent_sample_id": row["parent_sample_id"], "split": row["split"],
                            "target_sentiment": row["sentiment"], "class_7_index": row["class_7_index"],
                            "prediction": pred, "classification_logits": logits,
                            "aggregation_weight": row["aggregation_weight"]})
    utterances = aggregate_windows(records)
    video_ids = {r["parent_sample_id"]: r["video_id"] for r in loader.dataset.rows}
    for r in utterances:
        r["video_id"] = video_ids[r["parent_sample_id"]]
    metrics = sentiment_metrics([r["target_sentiment"] for r in utterances], [r["prediction"] for r in utterances])
    return metrics, utterances


def trainable_state(model):
    return {k: p.detach().cpu().clone() for k, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, state):
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError(f"trainable checkpoint keys differ: missing={expected-set(state)}, extra={set(state)-expected}")
    model.load_state_dict(state, strict=False)  # Only known frozen backbone keys may be absent.


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def protocol(args):
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in {"resume", "stop_after_epoch", "dry_run", "progress_every"}}
    config.update(protocol="video-source-v2-student", input_subset="ta" if args.mode == "ta_only" else "tav",
                  teacher_subset=args.teacher_subset, video_frames=16,
                  alpha_ce=0.5, lambda_full=1.0, kd_temperature=2.0, gradient_accumulation=1,
                  checkpoint_selection="valid_mae", temperature_policy="matching_timing_v2_probe_report",
                  ta_backbone_dropout="eval_except_lora", ta_gradient_checkpointing=False,
                  official_test_evaluated=False, torch_version=torch.__version__)
    paths = [args.manifest, args.teacher_targets, args.teacher_probe_report, Path(__file__),
             ROOT / "project/src/rdid_mosei/video_adaptation.py", ROOT / "project/src/rdid_mosei/student.py",
             ROOT / "project/src/rdid_mosei/metrics.py", ROOT / "project/scripts/train_student_baseline.py"]
    model_paths = [args.text_model, args.audio_model] + ([] if args.mode == "ta_only" else [args.video_model])
    for directory in model_paths:
        paths.extend(sorted(directory.glob("*.json")))
        weights = sorted(directory.glob("*.safetensors")) or sorted(directory.glob("pytorch_model*.bin"))
        if not weights:
            raise FileNotFoundError(f"local model weights required: {directory}")
        paths.extend(weights)
    config["input_sha256"] = {str(p.resolve()): sha256(p) for p in paths}
    return config


def build_model(args, device):
    from transformers import AutoModel
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    text = AutoModel.from_pretrained(args.text_model, local_files_only=True, dtype=dtype)
    audio = AutoModel.from_pretrained(args.audio_model, local_files_only=True, dtype=dtype)
    # Loading V must not consume the RNG stream used by common modules.
    video = None
    if args.mode != "ta_only":
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(args.seed + 100000)
            video = AutoModel.from_pretrained(args.video_model, local_files_only=True, dtype=dtype,
                                             attn_implementation="sdpa")
    text.config.use_cache = False
    return VideoAdaptationStudent(text, audio, video, mode=args.mode, video_layers=args.video_layers,
                                  video_rank=args.video_rank, video_alpha=args.video_alpha,
                                  video_dropout=args.video_dropout, video_seed=args.seed + 100000,
                                  video_checkpointing=args.video_checkpointing).to(device)


def train(args, output):
    seed_everything(args.seed, True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    train_rows, valid_rows = load_rows(args)
    config = protocol(args)
    cp = output / "run_config.json"
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise ValueError("resume protocol/input fingerprint differs")
    if args.resume and not cp.exists():
        raise FileNotFoundError("--resume requires an existing configured run")
    if args.resume and (output / "report.json").is_file():
        if json.loads((output / "status.json").read_text())["status"] == "complete":
            print(json.dumps({"status": "already_complete", "output": str(output)}), flush=True)
            return
    atomic_json(config, cp)
    model = build_model(args, device)
    from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    audio_processor = AutoFeatureExtractor.from_pretrained(args.audio_model, local_files_only=True)
    video_processor = None if args.mode == "ta_only" else AutoImageProcessor.from_pretrained(args.video_model, local_files_only=True, use_fast=False)
    collator = Collator(tokenizer, audio_processor)
    generator = torch.Generator().manual_seed(args.seed)
    kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collator,
                  pin_memory=device.type == "cuda")
    train_loader = DataLoader(VideoDataset(train_rows, video_processor, mode=args.mode), shuffle=True,
                              generator=generator, **kwargs)
    valid_loader = DataLoader(VideoDataset(valid_rows, video_processor, mode=args.mode), shuffle=False, **kwargs)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    best_mae, best_epoch, stale, history, start_epoch = math.inf, 0, 0, [], 1
    elapsed_before = 0.0
    last = output / "last.pt"
    if args.resume:
        saved = torch.load(last, map_location="cpu", weights_only=False)
        load_trainable(model, saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        best_mae, best_epoch, stale = saved["best_mae"], saved["best_epoch"], saved["stale"]
        history, start_epoch = saved["history"], saved["epoch"] + 1
        elapsed_before = saved["elapsed_seconds"]
        restore_rng(saved["rng"], generator)
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    atomic_json({"status": "training", "mode": args.mode, "start_epoch": start_epoch}, output / "status.json")
    print(json.dumps({"status": "initialized", "mode": args.mode, "train_windows": len(train_rows),
                      "valid_windows": len(valid_rows), "trainable_parameters": sum(p.numel() for p in parameters),
                      "video_lora_modules": model.video_lora_modules}), flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        if stale >= args.patience:
            break
        model.train()
        total_loss = total_weight = 0.0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                predictions = model(**batch_inputs(batch, device))
                loss = compute_loss(predictions, batch, device, args.teacher_calibration_temperature)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            if epoch == 1 and step == 1 and args.mode == "video_lora":
                grads = {n: float(p.grad.float().norm()) if p.grad is not None else None
                         for n, p in model.named_parameters() if n.startswith("video_encoder.") and p.requires_grad}
                if not all(v is not None and math.isfinite(v) and v > 0
                           for n, v in grads.items() if "lora_B" in n):
                    raise RuntimeError("some visual LoRA B projections received no finite nonzero gradient")
                atomic_json(grads, output / "video_gradient_audit.json")
            optimizer.step()
            weight = float(batch["weights"].sum())
            total_loss += float(loss.detach()) * weight
            total_weight += weight
            if step == 1 or step % args.progress_every == 0:
                progress = {"status": "training", "epoch": epoch, "step": step,
                            "steps": len(train_loader), "loss": float(loss.detach()),
                            "elapsed_seconds": elapsed_before + time.time() - started}
                print(json.dumps(progress), flush=True)
                atomic_json(progress, output / "status.json")
        metrics, _ = evaluate(model, valid_loader, device)
        row = {"epoch": epoch, "train_loss": total_loss / total_weight, "valid_metrics": metrics,
               "gradient_norm_last": float(norm), "elapsed_seconds": elapsed_before + time.time() - started}
        history.append(row)
        if metrics["mae"] < best_mae - 1e-5:
            best_mae, best_epoch, stale = metrics["mae"], epoch, 0
            atomic_save({"model": trainable_state(model), "epoch": epoch}, output / "best.pt")
        else:
            stale += 1
        atomic_save({"model": trainable_state(model), "optimizer": optimizer.state_dict(), "epoch": epoch,
                     "best_mae": best_mae, "best_epoch": best_epoch, "stale": stale, "history": history,
                     "rng": rng_state(generator), "elapsed_seconds": row["elapsed_seconds"]}, last)
        atomic_json(history, output / "history.json")
        print(json.dumps(row), flush=True)
        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
            atomic_json({"status": "paused", "epoch": epoch}, output / "status.json")
            return
    load_trainable(model, torch.load(output / "best.pt", map_location="cpu", weights_only=True)["model"])
    metrics, predictions = evaluate(model, valid_loader, device)
    tmp = output / "predictions.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in predictions))
    os.replace(tmp, output / "predictions.jsonl")
    diagnostics = {}
    if args.diagnostics and args.mode != "ta_only":
        perturbations = [("shuffle_video", n) for n in range(args.diagnostic_repetitions)] + [("repeat_frame", 0), ("shuffle_frames", 0)]
        for perturbation, repetition in perturbations:
            diagnostic_name = f"{perturbation}_{repetition}" if perturbation == "shuffle_video" else perturbation
            loader = DataLoader(VideoDataset(valid_rows, video_processor, mode=args.mode,
                                             perturbation=perturbation, seed=args.seed + 200000 + repetition), shuffle=False, **kwargs)
            dm, dr = evaluate(model, loader, device)
            diagnostics[diagnostic_name] = {"metrics": dm, "delta_mae_vs_normal": dm["mae"] - metrics["mae"]}
            atomic_json(dr, output / f"diagnostic_{diagnostic_name}.json")
    report = {"mode": args.mode, "seed": args.seed, "best_epoch": best_epoch, "epochs_run": len(history),
              "valid_metrics": metrics, "diagnostics": diagnostics, "train_windows": len(train_rows),
              "valid_windows": len(valid_rows), "video_lora_modules": model.video_lora_modules,
              "video_trainable_parameters": sum(p.numel() for n,p in model.named_parameters() if n.startswith("video_encoder.") and p.requires_grad),
              "elapsed_seconds": elapsed_before + time.time() - started,
              "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
              "official_test_evaluated": False}
    atomic_json(report, output / "report.json")
    atomic_json({"status": "complete", "best_epoch": best_epoch}, output / "status.json")
    print(json.dumps(report), flush=True)


def main():
    args = parse_args()
    if args.dry_run:
        tr, va = load_rows(args)
        print(json.dumps({"mode": args.mode, "teacher_subset": args.teacher_subset,
                          "train_windows": len(tr), "valid_windows": len(va), "protocol": protocol(args)}, default=str, indent=2))
        return
    with run_lock(args.output.resolve(), args.resume):
        try:
            train(args, args.output.resolve())
        except Exception as exc:
            atomic_json({"status": "failed", "error": repr(exc)}, args.output / "status.json")
            raise


if __name__ == "__main__":
    main()
