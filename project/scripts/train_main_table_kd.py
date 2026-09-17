#!/usr/bin/env python3
"""Train one fixed-student main-table or mechanism-ablation experiment."""
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
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "project/src"), str(ROOT / "project/scripts")]

from rdid_mosei.main_table_kd import (
    MAIN_TABLE_METHODS,
    SUBSET_METHODS,
    MainTableKDStudent,
    adapted_interaction_loss,
    entropy_adaptive_weights,
    kd_per_sample,
    probability_kd_per_sample,
    rld_per_sample,
    skd_losses,
    subset_regression_loss,
    teacher_supervision_metadata,
    weighted_mean,
)
from train_student_baseline import (
    aggregate_windows,
    seed_everything,
    weighted_full_kd_loss,
    weighted_task_loss,
)
from train_tav_distillation import (
    Collator,
    VideoDataset,
    atomic_json,
    atomic_save,
    batch_inputs,
    load_rows as load_base_rows,
    restore_rng,
    rng_state,
    sha256,
)
from rdid_mosei.metrics import sentiment_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=MAIN_TABLE_METHODS, required=True)
    parser.add_argument("--assets", type=Path, default=ROOT / "outputs/experiments/main_table_v1/mosei/assets/protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, default=ROOT / "model/Qwen3-0.6B-Base")
    parser.add_argument("--audio-model", type=Path, default=ROOT / "model/WavLM-Base-Plus")
    parser.add_argument("--video-model", type=Path, default=ROOT / "model/VideoMAE-Base")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--video-layers", type=int, default=12)
    parser.add_argument("--video-rank", type=int, default=8)
    parser.add_argument("--video-alpha", type=float, default=16)
    parser.add_argument("--video-dropout", type=float, default=0.05)
    parser.add_argument("--video-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--lambda-feature", type=float)
    parser.add_argument("--lambda-cafd", type=float)
    parser.add_argument("--cmad-tau", type=float)
    parser.add_argument("--lambda-subset", type=float)
    parser.add_argument("--lambda-interaction", type=float)
    parser.add_argument("--ea-entropy-temperature", type=float)
    parser.add_argument("--rld-temperature", type=float)
    parser.add_argument("--rld-alpha", type=float)
    parser.add_argument("--rld-beta", type=float)
    parser.add_argument("--rld-warmup-epochs", type=int)
    parser.add_argument("--skd-temperature", type=float)
    parser.add_argument("--skd-tik-factor", type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.assets = args.assets.resolve()
    assets = json.loads(args.assets.read_text())
    args.manifest = Path(assets["manifest"])
    args.teacher_targets = Path(assets["teacher_targets"])
    args.teacher_probe_report = Path(assets["teacher_probe_report"])
    args.teacher_subset = "tav"
    args.teacher_calibration_temperature = float(assets["teacher_temperature"])
    args.mode = "video_lora"
    defaults = {
        "lambda_feature": assets["lambda_feature"],
        "lambda_cafd": assets["cmad_cafd"]["lambda"],
        "cmad_tau": assets["cmad_cafd"]["tau"],
        "lambda_subset": assets["lambda_subset"],
        "lambda_interaction": assets["lambda_interaction"],
        "ea_entropy_temperature": assets["ea_kd"]["entropy_temperature"],
        "rld_temperature": assets["rld"]["temperature"],
        "rld_alpha": assets["rld"]["alpha"],
        "rld_beta": assets["rld"]["beta"],
        "rld_warmup_epochs": assets["rld"]["warmup_epochs"],
        "skd_temperature": assets["skd"]["temperature"],
        "skd_tik_factor": assets["skd"]["tik_factor"],
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.seed not in assets["seeds"]:
        parser.error(f"seed must be one of frozen seeds {assets['seeds']}")
    if min(args.batch_size, args.epochs, args.patience, args.video_layers, args.video_rank, args.progress_every) <= 0:
        parser.error("counts must be positive")
    if args.num_workers < 0 or args.learning_rate <= 0:
        parser.error("invalid worker count or learning rate")
    positive = (args.lambda_feature, args.lambda_cafd, args.cmad_tau, args.lambda_subset, args.lambda_interaction,
                args.ea_entropy_temperature, args.rld_temperature, args.rld_alpha,
                args.rld_beta, args.rld_warmup_epochs, args.skd_temperature, args.skd_tik_factor)
    if any(value <= 0 for value in positive):
        parser.error("method hyperparameters must be positive")
    if args.limit_per_split is not None and args.limit_per_split <= 0:
        parser.error("limit-per-split must be positive")
    return args


def load_rows(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    train, valid = load_base_rows(args)
    assets = json.loads(args.assets.read_text())
    ensemble = {}
    for line in Path(assets["ensemble_targets"]).open():
        item = json.loads(line)
        ensemble[item["parent_sample_id"]] = item
    feature_index = {
        item["sample_id"]: int(item["job_index"])
        for item in (json.loads(line) for line in Path(assets["teacher_feature_index"]).open())
    }
    for row in train + valid:
        target = ensemble[row["parent_sample_id"]]
        if target["split"] != row["split"] or target["video_id"] != row["video_id"]:
            raise ValueError("ensemble/manifest identity mismatch")
        row["teacher_subset_scores"] = target["subset_score_mean"]
        row["teacher_ensemble_probabilities"] = target["tav_calibrated_probability_mean"]
        row["teacher_feature_index"] = feature_index[row["sample_id"]]
        row["weight_interaction_mean"] = row["interaction_mean"]
        row["weight_interaction_variance"] = row["interaction_variance"]

    if args.method == "shuffled_r_u_interaction":
        # Shuffle whole seven-dimensional reliability vectors between train
        # utterances while forbidding same-video donors. All windows of one
        # utterance receive the same donor vector.
        parents = {}
        for row in train:
            parents.setdefault(row["parent_sample_id"], row)
        rng = np.random.default_rng(args.seed + 31000)
        groups = {}
        for key, row in parents.items():
            groups.setdefault(row["video_id"], []).append(key)
        video_order = list(groups)
        rng.shuffle(video_order)
        keys = []
        for video in video_order:
            group = groups[video]
            rng.shuffle(group)
            keys.extend(group)
        maximum_group = max(map(len, groups.values()))
        if maximum_group * 2 > len(keys):
            raise ValueError("one source video owns too many utterances for cross-video shuffling")
        donor_keys = keys[maximum_group:] + keys[:maximum_group]
        donors = {key: parents[donor] for key, donor in zip(keys, donor_keys)}
        for row in train:
            donor = donors[row["parent_sample_id"]]
            row["weight_interaction_mean"] = donor["interaction_mean"]
            row["weight_interaction_variance"] = donor["interaction_variance"]
            row["reliability_donor_parent"] = donor["parent_sample_id"]
            if donor["video_id"] == row["video_id"]:
                raise AssertionError("reliability donor must come from another source video")
    return train, valid


class MainTableCollator(Collator):
    def __init__(self, tokenizer, audio_processor, teacher_features: Path | None):
        super().__init__(tokenizer, audio_processor)
        self.teacher_features_path = teacher_features
        self._teacher_features = None

    def __call__(self, items):
        batch = super().__call__(items)
        rows = batch["rows"]
        batch["teacher_subset_scores"] = torch.tensor([row["teacher_subset_scores"] for row in rows], dtype=torch.float32)
        batch["teacher_ensemble_probabilities"] = torch.tensor([row["teacher_ensemble_probabilities"] for row in rows], dtype=torch.float32)
        batch["interaction_mean"] = torch.tensor([row["interaction_mean"] for row in rows], dtype=torch.float32)
        batch["interaction_variance"] = torch.tensor([row["interaction_variance"] for row in rows], dtype=torch.float32)
        batch["weight_interaction_mean"] = torch.tensor([row["weight_interaction_mean"] for row in rows], dtype=torch.float32)
        batch["weight_interaction_variance"] = torch.tensor([row["weight_interaction_variance"] for row in rows], dtype=torch.float32)
        if self.teacher_features_path is not None:
            if self._teacher_features is None:
                self._teacher_features = np.load(self.teacher_features_path, mmap_mode="r")
            indices = [row["teacher_feature_index"] for row in rows]
            batch["teacher_features"] = torch.from_numpy(np.asarray(self._teacher_features[indices], dtype=np.float32))
        return batch


def regression_kd(output, scores, weights):
    return weighted_mean(F.smooth_l1_loss(output["regression"].float(), scores.float(), reduction="none"), weights)


def checkpoint_state(model):
    state = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    if model.feature_loss is not None:
        state.update({f"__buffer__.{name}": value.detach().cpu().clone() for name, value in model.feature_loss.named_buffers()})
    return state


def load_checkpoint_state(model, state):
    parameters = {name for name, value in model.named_parameters() if value.requires_grad}
    buffer_names = {f"__buffer__.{name}" for name, _ in model.feature_loss.named_buffers()} if model.feature_loss is not None else set()
    if set(state) != parameters | buffer_names:
        raise ValueError("main-table checkpoint keys differ from active trainable state")
    model.load_state_dict({name: value for name, value in state.items() if not name.startswith("__buffer__.")}, strict=False)
    if model.feature_loss is not None:
        buffers = dict(model.feature_loss.named_buffers())
        for name in buffer_names:
            buffers[name.removeprefix("__buffer__.")].copy_(state[name])


def compute_loss(model, outputs, batch, device, args, assets, epoch):
    method = args.method
    output = outputs["tav"] if method in SUBSET_METHODS else outputs
    weights = batch["weights"].to(device)
    sentiment = batch["sentiment"].to(device)
    classes = batch["classes"].to(device)
    teacher_scores = batch["teacher_scores"].to(device)
    teacher_logits = batch["teacher_logits"].to(device)
    task, _, _ = weighted_task_loss(output, sentiment, classes, weights, assets["alpha_ce"])
    if method == "adapted_student":
        return task

    if method == "ensemble_full":
        ensemble_scores = batch["teacher_subset_scores"].to(device)[:, -1]
        regression = regression_kd(output, ensemble_scores, weights)
        classification = weighted_mean(
            probability_kd_per_sample(output["classification_logits"], batch["teacher_ensemble_probabilities"].to(device), assets["kd_temperature"]),
            weights,
        )
        return task + regression + classification

    if method in {"ea_kd", "ea_kd_full"}:
        per_regression = F.smooth_l1_loss(output["regression"].float(), teacher_scores, reduction="none")
        per_classification = kd_per_sample(output["classification_logits"], teacher_logits, assets["kd_temperature"], args.teacher_calibration_temperature)
        adaptive = entropy_adaptive_weights(output["classification_logits"], teacher_logits / args.teacher_calibration_temperature, args.ea_entropy_temperature)
        regression = weighted_mean(per_regression * (adaptive if method == "ea_kd_full" else 1.0), weights)
        classification = weighted_mean(per_classification * adaptive, weights)
        return task + regression + classification

    if method == "rld":
        settings = assets["rld"]
        regression = regression_kd(output, teacher_scores, weights)
        refined = rld_per_sample(
            output["classification_logits"], teacher_logits / args.teacher_calibration_temperature, classes,
            alpha=args.rld_alpha, beta=args.rld_beta, temperature=args.rld_temperature,
            confidence_temperature=settings["confidence_temperature"], standardize_logits=settings["standardize_logits"],
        )
        warmup = min(epoch / args.rld_warmup_epochs, 1.0)
        return task + regression + warmup * weighted_mean(refined, weights)

    if method == "skd":
        settings = assets["skd"]
        regression = regression_kd(output, teacher_scores, weights)
        instance, direction, mask = skd_losses(
            output["classification_logits"], teacher_logits / args.teacher_calibration_temperature,
            temperature=args.skd_temperature, tik_factor=args.skd_tik_factor,
        )
        selected = weights * mask.to(weights)
        instance_loss = (instance * selected).sum() / selected.sum().clamp_min(1e-8)
        direction_loss = weighted_mean(direction, weights)
        return task + regression + instance_loss + direction_loss

    full, _, _ = weighted_full_kd_loss(
        output, teacher_scores, teacher_logits, weights, assets["kd_temperature"],
        args.teacher_calibration_temperature, 1.0, 1.0,
    )
    loss = task + full
    if method == "projector":
        loss = loss + args.lambda_feature * model.feature_loss(
            output["fused"], batch["teacher_features"].to(device)
        )
    elif method == "cmad_cafd":
        loss = loss + args.lambda_cafd * model.feature_loss(
            output["fused"], batch["teacher_features"].to(device)
        )
    elif method == "subset7":
        loss = loss + args.lambda_subset * subset_regression_loss(
            outputs, batch["teacher_subset_scores"].to(device), weights
        )
    elif method in SUBSET_METHODS:
        utility_key = "coarse_utility_normalized" if method == "coarse_u_interaction" else "utility_normalized"
        interaction, _ = adapted_interaction_loss(
            outputs,
            batch["interaction_mean"].to(device),
            batch["interaction_variance"].to(device),
            torch.tensor(assets[utility_key], device=device),
            weights,
            assets["student_empty_baseline"],
            method,
            batch["weight_interaction_mean"].to(device),
            batch["weight_interaction_variance"].to(device),
        )
        loss = loss + args.lambda_interaction * interaction
    return loss


@torch.inference_mode()
def evaluate(model, loader, device, timing=None):
    model.eval()
    records = []
    for batch in loader:
        if timing is not None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_started = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            raw = model(**batch_inputs(batch, device))
            output = raw["tav"] if model.training and model.comparison_method in SUBSET_METHODS else raw
        if timing is not None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timing["model_only_seconds"] = timing.get("model_only_seconds", 0.0) + time.perf_counter() - model_started
            timing["model_batches"] = timing.get("model_batches", 0) + 1
            timing["model_windows"] = timing.get("model_windows", 0) + len(batch["rows"])
        for row, prediction, logits in zip(batch["rows"], output["regression"].float().cpu().tolist(), output["classification_logits"].float().cpu().tolist()):
            records.append({"parent_sample_id": row["parent_sample_id"], "split": row["split"], "target_sentiment": row["sentiment"], "class_7_index": row["class_7_index"], "prediction": prediction, "classification_logits": logits, "aggregation_weight": row["aggregation_weight"]})
    utterances = aggregate_windows(records)
    videos = {row["parent_sample_id"]: row["video_id"] for row in loader.dataset.rows}
    for row in utterances:
        row["video_id"] = videos[row["parent_sample_id"]]
    metrics = sentiment_metrics([row["target_sentiment"] for row in utterances], [row["prediction"] for row in utterances])
    return metrics, utterances


def build_model(args, assets, device):
    from transformers import AutoModel
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    text = AutoModel.from_pretrained(args.text_model, local_files_only=True, dtype=dtype)
    audio = AutoModel.from_pretrained(args.audio_model, local_files_only=True, dtype=dtype)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(args.seed + 100000)
        video = AutoModel.from_pretrained(args.video_model, local_files_only=True, dtype=dtype, attn_implementation="sdpa")
    text.config.use_cache = False
    return MainTableKDStudent(
        text, audio, video, comparison_method=args.method,
        teacher_feature_dim=assets["teacher_feature_dimension"],
        cmad_tau=args.cmad_tau,
        video_layers=args.video_layers, video_rank=args.video_rank, video_alpha=args.video_alpha,
        video_dropout=args.video_dropout, video_seed=args.seed + 100000,
        video_checkpointing=args.video_checkpointing,
    ).to(device)


def run_protocol(args, assets):
    excluded = {"resume", "stop_after_epoch", "dry_run", "progress_every"}
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items() if key not in excluded}
    config.update(
        schema="rdid-msa-fixed-student-main-table-v1",
        checkpoint_selection="valid_mae",
        official_test_evaluated=False,
        common_student="Qwen3-0.6B + WavLM-Base-Plus + VideoMAE-Base; T/A/V LoRA adaptation",
        frozen_assets=assets,
        upstream_sources=json.loads(Path(assets["upstream_sources"]).read_text()),
        **teacher_supervision_metadata(args.method),
    )
    paths = [args.assets, Path(__file__), ROOT / "project/src/rdid_mosei/main_table_kd.py"]
    config["input_sha256"] = {str(path.resolve()): sha256(path) for path in paths}
    return config


@contextmanager
def run_lock(output: Path, resume: bool):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "run_config.json").exists() and not resume:
            raise FileExistsError(f"existing run: use --resume or a new output: {output}")
        yield


def train(args, output):
    seed_everything(args.seed, True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    assets = json.loads(args.assets.read_text())
    train_rows, valid_rows = load_rows(args)
    config = run_protocol(args, assets)
    config_path = output / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("resume protocol/input fingerprint differs")
    if args.resume and not config_path.exists():
        raise FileNotFoundError("--resume requires an existing configured run")
    atomic_json(config, config_path)

    model = build_model(args, assets, device)
    from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    audio_processor = AutoFeatureExtractor.from_pretrained(args.audio_model, local_files_only=True)
    video_processor = AutoImageProcessor.from_pretrained(args.video_model, local_files_only=True, use_fast=False)
    feature_path = Path(assets["teacher_features"]) if args.method in {"projector", "cmad_cafd"} else None
    collator = MainTableCollator(tokenizer, audio_processor, feature_path)
    generator = torch.Generator().manual_seed(args.seed)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collator, pin_memory=device.type == "cuda")
    train_loader = DataLoader(VideoDataset(train_rows, video_processor, mode="video_lora"), shuffle=True, generator=generator, **loader_args)
    valid_loader = DataLoader(VideoDataset(valid_rows, video_processor, mode="video_lora"), shuffle=False, **loader_args)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    best_mae, best_epoch, stale, history, start_epoch, elapsed_before = math.inf, 0, 0, [], 1, 0.0
    last = output / "last.pt"
    if args.resume:
        saved = torch.load(last, map_location="cpu", weights_only=False)
        load_checkpoint_state(model, saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        best_mae, best_epoch, stale = saved["best_mae"], saved["best_epoch"], saved["stale"]
        history, start_epoch, elapsed_before = saved["history"], saved["epoch"] + 1, saved["elapsed_seconds"]
        restore_rng(saved["rng"], generator)
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    atomic_json({"status": "training", "start_epoch": start_epoch}, output / "status.json")
    for epoch in range(start_epoch, args.epochs + 1):
        if stale >= args.patience:
            break
        model.train()
        total_loss = total_weight = 0.0
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(**batch_inputs(batch, device))
                loss = compute_loss(model, outputs, batch, device, args, assets, epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            weight = float(batch["weights"].sum())
            total_loss += float(loss.detach()) * weight
            total_weight += weight
            if step == 1 or step % args.progress_every == 0:
                progress = {"status": "training", "method": args.method, "epoch": epoch, "step": step, "steps": len(train_loader), "loss": float(loss.detach()), "elapsed_seconds": elapsed_before + time.time() - started}
                atomic_json(progress, output / "status.json")
                print(json.dumps(progress), flush=True)
        metrics, _ = evaluate(model, valid_loader, device)
        row = {"epoch": epoch, "train_loss": total_loss / total_weight, "valid_metrics": metrics, "gradient_norm_last": float(norm), "elapsed_seconds": elapsed_before + time.time() - started}
        history.append(row)
        if metrics["mae"] < best_mae - 1e-5:
            best_mae, best_epoch, stale = metrics["mae"], epoch, 0
            atomic_save({"model": checkpoint_state(model), "epoch": epoch, "protocol": config}, output / "best.pt")
        else:
            stale += 1
        atomic_save({"model": checkpoint_state(model), "protocol": config, "optimizer": optimizer.state_dict(), "epoch": epoch, "best_mae": best_mae, "best_epoch": best_epoch, "stale": stale, "history": history, "rng": rng_state(generator), "elapsed_seconds": row["elapsed_seconds"]}, last)
        atomic_json(history, output / "history.json")
        print(json.dumps(row), flush=True)
        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
            atomic_json({"status": "paused", "epoch": epoch}, output / "status.json")
            return

    load_checkpoint_state(model, torch.load(output / "best.pt", map_location="cpu", weights_only=True)["model"])
    metrics, predictions = evaluate(model, valid_loader, device)
    temporary = output / "predictions.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
    os.replace(temporary, output / "predictions.jsonl")
    report = {
        "method": args.method, "seed": args.seed, "best_epoch": best_epoch, "epochs_run": len(history),
        "valid_metrics": metrics, "train_windows": len(train_rows), "valid_windows": len(valid_rows),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "projector_parameters": 0 if args.method != "projector" else sum(parameter.numel() for parameter in model.feature_loss.parameters()),
        "training_only_feature_parameters": 0 if model.feature_loss is None else sum(parameter.numel() for parameter in model.feature_loss.parameters()),
        "deployed_parameters": sum(parameter.numel() for parameter in model.parameters()) - (0 if model.feature_loss is None else sum(parameter.numel() for parameter in model.feature_loss.parameters())),
        "elapsed_seconds": elapsed_before + time.time() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
        "official_test_evaluated": False,
        **teacher_supervision_metadata(args.method),
    }
    atomic_json(report, output / "report.json")
    atomic_json({"status": "complete", "best_epoch": best_epoch}, output / "status.json")
    print(json.dumps(report), flush=True)


def main():
    args = parse_args()
    if args.dry_run:
        train_rows, valid_rows = load_rows(args)
        assets = json.loads(args.assets.read_text())
        print(json.dumps({"method": args.method, "seed": args.seed, "train_windows": len(train_rows), "valid_windows": len(valid_rows), "protocol": run_protocol(args, assets)}, ensure_ascii=False, indent=2))
        return
    output = args.output.resolve()
    with run_lock(output, args.resume):
        try:
            train(args, output)
        except Exception as error:
            atomic_json({"status": "failed", "error": repr(error)}, output / "status.json")
            raise


if __name__ == "__main__":
    main()
