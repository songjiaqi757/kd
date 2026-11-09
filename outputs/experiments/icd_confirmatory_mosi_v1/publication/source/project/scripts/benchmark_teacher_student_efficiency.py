#!/usr/bin/env python3
"""Single-GPU Teacher/Student deployment benchmark on fixed MOSEI TAV utterances."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "project/src"), str(ROOT / "project/scripts")]

from rdid_mosei.metrics import sentiment_metrics
from rdid_mosei.probe import TeacherProbe, extract_thinker_last_input_state
from rdid_mosei.subsets import build_conversation
from rdid_mosei.teacher_video import read_teacher_media
from train_tav_distillation import VideoDataset, batch_inputs, build_model, load_trainable
from evaluate_tav_test import OfficialTestCollator


DEFAULT_OUTPUT = ROOT / "outputs/experiments/efficiency_benchmark_v1/mosei"
PAPER_TABLE = ROOT / "project/reports/paper_figures/table_teacher_student_efficiency_mosei.md"
PAPER_CAPTION = ROOT / "project/reports/paper_figures/table_teacher_student_efficiency_mosei_caption.md"
MANIFEST = ROOT / "dataset/cmu_mosei/manifests/official_train_valid_windowed.jsonl"
TEACHER_MODEL = ROOT / "model/Qwen3-Omni-30B-A3B-Instruct"
TEACHER_PROBE = ROOT / "outputs/experiments/video_source_v2/probes/seed2026/teacher_probe.pt"
STUDENT_RUNS = {
    "full_kd": ROOT / "outputs/experiments/tav_main_v1/students/M3_seed13",
    "uniform": ROOT / "outputs/experiments/tav_main_v1/students/M4_seed13",
}
BUCKETS = ("lt5", "5to15", "15to30")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def timing_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_seconds": statistics.mean(values),
        "p50_seconds": percentile(values, 50),
        "p95_seconds": percentile(values, 95),
        "max_seconds": max(values),
        "throughput_utterances_per_second": len(values) / sum(values),
    }


def prepare_rows(output: Path, samples_per_bucket: int, seed: int) -> tuple[list[dict], list[dict]]:
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    eligible = [row for row in rows if row["split"] == "valid" and int(row["window_count"]) == 1]
    rng = np.random.default_rng(seed)
    measured: list[dict] = []
    warmup: list[dict] = []
    for bucket in BUCKETS:
        candidates = [row for row in eligible if row["duration_bucket"] == bucket]
        candidates.sort(key=lambda row: str(row["sample_id"]))
        order = rng.permutation(len(candidates)).tolist()
        required = samples_per_bucket + 1
        if len(order) < required:
            raise RuntimeError(f"not enough single-window validation rows in {bucket}: {len(order)}")
        warmup.append(candidates[order[0]])
        measured.extend(candidates[index] for index in order[1:required])
    measured.sort(key=lambda row: (BUCKETS.index(row["duration_bucket"]), str(row["sample_id"])))
    selection = {
        "schema": "rdid-msa-efficiency-selection-v1",
        "source_manifest": str(MANIFEST),
        "source_manifest_sha256": sha256(MANIFEST),
        "split": "valid",
        "single_window_only": True,
        "seed": seed,
        "samples_per_duration_bucket": samples_per_bucket,
        "duration_buckets": list(BUCKETS),
        "warmup_sample_ids": [row["sample_id"] for row in warmup],
        "measurement_sample_ids": [row["sample_id"] for row in measured],
    }
    selection_path = output / "selection.json"
    if selection_path.is_file() and json.loads(selection_path.read_text()) != selection:
        raise ValueError(f"existing selection differs: {selection_path}")
    atomic_json(selection, selection_path)
    manifest_path = output / "selected_valid_tav.jsonl"
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in measured)
    if manifest_path.is_file() and manifest_path.read_text() != payload:
        raise ValueError(f"existing selected manifest differs: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(payload)
    return warmup, measured


def common_report(mode: str, rows: list[dict]) -> dict:
    return {
        "schema": "rdid-msa-teacher-student-efficiency-v1",
        "model": mode,
        "dataset": "CMU-MOSEI",
        "split": "official-validation",
        "input_subset": "TAV",
        "batch_size": 1,
        "utterances": len(rows),
        "single_window_only": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "host": socket.gethostname(),
        "selection": str((DEFAULT_OUTPUT / "selection.json").resolve()),
        "latency_protocol": {
            "warmup": "one held-out utterance per duration bucket",
            "synchronization": "torch.cuda.synchronize before and after each timed forward",
            "model_only": "preprocessed host tensors through regression output; includes host-to-device transfer",
            "end_to_end": "media read/decode, processor transforms, host-to-device transfer, and model forward",
        },
    }


def teacher_inputs(processor, row: dict):
    conversation = build_conversation(row, "tav", prompt_version="v1", video_fps=None)
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos, video_kwargs = read_teacher_media(conversation, "sampled_fps_v2")
    return processor(
        text=prompt, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=False, **video_kwargs,
    )


def load_probe(device: torch.device, dtype: torch.dtype) -> TeacherProbe:
    checkpoint = torch.load(TEACHER_PROBE, map_location="cpu", weights_only=True)
    probe = TeacherProbe(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        classes=int(checkpoint["classes"]),
        dropout=float(checkpoint["dropout"]),
    )
    probe.load_state_dict(checkpoint["model_state_dict"])
    probe.requires_grad_(False).eval().to(device=device, dtype=dtype)
    return probe


def teacher_active_equivalent(model, probe: TeacherProbe) -> dict[str, float | int | str]:
    total = 0
    expert = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if "experts" in name:
            expert += count
    probe_count = sum(parameter.numel() for parameter in probe.parameters())
    experts = int(model.config.thinker_config.text_config.num_experts)
    selected = int(model.config.thinker_config.text_config.num_experts_per_tok)
    equivalent = total - expert + expert * selected / experts + probe_count
    return {
        "expert_parameter_count": expert,
        "num_experts": experts,
        "experts_selected_per_token": selected,
        "active_parameter_equivalent": equivalent,
        "definition": "all non-expert parameters plus selected/total fraction of routed expert tensors; an architecture-equivalent count, not the number touched uniformly by every multimodal token",
    }


def run_teacher(output: Path, warmup_rows: list[dict], rows: list[dict]) -> None:
    from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    status_path = output / "teacher/status.json"
    atomic_json({"status": "loading"}, status_path)
    load_started = time.perf_counter()
    processor = Qwen3OmniMoeProcessor.from_pretrained(TEACHER_MODEL, local_files_only=True)
    config = Qwen3OmniMoeConfig.from_pretrained(TEACHER_MODEL, local_files_only=True)
    config.enable_audio_output = False
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        TEACHER_MODEL, config=config, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    model.requires_grad_(False).eval()
    device = torch.device("cuda:0")
    probe = load_probe(device, torch.bfloat16)
    load_seconds = time.perf_counter() - load_started

    def forward(inputs) -> float:
        inputs = inputs.to(device).to(model.dtype)
        with torch.inference_mode():
            hidden = extract_thinker_last_input_state(model, inputs)
            prediction = probe(hidden)["regression"]
        return float(prediction.float().cpu()[0])

    for row in warmup_rows:
        forward(teacher_inputs(processor, row))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    model_times: list[float] = []
    end_times: list[float] = []
    predictions: list[float] = []
    records = []
    for index, row in enumerate(rows, start=1):
        end_started = time.perf_counter()
        inputs = teacher_inputs(processor, row)
        torch.cuda.synchronize()
        model_started = time.perf_counter()
        prediction = forward(inputs)
        torch.cuda.synchronize()
        model_seconds = time.perf_counter() - model_started
        end_seconds = time.perf_counter() - end_started
        model_times.append(model_seconds)
        end_times.append(end_seconds)
        predictions.append(prediction)
        records.append({
            "sample_id": row["sample_id"], "duration_bucket": row["duration_bucket"],
            "target": float(row["sentiment"]), "prediction": prediction,
            "model_only_seconds": model_seconds, "end_to_end_seconds": end_seconds,
        })
        atomic_json({"status": "running", "completed": index, "total": len(rows)}, status_path)
        del inputs
    model_count = sum(parameter.numel() for parameter in model.parameters())
    probe_count = sum(parameter.numel() for parameter in probe.parameters())
    report = common_report("teacher", rows) | {
        "teacher_path": "Qwen3-Omni Thinker last-valid-input hidden state followed by the frozen seed2026 supervised Probe",
        "model_path": str(TEACHER_MODEL),
        "probe_path": str(TEACHER_PROBE),
        "load_seconds": load_seconds,
        "total_parameters": model_count + probe_count,
        "base_model_parameters": model_count,
        "probe_parameters": probe_count,
        "trainable_parameters_at_deployment": 0,
        "active_parameters": teacher_active_equivalent(model, probe),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "model_only": timing_summary(model_times),
        "end_to_end": timing_summary(end_times),
        "metrics": sentiment_metrics([row["sentiment"] for row in rows], predictions),
        "records": records,
    }
    atomic_json(report, output / "teacher/report.json")
    atomic_json({"status": "complete"}, status_path)


def student_components(run: Path, device: torch.device):
    config = json.loads((run / "run_config.json").read_text())
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    model_args = SimpleNamespace(**{
        key: value for key, value in config.items() if key not in {"frozen_assets", "input_sha256"}
    })
    for name in ("text_model", "audio_model", "video_model", "assets"):
        setattr(model_args, name, Path(getattr(model_args, name)))
    model = build_model(model_args, device)
    load_trainable(model, checkpoint["model"])
    model.eval()
    from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.text_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    audio_processor = AutoFeatureExtractor.from_pretrained(model_args.audio_model, local_files_only=True)
    video_processor = AutoImageProcessor.from_pretrained(model_args.video_model, local_files_only=True, use_fast=False)
    collator = OfficialTestCollator(tokenizer, audio_processor)
    return model, model_args, video_processor, collator


def run_student(mode: str, output: Path, warmup_rows: list[dict], rows: list[dict]) -> None:
    run = STUDENT_RUNS[mode]
    status_path = output / mode / "status.json"
    atomic_json({"status": "loading"}, status_path)
    device = torch.device("cuda:0")
    load_started = time.perf_counter()
    model, model_args, video_processor, collator = student_components(run, device)
    load_seconds = time.perf_counter() - load_started

    def one(row: dict) -> tuple[float, float, float]:
        end_started = time.perf_counter()
        item = VideoDataset([row], video_processor, mode=model_args.mode)[0]
        batch = collator([item])
        torch.cuda.synchronize()
        model_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(**batch_inputs(batch, device))["regression"]
        torch.cuda.synchronize()
        return float(prediction.float().cpu()[0]), time.perf_counter() - model_started, time.perf_counter() - end_started

    for row in warmup_rows:
        one(row)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    model_times: list[float] = []
    end_times: list[float] = []
    predictions: list[float] = []
    records = []
    for index, row in enumerate(rows, start=1):
        prediction, model_seconds, end_seconds = one(row)
        predictions.append(prediction)
        model_times.append(model_seconds)
        end_times.append(end_seconds)
        records.append({
            "sample_id": row["sample_id"], "duration_bucket": row["duration_bucket"],
            "target": float(row["sentiment"]), "prediction": prediction,
            "model_only_seconds": model_seconds, "end_to_end_seconds": end_seconds,
        })
        atomic_json({"status": "running", "completed": index, "total": len(rows)}, status_path)
    report = common_report(mode, rows) | {
        "source_run": str(run),
        "checkpoint": str(run / "best.pt"),
        "checkpoint_selection": "validation_mae",
        "source_best_epoch": json.loads((run / "status.json").read_text())["best_epoch"],
        "load_seconds": load_seconds,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "deployed_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "teacher_required_at_inference": False,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "model_only": timing_summary(model_times),
        "end_to_end": timing_summary(end_times),
        "metrics": sentiment_metrics([row["sentiment"] for row in rows], predictions),
        "records": records,
    }
    atomic_json(report, output / mode / "report.json")
    atomic_json({"status": "complete"}, status_path)


def summarize(output: Path) -> None:
    reports = {name: json.loads((output / name / "report.json").read_text()) for name in ("teacher", "full_kd", "uniform")}
    selection = json.loads((output / "selection.json").read_text())
    hosts = {report["host"] for report in reports.values()}
    gpus = {report["gpu_name"] for report in reports.values()}
    if len(hosts) != 1 or len(gpus) != 1:
        raise ValueError(f"benchmark devices differ: hosts={hosts}, gpus={gpus}")
    protocol = {
        "hardware": {
            "host": next(iter(hosts)),
            "gpu": next(iter(gpus)),
            "physical_device": "GPU 1 (exposed as logical cuda:0)",
            "same_machine_and_gpu": True,
            "execution": "teacher and students measured sequentially",
        },
        "precision": {
            "teacher": "BF16 model weights and BF16 Probe",
            "students": "BF16 autocast inference",
        },
        "batch_size": 1,
        "measurement_set": {
            "dataset": "CMU-MOSEI",
            "split": "official-validation",
            "utterances": len(selection["measurement_sample_ids"]),
            "repetitions_per_utterance": 1,
            "sampling": "distinct single-window TAV utterances: 30 each from <5 s, 5-15 s, and 15-30 s",
        },
        "warmup": {
            "utterances": len(selection["warmup_sample_ids"]),
            "sampling": "one held-out utterance per duration bucket",
            "included_in_statistics": False,
        },
        "timing": {
            "cuda_synchronization": "torch.cuda.synchronize immediately before and after every timed model forward",
            "model_latency": "preprocessed host tensors, host-to-device transfer, and model forward through the scalar regression output; excludes media I/O, decoding, tokenization, and modality preprocessing",
            "end_to_end": "audio loading, video decoding/frame sampling, tokenization, modality preprocessing, host-to-device transfer, and model forward; excludes DataLoader worker queuing and multi-sample batching",
            "throughput": "number of utterances divided by the sum of per-utterance end-to-end times",
        },
        "peak_gpu_memory": {
            "metric": "torch.cuda.max_memory_allocated()",
            "reset": "after warmup",
            "scope": "loaded model weights plus peak batch-1 inference allocations",
            "excludes": "CUDA reserved/cache memory and non-PyTorch allocations",
        },
        "checkpoint_selection": {
            "full_kd": "seed13, validation-MAE-selected checkpoint",
            "icd": "seed13, validation-MAE-selected checkpoint",
        },
        "historical_memory_note": "The earlier 5.424705 GiB student value used the same PyTorch allocated-memory metric but batch size 8 on the full official-test evaluator; it is not directly comparable to this batch-1 value.",
    }
    combined = {
        "schema": "rdid-msa-teacher-student-efficiency-summary-v1",
        "selection": selection,
        "protocol": protocol,
        "reports": reports,
        "ratios": {
            name: {
                "parameter_compression_teacher_over_student": reports["teacher"]["total_parameters"] / reports[name]["deployed_parameters"],
                "peak_memory_reduction_teacher_over_student": reports["teacher"]["peak_gpu_memory_gib"] / reports[name]["peak_gpu_memory_gib"],
                "model_p50_speedup_teacher_over_student": reports["teacher"]["model_only"]["p50_seconds"] / reports[name]["model_only"]["p50_seconds"],
                "model_p95_speedup_teacher_over_student": reports["teacher"]["model_only"]["p95_seconds"] / reports[name]["model_only"]["p95_seconds"],
                "end_to_end_throughput_speedup_student_over_teacher": reports[name]["end_to_end"]["throughput_utterances_per_second"] / reports["teacher"]["end_to_end"]["throughput_utterances_per_second"],
            }
            for name in ("full_kd", "uniform")
        },
    }
    atomic_json(combined, output / "summary.json")
    atomic_json(protocol, output / "protocol.json")
    lines = [
        "# Teacher–Student Compression and Deployment Efficiency", "",
        "All models were measured sequentially on the same NVIDIA RTX 6000D GPU of the same machine, using batch size 1.", "",
        "| Model | Params ↓ | Peak GPU Mem. (GiB) ↓ | Model Latency p50 / p95 (ms) ↓ | E2E p50 / p95 (ms) ↓ | E2E Throughput (utt/s) ↑ | Diagnostic MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"teacher": "Qwen3-Omni Teacher + Probe", "full_kd": "Full KD Student", "uniform": "ICD Student"}
    for name in ("teacher", "full_kd", "uniform"):
        row = reports[name]
        params = row.get("deployed_parameters", row["total_parameters"])
        lines.append(
            f"| {labels[name]} | {params:,} | {row['peak_gpu_memory_gib']:.3f} | "
            f"{1000*row['model_only']['p50_seconds']:.2f}/{1000*row['model_only']['p95_seconds']:.2f} | "
            f"{1000*row['end_to_end']['p50_seconds']:.2f}/{1000*row['end_to_end']['p95_seconds']:.2f} | "
            f"{row['end_to_end']['throughput_utterances_per_second']:.3f} | {row['metrics']['mae']:.6f} |"
        )
    lines.extend([
        "", "## Measurement protocol", "",
        f"- Hardware: `{protocol['hardware']['gpu']}` (physical GPU 1) on `{protocol['hardware']['host']}`; teacher and students were measured sequentially on this same GPU and machine.",
        "- Precision: teacher weights and Probe use BF16; students use BF16 autocast inference.",
        "- Inputs: 90 distinct single-window official-validation TAV utterances (30 each from `<5 s`, `5–15 s`, and `15–30 s`), one timed pass per utterance at batch size 1.",
        "- Warm-up: three held-out utterances, one per duration bucket; warm-up samples are excluded from all statistics.",
        "- Timing: `torch.cuda.synchronize()` immediately before and after every timed model forward.",
        "- Model latency starts from preprocessed host tensors and includes host-to-device transfer plus model forward through the scalar regression output. It excludes media I/O/decoding, tokenization, and modality preprocessing.",
        "- E2E throughput additionally includes audio loading, video decoding/frame sampling, tokenization, modality preprocessing, transfer, and forward. It excludes DataLoader worker queuing and multi-sample batching.",
        "- Peak GPU memory is `torch.cuda.max_memory_allocated()` after warm-up: loaded weights plus peak batch-1 inference allocations, excluding CUDA reserved/cache memory and non-PyTorch allocations.",
        "- Full KD and ICD use their seed13 validation-MAE-selected checkpoints. ICD denotes Uniform Interaction Distillation.",
        "- The earlier 5.425 GiB student result used batch size 8 in the full official-test evaluator; the present 2.091 GiB result is batch-1 peak allocated memory, so the two numbers are not directly comparable.",
        "- The 90-sample MAE is a same-input diagnostic only and does not replace full official-validation or official-test performance.",
    ])
    (output / "summary.md").write_text("\n".join(lines) + "\n")

    paper_lines = [
        "# Teacher–student compression and deployment efficiency on CMU-MOSEI", "",
        "| Model | Params ↓ | Peak GPU Mem. (GiB) ↓ | Model Latency p50 / p95 (ms) ↓ | E2E Throughput (utt/s) ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    rounded_p50 = {name: round(1000 * reports[name]["model_only"]["p50_seconds"]) for name in reports}
    rounded_p95 = {name: round(1000 * reports[name]["model_only"]["p95_seconds"]) for name in reports}
    rounded_throughput = {
        name: round(reports[name]["end_to_end"]["throughput_utterances_per_second"], 3)
        for name in reports
    }
    for name in ("teacher", "full_kd", "uniform"):
        row = reports[name]
        params = row.get("deployed_parameters", row["total_parameters"])
        params_label = f"{params / 1e9:.3f}B" if name == "teacher" else f"{params / 1e6:.3f}M"
        p50 = str(rounded_p50[name])
        p95 = str(rounded_p95[name])
        if rounded_p50[name] == min(rounded_p50.values()):
            p50 = f"**{p50}**"
        if rounded_p95[name] == min(rounded_p95.values()):
            p95 = f"**{p95}**"
        latency = f"{p50} / {p95}"
        throughput = f"{rounded_throughput[name]:.3f}"
        if rounded_throughput[name] == max(rounded_throughput.values()):
            throughput = f"**{throughput}**"
        paper_lines.append(
            f"| {'**ICD Student**' if name == 'uniform' else labels[name]} | {params_label} | "
            f"{row['peak_gpu_memory_gib']:.3f} | "
            f"{latency} | {throughput} |"
        )
    caption = "**Caption.** Teacher–student compression and deployment efficiency on CMU-MOSEI. All models were measured sequentially on the same physical NVIDIA RTX 6000D GPU (GPU 1) of the same machine, using batch size 1. The teacher and Probe use BF16; students use BF16 autocast inference. Statistics are computed over one timed pass through 90 distinct single-window official-validation TAV utterances (30 each from <5 s, 5–15 s, and 15–30 s), after three held-out warm-up utterances (one per bucket). CUDA was synchronized immediately before and after each timed model forward. Model latency covers preprocessed host tensors, host-to-device transfer, and model forward through the scalar regression output, excluding media I/O/decoding and preprocessing. E2E throughput additionally includes audio loading, video decoding/frame sampling, tokenization, multimodal preprocessing, transfer, and model forward; it excludes DataLoader worker queuing and multi-sample batching. Peak GPU memory is peak allocated memory measured by `torch.cuda.max_memory_allocated()` after warm-up, including model weights and batch-1 inference tensors. Full KD and ICD use seed13 checkpoints selected by validation MAE. ICD denotes Uniform Interaction Distillation."
    appendix_note = "**Appendix note.** The earlier 5.425 GiB student figure used batch size 8 in the full official-test evaluator, whereas 2.091 GiB here is batch-1 peak allocated memory. The difference is therefore expected and the two figures should not be compared directly."
    paper_lines.extend([
        "", caption,
        "", appendix_note,
    ])
    PAPER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    PAPER_TABLE.write_text("\n".join(paper_lines) + "\n")
    PAPER_CAPTION.write_text(caption + "\n\n" + appendix_note + "\n")
    atomic_json({"status": "complete", "summary": str(output / "summary.json")}, output / "status.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "teacher", "full_kd", "uniform", "summarize"), default="all")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-bucket", type=int, default=30)
    parser.add_argument("--selection-seed", type=int, default=20260924)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    warmup, rows = prepare_rows(output, args.samples_per_bucket, args.selection_seed)
    if args.mode == "all":
        atomic_json({"status": "running", "phases": ["teacher", "full_kd", "uniform"]}, output / "status.json")
        for mode in ("teacher", "full_kd", "uniform"):
            report = output / mode / "report.json"
            if report.is_file():
                continue
            subprocess.run([
                sys.executable, "-u", str(Path(__file__).resolve()), "--mode", mode,
                "--output", str(output), "--samples-per-bucket", str(args.samples_per_bucket),
                "--selection-seed", str(args.selection_seed),
            ], check=True)
        summarize(output)
        return
    if args.mode == "summarize":
        summarize(output)
    elif args.mode == "teacher":
        run_teacher(output, warmup, rows)
    else:
        run_student(args.mode, output, warmup, rows)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
