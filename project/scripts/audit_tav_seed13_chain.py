#!/usr/bin/env python3
"""Technical promotion audit; no MAE improvement or method-ranking gate."""
import json
import math
from pathlib import Path

from train_video_adaptation_v2 import atomic_json, sha256

CORE_METHODS = ("M0", "M1", "M3", "M4", "M6")


def audit(base, assets, methods=CORE_METHODS):
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or not set(methods).issubset(CORE_METHODS):
        raise ValueError("unique core methods required")
    manifest = [json.loads(x) for x in Path(assets["manifest"]).read_text().splitlines() if x.strip()]
    valid = {r["parent_sample_id"]: r for r in manifest if r["split"] == "valid"}
    windows = {s: sum(r["split"] == s for r in manifest) for s in ("train", "valid")}
    records = {}
    for method in methods:
        output = base / "students" / f"{method}_seed13"
        status = json.loads((output / "status.json").read_text())
        report = json.loads((output / "report.json").read_text())
        config = json.loads((output / "run_config.json").read_text())
        if status["status"] != "complete" or report["method"] != method or report["seed"] != 13:
            raise ValueError(f"incomplete/incorrect seed13 run: {method}")
        if report["protocol"] != config or config["frozen_assets"] != assets:
            raise ValueError(f"protocol differs: {method}")
        expected = {"input_subset":"tav", "teacher_subset":"tav", "batch_size":8,
                    "gradient_accumulation":1, "learning_rate":1e-4, "weight_decay":.01,
                    "epochs":30, "patience":7, "video_layers":12, "video_frames":16,
                    "video_rank":8, "video_alpha":16, "video_dropout":.05,
                    "teacher_probe_seed":2026, "teacher_temperature":assets["teacher_temperature"],
                    "teacher_calibration_temperature":assets["teacher_temperature"],
                    "kd_temperature":2., "alpha_ce":.5, "lambda_full":1.,
                    "diagnostics":False, "limit_per_split":None, "official_test_evaluated":False}
        if any(config.get(k) != v for k,v in expected.items()):
            raise ValueError(f"training input/temperature/LoRA protocol mismatch: {method}")
        for name, digest in config["input_sha256"].items():
            if sha256(name) != digest:
                raise ValueError(f"changed run input: {name}")
        if (report["train_windows"] != windows["train"] or report["valid_windows"] != windows["valid"]
                or report["official_test_evaluated"]):
            raise ValueError(f"data coverage mismatch: {method}")
        rows = [json.loads(x) for x in (output / "predictions.jsonl").read_text().splitlines() if x.strip()]
        if len(rows) != len(valid) or {r["parent_sample_id"] for r in rows} != set(valid):
            raise ValueError(f"valid prediction coverage differs: {method}")
        for row in rows:
            original = valid[row["parent_sample_id"]]
            if (row["split"] != "valid" or row["video_id"] != original["video_id"]
                    or row["target_sentiment"] != original["sentiment"]
                    or not math.isfinite(row["prediction"])):
                raise ValueError(f"invalid prediction identity/value: {method}")
        gradients = json.loads((output / "adapter_gradient_audit.json").read_text())
        expected_counts = {"text_encoder.":0, "audio_encoder.":0, "video_encoder.":0} if method in ("M0", "M1") else {
            "text_encoder.":112, "audio_encoder.":48, "video_encoder.":48}
        for prefix, count in expected_counts.items():
            values = [v for n,v in gradients.items() if n.startswith(prefix) and "lora_B" in n]
            if len(values) != count or any(v is None or not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f"LoRA gradient coverage mismatch: {method}/{prefix}")
        if report["video_trainable_parameters"] != (0 if method in ("M0", "M1") else 589824):
            raise ValueError(f"Video adaptation mismatch: {method}")
        records[method] = {"valid_mae_observation_only":report["valid_metrics"]["mae"],
                           "protocol_sha256":sha256(output / "run_config.json"), "checks":"pass"}
    result = {"status":"pass", "promotion_basis":"technical_integrity_only", "performance_gate":False,
              "M4_improvement_required":False, "runs":records, "official_test_evaluated":False}
    if methods == CORE_METHODS:
        output = base / "seed13_chain_audit.json"
    else:
        directory = base / "seed13_method_audits"
        directory.mkdir(exist_ok=True)
        output = directory / ("_".join(methods) + ".json")
    atomic_json(result, output)
    return result
