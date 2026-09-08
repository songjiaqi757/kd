#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rdid_mosei.student import CachedStudentCore, SUBSETS
from train_student_baseline import (
    FeatureDataset,
    add_window_weights,
    aggregate_windows,
    collate_features,
    to_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all seven subset forwards from a saved student checkpoint")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    config = json.loads((args.features / "run_config.json").read_text(encoding="utf-8"))
    if config.get("status") != "complete":
        raise RuntimeError("feature cache is not complete")
    rows = read_jsonl(args.features / "index.jsonl")
    if {str(row["split"]) for row in rows} != {"train", "valid"}:
        raise RuntimeError("expected exactly official train and valid splits")
    add_window_weights(rows)
    loaders = [
        DataLoader(
            FeatureDataset([row for row in rows if row["split"] == split]),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_features,
            pin_memory=args.device.startswith("cuda"),
        )
        for split in ("train", "valid")
    ]
    device = torch.device(args.device)
    model = CachedStudentCore(
        text_hidden_size=int(config["hidden_sizes"]["t"]),
        audio_hidden_size=int(config["hidden_sizes"]["a"]),
        video_hidden_size=int(config["hidden_sizes"]["v"]),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    records = []
    for loader in loaders:
        for batch in loader:
            hidden, masks = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(hidden, masks, subsets=SUBSETS)
            subset_values = {
                subset: outputs[subset]["regression"].float().cpu().tolist() for subset in SUBSETS
            }
            tav_logits = outputs["tav"]["classification_logits"].float().cpu().tolist()
            for index, row in enumerate(batch["rows"]):
                records.append({
                    "sample_id": row["sample_id"],
                    "parent_sample_id": row["parent_sample_id"],
                    "split": row["split"],
                    "target_sentiment": float(row["sentiment"]),
                    "class_7_index": int(row["class_7_index"]),
                    "aggregation_weight": float(row["aggregation_weight"]),
                    "prediction": float(subset_values["tav"][index]),
                    "classification_logits": tav_logits[index],
                    "subset_predictions": {
                        subset: float(subset_values[subset][index]) for subset in SUBSETS
                    },
                })
    utterances = aggregate_windows(records)
    reference = {str(row["parent_sample_id"]): row for row in read_jsonl(args.reference_predictions)}
    generated = {str(row["parent_sample_id"]): row for row in utterances}
    if set(reference) != set(generated):
        raise RuntimeError("generated and reference utterance IDs differ")
    differences = np.asarray([
        abs(float(generated[sample_id]["prediction"]) - float(reference[sample_id]["prediction"]))
        for sample_id in sorted(reference)
    ])
    if float(differences.max()) > 1e-6:
        raise RuntimeError(f"TAV reproduction mismatch: max abs difference {differences.max()}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in utterances:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    audit = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "features": str(args.features),
        "splits": sorted({str(row["split"]) for row in utterances}),
        "windows": len(rows),
        "utterances": len(utterances),
        "subsets": list(SUBSETS),
        "tav_reference_max_abs_difference": float(differences.max()),
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
