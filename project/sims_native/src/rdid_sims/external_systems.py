"""Auditing and configuration helpers for native DLF/GsiT/DPDF-LQ runs."""
from __future__ import annotations

import copy
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np


SYSTEMS = ("dlf", "gsit", "dpdf_lq")
DATASETS = ("mosi", "mosei")
EXPECTED_COUNTS = {
    "mosi": {"train": 1284, "valid": 229, "test": 686},
    "mosei": {"train": 16326, "valid": 1871, "test": 4659},
}
EXPECTED_DIMS = {"mosi": (5, 20), "mosei": (74, 35)}


def canonical_id(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value)
    match = re.fullmatch(r"(.+?)\$_\$(\d+)", value)
    if match:
        return f"{match.group(1)}[{int(match.group(2))}]"
    match = re.fullmatch(r"(.+?)\[(\d+)\]", value)
    if match:
        return f"{match.group(1)}[{int(match.group(2))}]"
    return value


def manifest_rows(dataset_root: Path) -> dict[str, dict[str, dict]]:
    result = {}
    for split in ("train", "valid", "test"):
        path = dataset_root / "manifests" / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        mapped = {canonical_id(row["sample_id"]): row for row in rows}
        if len(mapped) != len(rows):
            raise ValueError(f"duplicate canonical IDs in {path}")
        result[split] = mapped
    return result


def audit_standard_features(feature_path: Path, dataset: str, dataset_root: Path) -> dict:
    if dataset not in DATASETS:
        raise ValueError(dataset)
    with feature_path.open("rb") as handle:
        data = pickle.load(handle)
    manifests = manifest_rows(dataset_root)
    report = {"feature_path": str(feature_path), "dataset": dataset, "splits": {}, "status": "pass"}
    audio_dim, vision_dim = EXPECTED_DIMS[dataset]
    for split, expected_count in EXPECTED_COUNTS[dataset].items():
        if split not in data:
            raise ValueError(f"feature file has no {split} split")
        section = data[split]
        required = {"id", "audio", "vision", "text_bert", "regression_labels"}
        if missing := required - set(section):
            raise ValueError(f"{split} missing fields: {sorted(missing)}")
        ids = [canonical_id(value) for value in section["id"]]
        if len(ids) != expected_count or len(set(ids)) != len(ids):
            raise ValueError(f"{split} count/ID uniqueness differs: {len(ids)} != {expected_count}")
        expected_ids = set(manifests[split])
        if set(ids) != expected_ids:
            raise ValueError(
                f"{split} ID coverage differs: missing={len(expected_ids-set(ids))}, extra={len(set(ids)-expected_ids)}"
            )
        labels = np.asarray(section["regression_labels"], dtype=np.float64).reshape(-1)
        if labels.shape != (expected_count,) or not np.isfinite(labels).all():
            raise ValueError(f"invalid {split} labels")
        label_by_id = dict(zip(ids, labels.tolist()))
        mismatches = [
            sample_id for sample_id, row in manifests[split].items()
            if abs(label_by_id[sample_id] - float(row["sentiment"])) > 1e-5
        ]
        if mismatches:
            raise ValueError(f"{split} label mismatch for {len(mismatches)} IDs")
        audio = np.asarray(section["audio"])
        vision = np.asarray(section["vision"])
        text = np.asarray(section["text_bert"])
        if audio.shape[0] != expected_count or audio.shape[-1] != audio_dim:
            raise ValueError(f"{split} audio shape differs: {audio.shape}")
        if vision.shape[0] != expected_count or vision.shape[-1] != vision_dim:
            raise ValueError(f"{split} vision shape differs: {vision.shape}")
        if text.shape[0] != expected_count or text.ndim != 3 or text.shape[1] != 3:
            raise ValueError(f"{split} text_bert shape differs: {text.shape}")
        report["splits"][split] = {
            "samples": expected_count,
            "audio_shape": list(audio.shape),
            "vision_shape": list(vision.shape),
            "text_bert_shape": list(text.shape),
            "id_coverage": "exact",
            "label_match": "exact_with_tolerance_1e-5",
        }
    return report


def build_native_config(
    system: str,
    dataset: str,
    feature_path: Path,
    bert_model: Path,
    repositories_root: Path,
) -> tuple[dict, str]:
    if system not in SYSTEMS or dataset not in DATASETS:
        raise ValueError(f"unknown system/dataset: {system}/{dataset}")
    if system == "dlf":
        source = repositories_root / "DLF/config/config.json"
        config = json.loads(source.read_text())
        config["datasetCommonParams"]["dataset_root_dir"] = str(feature_path.parent)
        for alignment in ("aligned", "unaligned"):
            config["datasetCommonParams"][dataset][alignment]["featurePath"] = feature_path.name
        config["DLF"]["datasetParams"][dataset]["pretrained"] = str(bert_model)
        return config, "json"
    if system == "gsit":
        source = repositories_root / "GsiT/src/MMSA-GsiT/config/config_regression.json"
        config = json.loads(source.read_text())
        config["datasetCommonParams"]["dataset_root_dir"] = str(feature_path.parent)
        for alignment in ("aligned", "unaligned"):
            config["datasetCommonParams"][dataset][alignment]["featurePath"] = feature_path.name
        # config.py resolves <root>/bert/bert-base-uncased.
        config["pretrainedWeights"]["weights_root_dir"] = str(bert_model.parent.parent)
        return config, "json"

    import yaml

    source = repositories_root / f"DPDF-LQ/configs/{dataset}.yaml"
    config = yaml.safe_load(source.read_text())
    config["dataset"]["dataPath"] = str(feature_path)
    config["model"]["bert_pretrained"] = str(bert_model)
    return config, "yaml"


def write_native_config(config: dict, kind: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "json":
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    elif kind == "yaml":
        import yaml
        path.write_text(yaml.safe_dump(config, sort_keys=False))
    else:
        raise ValueError(kind)
