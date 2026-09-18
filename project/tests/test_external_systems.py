from pathlib import Path
from argparse import Namespace
import json
import pickle
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import rdid_mosei.external_systems as external
from build_main_table_plan import build
from run_external_system import locked_checkpoint


def test_canonical_native_ids_match_project_ids():
    assert external.canonical_id(b"video$_$03") == "video[3]"
    assert external.canonical_id("video[3]") == "video[3]"


def test_standard_feature_audit_checks_ids_labels_and_shapes(tmp_path, monkeypatch):
    monkeypatch.setitem(external.EXPECTED_COUNTS, "mosi", {"train": 2, "valid": 1, "test": 1})
    dataset = tmp_path / "cmu_mosi"
    manifests = dataset / "manifests"
    manifests.mkdir(parents=True)
    split_ids = {"train": ["a[0]", "b[1]"], "valid": ["c[0]"], "test": ["d[2]"]}
    split_labels = {"train": [0.5, -1.0], "valid": [1.25], "test": [0.0]}
    payload = {}
    for split, ids in split_ids.items():
        rows = [{"sample_id": sample_id, "sentiment": label} for sample_id, label in zip(ids, split_labels[split])]
        (manifests / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        count = len(ids)
        payload[split] = {
            "id": np.asarray([sample_id.replace("[", "$_$").replace("]", "") for sample_id in ids]),
            "regression_labels": np.asarray(split_labels[split])[:, None],
            "audio": np.zeros((count, 4, 5), dtype=np.float32),
            "vision": np.zeros((count, 4, 20), dtype=np.float32),
            "text_bert": np.zeros((count, 3, 50), dtype=np.float32),
        }
    feature_file = tmp_path / "aligned_50.pkl"
    with feature_file.open("wb") as handle:
        pickle.dump(payload, handle)
    report = external.audit_standard_features(feature_file, "mosi", dataset)
    assert report["status"] == "pass"
    assert report["splits"]["test"]["id_coverage"] == "exact"

    payload["valid"]["regression_labels"][0, 0] = -2.0
    with feature_file.open("wb") as handle:
        pickle.dump(payload, handle)
    with pytest.raises(ValueError, match="label mismatch"):
        external.audit_standard_features(feature_file, "mosi", dataset)


def test_full_plan_has_no_training_command_that_unlocks_test():
    matrix = json.loads((ROOT / "configs/main_table_v1/experiments.json").read_text())
    runs = build(matrix, "mosei")
    assert len(runs) == 57
    assert sum(run["kind"] in {"student", "selected_replication", "native_system"} for run in runs) == 53
    commands = [run["command"] for run in runs if "command" in run]
    assert all("--allow-official-test" not in command for command in commands)
    assert {run.get("system") for run in runs if run["kind"] == "native_system"} == {"dlf", "dpdf_lq"}
    main_rows = {item["row"]: item["method"] for item in matrix["main_table_a"]}
    assert main_rows["A6"] == "uniform_interaction"
    assert main_rows["A7"] == "ru_interaction"
    assert {run["method"] for run in runs if run["kind"] == "selection_gate"} == {
        "projector", "ea_kd", "cmad_cafd", "subset7",
    }


def test_student_plan_readiness_comes_from_frozen_asset_file():
    matrix = json.loads((ROOT / "configs/main_table_v1/experiments.json").read_text())
    for dataset in ("mosei", "mosi"):
        assets = ROOT.parent / matrix["datasets"][dataset]["assets"]
        student = [run for run in build(matrix, dataset) if run["kind"] == "student"]
        assert student
        assert all(run["ready"] == assets.is_file() for run in student)


def test_native_source_patches_remove_test_from_epoch_selection():
    workspace = ROOT.parent
    dlf = (workspace / "external/original/DLF/trains/singleTask/DLF.py").read_text()
    gsit = (workspace / "external/original/GsiT/src/MMSA-GsiT/trains/custom/GSIT.py").read_text()
    dpdf = (workspace / "external/original/DPDF-LQ/train.py").read_text()
    assert "dataloader['test']" not in dlf
    assert "dataloader['test']" not in gsit
    assert "test_ret = evaluate" not in dpdf
    assert "current_valid_mae" in dpdf


def test_native_final_test_requires_locked_valid_checkpoint(tmp_path):
    source = tmp_path / "source"
    (source / "checkpoints").mkdir(parents=True)
    (source / "status.json").write_text(json.dumps({"status": "complete"}))
    (source / "run_config.json").write_text(json.dumps({
        "system": "dlf", "dataset": "mosei", "seed": 13, "phase": "train_valid",
    }))
    checkpoint = source / "checkpoints" / "DLF-mosei.pth"
    checkpoint.write_bytes(b"locked")
    args = Namespace(phase="final_test", source_run=source, system="dlf", dataset="mosei", seed=13)
    resolved_source, resolved_checkpoint = locked_checkpoint(args)
    assert resolved_source == source.resolve()
    assert resolved_checkpoint == checkpoint.resolve()

    (source / "status.json").write_text(json.dumps({"status": "training"}))
    with pytest.raises(ValueError, match="completed"):
        locked_checkpoint(args)
