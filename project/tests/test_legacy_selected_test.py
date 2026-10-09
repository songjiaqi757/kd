"""Validate historical checkpoint identity before expensive test inference."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

path=Path(__file__).parents[1]/'scripts/evaluate_legacy_selected_test.py'
spec=importlib.util.spec_from_file_location('legacy_selected_test',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_run(tmp_path):
    history=[{'epoch':1,'valid_metrics':{'mae':.7,'acc2_nonzero':.8}},
             {'epoch':2,'valid_metrics':{'mae':.8,'acc2_nonzero':.9}}]
    (tmp_path/'history.json').write_text(json.dumps(history))
    (tmp_path/'report.json').write_text(json.dumps({'best_epoch':1,'valid_metrics':history[0]['valid_metrics']}))
    return history


def test_legacy_epoch_only_best_is_accepted_with_matching_report(tmp_path):
    fixture_run(tmp_path);torch.save({'epoch':1,'model':{}},tmp_path/'best.pt')
    _,row=module.load_selected(tmp_path,tmp_path/'best.pt','valid_mae')
    assert row['epoch']==1


def test_mae_best_cannot_replace_missing_acc_best(tmp_path):
    fixture_run(tmp_path);torch.save({'epoch':1,'model':{}},tmp_path/'best.pt')
    with pytest.raises(ValueError,match='Checkpoint epoch'):
        module.load_selected(tmp_path,tmp_path/'best.pt','valid_acc2_nonzero')


def test_conflicting_embedded_metrics_are_rejected(tmp_path):
    fixture_run(tmp_path)
    torch.save({'epoch':1,'model':{},'valid_metrics':{'mae':.6,'acc2_nonzero':.8}},tmp_path/'best.pt')
    with pytest.raises(ValueError,match='metrics differ'):
        module.load_selected(tmp_path,tmp_path/'best.pt','valid_mae')


def test_epoch_only_untracked_alias_is_rejected(tmp_path):
    fixture_run(tmp_path);torch.save({'epoch':1,'model':{}},tmp_path/'other.pt')
    with pytest.raises(ValueError,match='matching original best report'):
        module.load_selected(tmp_path,tmp_path/'other.pt','valid_mae')
