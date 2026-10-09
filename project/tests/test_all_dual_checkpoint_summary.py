"""Guard against historical checkpoint/result selection regressions."""
import importlib.util
from pathlib import Path

path = Path(__file__).parents[1] / 'scripts/summarize_all_dual_checkpoints.py'
spec = importlib.util.spec_from_file_location('dual_summary', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_acc_ties_do_not_silently_use_old_f1_or_test_tiebreak():
    history = [
        {'epoch': 2, 'valid_metrics': {'mae': .8, 'acc2_nonzero': .9, 'f1_weighted_nonzero': .89}, 'test_metrics': {'mae': .9}},
        {'epoch': 3, 'valid_metrics': {'mae': .7, 'acc2_nonzero': .9, 'f1_weighted_nonzero': .91}, 'test_metrics': {'mae': .6}},
    ]
    assert module.select(history, 'valid_acc2_nonzero')['epoch'] == 2
    assert module.select(history, 'valid_mae')['epoch'] == 3


def test_legacy_flat_history_preserves_missing_f1():
    history = [{'epoch': 1, 'valid_mae': .7, 'valid_acc2_nonzero': .8},
               {'epoch': 2, 'valid_mae': .8, 'valid_acc2_nonzero': .9}]
    selected = module.select(history, 'valid_acc2_nonzero')
    assert module.metrics(selected) == {'mae': .8, 'acc2_nonzero': .9}


def test_source_priority_does_not_choose_better_test_score():
    sweep = {'source': '/sweep/epoch_002/report.json', 'metrics': {'mae': .5}, 'exploratory': True}
    official = {'source': '/official/report.json', 'metrics': {'mae': .8}, 'exploratory': False}
    result, variation, unresolved = module.choose_test([sweep, official])
    assert result == official and variation and not unresolved


def test_fixed_epoch_preferred_over_mutable_alias_and_real_conflict_blocked():
    epoch = {'source': '/checkpoints/epoch_006/report.json', 'metrics': {'mae': .7}, 'exploratory': True}
    alias = {'source': '/checkpoints/best/report.json', 'metrics': {'mae': .6}, 'exploratory': True}
    assert module.choose_test([alias, epoch]) == (epoch, True, False)
    conflicting = {**epoch, 'source': '/another/epoch_006/report.json', 'metrics': {'mae': .8}}
    assert module.choose_test([epoch, conflicting]) == (None, True, True)
