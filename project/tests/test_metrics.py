import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.metrics import polarity_metrics, sentiment_metrics


def test_sentiment_metrics_perfect_predictions() -> None:
    metrics = sentiment_metrics([-3, -1, 0, 1, 3], [-3, -1, 0, 1, 3])
    assert metrics["mae"] == 0.0
    assert metrics["pearson"] == 1.0
    assert metrics["acc2_nonzero"] == 1.0
    assert metrics["f1_weighted_nonzero"] == 1.0
    assert metrics["acc2_has_zero"] == 1.0
    assert metrics["acc7"] == 1.0


def test_nonzero_policy_excludes_only_true_zero_and_predicts_zero_as_nonnegative() -> None:
    metrics = sentiment_metrics([-1, 0, 1], [0, -1, 0])
    assert metrics["nonzero_count"] == 2
    assert metrics["acc2_nonzero"] == 0.5
    assert metrics["acc2_has_zero"] == 1 / 3


def test_constant_predictions_have_undefined_correlation() -> None:
    metrics = sentiment_metrics([-1, 0, 1], [0, 0, 0])
    assert math.isnan(metrics["pearson"])


def test_polarity_metrics_exclude_zero_targets() -> None:
    metrics = polarity_metrics(
        [-1.0, 0.0, 1.0],
        [[3.0, -1.0], [10.0, -10.0], [-2.0, 2.0]],
    )
    assert metrics["binary_count"] == 2
    assert metrics["binary_acc2"] == 1.0
    assert metrics["binary_f1_weighted"] == 1.0
    assert metrics["binary_f1_macro"] == 1.0
    assert metrics["binary_negative_recall"] == 1.0
    assert metrics["binary_positive_recall"] == 1.0
    assert metrics["binary_nll"] > 0.0


def test_polarity_metrics_report_class_imbalance_failures() -> None:
    metrics = polarity_metrics(
        [-1.0, -0.5, 1.0, 2.0],
        [[0.0, 1.0]] * 4,
    )
    assert metrics["binary_acc2"] == 0.5
    assert metrics["binary_negative_recall"] == 0.0
    assert metrics["binary_positive_recall"] == 1.0
