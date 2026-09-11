import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from analyze_stage_d_significance import cluster_comparison, exact_mcnemar, load_predictions


def rows(predictions):
    return {
        str(i): {
            "parent_sample_id": str(i),
            "video_id": f"v{i // 2}",
            "split": "valid",
            "target_sentiment": target,
            "prediction": prediction,
        }
        for i, (target, prediction) in enumerate(predictions)
    }


def test_cluster_bootstrap_is_deterministic_and_reports_video_count():
    baseline = rows([(1, -1), (1, 0.2), (-1, 1), (-1, -0.2)])
    candidate = rows([(1, 0.8), (1, 0.5), (-1, -0.8), (-1, -0.5)])
    left = cluster_comparison(baseline, candidate, 200, 13)
    right = cluster_comparison(baseline, candidate, 200, 13)
    assert left == right
    assert left["video_clusters"] == 2
    assert left["comparison"]["mae"]["candidate_minus_baseline"] < 0
    assert left["acc2_nonzero_mcnemar"]["baseline_wrong_candidate_correct"] == 2


def test_exact_mcnemar_all_discordance_favors_candidate():
    result = exact_mcnemar(np.ones(4), -np.ones(4), np.ones(4))
    assert result["discordant"] == 4
    assert result["exact_two_sided_p_value"] == pytest.approx(0.125)


def test_prediction_manifest_mismatch_is_rejected(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"parent_sample_id": "x", "split": "valid", "target_sentiment": 2, "prediction": 0}) + "\n")
    identities = {"x": {"video_id": "v", "split": "valid", "target_sentiment": 1.0}}
    with pytest.raises(ValueError, match="mismatch"):
        load_predictions(path, identities)
