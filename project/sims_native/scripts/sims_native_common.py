"""Native SIMS labels and fixed official evaluation definitions."""
import math

METRIC_PROTOCOL = {
    'native_label_range': [-1.0, 1.0], 'label_scale': 1.0,
    'regression_output': 'tanh; no label or prediction rescaling',
    'auxiliary_classification_classes': 5,
    'class_boundaries': [-0.7, -0.1, 0.1, 0.7],
    'boundary_policy': 'open_left_closed_right',
    'mae_units': 'native [-1,1] scores',
    'acc2_best': 'validation SIMS official Acc2; label<=0 vs label>0; zeros included',
    'metric_source': 'https://github.com/thuiar/MMSA/blob/master/src/MMSA/utils/metricsTop.py',
    'test_policy': 'evaluate validation-MAE-best and validation-Acc2-best separately, never select by test',
}


def five_class_index(label):
    if not math.isfinite(label) or not -1 <= label <= 1:
        raise ValueError('SIMS label must be finite and in [-1,1]')
    return sum(label > boundary for boundary in METRIC_PROTOCOL['class_boundaries'])
