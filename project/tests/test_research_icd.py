from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rdid_mosei.research_icd import (
    encoder_objective, corrected_coordinates, decay_coefficient, assign_routed_gradients,
)
from rdid_mosei.interaction import inverse_mobius, mobius_transform


@pytest.mark.parametrize('variant,indices', [('order_routed', [0, 1, 2]),
                                            ('scrambled_routed', [0, 3, 6]), ('fusion_only', [])])
def test_routing_updates_only_intended_parameter_group(variant, indices):
    enc = torch.nn.Parameter(torch.tensor(2.0))
    fus = torch.nn.Parameter(torch.tensor(3.0))
    common = (enc * fus - 1).square()
    coordinates = torch.stack([(i + 1) * (enc + fus).square() for i in range(7)])
    encoder_loss = encoder_objective(variant, common, coordinates, 0.7)
    fusion_loss = common + 0.7 * coordinates.mean()
    expected_enc = 2*(2*3-1)*3 + 0.7*sum(i+1 for i in indices)/7*2*(2+3)
    expected_fus = 2*(2*3-1)*2 + 0.7*4*2*(2+3)
    assign_routed_gradients(encoder_loss, fusion_loss, [enc], [fus])
    assert enc.grad.item() == pytest.approx(expected_enc)
    assert fus.grad.item() == pytest.approx(expected_fus)


def test_dgl_encoder_receives_unimodal_loss_only():
    enc = torch.nn.Parameter(torch.tensor(2.0))
    fus = torch.nn.Parameter(torch.tensor(3.0))
    common = enc.square() + fus.square()
    coordinates = torch.stack([enc*fus]*7)
    unimodal = (enc + fus - 9).square()
    loss = encoder_objective('dgl_icd', common, coordinates, 1, unimodal)
    assign_routed_gradients(loss, common+coordinates.mean(), [enc], [fus])
    assert enc.grad.item() == -8
    assert fus.grad.item() == 8


def test_routing_rejects_overlapping_groups():
    p = torch.nn.Parameter(torch.tensor(1.0))
    with pytest.raises(ValueError, match='disjoint'):
        assign_routed_gradients(p**2, p**2, [p], [p])


@pytest.mark.parametrize('variant', ['target_uniform', 'target_variance'])
def test_target_correction_preserves_same_full_label_mix_and_reconstruction(variant):
    mean = torch.randn(6, 7, requires_grad=True)
    variance = torch.rand(6, 7, requires_grad=True)
    labels = torch.randn(6, requires_grad=True)
    result = corrected_coordinates(mean, variance, labels, .17, variant)
    assert not result.requires_grad
    original_full = .17 + mean.detach().sum(-1)
    torch.testing.assert_close(.17 + result.sum(-1), 0.5*original_full + 0.5*labels.detach())
    torch.testing.assert_close(mobius_transform(inverse_mobius(result, .17), .17), result,
                               atol=2e-6, rtol=1e-5)


def test_variance_control_is_uniform_when_all_variances_are_zero():
    mean = torch.randn(4, 7)
    var = torch.zeros_like(mean)
    labels = torch.randn(4)
    a = corrected_coordinates(mean, var, labels, .17, 'target_variance')
    b = corrected_coordinates(mean, var, labels, .17, 'target_uniform')
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(corrected_coordinates(mean, var, labels, .17, 'ensemble_source'), mean)


def test_variance_allocation_favors_less_stable_coordinates():
    mean = torch.zeros(1, 7)
    var = torch.tensor([[10., 0., 0., 0., 0., 0., 0.]])
    result = corrected_coordinates(mean, var, torch.tensor([2.]), 0, 'target_variance')
    assert result[0, 0] > result[0, 1]
    assert result.sum().item() == pytest.approx(1)
    with pytest.raises(ValueError, match='Negative'):
        corrected_coordinates(mean, -var, torch.tensor([2.]), 0, 'target_variance')


def test_decay_is_fixed_by_epoch_not_validation_metrics():
    assert [decay_coefficient(epoch, 20) for epoch in range(1, 11)] == [1]*10
    assert decay_coefficient(11, 20) == .9
    assert decay_coefficient(20, 20) == 0
