#!/usr/bin/env python3
"""Isolated process-local ICD research hooks; legacy trainer files stay unchanged."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'project/scripts'), str(ROOT / 'project/src')]
import torch
from torch.nn import functional as F
import train_main_table_kd as trainer
from rdid_mosei.interaction import mobius_transform
from rdid_mosei.main_table_kd import probability_kd_per_sample, weighted_mean
from rdid_mosei.student import SUBSETS
import rdid_mosei.research_icd as research
from rdid_mosei.decision_icd import error_strata

ORIGINAL_BUILD = trainer.build_model
ORIGINAL_COMPONENTS = trainer.uniform_interaction_components
ORIGINAL_PROTOCOL = trainer.run_protocol
ORIGINAL_SAVE = trainer.atomic_save
ORIGINAL_EVALUATE = trainer.evaluate


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--research-variant', choices=research.VARIANTS, required=True)
    extra, remaining = parser.parse_known_args()
    previous = sys.argv
    try:
        sys.argv = [previous[0], *remaining]
        args = trainer.parse_args()
    finally:
        sys.argv = previous
    if args.method != 'uniform_interaction' or not args.retain_best_mae_and_acc2 or args.save_every_epoch:
        parser.error('Require uniform_interaction with dual-best retention and no epoch snapshots')
    if args.min_epochs != args.epochs or args.epochs != 20:
        parser.error('This frozen pilot requires 20 fixed epochs')
    vars(args).update(vars(extra))
    return args


class ResearchHooks:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.step = 0
        self.latest_predictions = None

    def build(self, args, assets, device):
        self.model = ORIGINAL_BUILD(args, assets, device)
        named = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]
        self.enc = [p for name, p in named if not name.startswith('fusion.')]
        self.fus = [p for name, p in named if name.startswith('fusion.')]
        self.groups = {prefix: [p for name, p in named if name.startswith(prefix + '.')]
                       for prefix in ('text_encoder', 'audio_encoder', 'video_encoder', 'pools', 'fusion')}
        grouped = {id(p) for values in self.groups.values() for p in values}
        if grouped != {id(p) for _, p in named}:
            raise ValueError('Unaccounted trainable parameter group')
        return self.model

    def protocol(self, args, assets):
        config = ORIGINAL_PROTOCOL(args, assets)
        config.update(training_variant='research-icd-v1', experiment_method=args.research_variant,
                      development_only=True, target_correction_alpha=0.5,
                      target_variance_floor='0.1 * per_sample_variance_mean + 1e-6',
                      routing_control_indices=[0, 3, 6], dgl_unimodal_scale=1.0,
                      selection_tie_policy='improvement > 1e-5, earlier epoch retained')
        for path in (Path(__file__), Path(research.__file__)):
            config['input_sha256'][str(path.resolve())] = trainer.sha256(path)
        return config

    def components(self, outputs, batch, device, args, assets):
        common, coordinates = ORIGINAL_COMPONENTS(outputs, batch, device, args, assets)
        self.step += 1
        unimodal = None
        if args.research_variant == 'dgl_icd':
            weights = batch['weights'].to(device)
            tasks = [trainer.weighted_task_loss(outputs[s], batch['sentiment'].to(device),
                     batch['classes'].to(device), weights, assets['alpha_ce'])[0] for s in ('t', 'a', 'v')]
            unimodal = torch.stack(tasks).mean()
        self.common, self.coordinates = common, coordinates
        self.encoder_loss = research.encoder_objective(args.research_variant, common, coordinates,
                                                       args.lambda_interaction, unimodal)
        self.fusion_loss = common + args.lambda_interaction * coordinates.mean()
        return common, coordinates

    def backward(self, base_loss, auxiliary_loss, parameters):
        diagnostics = {'routing_variant': self.args.research_variant}
        if self.step == 1 or self.step % 100 == 0:
            params = self.enc + self.fus
            indices = {id(p): i for i, p in enumerate(params)}
            losses = {'common': self.common, 'first': self.coordinates[:3].sum()/7,
                      'higher': self.coordinates[3:].sum()/7}
            grads = {name: torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
                     for name, value in losses.items()}
            for group, group_params in self.groups.items():
                ix = [indices[id(p)] for p in group_params]
                if not ix:
                    continue
                left = [grads['common'][i] for i in ix]
                for order in ('first', 'higher'):
                    right = [grads[order][i] for i in ix]
                    dot, n0, n1 = trainer._gradient_geometry(left, right)
                    diagnostics[f'{group}_{order}_cosine'] = float((dot/(n0*n1).sqrt().clamp_min(1e-12)).detach())
                    diagnostics[f'{group}_{order}_norm'] = float(n1.sqrt().detach())
                    diagnostics[f'{group}_common_norm'] = float(n0.sqrt().detach())
            directory = self.args.output / 'diagnostics'
            directory.mkdir(exist_ok=True)
            trainer.atomic_json({'step': self.step, **diagnostics}, directory/f'step_{self.step:06d}.json')
            del grads
        research.assign_routed_gradients(self.encoder_loss, self.fusion_loss, self.enc, self.fus)
        self.encoder_loss = self.fusion_loss = self.common = self.coordinates = None
        return diagnostics

    def loss(self, model, outputs, batch, device, args, assets, epoch):
        variant = args.research_variant
        if variant == 'late_decay':
            common, coordinates = ORIGINAL_COMPONENTS(outputs, batch, device, args, assets)
            return common + args.lambda_interaction * research.decay_coefficient(epoch, args.epochs) * coordinates.mean()
        if variant not in research.TARGET_VARIANTS:
            raise ValueError(variant)
        if any(row['split'] != 'train' for row in batch['rows']):
            raise ValueError('Target correction may only use training labels')
        weights = batch['weights'].to(device)
        labels = batch['sentiment'].to(device)
        baseline = assets['student_empty_baseline']
        mean = mobius_transform(batch['teacher_subset_scores'].to(device).float(), baseline)
        corrected = research.corrected_coordinates(mean, batch['interaction_variance'].to(device),
                                                    labels, baseline, variant)
        full_target = baseline + corrected.sum(-1)
        task = trainer.weighted_task_loss(outputs['tav'], labels, batch['classes'].to(device),
                                          weights, assets['alpha_ce'])[0]
        full = trainer.regression_kd(outputs['tav'], full_target, weights)
        classification = weighted_mean(probability_kd_per_sample(outputs['tav']['classification_logits'],
            batch['teacher_ensemble_probabilities'].to(device), assets['kd_temperature']), weights)
        scores = torch.stack([outputs[s]['regression'].float() for s in SUBSETS], -1)
        errors = F.smooth_l1_loss(mobius_transform(scores, baseline), corrected, reduction='none').mean(-1)
        return task + full + classification + args.lambda_interaction * weighted_mean(errors, weights)

    def evaluate(self, model, loader, device, timing=None):
        metrics, predictions = ORIGINAL_EVALUATE(model, loader, device, timing)
        self.latest_predictions = predictions
        return metrics, predictions

    def save(self, payload, path):
        ORIGINAL_SAVE(payload, path)
        path = Path(path)
        if path.name not in {'best_valid_mae.pt', 'best_valid_acc2_nonzero.pt'}:
            return
        if self.latest_predictions is None:
            raise RuntimeError('Missing corresponding validation predictions')
        policy = path.stem.removeprefix('best_')
        directory = self.args.output / 'validation' / policy
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / 'predictions.jsonl.tmp'
        temporary.write_text(''.join(json.dumps(row)+'\n' for row in self.latest_predictions))
        temporary.replace(directory / 'predictions.jsonl')
        trainer.atomic_json({'experiment_method': self.args.research_variant, 'seed': self.args.seed,
            'selected_epoch': payload['epoch'], 'checkpoint_selection': policy,
            'metrics': payload['valid_metrics'], 'official_test_evaluated': False,
            'error_strata': error_strata(self.latest_predictions)}, directory / 'report.json')


def main():
    args = parse_args()
    hooks = ResearchHooks(args)
    trainer.build_model = hooks.build
    trainer.run_protocol = hooks.protocol
    trainer.evaluate = hooks.evaluate
    trainer.atomic_save = hooks.save
    if args.research_variant in research.ROUTED_VARIANTS:
        # The original loop's custom-backward branch is reused only in this process.
        trainer.GRADIENT_PROJECTED_METHODS = trainer.GRADIENT_PROJECTED_METHODS | {'uniform_interaction'}
        trainer.uniform_interaction_components = hooks.components
        trainer.projected_auxiliary_backward = hooks.backward
    else:
        trainer.compute_loss = hooks.loss
    if args.dry_run:
        train, valid = trainer.load_rows(args)
        print(json.dumps({'train_windows': len(train), 'valid_windows': len(valid),
                          'protocol': hooks.protocol(args, json.loads(args.assets.read_text()))}))
        return
    with trainer.run_lock(args.output.resolve(), args.resume):
        try:
            trainer.train(args, args.output.resolve())
        except Exception as error:
            trainer.atomic_json({'status': 'failed', 'error': repr(error)}, args.output/'status.json')
            raise


if __name__ == '__main__':
    main()
