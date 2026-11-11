"""Frozen MOSI controls: identical ensemble base loss, explicit error transforms."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from rdid_mosei.interaction import mobius_matrix, random_orthogonal_matrix
from rdid_mosei.main_table_kd import probability_kd_per_sample, weighted_mean

BASE = Path(__file__).resolve().parent
NAMES = ('t', 'a', 'v', 'ta', 'tv', 'av', 'tav')
MECHANISMS = ('none', 'icd', 'subset', 'orthogonal', 'matched', 'diagonal',
              'weighted', 'normalized_icd', 'normalized_subset')

def transform(args, dtype=torch.float64, device='cpu'):
    m = mobius_matrix(dtype=torch.float64)
    name = args.mechanism
    if name == 'icd':
        matrix = m
    elif name == 'matched':
        sigma = torch.linalg.svdvals(m)
        q1 = random_orthogonal_matrix(seed=args.matrix_seed)
        q2 = random_orthogonal_matrix(seed=args.matrix_seed + 1000003)
        matrix = q1 @ torch.diag(sigma) @ q2.T
    elif name == 'orthogonal':
        matrix = random_orthogonal_matrix(seed=args.matrix_seed)
    elif name in ('diagonal', 'weighted'):
        weights = (m.T @ m).diag()
        if name == 'weighted':
            weights = weights / weights.mean()
        matrix = torch.diag(weights.sqrt())
    elif name in ('normalized_icd', 'normalized_subset'):
        stats = json.loads((BASE / 'assets/normalization.json').read_text())
        key = 'coordinate_std' if name == 'normalized_icd' else 'subset_std'
        std = torch.tensor(stats[key], dtype=torch.float64).clamp_min(stats['epsilon'])
        matrix = torch.diag(1 / std) @ (m if name == 'normalized_icd' else torch.eye(7, dtype=torch.float64))
    else:
        matrix = torch.eye(7, dtype=torch.float64)
    return matrix.to(device=device, dtype=dtype)

def transform_metadata(args):
    a = transform(args)
    return dict(mechanism=args.mechanism, loss_kind=args.loss_kind, matrix_seed=args.matrix_seed,
                subset_order=NAMES, matrix=a.tolist(), sha256=hashlib.sha256(a.numpy().tobytes()).hexdigest(),
                singular_values=torch.linalg.svdvals(a).tolist(), condition_number=float(torch.linalg.cond(a)),
                normalization_split='train_utterances_only', mse_definition='0.5 * mean((A @ error)^2)',
                huber_definition='mean(SmoothL1(A @ error, beta=1))')

def terms(outputs, batch, device, args, assets):
    output = outputs['tav'] if 'tav' in outputs else outputs
    weights = batch['weights'].to(device)
    reg = weighted_mean(F.smooth_l1_loss(output['regression'].float(), batch['sentiment'].to(device), reduction='none'), weights)
    ce = weighted_mean(F.cross_entropy(output['classification_logits'].float(), batch['classes'].to(device), reduction='none'), weights)
    teacher = batch['teacher_subset_scores'].to(device)
    kd = weighted_mean(F.smooth_l1_loss(output['regression'].float(), teacher[:, -1], reduction='none'), weights)
    kl = weighted_mean(probability_kd_per_sample(output['classification_logits'], batch['teacher_ensemble_probabilities'].to(device), assets['kd_temperature']), weights)
    if 'tav' in outputs:
        student = torch.stack([outputs[s]['regression'].float() for s in NAMES], -1)
        error = (student - teacher.float()) @ transform(args, torch.float32, device).T
        losses = 0.5 * error.square() if args.loss_kind == 'mse' else F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none')
        coordinates = (losses * weights[:, None]).sum(0) / weights.sum().clamp_min(1e-8)
    else:
        error = reg.new_zeros((len(weights), 7))
        coordinates = reg.new_zeros(7)
    return reg, ce, kd, kl, coordinates, error

def compute_loss(model, outputs, batch, device, args, assets, epoch):
    reg, ce, kd, kl, coords, error = terms(outputs, batch, device, args, assets)
    base = reg + args.task_ce * ce + kd + args.kd_kl * kl
    loss = base + args.lambda_interaction * coords.mean()
    step = getattr(args, '_diagnostic_step', 0) + 1
    args._diagnostic_step = step
    if step == 1 or step % args.diagnostic_every == 0:
        row = dict(epoch=epoch, step=step, base_loss=float(base.detach()), auxiliary=float(coords.mean().detach()),
                   total=float(loss.detach()), coordinate_losses=coords.detach().cpu().tolist(),
                   transformed_error_abs_quantiles=torch.quantile(error.detach().abs().float(), torch.tensor([.25,.5,.75,.95], device=device)).cpu().tolist(),
                   quadratic_region_fraction=(error.detach().abs() < 1).float().mean(0).cpu().tolist())
        with (args.output / 'loss_diagnostics.jsonl').open('a') as h:
            h.write(json.dumps(row) + '\n')
    return loss

def fixed_gradient_diagnostics(model, dataset, collator, device, args, assets, epoch, generator):
    from train_tav_distillation import rng_state, restore_rng, batch_inputs
    from train_student_baseline import seed_everything
    from rdid_mosei.tav_distillation import TAVDistillationStudent
    saved = rng_state(generator)
    was_training = model.training
    try:
        seed_everything(130013, True)
        model.train()
        batch = collator([dataset[i] for i in range(min(8, len(dataset)))])
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            outputs = TAVDistillationStudent.forward(model, **batch_inputs(batch, device), all_subsets=True)
            reg, ce, kd, kl, coords, error = terms(outputs, batch, device, args, assets)
            base = reg + args.task_ce * ce + kd + args.kd_kl * kl
            aux = coords.mean() * args.lambda_interaction
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        parameters = [p for _, p in named]
        gb = torch.autograd.grad(base, parameters, retain_graph=True, allow_unused=True)
        ga = torch.autograd.grad(aux, parameters, retain_graph=True, allow_unused=True)
        scopes = dict(all=range(len(named)), fusion=[i for i,(n,p) in enumerate(named) if n.startswith('fusion.')],
                      encoder_lora=[i for i,(n,p) in enumerate(named) if 'lora_' in n])
        geometry = {}
        for scope, indices in scopes.items():
            dot = base.new_zeros((), dtype=torch.float64)
            nb, na = dot.clone(), dot.clone()
            for i in indices:
                if gb[i] is not None: nb += gb[i].float().square().sum().double()
                if ga[i] is not None: na += ga[i].float().square().sum().double()
                if gb[i] is not None and ga[i] is not None: dot += (gb[i].float() * ga[i].float()).sum().double()
            geometry[scope] = dict(base_norm=float(nb.sqrt()), auxiliary_norm=float(na.sqrt()),
                                   norm_ratio=float(na.sqrt()/nb.sqrt().clamp_min(1e-12)),
                                   cosine=float(dot/(nb*na).sqrt().clamp_min(1e-12)))
        # Per-coordinate compatibility is diagnostic only; never adjusts losses.
        selected = [i for i,(n,p) in enumerate(named) if 'fusion.final_norm' in n]
        coordinate_cosines = []
        for term in coords:
            gs = torch.autograd.grad(term, [parameters[i] for i in selected], retain_graph=True, allow_unused=True)
            dot = sum((gb[i].float()*g.float()).sum() for i,g in zip(selected,gs) if g is not None and gb[i] is not None)
            nb = sum(gb[i].float().square().sum() for i in selected if gb[i] is not None)
            ng = sum(g.float().square().sum() for g in gs if g is not None)
            coordinate_cosines.append(float(dot/(nb*ng).sqrt().clamp_min(1e-12)))
        row = dict(epoch=epoch, diagnostic_seed=130013, sample_ids=[r['sample_id'] for r in batch['rows']],
                   gradient_geometry=geometry, coordinate_cosines_fusion_final_norm=coordinate_cosines,
                   coordinate_losses=coords.detach().cpu().tolist(),
                   quadratic_region_fraction=(error.detach().abs()<1).float().mean(0).cpu().tolist())
        with (args.output / 'diagnostics/gradient_geometry.jsonl').open('a') as h:
            h.write(json.dumps(row)+'\n')
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        restore_rng(saved, generator)
