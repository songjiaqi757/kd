"""Campaign objectives. All label-dependent correction is restricted to train."""
import json
import torch
from torch.nn import functional as F


def api():
    import sys
    return sys.modules['__main__']


def coordinates(outputs, batch, device, baseline):
    from rdid_mosei.interaction import mobius_transform
    from rdid_mosei.student import SUBSETS
    scores = torch.stack([outputs[s]['regression'].float() for s in SUBSETS], -1)
    return mobius_transform(scores, baseline)


def corrected_target(mean, variance, labels, baseline, variant, alpha):
    if variant not in ('target_uniform','target_variance') or not 0 <= alpha <= 1:
        raise ValueError('Invalid correction configuration')
    mean, variance, labels = mean.detach().float(), variance.detach().float(), labels.detach().float()
    if mean.shape != variance.shape or mean.shape[-1] != 7 or labels.shape != mean.shape[:-1]:
        raise ValueError('Target dimensions differ')
    if not all(torch.isfinite(x).all() for x in (mean,variance,labels)) or (variance < 0).any():
        raise ValueError('Invalid target inputs')
    weights = variance + 1e-6 if variant == 'target_variance' else torch.ones_like(mean)
    weights = weights / weights.sum(-1, keepdim=True)
    return mean + alpha * weights * (labels-baseline-mean.sum(-1)).unsqueeze(-1)


def compatibility(task, losses, model):
    # Fixed, shared parameter subset. No per-sample backward, no encoder gradients.
    params = [p for p in model.fusion.final_norm.parameters() if p.requires_grad]
    if not params:
        raise ValueError('Missing fixed fusion.final_norm parameter subset')
    gt = torch.autograd.grad(task, params, retain_graph=True, allow_unused=True)
    cosines = []
    for loss in losses:
        gc = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        dot = task.new_zeros(())
        nt = task.new_zeros(())
        nc = task.new_zeros(())
        for left,right in zip(gt,gc):
            if left is not None: nt = nt + left.float().square().sum()
            if right is not None: nc = nc + right.float().square().sum()
            if left is not None and right is not None: dot = dot + (left.float()*right.float()).sum()
        cosines.append((dot/(nt*nc).sqrt().clamp_min(1e-12)).clamp(-1,1))
    return torch.stack(cosines).detach()


def compute_loss(model, outputs, batch, device, args, assets, epoch):
    t = api()
    out = outputs['tav'] if args.method in t.SUBSET_METHODS else outputs
    w = batch['weights'].to(device)
    labels = batch['sentiment'].to(device)
    _,reg,ce = t.weighted_task_loss(out, labels, batch['classes'].to(device), w, args.task_ce)
    kdreg = t.regression_kd(out, batch['teacher_subset_scores'].to(device)[:,-1], w)
    probability = batch['teacher_ensemble_probabilities'].to(device).float()
    kl = t.weighted_mean(t.probability_kd_per_sample(out['classification_logits'], probability, assets['kd_temperature']),w)
    loss = reg + args.task_ce*ce + kdreg + args.kd_kl*kl
    raw_coordinates = reg.new_zeros(7)
    extra = {}
    if args.method in ('rld','skd','ea_kd_full'):
        # Logits corresponding exactly to the frozen three-probe mixture at KD temperature.
        teacher_logits = assets['kd_temperature'] * probability.clamp_min(1e-12).log()
        if args.method == 'rld':
            cfg=assets['rld']
            refined=t.rld_per_sample(out['classification_logits'],teacher_logits,batch['classes'].to(device),
                alpha=args.rld_alpha,beta=args.rld_beta,temperature=args.rld_temperature,
                confidence_temperature=cfg['confidence_temperature'],standardize_logits=cfg['standardize_logits'])
            loss=reg+args.task_ce*ce+kdreg+min(epoch/args.rld_warmup_epochs,1)*t.weighted_mean(refined,w)
        elif args.method == 'skd':
            instance,direction,mask=t.skd_losses(out['classification_logits'],teacher_logits,
                temperature=args.skd_temperature,tik_factor=args.skd_tik_factor)
            selected=w*mask.to(w)
            loss=reg+args.task_ce*ce+kdreg+(instance*selected).sum()/selected.sum().clamp_min(1e-8)+t.weighted_mean(direction,w)
        else:
            adaptive=t.entropy_adaptive_weights(out['classification_logits'],teacher_logits,args.ea_entropy_temperature)
            perreg=F.smooth_l1_loss(out['regression'].float(),batch['teacher_subset_scores'].to(device)[:,-1],reduction='none')
            perkl=t.probability_kd_per_sample(out['classification_logits'],probability,assets['kd_temperature'])
            loss=reg+args.task_ce*ce+t.weighted_mean(adaptive*(perreg+perkl),w)
    if args.method in t.SUBSET_METHODS:
        student=coordinates(outputs,batch,device,assets['student_empty_baseline'])
        target=batch['interaction_mean'].to(device).float()
        if args.variant.startswith('target_'):
            if any(row['split'] != 'train' for row in batch['rows']):
                raise ValueError('Label-based correction is restricted to training')
            target=corrected_target(target,batch['interaction_variance'].to(device),labels,
                assets['student_empty_baseline'],args.variant,args.correction_alpha)
        per=F.smooth_l1_loss(student,target,reduction='none')
        raw_coordinates=torch.stack([t.weighted_mean(per[:,k],w) for k in range(7)])
        interaction=raw_coordinates.mean()
        if args.variant in ('compatibility','absorbability'):
            cosines=compatibility(reg,raw_coordinates,model)
            current=raw_coordinates.detach().float()
            old=getattr(args,'_absorbability_state',None)
            previous=torch.tensor(old['loss_ema'],device=device) if old else current
            progress=(previous-current)/previous.clamp_min(1e-6)
            difficulty=current.sqrt()
            def standardize(x): return (x-x.mean())/x.std(unbiased=False).clamp_min(1e-3)
            signal=cosines if args.variant=='compatibility' else cosines+standardize(progress)-standardize(difficulty)
            gates=torch.sigmoid(signal).detach()
            gates=gates/gates.mean().clamp_min(1e-8)
            previous_weights=torch.tensor(old['weight_ema'],device=device) if old else gates
            gates=(0.9*previous_weights+0.1*gates).detach()
            gates=gates/gates.mean().clamp_min(1e-8)
            args._absorbability_state={'loss_ema':(0.9*previous+0.1*current).tolist(),'weight_ema':gates.tolist()}
            interaction=(gates*raw_coordinates).mean()
            extra=dict(compatibility=cosines.tolist(),progress=progress.tolist(),difficulty=difficulty.tolist(),weights=gates.tolist())
        loss=loss+args.lambda_interaction*interaction
    if args.variant=='consistency':
        if out['classification_logits'].shape[-1] != 7:
            raise ValueError('Consistency requires seven classes')
        expected=(F.softmax(out['classification_logits'].float(),-1)*torch.arange(-3,4,device=device)).sum(-1)
        cons=t.weighted_mean(F.smooth_l1_loss(out['regression'].float(),expected,reduction='none'),w)
        loss=loss+args.lambda_consistency*cons
        extra['consistency']=float(cons.detach())
    step=getattr(args,'_diagnostic_step',0)+1
    args._diagnostic_step=step
    if step==1 or step%args.diagnostic_every==0:
        row=dict(epoch=epoch,step=step,task_reg=float(reg.detach()),task_ce=float(ce.detach()),
                 kd_reg=float(kdreg.detach()),kd_kl=float(kl.detach()),coordinates=raw_coordinates.detach().tolist(),total=float(loss.detach()),**extra)
        with (args.output/'loss_diagnostics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    return loss


def fixed_gradient_diagnostics(model,dataset,collator,device,args,assets,epoch,generator):
    t=api()
    from rdid_mosei.tav_distillation import TAVDistillationStudent
    saved=t.rng_state(generator)
    was_training=model.training
    rows=[]
    try:
        t.seed_everything(130013,True)
        model.train()
        for start in range(0,min(24,len(dataset)),8):
            batch=collator([dataset[i] for i in range(start,min(start+8,len(dataset)))])
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                out=TAVDistillationStudent.forward(model,**t.batch_inputs(batch,device),all_subsets=True)
                weights=batch['weights'].to(device)
                task=t.weighted_task_loss(out['tav'],batch['sentiment'].to(device),batch['classes'].to(device),weights,0)[1]
                student=coordinates(out,batch,device,assets['student_empty_baseline'])
                per=F.smooth_l1_loss(student,batch['interaction_mean'].to(device),reduction='none')
                reduced=torch.stack([t.weighted_mean(per[:,k],weights) for k in range(7)])
                cos=compatibility(task,reduced,model)
            rows.append(dict(coordinate_losses=reduced.detach().tolist(),cosines=cos.tolist(),sample_ids=[r['sample_id'] for r in batch['rows']]))
            del out,task,student,per,reduced
        record=dict(epoch=epoch,parameter_subset='fusion.final_norm',task='train SmoothL1 regression only',batches=rows)
        with (args.output/'diagnostics/absorbability.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        t.restore_rng(saved,generator)
