#!/usr/bin/env python3
"""Isolated decision-ICD extension of the unchanged running main-table trainer.

Like train_main_table_all_epochs.py, this installs process-local trainer hooks.
Existing experiments and their source fingerprints are not modified. The model
is always uniform_interaction; decision_mode names the training-loss ablation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "project/src"), str(ROOT / "project/scripts")]
import torch
import train_main_table_kd as trainer
from rdid_mosei.decision_icd import DECISION_MODES, decision_distillation_loss, error_strata
import rdid_mosei.decision_icd as decision_module

ORIGINAL_LOSS = trainer.compute_loss
ORIGINAL_PROTOCOL = trainer.run_protocol
ORIGINAL_EVALUATE = trainer.evaluate
ORIGINAL_SAVE = trainer.atomic_save


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--decision-mode", choices=DECISION_MODES, default="none")
    parser.add_argument("--decision-temperature", type=float, default=1.0)
    parser.add_argument("--lambda-decision", type=float, default=1.0)
    parser.add_argument("--gradient-diagnostic-every", type=int, default=100)
    extra, remaining = parser.parse_known_args()
    if any(not math.isfinite(v) or v <= 0 for v in (extra.decision_temperature, extra.lambda_decision)):
        parser.error("decision temperature and weight must be finite and positive")
    if extra.gradient_diagnostic_every < 1:
        parser.error("diagnostic interval must be positive")
    previous = sys.argv
    try:
        sys.argv = [previous[0], *remaining]
        args = trainer.parse_args()
    finally:
        sys.argv = previous
    if args.method != "uniform_interaction":
        parser.error("decision controls require --method uniform_interaction")
    if not args.retain_best_mae_and_acc2 or args.save_every_epoch:
        parser.error("require --retain-best-mae-and-acc2 without --save-every-epoch")
    if args.min_epochs != args.epochs:
        parser.error("use a fixed common epoch budget: --min-epochs must equal --epochs")
    vars(args).update(vars(extra))
    return args


class DecisionHooks:
    def __init__(self, args):
        self.args = args
        self.output = args.output.resolve()
        self.epoch = 0
        self.step = 0
        self.samples = []
        self.latest_predictions = None

    def protocol(self, args, assets):
        config = ORIGINAL_PROTOCOL(args, assets)
        config.update(
            training_variant="decision-icd-v1",
            experiment_method=f"icd_decision_{args.decision_mode}",
            decision_teacher_target="sigmoid(ensemble_mean_subset_score / tau)",
            decision_empty_baseline="sigmoid(shared_student_empty_baseline / tau)",
            decision_distance="mean smooth_l1 over selected coordinates; no per-row rescaling",
            selection_tie_policy="first improvement exceeding 1e-5, matching legacy trainer",
            development_only=True,
        )
        for path in (Path(__file__), Path(decision_module.__file__),
                     Path(assets["ensemble_targets"]), Path(assets["interaction_targets"])):
            config["input_sha256"][str(path.resolve())] = trainer.sha256(path)
        return config

    def loss(self, model, outputs, batch, device, args, assets, epoch):
        if epoch != self.epoch:
            self.epoch, self.step, self.samples = epoch, 0, []
        self.step += 1
        base = ORIGINAL_LOSS(model, outputs, batch, device, args, assets, epoch)
        decision = decision_distillation_loss(
            outputs, batch["teacher_subset_scores"].to(device), batch["weights"].to(device),
            assets["student_empty_baseline"], mode=args.decision_mode,
            temperature=args.decision_temperature,
        )
        if self.step == 1 or self.step % args.gradient_diagnostic_every == 0:
            weights = batch["weights"].to(device)
            task, _, _ = trainer.weighted_task_loss(
                outputs["tav"], batch["sentiment"].to(device), batch["classes"].to(device),
                weights, assets["alpha_ce"],
            )
            common, coordinates = trainer.uniform_interaction_components(outputs, batch, device, args, assets)
            losses = {"task": task, "task_full_kd": common,
                      "interaction": args.lambda_interaction * coordinates.mean()}
            if args.decision_mode != "none":
                losses["decision"] = args.lambda_decision * decision
            parameters = [p for p in model.fusion.parameters() if p.requires_grad]
            grads = {name: torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
                     for name, value in losses.items()}
            sample = {"step": self.step, "losses": {k: float(v.detach()) for k, v in losses.items()}}
            for left in ("task", "task_full_kd"):
                for right in ("interaction", "decision"):
                    if right not in grads:
                        continue
                    dot, a, b = trainer._gradient_geometry(grads[left], grads[right])
                    sample[f"{left}_vs_{right}_cosine"] = float((dot / (a.sqrt()*b.sqrt()).clamp_min(1e-12)).detach())
            self.samples.append(sample)
        return base + args.lambda_decision * decision

    def evaluate(self, model, loader, device, timing=None):
        metrics, predictions = ORIGINAL_EVALUATE(model, loader, device, timing)
        self.latest_predictions = predictions
        if self.samples:
            directory = self.output / "diagnostics"
            directory.mkdir(parents=True, exist_ok=True)
            cosine_keys = [k for k in self.samples[0] if k.endswith("_cosine")]
            summary = {k: {
                "mean": sum(r[k] for r in self.samples)/len(self.samples),
                "conflict_fraction": sum(r[k] < 0 for r in self.samples)/len(self.samples),
            } for k in cosine_keys}
            trainer.atomic_json({"epoch": self.epoch, "gradient_scope": "fusion parameters",
                                 "gradient_batches": len(self.samples), "summary": summary,
                                 "samples": self.samples}, directory / f"epoch_{self.epoch:03d}.json")
            self.samples = []
        return metrics, predictions

    def save(self, payload, path):
        ORIGINAL_SAVE(payload, path)
        path = Path(path)
        if path.name not in {"best_valid_mae.pt", "best_valid_acc2_nonzero.pt"}:
            return
        if self.latest_predictions is None:
            raise RuntimeError("best checkpoint has no matching validation predictions")
        policy = path.stem.removeprefix("best_")
        destination = self.output / "validation" / policy
        destination.mkdir(parents=True, exist_ok=True)
        temporary = destination / "predictions.jsonl.tmp"
        temporary.write_text("".join(json.dumps(row) + "\n" for row in self.latest_predictions))
        os.replace(temporary, destination / "predictions.jsonl")
        trainer.atomic_json({
            "experiment_method": f"icd_decision_{self.args.decision_mode}",
            "seed": self.args.seed, "selected_epoch": payload["epoch"],
            "checkpoint_selection": policy, "metrics": payload["valid_metrics"],
            "official_test_evaluated": False,
            "error_strata": error_strata(self.latest_predictions),
        }, destination / "report.json")


def main():
    args = parse_args()
    hooks = DecisionHooks(args)
    trainer.compute_loss = hooks.loss
    trainer.run_protocol = hooks.protocol
    trainer.evaluate = hooks.evaluate
    trainer.atomic_save = hooks.save
    if args.dry_run:
        train, valid = trainer.load_rows(args)
        assets = json.loads(args.assets.read_text())
        print(json.dumps({"train_windows": len(train), "valid_windows": len(valid),
                          "protocol": hooks.protocol(args, assets)}, indent=2))
        return
    with trainer.run_lock(args.output.resolve(), args.resume):
        try:
            trainer.train(args, args.output.resolve())
        except Exception as error:
            trainer.atomic_json({"status": "failed", "error": repr(error)}, args.output / "status.json")
            raise


if __name__ == "__main__":
    main()
