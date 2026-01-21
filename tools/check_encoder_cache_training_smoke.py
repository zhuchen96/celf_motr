import argparse
import copy
import sys
from pathlib import Path
from typing import Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.data_prefetcher import data_dict_to_cuda
from tools.check_encoder_cache_parity import (
    apply_runtime_defaults,
    build_loader,
    build_models,
    capture_rng_state,
    compare_section,
    print_section_report,
    restore_rng_state,
)
from train_bft import get_args_parser


def get_parser():
    parser = argparse.ArgumentParser(
        "Encoder-cache training smoke",
        parents=[get_args_parser()],
    )
    parser.add_argument("--num_update_steps", type=int, default=3)
    parser.add_argument("--parity_num_workers", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser


def match_name_keywords(name, name_keywords):
    return any(keyword in name for keyword in name_keywords)


def build_optimizer(model, args):
    param_dicts = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and p.requires_grad
            ],
            "lr": args.lr,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad
            ],
            "lr": args.lr * args.lr_linear_proj_mult,
        },
    ]

    if args.sgd:
        return torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    return torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)


def weighted_loss(loss_dict: Dict[str, torch.Tensor], weight_dict: Dict[str, float]) -> torch.Tensor:
    return sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)


def run_training_microbatch(model, criterion_detect, batch_cpu, device, score_threshold, lambda_detect, use_cache, accum_iter):
    criterion = model.criterion
    model.train()
    criterion.train()
    criterion_detect.train()

    data_dict = data_dict_to_cuda(copy.deepcopy(batch_cpu), device)

    if use_cache:
        outputs_detector, proposals_detector, encoder_cache = model.forward_detect_self_light(
            data_dict, score_threshold=score_threshold
        )
    else:
        outputs_detector, proposals_detector = model.forward_detect_self(
            data_dict, score_threshold=score_threshold
        )
        encoder_cache = None

    targets = data_dict["gt_instances"]
    loss_dict_detect = criterion_detect.forward_detect(outputs_detector, targets)
    weight_dict_detect = criterion_detect.weight_dict

    for inst in data_dict["gt_instances"]:
        if "area" in inst._fields:
            inst._fields.pop("area")

    data_dict["proposals"] = proposals_detector

    if use_cache:
        outputs_track = model.forward_with_encoder_cache(data_dict, encoder_cache)
    else:
        outputs_track = model(data_dict)

    loss_dict_track = criterion(outputs_track, data_dict)
    weight_dict_track = criterion.weight_dict

    loss_track_raw = weighted_loss(loss_dict_track, weight_dict_track)
    loss_detect_raw = lambda_detect * weighted_loss(loss_dict_detect, weight_dict_detect)
    total_raw = loss_track_raw + loss_detect_raw

    (total_raw / accum_iter).backward()

    return {
        "loss_track_raw": float(loss_track_raw.detach().cpu()),
        "loss_detect_raw": float(loss_detect_raw.detach().cpu()),
        "loss_total_raw": float(total_raw.detach().cpu()),
    }


def collect_named_parameters(model):
    return {name: param.detach().cpu().clone() for name, param in model.named_parameters()}


def main():
    parser = get_parser()
    args = parser.parse_args()
    args = apply_runtime_defaults(args)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")

    total_microbatches = args.num_update_steps * args.accum_iter

    model_old, model_new, criterion_detect_old, criterion_detect_new = build_models(args, device)
    optimizer_old = build_optimizer(model_old, args)
    optimizer_new = build_optimizer(model_new, args)

    data_loader = build_loader(args)
    batch_iter = iter(data_loader)
    batches = []
    for _ in range(total_microbatches):
        try:
            batches.append(next(batch_iter))
        except StopIteration as exc:
            raise RuntimeError(
                f"Requested {total_microbatches} microbatches but data loader ran out early"
            ) from exc

    overall_ok = True

    for step_idx in range(args.num_update_steps):
        print(f"Update step {step_idx}")
        optimizer_old.zero_grad()
        optimizer_new.zero_grad()

        for accum_idx in range(args.accum_iter):
            batch_cpu = batches[step_idx * args.accum_iter + accum_idx]
            rng_state = capture_rng_state(device)

            restore_rng_state(rng_state, device)
            old_metrics = run_training_microbatch(
                model_old, criterion_detect_old, batch_cpu, device,
                args.score_threshold, args.lambda_detect, False, args.accum_iter
            )

            restore_rng_state(rng_state, device)
            new_metrics = run_training_microbatch(
                model_new, criterion_detect_new, batch_cpu, device,
                args.score_threshold, args.lambda_detect, True, args.accum_iter
            )

            for report in (
                compare_section(f"microbatch_{accum_idx}.loss_track_raw", old_metrics["loss_track_raw"], new_metrics["loss_track_raw"], args.atol, args.rtol),
                compare_section(f"microbatch_{accum_idx}.loss_detect_raw", old_metrics["loss_detect_raw"], new_metrics["loss_detect_raw"], args.atol, args.rtol),
                compare_section(f"microbatch_{accum_idx}.loss_total_raw", old_metrics["loss_total_raw"], new_metrics["loss_total_raw"], args.atol, args.rtol),
            ):
                print_section_report(report)
                overall_ok = overall_ok and report["ok"]

        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model_old.parameters(), args.clip_max_norm)
            torch.nn.utils.clip_grad_norm_(model_new.parameters(), args.clip_max_norm)

        optimizer_old.step()
        optimizer_new.step()

        param_report = compare_section(
            f"update_{step_idx}.parameters",
            collect_named_parameters(model_old),
            collect_named_parameters(model_new),
            args.atol,
            args.rtol,
        )
        print_section_report(param_report)
        overall_ok = overall_ok and param_report["ok"]

    if not overall_ok:
        raise SystemExit(1)

    print(f"Training smoke passed for {args.num_update_steps} optimizer step(s).")


if __name__ == "__main__":
    main()
