import argparse
import copy
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import util.misc as utils
from datasets import build_dataset
from datasets.data_prefetcher import data_dict_to_cuda
from models import build_model
from models.deformable_detrv2_det import build as build_detect
from train_bft import get_args_parser
from util.tool import load_model, load_torch_checkpoint


def get_parser():
    parser = argparse.ArgumentParser(
        "Encoder-cache parity check",
        parents=[get_args_parser()],
    )
    parser.add_argument("--num_batches", type=int, default=2)
    parser.add_argument("--parity_num_workers", type=int, default=0)
    parser.add_argument("--compare_grads", action="store_true", default=False)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser


def apply_runtime_defaults(args):
    args.distributed = False
    args.rank = 0
    args.world_size = 1
    args.gpu = 0
    args.dist_url = getattr(args, "dist_url", "env://")
    args.dist_backend = getattr(args, "dist_backend", "nccl")
    return args


def load_training_weights(model, args):
    if args.frozen_weights is not None:
        checkpoint = load_torch_checkpoint(args.frozen_weights, map_location="cpu", weights_only=False)
        model.detr.load_state_dict(checkpoint["model"])

    if args.pretrained is not None:
        model = load_model(model, args.pretrained)

    if args.resume:
        checkpoint = load_torch_checkpoint(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


def clone_to_cpu(data: Any) -> Any:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().clone()
    if isinstance(data, dict):
        return {k: clone_to_cpu(v) for k, v in data.items()}
    if isinstance(data, list):
        return [clone_to_cpu(v) for v in data]
    if isinstance(data, tuple):
        return tuple(clone_to_cpu(v) for v in data)
    return copy.deepcopy(data)


def capture_rng_state(device: torch.device) -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any], device: torch.device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])


def collect_grads(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().cpu().clone()
    return grads


def weighted_loss(loss_dict: Dict[str, torch.Tensor], weight_dict: Dict[str, float]) -> torch.Tensor:
    return sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)


def run_branch(model, criterion_detect, batch_cpu, device, score_threshold, lambda_detect, use_cache, compare_grads):
    criterion = model.criterion
    model.train()
    criterion.train()
    criterion_detect.train()
    model.zero_grad(set_to_none=True)

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

    detect_outputs_snapshot = clone_to_cpu(outputs_detector)
    proposals_snapshot = clone_to_cpu(proposals_detector)
    detect_losses_snapshot = clone_to_cpu(loss_dict_detect)

    for inst in data_dict["gt_instances"]:
        if "area" in inst._fields:
            inst._fields.pop("area")

    data_dict["proposals"] = proposals_detector

    if use_cache:
        outputs_track = model.forward_with_encoder_cache(data_dict, encoder_cache)
    else:
        outputs_track = model(data_dict)

    track_outputs_snapshot = clone_to_cpu(outputs_track)
    loss_dict_track = criterion(outputs_track, data_dict)
    weight_dict_track = criterion.weight_dict
    track_losses_snapshot = clone_to_cpu(loss_dict_track)

    total_loss = weighted_loss(loss_dict_track, weight_dict_track) + lambda_detect * weighted_loss(
        loss_dict_detect, weight_dict_detect
    )
    total_loss_value = float(total_loss.detach().cpu())

    grads_snapshot = None
    if compare_grads:
        total_loss.backward()
        grads_snapshot = collect_grads(model)

    return {
        "detect_outputs": detect_outputs_snapshot,
        "proposals": proposals_snapshot,
        "track_outputs": track_outputs_snapshot,
        "detect_losses": detect_losses_snapshot,
        "track_losses": track_losses_snapshot,
        "total_loss": total_loss_value,
        "grads": grads_snapshot,
    }


def compare_tensors(path: str, lhs: torch.Tensor, rhs: torch.Tensor, atol: float, rtol: float):
    if lhs.shape != rhs.shape:
        return {
            "ok": False,
            "path": path,
            "reason": f"shape mismatch {tuple(lhs.shape)} != {tuple(rhs.shape)}",
            "max_abs": float("inf"),
        }
    if lhs.dtype != rhs.dtype:
        return {
            "ok": False,
            "path": path,
            "reason": f"dtype mismatch {lhs.dtype} != {rhs.dtype}",
            "max_abs": float("inf"),
        }

    if lhs.numel() == 0:
        return {"ok": True, "path": path, "reason": "", "max_abs": 0.0}

    if torch.is_floating_point(lhs):
        diff = (lhs - rhs).abs()
        max_abs = float(diff.max().item())
        ok = torch.allclose(lhs, rhs, atol=atol, rtol=rtol)
        reason = "" if ok else f"max_abs={max_abs:.6e}"
        return {"ok": ok, "path": path, "reason": reason, "max_abs": max_abs}

    if lhs.dtype == torch.bool:
        ok = torch.equal(lhs, rhs)
        return {"ok": ok, "path": path, "reason": "" if ok else "bool tensor mismatch", "max_abs": 0.0 if ok else 1.0}

    diff = (lhs.to(torch.int64) - rhs.to(torch.int64)).abs()
    max_abs = float(diff.max().item())
    ok = torch.equal(lhs, rhs)
    return {"ok": ok, "path": path, "reason": "" if ok else f"max_abs={max_abs:.6e}", "max_abs": max_abs}


def compare_nested(path: str, lhs: Any, rhs: Any, atol: float, rtol: float, failures: List[Dict[str, Any]]):
    max_abs = 0.0

    if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
        result = compare_tensors(path, lhs, rhs, atol, rtol)
        max_abs = max(max_abs, result["max_abs"])
        if not result["ok"]:
            failures.append(result)
        return max_abs

    if isinstance(lhs, dict) and isinstance(rhs, dict):
        lhs_keys = sorted(lhs.keys())
        rhs_keys = sorted(rhs.keys())
        if lhs_keys != rhs_keys:
            failures.append({
                "ok": False,
                "path": path,
                "reason": f"dict keys mismatch {lhs_keys} != {rhs_keys}",
                "max_abs": float("inf"),
            })
            return float("inf")
        for key in lhs_keys:
            max_abs = max(max_abs, compare_nested(f"{path}.{key}", lhs[key], rhs[key], atol, rtol, failures))
        return max_abs

    if isinstance(lhs, (list, tuple)) and isinstance(rhs, (list, tuple)):
        if len(lhs) != len(rhs):
            failures.append({
                "ok": False,
                "path": path,
                "reason": f"length mismatch {len(lhs)} != {len(rhs)}",
                "max_abs": float("inf"),
            })
            return float("inf")
        for idx, (left_item, right_item) in enumerate(zip(lhs, rhs)):
            max_abs = max(max_abs, compare_nested(f"{path}[{idx}]", left_item, right_item, atol, rtol, failures))
        return max_abs

    if isinstance(lhs, float) and isinstance(rhs, float):
        max_abs = abs(lhs - rhs)
        if not np.isclose(lhs, rhs, atol=atol, rtol=rtol):
            failures.append({
                "ok": False,
                "path": path,
                "reason": f"scalar mismatch {lhs:.6e} != {rhs:.6e}",
                "max_abs": max_abs,
            })
        return max_abs

    if lhs != rhs:
        failures.append({
            "ok": False,
            "path": path,
            "reason": f"value mismatch {lhs!r} != {rhs!r}",
            "max_abs": float("inf"),
        })
        return float("inf")

    return max_abs


def compare_section(name: str, lhs: Any, rhs: Any, atol: float, rtol: float):
    failures: List[Dict[str, Any]] = []
    max_abs = compare_nested(name, lhs, rhs, atol, rtol, failures)
    return {
        "name": name,
        "ok": len(failures) == 0,
        "max_abs": max_abs,
        "failures": failures[:10],
    }


def print_section_report(report: Dict[str, Any]):
    status = "PASS" if report["ok"] else "FAIL"
    print(f"  {report['name']}: {status} (max_abs={report['max_abs']:.6e})")
    for failure in report["failures"]:
        print(f"    {failure['path']}: {failure['reason']}")


def build_models(args, device):
    base_model, _, _ = build_model(args)
    base_model = load_training_weights(base_model, args)

    model_old = base_model
    model_new = copy.deepcopy(base_model)

    model_old.to(device)
    model_new.to(device)

    _, criterion_detect_old, _ = build_detect(args)
    criterion_detect_new = copy.deepcopy(criterion_detect_old)
    criterion_detect_old.to(device)
    criterion_detect_new.to(device)

    return model_old, model_new, criterion_detect_old, criterion_detect_new


def build_loader(args):
    dataset_train = build_dataset(image_set="train", args=args)
    sampler = torch.utils.data.SequentialSampler(dataset_train)
    batch_sampler = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=True)
    return DataLoader(
        dataset_train,
        batch_sampler=batch_sampler,
        collate_fn=utils.mot_collate_fn,
        num_workers=args.parity_num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main():
    parser = get_parser()
    args = parser.parse_args()
    args = apply_runtime_defaults(args)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_old, model_new, criterion_detect_old, criterion_detect_new = build_models(args, device)
    data_loader = build_loader(args)

    overall_ok = True
    processed = 0

    for batch_idx, batch_cpu in enumerate(data_loader):
        if batch_idx >= args.num_batches:
            break

        processed += 1
        print(f"Batch {batch_idx}")
        rng_state = capture_rng_state(device)

        restore_rng_state(rng_state, device)
        old_result = run_branch(
            model_old, criterion_detect_old, batch_cpu, device,
            args.score_threshold, args.lambda_detect, False, args.compare_grads
        )

        restore_rng_state(rng_state, device)
        new_result = run_branch(
            model_new, criterion_detect_new, batch_cpu, device,
            args.score_threshold, args.lambda_detect, True, args.compare_grads
        )

        reports = [
            compare_section("detect_outputs", old_result["detect_outputs"], new_result["detect_outputs"], args.atol, args.rtol),
            compare_section("proposals", old_result["proposals"], new_result["proposals"], args.atol, args.rtol),
            compare_section("track_outputs", old_result["track_outputs"], new_result["track_outputs"], args.atol, args.rtol),
            compare_section("detect_losses", old_result["detect_losses"], new_result["detect_losses"], args.atol, args.rtol),
            compare_section("track_losses", old_result["track_losses"], new_result["track_losses"], args.atol, args.rtol),
            compare_section("total_loss", old_result["total_loss"], new_result["total_loss"], args.atol, args.rtol),
        ]

        if args.compare_grads:
            reports.append(compare_section("grads", old_result["grads"], new_result["grads"], args.atol, args.rtol))

        for report in reports:
            print_section_report(report)
            overall_ok = overall_ok and report["ok"]

    if processed == 0:
        raise RuntimeError("No batches were processed. Check dataset configuration or batch size.")

    if not overall_ok:
        raise SystemExit(1)

    print(f"Parity check passed for {processed} batch(es).")


if __name__ == "__main__":
    main()
