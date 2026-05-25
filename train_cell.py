"""
Training script for SelfMOTR on CTC cell-tracking data (Cell-TRACTR layout).

Minimal usage (single GPU):
    python train_cell.py \
        --mot_path /srv/home/chen/Cell-TRACTR/data/moma/COCO \
        --output_dir outputs/cell_test \
        --num_queries 50 \
        --num_queries_detect 100 \
        --epochs 20 \
        --batch_size 1 \
        --sampler_lengths 2 \
        --sample_mode fixed_interval \
        --sample_interval 1

Multi-GPU (e.g. 2 GPUs):
    torchrun --nproc_per_node=2 train_cell.py [same args above]

Point --mot_path at the COCO subfolder produced by
Cell-TRACTR/scripts/create_coco_dataset_from_CTC.py, i.e. the folder that
contains  annotations/train/anno.json  and  train/img/*.tif.
"""

import argparse
import csv
import datetime
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

from util.tool import load_model, load_torch_checkpoint
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset
from enginev2 import train_one_epoch_mot_self_proposal, eval_one_epoch_mot_self_proposal
from models import build_model
from models.deformable_detrv2_det import build as build_detect

import copy


def _run_val_inference(args, checkpoint_path: Path, epoch: int) -> dict:
    """Run eval_cell → diag_coverage → eval_division on val split.

    Returns a dict of float metrics (NaN on failure).
    Saves full stdout to <output_dir>/val_infer/epoch_XXXX.txt.
    """
    script_dir   = Path(__file__).parent
    eval_out_dir = Path(args.output_dir) / 'val_infer' / f'epoch_{epoch:04d}'
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    log_path     = eval_out_dir.parent / f'epoch_{epoch:04d}.txt'
    ctc_val_dir  = Path(args.mot_path).parent / 'CTC' / 'val'

    nan = float('nan')
    metrics = {
        'val_full_cov_pct': nan, 'val_mean_cov_pct': nan,
        'val_d1_detect_pct': nan, 'val_d2_detect_pct': nan,
        'val_both_linked_pct': nan,
    }

    def _run(cmd):
        return subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(script_dir))

    stdout_all = ''

    # 1. Inference
    cmd_eval = [sys.executable, str(script_dir / 'eval_cell.py'),
                '--config', args.infer_config,
                '--resume', str(checkpoint_path),
                '--output_dir', str(eval_out_dir),
                '--split', 'val']
    r = _run(cmd_eval)
    stdout_all += r.stdout + r.stderr
    if r.returncode != 0:
        print(f'  [val-infer] eval_cell.py failed (epoch {epoch})')
        log_path.write_text(stdout_all)
        return metrics

    # 2. Coverage
    cmd_cov = [sys.executable, str(script_dir / 'diag_coverage.py'),
               '--gt_dir', str(ctc_val_dir),
               '--res_dir', str(eval_out_dir)]
    r = _run(cmd_cov)
    stdout_all += r.stdout + r.stderr

    # 3. Division
    cmd_div = [sys.executable, str(script_dir / 'eval_division.py'),
               '--gt_dir', str(ctc_val_dir),
               '--res_dir', str(eval_out_dir)]
    r = _run(cmd_div)
    stdout_all += r.stdout + r.stderr

    log_path.write_text(stdout_all)

    # Parse key numbers from combined output
    def _pct(pattern):
        m = re.search(pattern, stdout_all)
        return float(m.group(1)) if m else nan

    metrics['val_full_cov_pct']    = _pct(r'Fully covered.*?\(\s*([\d.]+)%\)')
    metrics['val_mean_cov_pct']    = _pct(r'Mean\s+coverage per track\s*:\s*([\d.]+)')
    metrics['val_d1_detect_pct']   = _pct(r'D1 detected\s*:.*?\(([\d.]+)%\)')
    metrics['val_d2_detect_pct']   = _pct(r'D2 detected\s*:.*?\(([\d.]+)%\)')
    metrics['val_both_linked_pct'] = _pct(r'Both linked\s*:.*?\(([\d.]+)%\)')

    print(f'  [val-infer] epoch {epoch}: '
          f'full_cov={metrics["val_full_cov_pct"]:.1f}% '
          f'mean_cov={metrics["val_mean_cov_pct"]:.1f}% '
          f'D1={metrics["val_d1_detect_pct"]:.1f}% '
          f'D2={metrics["val_d2_detect_pct"]:.1f}% '
          f'both_linked={metrics["val_both_linked_pct"]:.1f}%')
    return metrics


def get_args_parser():
    parser = argparse.ArgumentParser('SelfMOTR — cell tracking', add_help=False)

    # Optimiser
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names',
                        default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr_drop', default=15, type=int)
    parser.add_argument('--save_period', default=5, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float)

    # Architecture
    parser.add_argument('--meta_arch', default='motrv2_self', type=str)
    parser.add_argument('--with_box_refine', default=True, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--shared_decoder', default=True, action='store_true')
    parser.add_argument('--accurate_ratio', default=False, action='store_true')
    parser.add_argument('--frozen_weights', type=str, default=None)
    parser.add_argument('--num_anchors', default=1, type=int)
    parser.add_argument('--sgd', action='store_true')

    # Backbone
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--enable_fpn', action='store_true')
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str,
                        choices=('sine', 'learned'))
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float)
    parser.add_argument('--num_feature_levels', default=4, type=int)

    # Transformer
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    # Track queries: number of objects the model can track simultaneously.
    # Set to ~max expected cells per frame (e.g. 30 for moma, 200 for deepcell).
    parser.add_argument('--num_queries', default=50, type=int,
                        help='Max simultaneous track queries (≈ max cells per frame)')
    # Detection queries: produces proposals fed to the tracker.
    parser.add_argument('--num_queries_detect', default=100, type=int,
                        help='Detection query count for self-generating proposals')
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--decoder_cross_self', default=False, action='store_true')
    parser.add_argument('--sigmoid_attn', default=False, action='store_true')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--cj', action='store_true')
    parser.add_argument('--extra_track_attn', action='store_true')
    parser.add_argument('--loss_normalizer', action='store_true')
    parser.add_argument('--max_size', default=1333, type=int)
    parser.add_argument('--val_width', default=800, type=int)
    parser.add_argument('--filter_ignore', action='store_true')
    parser.add_argument('--append_crowd', default=False, action='store_true')

    # Segmentation (unused for minimal testing)
    parser.add_argument('--masks', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')
    parser.add_argument('--mix_match', action='store_true')
    parser.add_argument('--set_cost_class', default=2, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # Dataset — cell-specific defaults
    parser.add_argument('--dataset_file', default='e2e_cell',
                        help='Use e2e_cell to load Cell-TRACTR CTC/COCO layout')
    parser.add_argument('--gt_file_train', type=str)
    parser.add_argument('--gt_file_val', type=str)
    # --mot_path is reused to point at the COCO root (contains annotations/ + train/ + val/)
    parser.add_argument('--mot_path', default='', type=str,
                        help='Path to Cell-TRACTR COCO root, e.g. data/moma/COCO')
    parser.add_argument('--coco_path', default='', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--det_db', default='', type=str)
    parser.add_argument('--data_txt_path_train', default='', type=str)
    parser.add_argument('--data_txt_path_val', default='', type=str)
    parser.add_argument('--img_path', default='')

    # Sampling / curriculum
    # --sampler_lengths: number of consecutive frames per training sample.
    #   Start with 2 for quick testing, increase to 4+ for real training.
    parser.add_argument('--sample_mode', type=str, default='fixed_interval')
    parser.add_argument('--sample_interval', type=int, default=1,
                        help='Frame stride when sampling clips (1 = consecutive frames)')
    parser.add_argument('--sampler_steps', type=int, nargs='*', default=None)
    parser.add_argument('--sampler_lengths', type=int, nargs='*', default=[2],
                        help='Clip length(s) in frames. Single value for fixed length.')

    # Query interaction
    parser.add_argument('--query_interaction_layer', default='QIMv2', type=str)
    parser.add_argument('--random_drop', type=float, default=0.1)
    parser.add_argument('--fp_ratio', type=float, default=0.1)
    parser.add_argument('--merger_dropout', type=float, default=0.1)
    parser.add_argument('--update_query_pos', action='store_true')

    # Memory bank (disabled by default for quick testing)
    parser.add_argument('--memory_bank_score_thresh', type=float, default=0.)
    parser.add_argument('--memory_bank_len', type=int, default=4)
    parser.add_argument('--memory_bank_type', type=str, default=None)
    parser.add_argument('--memory_bank_with_self_attn', action='store_true', default=False)

    # Training utilities
    parser.add_argument('--output_dir', default='outputs/cell_test')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pretrained', default=None)
    parser.add_argument('--cache_mode', default=False, action='store_true')
    parser.add_argument('--use_checkpoint', action='store_true', default=False)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--reuse_encoder_cache', action='store_true', default=False)
    parser.add_argument('--query_denoise', type=float, default=0.)
    parser.add_argument('--score_threshold', type=float, default=0.5)
    parser.add_argument('--lambda_detect', type=float, default=0.5)
    parser.add_argument('--exp_name', default='cell_test', type=str)
    # Division-ahead prediction loss coefficient.
    # Set > 0 to enable; requires man_track/<split>/<seq>.txt files in --mot_path.
    # Typical value: 1.0–2.0 (same scale as cls/bbox losses).
    parser.add_argument('--div_loss_coef', type=float, default=1.0,
                        help='Weight for the division-ahead BCE loss (0 to disable)')

    # Division-clip oversampling.
    # Fraction of each epoch's clips that are guaranteed to contain at least one
    # dividing cell.  0.0 = uniform sampling (default).  0.3 = 30% of clips
    # are drawn from the division-containing pool (with replacement if the pool
    # is smaller than needed).
    parser.add_argument('--div_ratio', type=float, default=0.0,
                        help='Fraction of training clips containing a division event (0–1)')

    # Periodic val inference
    parser.add_argument('--eval_period', type=int, default=0,
                        help='Run full val inference every N epochs (0 = disabled)')
    parser.add_argument('--infer_config', type=str, default=None,
                        help='Path to infer.yaml used for periodic val inference')

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is for segmentation only"

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    _, criterion_detect, _ = build_detect(args)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_parameters:,}')

    dataset_train = build_dataset(image_set='train', args=args)

    if args.distributed:
        sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train,
        collate_fn=utils.mot_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True)

    # Val dataloader — short fixed clips (length 2), no augmentation, no curriculum.
    # Gracefully disabled if annotations/val/anno.json is missing.
    data_loader_val = None
    try:
        args_val = copy.copy(args)
        args_val.sampler_lengths = [2]
        args_val.sampler_steps   = None
        args_val.random_drop     = 0.0
        args_val.fp_ratio        = 0.0
        args_val.div_ratio       = 0.0
        dataset_val = build_dataset(image_set='val', args=args_val)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            sampler=sampler_val,
            collate_fn=utils.mot_collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False)
        print(f'Val dataloader: {len(dataset_val)} clips')
    except (AssertionError, FileNotFoundError) as e:
        print(f'[train] Val dataloader disabled: {e}')

    def match_name_keywords(n, name_keywords):
        return any(b in n for b in name_keywords)

    param_dicts = [
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and p.requires_grad
            ],
            "lr": args.lr,
        },
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad
            ],
            "lr": args.lr * args.lr_linear_proj_mult,
        },
    ]
    optimizer = (
        torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        if args.sgd else
        torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                           find_unused_parameters=True)
        model_without_ddp = model.module

    if args.pretrained is not None:
        model_without_ddp = load_model(model_without_ddp, args.pretrained)

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = load_torch_checkpoint(args.resume, map_location='cpu')
        missing, unexpected = model_without_ddp.load_state_dict(
            checkpoint['model'], strict=False)
        unexpected = [k for k in unexpected
                      if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if missing:
            print('Missing keys:', missing)
        if unexpected:
            print('Unexpected keys:', unexpected)
        if not args.eval and 'optimizer' in checkpoint:
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = [g['initial_lr'] for g in optimizer.param_groups]
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

    # ------------------------------------------------------------------ #
    # Logging setup (TensorBoard + CSV, main process only)               #
    # ------------------------------------------------------------------ #
    writer = None
    log_csv_path = output_dir / 'train_log.csv'
    _CSV_HEADER = [
        'epoch', 'lr',
        'loss_total', 'loss_track', 'loss_detect',
        'loss_cls', 'loss_bbox', 'loss_giou',
        'loss_div_box', 'loss_div_class',
        'grad_norm',
        'val_loss_total', 'val_loss_track', 'val_loss_cls', 'val_loss_div_class',
        'val_full_cov_pct', 'val_mean_cov_pct',
        'val_d1_detect_pct', 'val_d2_detect_pct', 'val_both_linked_pct',
    ]
    best_val_div_class = float('inf')
    if utils.is_main_process():
        if _TB_AVAILABLE:
            writer = SummaryWriter(log_dir=str(output_dir / 'tb_logs'))
            print(f'TensorBoard logs → {output_dir / "tb_logs"}')
        if not log_csv_path.exists():
            with open(log_csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(_CSV_HEADER)
        print(f'CSV log         → {log_csv_path}')

    print("Start training")
    start_time = time.time()
    dataset_train.set_epoch(args.start_epoch)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch_mot_self_proposal(
            model, criterion, criterion_detect, data_loader_train, optimizer,
            args.score_threshold, device, epoch, args.clip_max_norm,
            args.accum_iter, args.lambda_detect,
            reuse_encoder_cache=args.reuse_encoder_cache)
        lr_scheduler.step()

        # -------------------------------------------------------------- #
        # Val loss (every epoch, skipped if no val dataloader)           #
        # -------------------------------------------------------------- #
        val_stats = {}
        if data_loader_val is not None:
            val_stats = eval_one_epoch_mot_self_proposal(
                model, criterion, criterion_detect, data_loader_val,
                args.score_threshold, device,
                lambda_detect=args.lambda_detect,
                reuse_encoder_cache=args.reuse_encoder_cache)

        # -------------------------------------------------------------- #
        # Periodic val inference (every eval_period epochs, main only)   #
        # -------------------------------------------------------------- #
        infer_metrics = {}
        run_infer = (utils.is_main_process()
                     and getattr(args, 'eval_period', 0) > 0
                     and getattr(args, 'infer_config', None) is not None
                     and (epoch + 1) % args.eval_period == 0)
        if run_infer:
            infer_metrics = _run_val_inference(
                args, output_dir / 'checkpoint.pth', epoch)

        # -------------------------------------------------------------- #
        # Log metrics (main process only)                                 #
        # -------------------------------------------------------------- #
        if utils.is_main_process():
            current_lr = optimizer.param_groups[0]['lr']

            def _sum_keys(prefix, stats=train_stats):
                return sum(v for k, v in stats.items() if prefix in k)

            nan = float('nan')
            row = {
                'epoch':          epoch,
                'lr':             current_lr,
                'loss_total':     train_stats.get('loss_overall', 0.0),
                'loss_track':     train_stats.get('loss', 0.0),
                'loss_detect':    train_stats.get('loss_detect', 0.0),
                'loss_cls':       _sum_keys('loss_ce'),
                'loss_bbox':      _sum_keys('loss_bbox'),
                'loss_giou':      _sum_keys('loss_giou'),
                'loss_div_box':   _sum_keys('loss_div_box'),
                'loss_div_class': _sum_keys('loss_div_class'),
                'grad_norm':      train_stats.get('grad_norm', 0.0),
                'val_loss_total':     val_stats.get('loss_overall', nan),
                'val_loss_track':     val_stats.get('loss', nan),
                'val_loss_cls':       _sum_keys('loss_ce', val_stats),
                'val_loss_div_class': _sum_keys('loss_div_class', val_stats),
                'val_full_cov_pct':    infer_metrics.get('val_full_cov_pct', nan),
                'val_mean_cov_pct':    infer_metrics.get('val_mean_cov_pct', nan),
                'val_d1_detect_pct':   infer_metrics.get('val_d1_detect_pct', nan),
                'val_d2_detect_pct':   infer_metrics.get('val_d2_detect_pct', nan),
                'val_both_linked_pct': infer_metrics.get('val_both_linked_pct', nan),
            }

            # TensorBoard
            if writer is not None:
                for k, v in row.items():
                    if k == 'epoch':
                        continue
                    if math.isnan(v):
                        continue
                    group = 'val' if k.startswith('val') else k.split('_')[0]
                    writer.add_scalar(f'{group}/{k}', v, epoch)

            # CSV (append one row)
            with open(log_csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([row[h] for h in _CSV_HEADER])

        if args.output_dir:
            checkpoint_payload = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if ((epoch + 1) % args.lr_drop == 0
                    or (epoch + 1) % args.save_period == 0
                    or (epoch + 1) % 5 == 0):
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04d}.pth')
            for cp in checkpoint_paths:
                utils.save_on_master(checkpoint_payload, cp)

            # Save best checkpoint based on val_loss_div_class.
            if val_stats and utils.is_main_process():
                val_div = row['val_loss_div_class']
                if not math.isnan(val_div) and val_div < best_val_div_class:
                    best_val_div_class = val_div
                    utils.save_on_master(checkpoint_payload,
                                         output_dir / 'checkpoint_best.pth')
                    print(f'  [val] New best val_loss_div_class={val_div:.4f} '
                          f'→ saved checkpoint_best.pth')

        dataset_train.step_epoch()

    if writer is not None:
        writer.close()
    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f'Training time {total_time}')


def _apply_yaml_defaults(parser: argparse.ArgumentParser, yaml_path: str):
    """Load a YAML config and set its values as parser defaults.

    CLI flags still override YAML values because set_defaults() only sets
    defaults; any flag explicitly provided on the command line wins.
    """
    if not _YAML_AVAILABLE:
        raise RuntimeError("PyYAML is not installed; run: pip install pyyaml")
    with open(yaml_path) as f:
        cfg = _yaml.safe_load(f)
    if cfg is None:
        return
    # argparse stores list args under their dest (underscored) name
    overrides = {}
    for key, val in cfg.items():
        dest = key.replace('-', '_')
        # None / null in YAML → keep argparse default
        if val is None:
            continue
        overrides[dest] = val
    parser.set_defaults(**overrides)


if __name__ == '__main__':
    # ---- pre-parse to find --config, if any ----
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None,
                     help='Path to a YAML config file; CLI args override YAML values')
    pre_args, remaining = pre.parse_known_args()

    parser = argparse.ArgumentParser(
        'SelfMOTR cell tracking', parents=[get_args_parser()])

    if pre_args.config is not None:
        _apply_yaml_defaults(parser, pre_args.config)

    args = parser.parse_args(remaining)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
