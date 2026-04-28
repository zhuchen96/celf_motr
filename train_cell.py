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
import datetime
import random
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

from util.tool import load_model, load_torch_checkpoint
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset
from enginev2 import train_one_epoch_mot_self_proposal
from models import build_model
from models.deformable_detrv2_det import build as build_detect

import copy


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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
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

    print("Start training")
    start_time = time.time()
    dataset_train.set_epoch(args.start_epoch)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_one_epoch_mot_self_proposal(
            model, criterion, criterion_detect, data_loader_train, optimizer,
            args.score_threshold, device, epoch, args.clip_max_norm,
            args.accum_iter, args.lambda_detect,
            reuse_encoder_cache=args.reuse_encoder_cache)
        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if ((epoch + 1) % args.lr_drop == 0
                    or (epoch + 1) % args.save_period == 0
                    or (epoch + 1) % 5 == 0):
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04d}.pth')
            for cp in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, cp)

        dataset_train.step_epoch()

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
