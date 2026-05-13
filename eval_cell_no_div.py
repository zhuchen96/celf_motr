"""
Inference script for the no-division variant of SelfMOTR.

Reads val/test frames, tracks cells frame-by-frame, and writes CTC results:
    <output_dir>/<seq_id>_RES/
        mask{t:03d}.tif   uint16 label images
        res_track.txt     L B E P  (all parent fields = 0, no division links)

Usage:
    python eval_cell_no_div.py --config configs/CellTracking/deepcell/infer_no_div.yaml

Or with overrides:
    python eval_cell_no_div.py --config configs/CellTracking/deepcell/infer_no_div.yaml \\
        --score_threshold 0.4
"""

# ---- patch model registry before any other import touches models ----
import models as _models
from models.motrv2_self_no_div import build as _build_no_div
from models.motrv2_self import RuntimeTrackerBase

_orig_build_model = _models.build_model
def _patched_build_model(args):
    if getattr(args, 'meta_arch', '') == 'motrv2_self_no_div':
        return _build_no_div(args)
    return _orig_build_model(args)
_models.build_model = _patched_build_model

# ---- patch dataset registry (not strictly needed for inference, but harmless) ----
import datasets as _datasets
from datasets.ctc_cell_no_div import build as _build_no_div_dataset

_orig_build_dataset = _datasets.build_dataset
def _patched_build_dataset(image_set, args):
    if getattr(args, 'dataset_file', '') == 'e2e_cell_no_div':
        return _build_no_div_dataset(image_set, args)
    return _orig_build_dataset(image_set, args)
_datasets.build_dataset = _patched_build_dataset

# ---- now the actual inference logic ----

import argparse
import json
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.ops as tvops
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from models.structures import Instances
from train_cell import get_args_parser
from util.box_ops import box_cxcywh_to_xyxy, box_iou
from util.tool import apply_checkpoint_model_args, load_model, load_torch_checkpoint


MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def load_and_preprocess(img_path: Path):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    tensor = TF.normalize(TF.to_tensor(img), MEAN, STD)
    return tensor.unsqueeze(0), (h, w)


def _recover_daughters(dying_box_xyxy, proposals, live_boxes, ori_size,
                       radius_factor, nms_iou_thresh, max_daughters):
    """
    Find detection proposals near a dying track that are not already covered by
    a live tracked cell.  These are candidate daughter cells.

    Parameters
    ----------
    dying_box_xyxy  : list[float]  last known box of the dying track (absolute xyxy)
    proposals       : Tensor [N, 5]  raw detection proposals (cxcywh-norm + score)
    live_boxes      : Tensor [M, 4] | None  boxes of currently live tracked cells (absolute xyxy)
    ori_size        : (H, W)
    radius_factor   : float  search radius = radius_factor × diagonal of dying_box
    nms_iou_thresh  : float  proposals with IoU > this vs any live track are discarded
    max_daughters   : int    maximum daughters to return

    Returns
    -------
    list of (box_xyxy_abs: list[float], proposal_cxcywh_norm: Tensor[4])
    """
    if proposals.shape[0] == 0:
        return []

    H, W = ori_size
    scale = torch.tensor([W, H, W, H], device=proposals.device, dtype=torch.float32)

    prop_abs = box_cxcywh_to_xyxy(proposals[:, :4]) * scale   # absolute xyxy
    prop_scores = proposals[:, 4]

    # --- spatial filter: centre must lie within radius_factor × mother diagonal ---
    dx1, dy1, dx2, dy2 = dying_box_xyxy
    diag = ((dx2 - dx1) ** 2 + (dy2 - dy1) ** 2) ** 0.5
    radius = diag * radius_factor
    mx, my = (dx1 + dx2) / 2, (dy1 + dy2) / 2

    pcx = (prop_abs[:, 0] + prop_abs[:, 2]) / 2
    pcy = (prop_abs[:, 1] + prop_abs[:, 3]) / 2
    nearby = ((pcx - mx) ** 2 + (pcy - my) ** 2) ** 0.5 <= radius
    if not nearby.any():
        return []

    prop_abs  = prop_abs[nearby]
    prop_scores = prop_scores[nearby]
    prop_norm   = proposals[nearby, :4]

    # --- suppress proposals already covered by a live tracked cell ---
    if live_boxes is not None and live_boxes.shape[0] > 0:
        iou, _ = box_iou(prop_abs, live_boxes)     # [N_nearby, M_live]
        covered = iou.max(dim=1).values >= nms_iou_thresh
        prop_abs    = prop_abs[~covered]
        prop_scores = prop_scores[~covered]
        prop_norm   = prop_norm[~covered]

    if prop_abs.shape[0] == 0:
        return []

    # --- NMS among the remaining proposals to avoid duplicate daughters ---
    keep = tvops.nms(prop_abs, prop_scores, nms_iou_thresh)
    prop_abs    = prop_abs[keep]
    prop_scores = prop_scores[keep]
    prop_norm   = prop_norm[keep]

    # Take top-max_daughters by score
    topk = min(max_daughters, prop_abs.shape[0])
    order = prop_scores.argsort(descending=True)[:topk]

    return [(prop_abs[i].tolist(), prop_norm[i]) for i in order.tolist()]


def track_sequence(model, img_dir: Path, frame_files: list,
                   proposal_threshold: float, score_threshold: float,
                   reuse_encoder_cache: bool, device: torch.device,
                   recovery_radius_factor: float = 0.0,
                   recovery_nms_iou: float = 0.4,
                   recovery_max_daughters: int = 2,
                   recovery_min_prev_score: float = 0.5):
    """
    Run no-div tracker over one sequence.

    Score-Drop Division Recovery (SDDR): when a tracked cell's score drops below
    the filter threshold (potential death or division), raw detection proposals
    near the cell's last known position are injected as daughter detections for
    that frame.  The dying track is killed immediately so it cannot suppress the
    daughters in subsequent frames.  Set recovery_radius_factor=0 to disable.

    Returns
    -------
    per_frame : list of lists
        per_frame[t] = [(track_id, box_xyxy), ...]
    """
    model.clear()
    track_instances = None
    per_frame = []

    filter_thresh = model.track_base.filter_score_thresh
    # obj_idx → (score, box_xyxy_absolute) from the last frame the track was healthy
    prev_alive = {}  # type: dict[int, tuple[float, list]]

    for fname in frame_files:
        img_tensor, ori_size = load_and_preprocess(img_dir / fname)
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            if reuse_encoder_cache:
                proposals, memory, spatial_shapes, level_start_index, \
                    valid_ratios, mask_flatten = \
                    model.inference_single_image_proposals_light_light(
                        img_tensor, ori_size,
                        score_threshold=proposal_threshold)
                res = model.inference_single_image_light_light(
                    img_tensor, ori_size, track_instances,
                    proposals, memory, spatial_shapes,
                    level_start_index, valid_ratios, mask_flatten)
            else:
                proposals = model.inference_single_image_proposals(
                    img_tensor, ori_size, score_threshold=proposal_threshold)
                res = model.inference_single_image(
                    img_tensor, ori_size, track_instances, proposals)

        track_instances = res['track_instances']

        # ── Score-Drop Division Recovery ─────────────────────────────────────
        extra_detections = []
        if recovery_radius_factor > 0 and proposals.shape[0] > 0:
            # Tracks whose score crossed the death threshold THIS frame:
            # disappear_time == 1 means track_base.update() just set it to 1
            # (was 0 last frame, i.e. the track was alive and healthy).
            just_died = (
                (track_instances.obj_idxes >= 0) &
                (track_instances.scores < filter_thresh) &
                (track_instances.disappear_time == 1)
            )

            if just_died.any():
                alive_mask = (
                    (track_instances.obj_idxes >= 0) &
                    (track_instances.scores >= filter_thresh)
                )
                live_boxes = track_instances.boxes[alive_mask] if alive_mask.any() else None

                for idx in just_died.nonzero(as_tuple=True)[0].tolist():
                    obj_id = track_instances.obj_idxes[idx].item()
                    prev = prev_alive.get(obj_id)
                    if prev is None:
                        continue
                    prev_score, last_box = prev
                    # Only recover if the track was confidently alive last frame.
                    # From the diagnostic: real mothers score ~0.70 at T-1;
                    # spurious short-lived tracks die barely above filter_thresh.
                    if prev_score < recovery_min_prev_score:
                        continue

                    daughters = _recover_daughters(
                        last_box, proposals, live_boxes, ori_size,
                        radius_factor=recovery_radius_factor,
                        nms_iou_thresh=recovery_nms_iou,
                        max_daughters=recovery_max_daughters,
                    )

                    for box_abs, _ in daughters:
                        fresh_id = model.track_base.max_obj_id
                        model.track_base.max_obj_id += 1
                        extra_detections.append((fresh_id, box_abs))

                    if daughters:
                        # Kill the mother immediately so she does not enter the
                        # next decoder and suppress the daughters' signals.
                        track_instances.obj_idxes[idx] = -1

        # Update prev_alive: only tracks with a confirmed high score this frame
        prev_alive = {
            track_instances.obj_idxes[i].item(): (
                track_instances.scores[i].item(),
                track_instances.boxes[i].tolist(),
            )
            for i in range(len(track_instances))
            if track_instances.obj_idxes[i].item() >= 0
            and track_instances.scores[i].item() >= filter_thresh
        }
        # ─────────────────────────────────────────────────────────────────────

        dt = deepcopy(track_instances)
        keep = (dt.obj_idxes >= 0) & (dt.scores > score_threshold)
        dt = dt[keep]

        frame_tracks = [(int(tid), box)
                        for tid, box in zip(dt.obj_idxes.tolist(), dt.boxes.tolist())]
        frame_tracks.extend(extra_detections)
        per_frame.append(frame_tracks)

    return per_frame


def _box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _box_diag(box):
    x1, y1, x2, y2 = box
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def _center_dist(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def close_gaps(per_frame: list, max_gap: int = 5,
               max_dist_factor: float = 1.5) -> list:
    """Stitch broken track fragments."""
    track_last  = {}
    track_first = {}
    for t, frame_tracks in enumerate(per_frame):
        for tid, box in frame_tracks:
            if tid not in track_first:
                track_first[tid] = (t, box)
            track_last[tid] = (t, box)

    parent = {}
    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x
    def union(a, b):
        parent[find(b)] = find(a)

    for tid_b, (t_b, box_b) in sorted(track_first.items(), key=lambda kv: kv[1][0]):
        best_tid, best_dist = None, float('inf')
        max_dist = _box_diag(box_b) * max_dist_factor
        for tid_a, (t_a, box_a) in track_last.items():
            if find(tid_a) == find(tid_b):
                continue
            gap = t_b - t_a
            if gap < 1 or gap > max_gap + 1:
                continue
            dist = _center_dist(_box_center(box_a), _box_center(box_b))
            if dist < max_dist and dist < best_dist:
                best_dist, best_tid = dist, tid_a
        if best_tid is not None:
            union(best_tid, tid_b)

    if not parent:
        return per_frame

    new_per_frame = []
    for frame_tracks in per_frame:
        seen = set()
        new_frame = []
        for tid, box in frame_tracks:
            root = find(tid)
            if root not in seen:
                new_frame.append((root, box))
                seen.add(root)
        new_per_frame.append(new_frame)

    merged = sum(1 for t in track_first if find(t) != t)
    print(f'    gap closing: merged {merged} fragments into longer tracks')
    return new_per_frame


def write_ctc_results(per_frame: list, img_dir: Path, frame_files: list,
                      out_dir: Path):
    """Write CTC-format mask tifs and res_track.txt (all parent = 0)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    first = Image.open(img_dir / frame_files[0]).convert('L')
    W, H = first.size

    track_span: dict = {}  # label → [first_frame, last_frame, last_box]

    for t, frame_tracks in enumerate(per_frame):
        label_img = np.zeros((H, W), dtype=np.uint16)

        for tid, box in frame_tracks:
            label = int(tid) + 1
            x1, y1, x2, y2 = box
            x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
            x2, y2 = min(W, int(round(x2))), min(H, int(round(y2)))
            if x2 > x1 and y2 > y1:
                label_img[y1:y2, x1:x2] = label

            if label not in track_span:
                track_span[label] = [t, t, box]
            else:
                track_span[label][1] = t
                track_span[label][2] = box

        cv2.imwrite(str(out_dir / f'mask{t:03d}.tif'), label_img)

    with open(out_dir / 'res_track.txt', 'w') as f:
        for label, (b, e, _) in sorted(track_span.items()):
            f.write(f'{label} {b} {e} 0\n')  # parent always 0 (no division)


def main(args):
    checkpoint = load_torch_checkpoint(args.resume, map_location='cpu',
                                       weights_only=False)
    args = apply_checkpoint_model_args(args, checkpoint, context='eval_cell_no_div')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, _, _ = _models.build_model(args)
    model.track_base = RuntimeTrackerBase(
        score_thresh=args.update_score_threshold,
        filter_score_thresh=args.update_score_threshold,
        miss_tolerance=args.miss_tolerance,
    )
    model = load_model(model, args.resume)
    model.eval()
    model.to(device)

    mot_root = Path(args.mot_path)
    ann_file = mot_root / 'annotations' / args.split / 'anno.json'
    img_dir  = mot_root / args.split / 'img'

    with open(ann_file) as f:
        data = json.load(f)

    seq_to_imgs = defaultdict(list)
    for img in data['images']:
        seq_key = img.get('ctc_id', img.get('man_track_id', 'unknown'))
        seq_to_imgs[seq_key].append(img)
    for seq_key in seq_to_imgs:
        seq_to_imgs[seq_key].sort(key=lambda x: x['frame_id'])

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Running no-div inference on {len(seq_to_imgs)} sequences "
          f"(split={args.split}, score_threshold={args.score_threshold})")

    for seq_key in tqdm(sorted(seq_to_imgs.keys()), desc='sequences'):
        imgs = seq_to_imgs[seq_key]
        frame_files = [img['file_name'] for img in imgs]

        per_frame = track_sequence(
            model, img_dir, frame_files,
            proposal_threshold=args.proposal_threshold,
            score_threshold=args.score_threshold,
            reuse_encoder_cache=args.reuse_encoder_cache,
            device=device,
            recovery_radius_factor=args.recovery_radius_factor,
            recovery_nms_iou=args.recovery_nms_iou,
            recovery_max_daughters=args.recovery_max_daughters,
            recovery_min_prev_score=args.recovery_min_prev_score,
        )

        if args.gap_close_frames > 0:
            per_frame = close_gaps(per_frame,
                                   max_gap=args.gap_close_frames,
                                   max_dist_factor=args.gap_close_dist_factor)

        out_dir = output_root / f'{seq_key}_RES'
        write_ctc_results(per_frame, img_dir, frame_files, out_dir)
        print(f"  seq {seq_key}: {len(frame_files)} frames → {out_dir}")


if __name__ == '__main__':
    try:
        import yaml as _yaml
        _YAML_AVAILABLE = True
    except ImportError:
        _YAML_AVAILABLE = False

    def _apply_yaml_defaults(parser, yaml_path):
        if not _YAML_AVAILABLE:
            raise RuntimeError('PyYAML not installed')
        with open(yaml_path) as f:
            cfg = _yaml.safe_load(f)
        if cfg is None:
            return
        overrides = {k.replace('-', '_'): v for k, v in cfg.items() if v is not None}
        parser.set_defaults(**overrides)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    pre_args, remaining = pre.parse_known_args()

    parser = argparse.ArgumentParser(
        'SelfMOTR no-div inference', parents=[get_args_parser()])
    parser.add_argument('--split', default='val', choices=['train', 'val'])
    parser.add_argument('--proposal_threshold', default=0.05, type=float)
    parser.add_argument('--update_score_threshold', default=0.5, type=float)
    parser.add_argument('--miss_tolerance', default=10, type=int)
    parser.add_argument('--gap_close_frames', default=5, type=int)
    parser.add_argument('--gap_close_dist_factor', default=1.5, type=float)
    # --- Score-Drop Division Recovery ---
    # Set recovery_radius_factor > 0 to enable (e.g. 2.0).
    # When a track's score drops below filter_thresh for the first time,
    # raw detection proposals within (radius_factor × dying-cell diagonal)
    # are injected as daughter detections and the dying track is killed
    # immediately so it no longer suppresses them in subsequent frames.
    parser.add_argument('--recovery_radius_factor', default=0.0, type=float,
                        help='Enable SDDR: search radius = N × dying track diagonal (0=off)')
    parser.add_argument('--recovery_nms_iou', default=0.4, type=float,
                        help='IoU threshold to discard proposals already covered by live tracks')
    parser.add_argument('--recovery_max_daughters', default=2, type=int,
                        help='Max daughter proposals to inject per dying track')
    parser.add_argument('--recovery_min_prev_score', default=0.5, type=float,
                        help='Min score at T-1 to trigger recovery (filters short-lived tracks)')
    # kept for apply_checkpoint_model_args compatibility; not used in no-div
    parser.add_argument('--div_score_thresh', default=0.5, type=float)
    parser.add_argument('--div_threshold', default=0.5, type=float)
    parser.add_argument('--max_div_dist_factor', default=2.0, type=float)

    if pre_args.config is not None:
        _apply_yaml_defaults(parser, pre_args.config)

    args = parser.parse_args(remaining)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
