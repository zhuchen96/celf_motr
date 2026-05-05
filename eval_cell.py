"""
Inference script for SelfMOTR on CTC cell-tracking data.

Reads the val (or test) split from the Cell-TRACTR COCO layout, runs the
tracker frame-by-frame for each sequence, and writes results in CTC format:

    <output_dir>/<seq_id>_RES/
        mask000.tif          uint16 label image  (pixel value = track ID)
        mask001.tif
        ...
        res_track.txt        L B E P  (CTC tracking-result summary)

Usage (single GPU):
    python eval_cell.py \\
        --resume outputs/cell_moma/checkpoint.pth \\
        --mot_path /srv/home/chen/Cell-TRACTR/data/moma/COCO \\
        --output_dir outputs/cell_moma/eval \\
        --split val \\
        --score_threshold 0.5 \\
        --reuse_encoder_cache

Then evaluate with the CTC evaluation tool:
    python Cell-TRACTR/scripts/evaluate_tracking.py \\
        --res_dir outputs/cell_moma/eval \\
        --gt_dir  /srv/home/chen/Cell-TRACTR/data/moma/CTC/val
"""

import argparse
import json
import os
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from models import build_model
from models.motrv2_self import RuntimeTrackerBase
from models.structures import Instances
from train_cell import get_args_parser
from util.tool import apply_checkpoint_model_args, load_model, load_torch_checkpoint


# --------------------------------------------------------------------------- #
# Image loading / preprocessing                                                 #
# --------------------------------------------------------------------------- #

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def load_and_preprocess(img_path: Path):
    """Load a .tif frame and return (tensor [1,3,H,W], (H, W))."""
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    tensor = TF.normalize(TF.to_tensor(img), MEAN, STD)
    return tensor.unsqueeze(0), (h, w)


# --------------------------------------------------------------------------- #
# Per-sequence tracker                                                          #
# --------------------------------------------------------------------------- #

def track_sequence(model, img_dir: Path, frame_files: list,
                   proposal_threshold: float, score_threshold: float,
                   reuse_encoder_cache: bool, device: torch.device):
    """
    Run the tracker over one sequence.

    Returns
    -------
    per_frame : list of lists
        per_frame[t] = [(track_id, box_xyxy, div_score, div_boxes_8d, parent_id), ...]
          track_id    : int, globally unique
          box_xyxy    : [x1, y1, x2, y2] in absolute pixel coords
          div_score   : float in [0,1] — sigmoid(pred_div_ahead)
          div_boxes_8d: list[8] normalised cxcywh for both daughters, or None
          parent_id   : int, parent track ID if this track was created by division, else -1
    """
    model.clear()
    track_instances = None
    per_frame = []

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

        dt = deepcopy(track_instances)
        keep = (dt.obj_idxes >= 0) & (dt.scores > score_threshold)
        dt = dt[keep]

        has_div    = dt.has('pred_div_ahead')
        has_db     = dt.has('pred_div_boxes')
        has_parent = dt.has('parent_obj_id')

        frame_tracks = []
        for idx, (tid, box) in enumerate(zip(dt.obj_idxes.tolist(), dt.boxes.tolist())):
            div_score  = float(torch.sigmoid(dt.pred_div_ahead[idx])) if has_div else 0.0
            # pred_boxes (4D, normalised cxcywh) = daughter1; pred_div_boxes (4D) = daughter2.
            # Concatenate to 8D for write_ctc_results which expects [d1(4), d2(4)].
            if has_db:
                d1 = dt.pred_boxes[idx].cpu().tolist()
                d2 = dt.pred_div_boxes[idx].cpu().tolist()
                div_boxes = d1 + d2
            else:
                div_boxes = None
            parent_id  = int(dt.parent_obj_id[idx]) if has_parent else -1
            frame_tracks.append((int(tid), box, div_score, div_boxes, parent_id))
        per_frame.append(frame_tracks)

    return per_frame


# --------------------------------------------------------------------------- #
# CTC result writing                                                            #
# --------------------------------------------------------------------------- #

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
    """
    Stitch broken track fragments that belong to the same physical cell.
    """
    track_last  = {}
    track_first = {}
    for t, frame_tracks in enumerate(per_frame):
        for tid, box, *_ in frame_tracks:
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
        for tid, box, div_score, div_boxes, par in frame_tracks:
            root = find(tid)
            if root not in seen:
                new_frame.append((root, box, div_score, div_boxes, par))
                seen.add(root)
        new_per_frame.append(new_frame)

    merged = sum(1 for t in track_first if find(t) != t)
    print(f'    gap closing: merged {merged} fragments into longer tracks')
    return new_per_frame


def write_ctc_results(per_frame: list, img_dir: Path, frame_files: list,
                      out_dir: Path, div_threshold: float = 0.5,
                      max_div_dist_factor: float = 2.0):
    """
    Write CTC-format results:
      mask{t:03d}.tif  — uint16 label images
      res_track.txt    — L B E P

    Division linking (Cell-TRACTR style)
    ------------------------------------
    When a track's last frame has div_score >= div_threshold, we look for
    daughter candidates in the next frame.

    If the track has pred_div_boxes (model-predicted daughter positions), we
    use those as anchors for the search — one candidate per predicted position,
    each within max_div_dist_factor × parent_box_diagonal.

    If pred_div_boxes is unavailable (old checkpoint), we fall back to the
    distance-to-parent-center heuristic.

    Tracks created by the model's query-duplication mechanism already have
    parent_id set; we use those directly where available.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    first = Image.open(img_dir / frame_files[0]).convert('L')
    W, H = first.size

    # label = tid + 1 (CTC labels must be > 0)
    # track_span[label] = [first_frame, last_frame, last_box, max_div_score, div_boxes_8d]
    track_span: dict = {}
    parent: dict = {}   # label → parent label (0 = no parent)

    for t, frame_tracks in enumerate(per_frame):
        label_img = np.zeros((H, W), dtype=np.uint16)
        active_labels = set()

        for tid, box, div_score, div_boxes, par_tid in frame_tracks:
            label = int(tid) + 1
            active_labels.add(label)
            x1, y1, x2, y2 = box
            x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
            x2, y2 = min(W, int(round(x2))), min(H, int(round(y2)))
            if x2 > x1 and y2 > y1:
                label_img[y1:y2, x1:x2] = label

            if label not in track_span:
                track_span[label] = [t, t, box, div_score, div_boxes]
                # Use model-provided parent ID if available
                parent[label] = (int(par_tid) + 1) if par_tid >= 0 else 0
            else:
                track_span[label][1] = t
                track_span[label][2] = box
                track_span[label][3] = div_score
                track_span[label][4] = div_boxes

        cv2.imwrite(str(out_dir / f'mask{t:03d}.tif'), label_img)

        # --- division linking for tracks without a model-provided parent ---
        if t > 0:
            prev_labels = {int(tid) + 1 for tid, *_ in per_frame[t - 1]}
            ended = prev_labels - active_labels
            new_labels = active_labels - prev_labels

            for plabel in ended:
                info = track_span.get(plabel)
                if info is None:
                    continue
                _, end_t, last_box, div_score, div_boxes = info
                if end_t != t - 1 or div_score < div_threshold:
                    continue

                # Collect unassigned new tracks
                unassigned = [dl for dl in new_labels if parent.get(dl, 0) == 0]

                if div_boxes is not None:
                    # Use model-predicted daughter positions as anchors
                    _link_daughters_by_prediction(
                        plabel, div_boxes, unassigned, track_span,
                        parent, W, H, max_div_dist_factor)
                else:
                    # Fallback: search near parent's last position
                    _link_daughters_by_proximity(
                        plabel, last_box, unassigned, track_span,
                        parent, max_div_dist_factor)

    n_div = sum(1 for p in parent.values() if p != 0)
    print(f'    division events: {n_div // 2} '
          f'({n_div} daughter tracks with parent links)')
    with open(out_dir / 'res_track.txt', 'w') as f:
        for label, (b, e, *_) in sorted(track_span.items()):
            f.write(f'{label} {b} {e} {parent.get(label, 0)}\n')


def _link_daughters_by_prediction(plabel, div_boxes_8d, unassigned, track_span,
                                   parent, W, H, max_dist_factor):
    """
    Link daughters using the model's predicted daughter positions.

    div_boxes_8d = [d1_cx, d1_cy, d1_w, d1_h, d2_cx, d2_cy, d2_w, d2_h]
    in normalised [0,1] coordinates.  We convert to pixel xyxy, then find the
    closest unassigned track to each predicted position.
    """
    def norm_to_px(cx, cy, bw, bh):
        x1 = (cx - bw / 2) * W
        y1 = (cy - bh / 2) * H
        x2 = (cx + bw / 2) * W
        y2 = (cy + bh / 2) * H
        return [x1, y1, x2, y2]

    d1_box = norm_to_px(*div_boxes_8d[:4])
    d2_box = norm_to_px(*div_boxes_8d[4:])

    # For each predicted daughter, find the closest unassigned new track
    chosen = []
    remaining = list(unassigned)
    for pred_box in (d1_box, d2_box):
        pred_ctr = _box_center(pred_box)
        pred_diag = _box_diag(pred_box)
        max_dist = pred_diag * max_dist_factor if pred_diag > 1 else float('inf')
        best, best_dist = None, float('inf')
        for dl in remaining:
            if dl in chosen:
                continue
            d_info = track_span.get(dl)
            if d_info is None:
                continue
            dist = _center_dist(pred_ctr, _box_center(d_info[2]))
            if dist < max_dist and dist < best_dist:
                best_dist, best = dist, dl
        if best is not None:
            chosen.append(best)

    for dl in chosen:
        parent[dl] = plabel


def _link_daughters_by_proximity(plabel, last_box, unassigned, track_span,
                                  parent, max_dist_factor):
    """Fallback: link daughters by proximity to parent's last position."""
    parent_ctr = _box_center(last_box)
    max_dist   = _box_diag(last_box) * max_dist_factor
    candidates = []
    for dl in unassigned:
        d_info = track_span.get(dl)
        if d_info is None:
            continue
        dist = _center_dist(parent_ctr, _box_center(d_info[2]))
        if dist <= max_dist:
            candidates.append((dist, dl))
    candidates.sort()
    for _, dl in candidates[:2]:
        parent[dl] = plabel


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main(args):
    checkpoint = load_torch_checkpoint(args.resume, map_location='cpu',
                                       weights_only=False)
    args = apply_checkpoint_model_args(args, checkpoint, context='eval_cell')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, _, _ = build_model(args)
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

    print(f"Running inference on {len(seq_to_imgs)} sequences "
          f"(split={args.split}, score_threshold={args.score_threshold}, "
          f"proposal_threshold={args.proposal_threshold})")

    for seq_key in tqdm(sorted(seq_to_imgs.keys()), desc='sequences'):
        imgs = seq_to_imgs[seq_key]
        frame_files = [img['file_name'] for img in imgs]

        per_frame = track_sequence(
            model, img_dir, frame_files,
            proposal_threshold=args.proposal_threshold,
            score_threshold=args.score_threshold,
            reuse_encoder_cache=args.reuse_encoder_cache,
            device=device,
        )

        if args.gap_close_frames > 0:
            per_frame = close_gaps(per_frame,
                                   max_gap=args.gap_close_frames,
                                   max_dist_factor=args.gap_close_dist_factor)

        out_dir = output_root / f'{seq_key}_RES'
        write_ctc_results(per_frame, img_dir, frame_files, out_dir,
                          div_threshold=args.div_threshold,
                          max_div_dist_factor=args.max_div_dist_factor)
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
        'SelfMOTR cell inference', parents=[get_args_parser()])
    parser.add_argument('--split', default='val', choices=['train', 'val'])
    parser.add_argument('--proposal_threshold', default=0.05, type=float)
    parser.add_argument('--update_score_threshold', default=0.5, type=float)
    parser.add_argument('--miss_tolerance', default=10, type=int)
    parser.add_argument('--div_score_thresh', default=0.4, type=float)
    parser.add_argument('--div_threshold', default=0.5, type=float)
    parser.add_argument('--max_div_dist_factor', default=2.0, type=float)
    parser.add_argument('--gap_close_frames', default=5, type=int)
    parser.add_argument('--gap_close_dist_factor', default=1.5, type=float)

    if pre_args.config is not None:
        _apply_yaml_defaults(parser, pre_args.config)

    args = parser.parse_args(remaining)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
