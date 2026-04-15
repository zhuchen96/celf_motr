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
# Image loading / preprocessing (mirrors ctc_cell.py / make_transforms_cell)  #
# --------------------------------------------------------------------------- #

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def load_and_preprocess(img_path: Path):
    """Load a .tif frame and return (tensor [1,3,H,W], (H, W))."""
    img = Image.open(img_path).convert('RGB')
    w, h = img.size          # PIL: (width, height)
    tensor = TF.normalize(TF.to_tensor(img), MEAN, STD)
    return tensor.unsqueeze(0), (h, w)   # ori_img_size = (H, W)


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
        per_frame[t] = [(track_id: int, box_xyxy: [x1,y1,x2,y2], div_score: float), ...]
        div_score = sigmoid(pred_div_ahead) in [0, 1]; high → cell is about to divide
    """
    model.track_base.clear()
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

        has_div = dt.has('pred_div_ahead')
        frame_tracks = []
        for i, (tid, box) in enumerate(zip(dt.obj_idxes.tolist(), dt.boxes.tolist())):
            div_score = float(torch.sigmoid(dt.pred_div_ahead[i])) if has_div else 0.0
            frame_tracks.append((int(tid), box, div_score))
        per_frame.append(frame_tracks)

    return per_frame


# --------------------------------------------------------------------------- #
# CTC result writing                                                            #
# --------------------------------------------------------------------------- #

def _box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _center_dist(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def write_ctc_results(per_frame: list, img_dir: Path, frame_files: list,
                      out_dir: Path, div_threshold: float = 0.5):
    """
    Write CTC-format results:
      mask{t:03d}.tif  — uint16 label images (rectangle fill from boxes)
      res_track.txt    — L B E P

    Division linking
    ----------------
    A track is flagged as a dividing parent when its div_score at its LAST
    observed frame exceeds div_threshold.  The two new tracks that start in
    the very next frame and are spatially closest to the parent's last position
    are assigned as daughters (P = parent label).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    first = Image.open(img_dir / frame_files[0]).convert('L')
    W, H = first.size   # PIL: (width, height)

    # label = tid + 1 (CTC labels must be > 0)
    # track_span[label] = [first_frame, last_frame, last_box, max_div_score_at_last_frame]
    track_span: dict = {}
    # label -> parent label (0 = no parent)
    parent: dict = {}

    for t, frame_tracks in enumerate(per_frame):
        label_img = np.zeros((H, W), dtype=np.uint16)
        active_labels = set()

        for tid, box, div_score in frame_tracks:
            label = int(tid) + 1
            active_labels.add(label)
            x1, y1, x2, y2 = box
            x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
            x2, y2 = min(W, int(round(x2))), min(H, int(round(y2)))
            if x2 > x1 and y2 > y1:
                label_img[y1:y2, x1:x2] = label

            if label not in track_span:
                track_span[label] = [t, t, box, div_score]
                parent[label] = 0
            else:
                track_span[label][1] = t          # update last frame
                track_span[label][2] = box        # update last box
                track_span[label][3] = div_score  # update div score at last frame

        cv2.imwrite(str(out_dir / f'mask{t:03d}.tif'), label_img)

        # --- division linking ---
        # Tracks that were active last frame but are gone now = ended at t-1
        if t > 0:
            prev_labels = {int(tid) + 1 for tid, _, _ in per_frame[t - 1]}
            ended = prev_labels - active_labels
            new_labels = active_labels - prev_labels

            for plabel in ended:
                info = track_span.get(plabel)
                if info is None:
                    continue
                _, end_t, last_box, div_score = info
                if end_t != t - 1:
                    continue   # ended earlier, already processed
                if div_score < div_threshold:
                    continue   # not predicted as dividing

                # Find the 2 new tracks closest to the parent's last centre
                parent_ctr = _box_center(last_box)
                new_with_dist = []
                for dlabel in new_labels:
                    if parent[dlabel] != 0:
                        continue  # already assigned a parent
                    d_info = track_span.get(dlabel)
                    if d_info is None:
                        continue
                    d_box = d_info[2]
                    dist = _center_dist(parent_ctr, _box_center(d_box))
                    new_with_dist.append((dist, dlabel))

                new_with_dist.sort()
                for _, dlabel in new_with_dist[:2]:
                    parent[dlabel] = plabel

    # Write res_track.txt:  L  B  E  P
    n_div = sum(1 for p in parent.values() if p != 0)
    print(f'    division events detected: {n_div // 2} '
          f'({n_div} daughter tracks with parent links)')
    with open(out_dir / 'res_track.txt', 'w') as f:
        for label, (b, e, _, _) in sorted(track_span.items()):
            f.write(f'{label} {b} {e} {parent.get(label, 0)}\n')


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main(args):
    # ------------------------------------------------------------------ #
    # Load checkpoint and rebuild model args                               #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Discover sequences from COCO annotation                              #
    # ------------------------------------------------------------------ #
    mot_root = Path(args.mot_path)
    ann_file = mot_root / 'annotations' / args.split / 'anno.json'
    img_dir  = mot_root / args.split / 'img'

    with open(ann_file) as f:
        data = json.load(f)

    # Group images by sequence, sorted by frame_id
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

        out_dir = output_root / f'{seq_key}_RES'
        write_ctc_results(per_frame, img_dir, frame_files, out_dir,
                          div_threshold=args.div_threshold)
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

    # pre-parse to find --config before building the full parser
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    pre_args, remaining = pre.parse_known_args()

    parser = argparse.ArgumentParser(
        'SelfMOTR cell inference', parents=[get_args_parser()])
    parser.add_argument('--split', default='val', choices=['train', 'val'],
                        help='Dataset split to run inference on')
    parser.add_argument('--proposal_threshold', default=0.05, type=float,
                        help='Confidence threshold for self-proposals')
    parser.add_argument('--update_score_threshold', default=0.5, type=float,
                        help='Score threshold used by RuntimeTrackerBase')
    parser.add_argument('--miss_tolerance', default=10, type=int,
                        help='Frames before a lost track is dropped')
    parser.add_argument('--div_threshold', default=0.5, type=float,
                        help='sigmoid(pred_div_ahead) >= this → flag as dividing parent')

    if pre_args.config is not None:
        _apply_yaml_defaults(parser, pre_args.config)

    args = parser.parse_args(remaining)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
