"""
Visualisation script for SelfMOTR cell tracking.

Runs inference on every sequence in the chosen split and writes one MP4 per
sequence showing:
  - Bounding boxes coloured by track ID (consistent across frames)
  - Track ID label in the top-left corner of each box
  - "M" marker + yellow box border on confirmed dividing cells (via parent_id linkage)

Usage:
    python viz_cell.py --config configs/CellTracking/infer.yaml

Override anything on the command line:
    python viz_cell.py --config configs/CellTracking/infer.yaml \\
        --div_threshold 0.4 --fps 8 --scale 8 --split val
"""

import argparse
import json
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from models import build_model
from models.motrv2_self import RuntimeTrackerBase
from train_cell import get_args_parser
from util.tool import apply_checkpoint_model_args, load_model, load_torch_checkpoint


# --------------------------------------------------------------------------- #
# Colour palette (BGR for cv2) — 20 visually distinct colours                 #
# --------------------------------------------------------------------------- #

_PALETTE_BGR = [
    (230,  25,  75), ( 60, 180,  75), (255, 225,  25), (  0, 130, 200),
    (245, 130,  48), (145,  30, 180), ( 70, 240, 240), (240,  50, 230),
    (210, 245,  60), (250, 190, 212), (  0, 128, 128), (220, 190, 255),
    (170, 110,  40), (255, 250, 200), (128,   0,   0), (170, 255, 195),
    (128, 128,   0), (255, 215, 180), (  0,   0, 128), (128, 128, 128),
]

def track_color(tid: int):
    """Return a BGR colour tuple for a given track ID."""
    return _PALETTE_BGR[int(tid) % len(_PALETTE_BGR)]


# --------------------------------------------------------------------------- #
# Image loading (mirrors ctc_cell.py)                                          #
# --------------------------------------------------------------------------- #

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def load_frame(img_path: Path):
    """Return (preprocessed tensor [1,3,H,W], raw uint8 numpy [H,W,3], (H,W))."""
    pil = Image.open(img_path).convert('RGB')
    w, h = pil.size
    tensor = TF.normalize(TF.to_tensor(pil), MEAN, STD).unsqueeze(0)
    raw = np.array(pil)          # H×W×3  uint8
    return tensor, raw, (h, w)


# --------------------------------------------------------------------------- #
# Per-sequence inference                                                        #
# --------------------------------------------------------------------------- #

def run_sequence(model, img_dir: Path, frame_files: list,
                 proposal_threshold: float, score_threshold: float,
                 reuse_encoder_cache: bool, device: torch.device):
    """
    Run the tracker over one sequence.

    Returns
    -------
    frames : list of dicts, one per frame
        {
          'raw'  : np.ndarray [H, W, 3] uint8,
          'ori_size': (H, W),
          'tracks': [(tid, box_xyxy, div_score), ...]
                    box_xyxy in absolute pixel coords of the COCO image
        }
    """
    model.clear()
    track_instances = None
    results = []

    for fname in frame_files:
        tensor, raw, ori_size = load_frame(img_dir / fname)
        tensor = tensor.to(device)

        with torch.no_grad():
            if reuse_encoder_cache:
                proposals, memory, spatial_shapes, level_start_index, \
                    valid_ratios, mask_flatten = \
                    model.inference_single_image_proposals_light_light(
                        tensor, ori_size, score_threshold=proposal_threshold)
                res = model.inference_single_image_light_light(
                    tensor, ori_size, track_instances,
                    proposals, memory, spatial_shapes,
                    level_start_index, valid_ratios, mask_flatten)
            else:
                proposals = model.inference_single_image_proposals(
                    tensor, ori_size, score_threshold=proposal_threshold)
                res = model.inference_single_image(
                    tensor, ori_size, track_instances, proposals)

        track_instances = res['track_instances']

        dt = deepcopy(track_instances)
        keep = (dt.obj_idxes >= 0) & (dt.scores > score_threshold)
        dt = dt[keep]

        tracks = []
        has_div    = dt.has('pred_div_score')
        has_db     = dt.has('pred_div_boxes')
        has_parent = dt.has('parent_obj_id')
        ori_h, ori_w = ori_size
        scale_fct = torch.tensor([ori_w, ori_h, ori_w, ori_h], dtype=torch.float32)
        for i, (tid, box) in enumerate(zip(dt.obj_idxes.tolist(), dt.boxes.tolist())):
            div_score = float(torch.sigmoid(dt.pred_div_score[i])) if has_div else 0.0
            # Store daughter positions as pixel xyxy so _find_daughters_by_prediction
            # can use proper Euclidean distance against track boxes (also pixel xyxy).
            # box (dt.boxes[i]) is already pixel xyxy via post_process.
            # pred_div_boxes is still normalised cxcywh → convert here.
            if has_db:
                from util import box_ops
                d1 = box   # pixel xyxy of daughter1 (current track position)
                d2_norm = dt.pred_div_boxes[i].cpu()
                d2 = (box_ops.box_cxcywh_to_xyxy(d2_norm) * scale_fct).tolist()
                div_boxes = list(d1) + d2
            else:
                div_boxes = None
            parent_id = int(dt.parent_obj_id[i]) if has_parent else -1
            tracks.append((int(tid), box, div_score, div_boxes, parent_id))

        results.append({'raw': raw, 'ori_size': ori_size, 'tracks': tracks})

    return results


# --------------------------------------------------------------------------- #
# Geometry helpers                                                              #
# --------------------------------------------------------------------------- #

def _box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _center_dist(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def _box_diag(box):
    x1, y1, x2, y2 = box
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


# --------------------------------------------------------------------------- #
# Gap closing                                                                   #
# --------------------------------------------------------------------------- #

def close_gaps(frames: list, max_gap: int = 5,
               max_dist_factor: float = 1.5) -> list:
    """
    Stitch broken track fragments that belong to the same physical cell.

    When track A ends at tA and track B starts at tB with tB - tA <= max_gap+1,
    and B's first position is within max_dist_factor × A's box diagonal of A's
    last centre, B is relabeled to A throughout the sequence.
    """
    track_last  = {}
    track_first = {}
    # Daughter tracks (parent_id >= 0) must not be merged into the mother — they
    # are intentional new tracks created by the division split mechanism.
    daughter_tids = set()
    for t, frame in enumerate(frames):
        for entry in frame['tracks']:
            tid, box = entry[0], entry[1]
            parent_id = entry[4] if len(entry) > 4 else -1
            if parent_id >= 0:
                daughter_tids.add(tid)
            if tid not in track_first:
                track_first[tid] = (t, box)
            track_last[tid] = (t, box)

    uf_parent = {}
    def find(x):
        while uf_parent.get(x, x) != x:
            uf_parent[x] = uf_parent.get(uf_parent[x], uf_parent[x])
            x = uf_parent[x]
        return x
    def union(a, b):
        uf_parent[find(b)] = find(a)

    for tid_b, (t_b, box_b) in sorted(track_first.items(), key=lambda kv: kv[1][0]):
        if tid_b in daughter_tids:
            continue  # never merge a daughter into its mother or another track
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

    if not uf_parent:
        return frames

    new_frames = []
    for frame in frames:
        seen = set()
        new_tracks = []
        for entry in frame['tracks']:
            tid = entry[0]
            root = find(tid)
            if root not in seen:
                new_tracks.append((root,) + entry[1:])
                seen.add(root)
        new_frames.append({**frame, 'tracks': new_tracks})

    merged = sum(1 for t in track_first if find(t) != t)
    print(f'  gap closing: merged {merged} fragments')
    return new_frames


# --------------------------------------------------------------------------- #
# Division linking                                                              #
# --------------------------------------------------------------------------- #

def find_division_pairs(frames: list, div_threshold: float,
                        max_dist_factor: float = 2.0, line_frames: int = 4):
    """
    Identify daughter-pair lines to draw after each division event.

    D1 (the mother query) continues with its existing track ID.
    D2 is spawned by the model with parent_obj_id = D1's track ID.
    Division pairs are read directly from parent_obj_id — no spatial search needed.

    Returns
    -------
    div_lines : dict  {frame_t: [(d1_tid, d2_tid), ...]}
    confirmed_parent_tids : set  — these tids show the yellow "M" marker
    """
    div_lines             = defaultdict(list)
    confirmed_parent_tids = set()
    d2_first_frame: dict  = {}   # d2_tid → (first_t, parent_tid)

    for t, frame in enumerate(frames):
        for entry in frame['tracks']:
            tid, box, div_score, div_boxes, parent_id = entry
            if parent_id >= 0 and tid not in d2_first_frame:
                d2_first_frame[tid] = (t, parent_id)

    for d2_tid, (t, parent_tid) in d2_first_frame.items():
        confirmed_parent_tids.add(parent_tid)
        for dt in range(line_frames):
            div_lines[t + dt].append((parent_tid, d2_tid))

    return div_lines, confirmed_parent_tids


def _find_daughters_by_prediction(div_boxes_8d, new_tids, active, assigned,
                                   max_dist_factor):
    """Find daughters closest to each predicted daughter position.

    div_boxes_8d: 8 floats [x1,y1,x2,y2, x1,y1,x2,y2] in pixel xyxy for d1 and d2.
    All boxes in active are also pixel xyxy, so distances are in the same space.
    """
    pred_ctrs = [
        ((div_boxes_8d[0] + div_boxes_8d[2]) / 2,
         (div_boxes_8d[1] + div_boxes_8d[3]) / 2),   # daughter 1 centre (pixels)
        ((div_boxes_8d[4] + div_boxes_8d[6]) / 2,
         (div_boxes_8d[5] + div_boxes_8d[7]) / 2),   # daughter 2 centre (pixels)
    ]
    chosen = []
    remaining = [d for d in new_tids if d not in assigned]
    for pred_cx, pred_cy in pred_ctrs:
        best, best_dist = None, float('inf')
        for d in remaining:
            if d in chosen:
                continue
            entry = active.get(d)
            if entry is None:
                continue
            box = entry[1]
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            # Enforce max distance: predicted daughter must be within max_dist_factor
            # × its own box diagonal to prevent linking far unrelated tracks.
            diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            max_dist = diag * max_dist_factor
            dist = ((cx - pred_cx) ** 2 + (cy - pred_cy) ** 2) ** 0.5
            if dist <= max_dist and dist < best_dist:
                best_dist, best = dist, d
        if best is not None:
            chosen.append(best)
    return chosen


def _find_daughters_by_proximity(last_box, new_tids, active, assigned,
                                  max_dist_factor):
    """Fallback: find daughters closest to parent's last centre."""
    parent_ctr = _box_center(last_box)
    max_dist   = _box_diag(last_box) * max_dist_factor
    candidates = []
    for d in new_tids:
        if d in assigned:
            continue
        entry = active.get(d)
        if entry is None:
            continue
        dist = _center_dist(parent_ctr, _box_center(entry[1]))
        if dist <= max_dist:
            candidates.append((dist, d))
    candidates.sort()
    return [d for _, d in candidates[:2]]


# --------------------------------------------------------------------------- #
# Video rendering                                                               #
# --------------------------------------------------------------------------- #

def render_video(frames: list, out_path: Path, fps: int, scale: int,
                 div_threshold: float, div_lines: dict = None,
                 confirmed_parent_tids: set = None):
    """
    Render one MP4 video from a list of annotated frames.

    Parameters
    ----------
    frames        : output of run_sequence()
    out_path      : path to write the .mp4 file
    fps           : frames per second
    scale         : integer upscale factor (images are tiny; 8× is readable)
    div_threshold : fallback threshold for mitotic marker when no parent_id linkage exists
    div_lines     : output of find_division_pairs(); draws lines between daughters
    """
    if not frames:
        return

    H, W = frames[0]['ori_size']
    out_H, out_W = H * scale, W * scale

    # MJPG + .avi is more reliable than mp4v on headless OpenCV builds
    # (mp4v produces wrong YUV color-space metadata → green video on most players)
    out_path = out_path.with_suffix('.avi')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (out_W, out_H))
    if not vw.isOpened():
        raise RuntimeError(f'VideoWriter failed to open: {out_path}')

    # Try to load a small font; fall back to cv2 default if unavailable
    font_scale = max(0.3, scale * 0.07)
    font_thick = max(1, scale // 6)
    cv2_font   = cv2.FONT_HERSHEY_SIMPLEX

    for t, frame in enumerate(frames):
        # --- base image: upscale the raw RGB frame ---
        raw = frame['raw']   # H×W×3 uint8 RGB
        if raw.ndim == 2:    # grayscale fallback
            raw = np.stack([raw] * 3, axis=-1)
        # Upscale with nearest-neighbour to keep cell borders crisp
        img_bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.resize(img_bgr, (out_W, out_H), interpolation=cv2.INTER_NEAREST)

        # --- draw tracks ---
        for tid, box, div_score, *_rest in frame['tracks']:
            x1, y1, x2, y2 = box
            # scale box to upscaled image coords
            sx1 = int(round(x1 * scale))
            sy1 = int(round(y1 * scale))
            sx2 = int(round(x2 * scale))
            sy2 = int(round(y2 * scale))
            sx1, sx2 = max(0, sx1), min(out_W - 1, sx2)
            sy1, sy2 = max(0, sy1), min(out_H - 1, sy2)

            color = track_color(tid)
            # D1's div_score is suppressed after spawning D2; use confirmed_parent_tids
            # (populated by find_division_pairs) to reliably mark dividing cells.
            if confirmed_parent_tids:
                mitotic = tid in confirmed_parent_tids
            else:
                mitotic = div_score >= div_threshold

            if mitotic:
                # bright yellow border for dividing cells
                cv2.rectangle(img_bgr, (sx1, sy1), (sx2, sy2), (0, 255, 255), font_thick + 1)
            else:
                cv2.rectangle(img_bgr, (sx1, sy1), (sx2, sy2), color, font_thick)

            # label: "ID" or "ID M" for mitotic
            label = f'{tid}{"  M" if mitotic else ""}'
            lx = max(sx1, 0)
            ly = max(sy1 - 2, 8)
            cv2.putText(img_bgr, label, (lx, ly),
                        cv2_font, font_scale, color, font_thick,
                        cv2.LINE_AA)

        # --- division lines: connect daughter pairs ---
        if div_lines:
            tid_to_box = {tid: box for tid, box, *_ in frame['tracks']}
            for d1, d2 in div_lines.get(t, []):
                box1 = tid_to_box.get(d1)
                box2 = tid_to_box.get(d2)
                if box1 is None or box2 is None:
                    continue
                cx1 = int(round(((box1[0] + box1[2]) / 2) * scale))
                cy1 = int(round(((box1[1] + box1[3]) / 2) * scale))
                cx2 = int(round(((box2[0] + box2[2]) / 2) * scale))
                cy2 = int(round(((box2[1] + box2[3]) / 2) * scale))
                # bright magenta line between daughter centres
                cv2.line(img_bgr, (cx1, cy1), (cx2, cy2), (255, 0, 255), max(1, font_thick), cv2.LINE_AA)
                # cyan box highlight on each daughter at birth
                for box in (box1, box2):
                    bx1 = max(0, int(round(box[0] * scale)))
                    by1 = max(0, int(round(box[1] * scale)))
                    bx2 = min(out_W - 1, int(round(box[2] * scale)))
                    by2 = min(out_H - 1, int(round(box[3] * scale)))
                    cv2.rectangle(img_bgr, (bx1, by1), (bx2, by2), (255, 255, 0), font_thick + 1)

        # --- frame number overlay ---
        cv2.putText(img_bgr, f't={t:03d}', (2, out_H - 4),
                    cv2_font, font_scale * 0.8, (200, 200, 200), 1, cv2.LINE_AA)

        vw.write(img_bgr)

    vw.release()


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main(args):
    checkpoint = load_torch_checkpoint(args.resume, map_location='cpu',
                                       weights_only=False)
    args = apply_checkpoint_model_args(args, checkpoint, context='viz_cell')

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
    # Discover sequences                                                   #
    # ------------------------------------------------------------------ #
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

    print(f"Visualising {len(seq_to_imgs)} sequences  "
          f"(split={args.split}, score_thr={args.score_threshold}, "
          f"scale={args.scale}×, fps={args.fps})")

    for seq_key in tqdm(sorted(seq_to_imgs.keys()), desc='sequences'):
        imgs     = seq_to_imgs[seq_key]
        fnames   = [img['file_name'] for img in imgs]

        frames = run_sequence(
            model, img_dir, fnames,
            proposal_threshold=args.proposal_threshold,
            score_threshold=args.score_threshold,
            reuse_encoder_cache=args.reuse_encoder_cache,
            device=device,
        )

        if args.gap_close_frames > 0:
            frames = close_gaps(frames,
                                max_gap=args.gap_close_frames,
                                max_dist_factor=args.gap_close_dist_factor)

        div_lines, confirmed_parents = find_division_pairs(
            frames, div_threshold=args.div_threshold,
            max_dist_factor=args.max_div_dist_factor)
        n_div = sum(len(v) for v in div_lines.values())
        if n_div:
            print(f"    {n_div // max(1, args.fps)} division events linked")

        out_path = output_root / f'{seq_key}.avi'
        render_video(frames, out_path,
                     fps=args.fps,
                     scale=args.scale,
                     div_threshold=args.div_threshold,
                     div_lines=div_lines,
                     confirmed_parent_tids=confirmed_parents)
        print(f"  seq {seq_key}: {len(fnames)} frames → {out_path}")


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
        'SelfMOTR cell visualisation', parents=[get_args_parser()])
    parser.add_argument('--split', default='val', choices=['train', 'val'])
    parser.add_argument('--proposal_threshold', default=0.05, type=float)
    parser.add_argument('--update_score_threshold', default=0.5, type=float)
    parser.add_argument('--miss_tolerance', default=10, type=int)
    parser.add_argument('--div_threshold', default=0.5, type=float,
                        help='fallback threshold for mitotic marker (unused when model provides parent_id)')
    parser.add_argument('--max_div_dist_factor', default=2.0, type=float,
                        help='Max daughter distance as multiple of parent box diagonal')
    parser.add_argument('--gap_close_frames', default=5, type=int,
                        help='Max frame gap to stitch broken tracks (0 = disabled)')
    parser.add_argument('--gap_close_dist_factor', default=1.5, type=float,
                        help='Max stitch distance as multiple of the box diagonal')
    parser.add_argument('--fps', default=8, type=int,
                        help='Frames per second of output video')
    parser.add_argument('--scale', default=8, type=int,
                        help='Integer upscale factor (images are ~32×256 px)')

    if pre_args.config is not None:
        _apply_yaml_defaults(parser, pre_args.config)

    args = parser.parse_args(remaining)
    # redirect output to a sibling "videos" folder so it doesn't mix with
    # the CTC-format results from eval_cell.py
    args.output_dir = str(args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
