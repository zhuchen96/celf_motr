"""
Visualisation script for SelfMOTR cell tracking.

Runs inference on every sequence in the chosen split and writes one MP4 per
sequence showing:
  - Bounding boxes coloured by track ID (consistent across frames)
  - Track ID label in the top-left corner of each box
  - "M" marker + yellow box border on cells with a high division score
    (pred_div_ahead sigmoid > --div_threshold)

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
    model.track_base.clear()
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
        has_div = dt.has('pred_div_ahead')
        if has_div and len(dt) > 0:
            scores_sigmoid = torch.sigmoid(dt.pred_div_ahead)
            if scores_sigmoid.max() > 0.3:   # only print when anything is notable
                print(f'    frame div scores: min={scores_sigmoid.min():.3f} '
                      f'max={scores_sigmoid.max():.3f} mean={scores_sigmoid.mean():.3f}')
        for i, (tid, box) in enumerate(zip(dt.obj_idxes.tolist(), dt.boxes.tolist())):
            div_score = float(torch.sigmoid(dt.pred_div_ahead[i])) if has_div else 0.0
            tracks.append((int(tid), box, div_score))

        results.append({'raw': raw, 'ori_size': ori_size, 'tracks': tracks})

    return results


# --------------------------------------------------------------------------- #
# Video rendering                                                               #
# --------------------------------------------------------------------------- #

def render_video(frames: list, out_path: Path, fps: int, scale: int,
                 div_threshold: float):
    """
    Render one MP4 video from a list of annotated frames.

    Parameters
    ----------
    frames      : output of run_sequence()
    out_path    : path to write the .mp4 file
    fps         : frames per second
    scale       : integer upscale factor (images are tiny; 8× is readable)
    div_threshold : sigmoid(pred_div_ahead) ≥ this → flag as mitotic
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
        for tid, box, div_score in frame['tracks']:
            x1, y1, x2, y2 = box
            # scale box to upscaled image coords
            sx1 = int(round(x1 * scale))
            sy1 = int(round(y1 * scale))
            sx2 = int(round(x2 * scale))
            sy2 = int(round(y2 * scale))
            sx1, sx2 = max(0, sx1), min(out_W - 1, sx2)
            sy1, sy2 = max(0, sy1), min(out_H - 1, sy2)

            color = track_color(tid)
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
          f"div_thr={args.div_threshold}, scale={args.scale}×, fps={args.fps})")

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

        out_path = output_root / f'{seq_key}.avi'
        render_video(frames, out_path,
                     fps=args.fps,
                     scale=args.scale,
                     div_threshold=args.div_threshold)
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
                        help='sigmoid(pred_div_ahead) >= this → mark as mitotic')
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
