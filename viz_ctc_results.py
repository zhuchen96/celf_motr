"""
Visualise CTC inference results written by eval_cell.py.

Reads mask{t:03d}.tif and res_track.txt from <res_dir>/<seq>_RES/,
overlays bounding boxes and division events on the original images,
and writes one AVI per sequence.

Usage:
    python viz_ctc_results.py \
        --res_dir  outputs/cell_deepcell_eval \
        --mot_path /srv/home/chen/Cell-TRACTR/data/deepcell/COCO \
        --split val \
        --fps 8 --scale 1
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


_PALETTE_BGR = [
    (230,  25,  75), ( 60, 180,  75), (255, 225,  25), (  0, 130, 200),
    (245, 130,  48), (145,  30, 180), ( 70, 240, 240), (240,  50, 230),
    (210, 245,  60), (250, 190, 212), (  0, 128, 128), (220, 190, 255),
    (170, 110,  40), (255, 250, 200), (128,   0,   0), (170, 255, 195),
    (128, 128,   0), (255, 215, 180), (  0,   0, 128), (128, 128, 128),
]

def track_color(label: int):
    return _PALETTE_BGR[int(label) % len(_PALETTE_BGR)]


def read_res_track(path: Path):
    """
    Parse res_track.txt (format: L B E P per line).

    Returns
    -------
    track_info : dict  label → {'begin': int, 'end': int, 'parent': int}
    children   : dict  parent_label → [child_label, ...]
    """
    track_info = {}
    children   = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            label, begin, end, par = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            track_info[label] = {'begin': begin, 'end': end, 'parent': par}
            if par > 0:
                children[par].append(label)
    return track_info, dict(children)


def extract_bboxes(mask: np.ndarray):
    """Return {label: [x1, y1, x2, y2]} for every non-zero label in the mask."""
    bboxes = {}
    for label in np.unique(mask):
        if label == 0:
            continue
        rows, cols = np.where(mask == label)
        bboxes[int(label)] = [int(cols.min()), int(rows.min()),
                               int(cols.max()), int(rows.max())]
    return bboxes


def render_sequence(seq_key, res_dir: Path, img_dir: Path, frame_files: list,
                    fps: int, scale: int, out_path: Path, line_frames: int):
    res_seq_dir = res_dir / f'{seq_key}_RES'
    if not res_seq_dir.exists():
        print(f"  WARNING: {res_seq_dir} not found, skipping")
        return

    track_info, children = read_res_track(res_seq_dir / 'res_track.txt')

    # Parent labels (tracks that divide) get the yellow M marker.
    dividing_labels = set(children.keys())

    # Division lines: drawn between the two sibling children for line_frames frames
    # starting from the children's first frame.
    div_lines: dict = defaultdict(list)   # frame_t → [(child_a, child_b), ...]
    for par_label, child_labels in children.items():
        if len(child_labels) < 2:
            continue
        birth_t = min(track_info[c]['begin'] for c in child_labels
                      if c in track_info)
        for dt in range(line_frames):
            div_lines[birth_t + dt].append((child_labels[0], child_labels[1]))

    font_scale = max(0.3, scale * 0.07)
    font_thick = max(1, scale // 6)
    cv2_font   = cv2.FONT_HERSHEY_SIMPLEX

    out_path = out_path.with_suffix('.avi')
    fourcc   = cv2.VideoWriter_fourcc(*'MJPG')
    vw       = None

    for t, fname in enumerate(frame_files):
        raw = np.array(Image.open(img_dir / fname).convert('RGB'))
        if raw.ndim == 2:
            raw = np.stack([raw] * 3, axis=-1)
        H, W = raw.shape[:2]
        out_H, out_W = H * scale, W * scale

        if vw is None:
            vw = cv2.VideoWriter(str(out_path), fourcc, fps, (out_W, out_H))
            if not vw.isOpened():
                raise RuntimeError(f'VideoWriter failed: {out_path}')

        img_bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.resize(img_bgr, (out_W, out_H), interpolation=cv2.INTER_NEAREST)

        mask_path = res_seq_dir / f'mask{t:03d}.tif'
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        else:
            mask = np.zeros((H, W), dtype=np.uint16)

        bboxes = extract_bboxes(mask)

        # Scale bboxes to output resolution and build lookup for division lines.
        scaled: dict = {}
        for label, (x1, y1, x2, y2) in bboxes.items():
            sx1 = max(0,        int(round(x1 * scale)))
            sy1 = max(0,        int(round(y1 * scale)))
            sx2 = min(out_W-1,  int(round(x2 * scale)))
            sy2 = min(out_H-1,  int(round(y2 * scale)))
            scaled[label] = (sx1, sy1, sx2, sy2)

        # Draw track boxes and labels.
        for label, (sx1, sy1, sx2, sy2) in scaled.items():
            color   = track_color(label)
            mitotic = label in dividing_labels
            border  = (0, 255, 255) if mitotic else color   # cyan = yellow in BGR
            thick   = font_thick + 1 if mitotic else font_thick
            cv2.rectangle(img_bgr, (sx1, sy1), (sx2, sy2), border, thick)
            text = f'{label}{"  M" if mitotic else ""}'
            cv2.putText(img_bgr, text, (max(sx1, 0), max(sy1 - 2, 8)),
                        cv2_font, font_scale, color, font_thick, cv2.LINE_AA)

        # Draw division lines between sibling children.
        for child_a, child_b in div_lines.get(t, []):
            box1 = scaled.get(child_a)
            box2 = scaled.get(child_b)
            if box1 is None or box2 is None:
                continue
            cx1 = (box1[0] + box1[2]) // 2
            cy1 = (box1[1] + box1[3]) // 2
            cx2 = (box2[0] + box2[2]) // 2
            cy2 = (box2[1] + box2[3]) // 2
            cv2.line(img_bgr, (cx1, cy1), (cx2, cy2),
                     (255, 0, 255), max(2, font_thick), cv2.LINE_AA)
            for bx in (box1, box2):
                cv2.rectangle(img_bgr, (bx[0], bx[1]), (bx[2], bx[3]),
                               (255, 255, 0), font_thick + 1)

        cv2.putText(img_bgr, f't={t:03d}', (2, out_H - 4),
                    cv2_font, font_scale * 0.8, (200, 200, 200), 1, cv2.LINE_AA)
        vw.write(img_bgr)

    if vw:
        vw.release()


def main():
    parser = argparse.ArgumentParser('Visualise CTC inference results')
    parser.add_argument('--res_dir',   default="/srv/home/chen/cell_motr/self/outputs/cell_deepcell_eval",
                        help='eval_cell.py output_dir (contains *_RES/ subdirs)')
    parser.add_argument('--mot_path',  default="/srv/home/chen/Cell-TRACTR/data/deepcell/COCO",
                        help='COCO dataset root (same as eval_cell.py --mot_path)')
    parser.add_argument('--split',     default='val', choices=['train', 'val'])
    parser.add_argument('--video_dir', default=None,
                        help='Where to write AVIs (default: <res_dir>/videos/)')
    parser.add_argument('--fps',         type=int, default=8)
    parser.add_argument('--scale',       type=int, default=1,
                        help='Integer upscale factor for small images')
    parser.add_argument('--line_frames', type=int, default=8,
                        help='Number of frames to draw the sibling division line')
    args = parser.parse_args()

    res_dir   = Path(args.res_dir)
    mot_root  = Path(args.mot_path)
    img_dir   = mot_root / args.split / 'img'
    ann_file  = mot_root / 'annotations' / args.split / 'anno.json'
    video_dir = Path(args.video_dir) if args.video_dir else res_dir / 'videos'
    video_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_file) as f:
        data = json.load(f)

    seq_to_imgs = defaultdict(list)
    for img in data['images']:
        seq_key = img.get('ctc_id', img.get('man_track_id', 'unknown'))
        seq_to_imgs[seq_key].append(img)
    for seq_key in seq_to_imgs:
        seq_to_imgs[seq_key].sort(key=lambda x: x['frame_id'])

    print(f"Visualising {len(seq_to_imgs)} sequences from {res_dir}")
    for seq_key in tqdm(sorted(seq_to_imgs.keys()), desc='sequences'):
        fnames = [img['file_name'] for img in seq_to_imgs[seq_key]]
        out_path = video_dir / f'{seq_key}.avi'
        render_sequence(seq_key, res_dir, img_dir, fnames,
                        fps=args.fps, scale=args.scale,
                        out_path=out_path, line_frames=args.line_frames)
        print(f"  seq {seq_key}: {len(fnames)} frames → {out_path}")


if __name__ == '__main__':
    main()
