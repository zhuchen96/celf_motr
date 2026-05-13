"""
Diagnostic: how does the tracker's confidence score evolve for cells that are
about to divide, compared with cells that don't?

For every GT birth event (a track_id that first appears after frame 0) in the
validation set, the script:
  1. Identifies the GT "mother" — the cell present at T-1 whose box is closest
     to the daughters at T and that disappears at T.
  2. Finds the matching model track at T-1 via IoU.
  3. Records that track's score at offsets T-W … T+W.

Control group: for each division event, the non-dividing tracks present in the
same frame window are recorded too.

Output
------
  • Per-offset statistics printed to stdout (mean ± std, alive-fraction)
  • Optional CSV saved to --out_csv for further plotting

Usage:
    python diag_division_scores.py \
        --config configs/CellTracking/deepcell/infer_no_div.yaml \
        --split val [--window 8] [--out_csv scores.csv]
"""

# ── registry patches (identical to eval_cell_no_div.py) ──────────────────────
import models as _models
from models.motrv2_self_no_div import build as _build_no_div
from models.motrv2_self import RuntimeTrackerBase

_orig_build_model = _models.build_model
def _patched_build_model(args):
    if getattr(args, 'meta_arch', '') == 'motrv2_self_no_div':
        return _build_no_div(args)
    return _orig_build_model(args)
_models.build_model = _patched_build_model

import datasets as _datasets
from datasets.ctc_cell_no_div import build as _build_no_div_dataset
_orig_build_dataset = _datasets.build_dataset
def _patched_build_dataset(image_set, args):
    if getattr(args, 'dataset_file', '') == 'e2e_cell_no_div':
        return _build_no_div_dataset(image_set, args)
    return _orig_build_dataset(image_set, args)
_datasets.build_dataset = _patched_build_dataset
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from train_cell import get_args_parser
from util.box_ops import box_cxcywh_to_xyxy, box_iou
from util.tool import apply_checkpoint_model_args, load_model, load_torch_checkpoint

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
IOU_MATCH_THRESH = 0.3   # min IoU to call a model track "the mother"


def load_and_preprocess(img_path: Path):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    t = TF.normalize(TF.to_tensor(img), MEAN, STD)
    return t.unsqueeze(0), (h, w)


# ── GT helpers ────────────────────────────────────────────────────────────────

def build_gt_structures(data):
    """
    Returns
    -------
    seq_to_imgs : seq_key → sorted list of image dicts
    gt_by_img   : image_id → list of {track_id, box_xyxy}
    """
    gt_by_img = defaultdict(list)
    for ann in data['annotations']:
        if ann.get('empty', False):
            continue
        x, y, w, h = ann['bbox']
        gt_by_img[ann['image_id']].append({
            'track_id': int(ann['track_id']),
            'box': [x, y, x + w, y + h],
        })

    seq_to_imgs = defaultdict(list)
    for img in data['images']:
        key = img.get('ctc_id', img.get('man_track_id', 'unknown'))
        seq_to_imgs[key].append(img)
    for key in seq_to_imgs:
        seq_to_imgs[key].sort(key=lambda x: x['id'])

    return seq_to_imgs, gt_by_img


def find_birth_events(imgs, gt_by_img):
    """
    Scan a sequence for frames where new track_ids appear after frame 0.

    Returns list of dicts:
        frame_pos  : int   index in imgs where daughters first appear
        daughters  : list of {track_id, box_xyxy}
        mother_box : [x1,y1,x2,y2] | None  (GT box closest to daughters at T-1)
        mother_tid : int | None
    """
    seen_tids = set()
    events = []

    for frame_pos, img_meta in enumerate(imgs):
        img_id = img_meta['id']
        entries = gt_by_img[img_id]
        cur_tids = {e['track_id'] for e in entries}
        new_tids = cur_tids - seen_tids

        if frame_pos > 0 and new_tids:
            daughters = [e for e in entries if e['track_id'] in new_tids]

            # Find which cell in T-1 "became" these daughters.
            # The mother is the T-1 cell whose centre is closest to the
            # daughters' combined centroid AND that disappears at T.
            prev_id = imgs[frame_pos - 1]['id']
            prev_entries = gt_by_img[prev_id]
            prev_tids = {e['track_id'] for e in prev_entries}
            disappeared = prev_tids - cur_tids

            mother_box = None
            mother_tid = None
            if disappeared:
                # daughters' centroid
                dcx = np.mean([(e['box'][0] + e['box'][2]) / 2 for e in daughters])
                dcy = np.mean([(e['box'][1] + e['box'][3]) / 2 for e in daughters])
                best_dist = float('inf')
                for pe in prev_entries:
                    if pe['track_id'] not in disappeared:
                        continue
                    bx = (pe['box'][0] + pe['box'][2]) / 2
                    by = (pe['box'][1] + pe['box'][3]) / 2
                    dist = ((bx - dcx) ** 2 + (by - dcy) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        mother_box = pe['box']
                        mother_tid = pe['track_id']

            events.append({
                'frame_pos': frame_pos,
                'daughters': daughters,
                'mother_box': mother_box,
                'mother_tid': mother_tid,
            })

        seen_tids.update(cur_tids)

    return events


# ── Tracker run ───────────────────────────────────────────────────────────────

@torch.no_grad()
def run_tracker_record(model, img_dir, imgs, proposal_threshold,
                       update_score_threshold, reuse_encoder_cache, device):
    """
    Run the tracker on one sequence and record per-frame state for every active
    track query.

    Returns
    -------
    frame_records : list of dicts  (one per frame)
        Each dict maps obj_idx (int) → {'score': float, 'box': [x1,y1,x2,y2]}
    """
    model.clear()
    model.track_base.score_thresh        = update_score_threshold
    model.track_base.filter_score_thresh = update_score_threshold

    track_instances = None
    frame_records = []

    for img_meta in imgs:
        img_path = img_dir / img_meta['file_name']
        img_tensor, ori_size = load_and_preprocess(img_path)
        img_tensor = img_tensor.to(device)

        if reuse_encoder_cache:
            proposals, memory, spatial_shapes, level_start_index, \
                valid_ratios, mask_flatten = \
                model.inference_single_image_proposals_light_light(
                    img_tensor, ori_size, score_threshold=proposal_threshold)
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

        # Record ALL queries that have an assigned obj_idx (alive or dying)
        record = {}
        for i in range(len(track_instances)):
            oid = track_instances.obj_idxes[i].item()
            if oid < 0:
                continue
            record[oid] = {
                'score': track_instances.scores[i].item(),
                'box':   track_instances.boxes[i].tolist(),
            }
        frame_records.append(record)

    return frame_records


# ── Matching ──────────────────────────────────────────────────────────────────

def find_mother_track(frame_records, birth_frame_pos, mother_box_gt, device):
    """
    At frame T-1 (birth_frame_pos - 1), find the model track with the highest
    IoU against mother_box_gt.  Returns (obj_idx, iou) or (None, 0).
    """
    if birth_frame_pos == 0:
        return None, 0.0
    prev_record = frame_records[birth_frame_pos - 1]
    if not prev_record:
        return None, 0.0

    gt_box = torch.tensor([mother_box_gt], dtype=torch.float32, device=device)
    pred_boxes = torch.tensor(
        [v['box'] for v in prev_record.values()], dtype=torch.float32, device=device)
    obj_ids = list(prev_record.keys())

    iou, _ = box_iou(pred_boxes, gt_box)   # [N, 1]
    iou = iou[:, 0]
    best_idx = iou.argmax().item()
    best_iou = iou[best_idx].item()

    if best_iou < IOU_MATCH_THRESH:
        return None, best_iou
    return obj_ids[best_idx], best_iou


def extract_score_trajectory(frame_records, obj_idx, birth_frame_pos, window):
    """
    Extract the score of obj_idx at offsets -window … +window relative to birth_frame_pos.
    Returns dict {offset: score | None}.
    """
    n = len(frame_records)
    traj = {}
    for offset in range(-window, window + 1):
        t = birth_frame_pos + offset
        if t < 0 or t >= n:
            traj[offset] = None
        elif obj_idx in frame_records[t]:
            traj[offset] = frame_records[t][obj_idx]['score']
        else:
            traj[offset] = None   # track no longer in record (killed or not yet born)
    return traj


def extract_control_scores(frame_records, birth_frame_pos, mother_obj_idx, window):
    """
    For the same frame window, collect scores of all OTHER tracks (not the mother).
    Returns list of (offset, score) pairs.
    """
    n = len(frame_records)
    pairs = []
    for offset in range(-window, window + 1):
        t = birth_frame_pos + offset
        if t < 0 or t >= n:
            continue
        for oid, info in frame_records[t].items():
            if oid != mother_obj_idx:
                pairs.append((offset, info['score']))
    return pairs


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    checkpoint = load_torch_checkpoint(args.resume, map_location='cpu',
                                       weights_only=False)
    args = apply_checkpoint_model_args(args, checkpoint, context='diag_division_scores')

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

    seq_to_imgs, gt_by_img = build_gt_structures(data)
    window = args.window

    # Accumulators keyed by offset
    div_scores   = defaultdict(list)   # offset → [scores] for mother tracks
    ctrl_scores  = defaultdict(list)   # offset → [scores] for non-mother tracks

    n_events_total  = 0
    n_events_matched = 0

    for seq_key in tqdm(sorted(seq_to_imgs.keys()), desc='sequences'):
        imgs = seq_to_imgs[seq_key]
        birth_events = find_birth_events(imgs, gt_by_img)
        if not birth_events:
            continue

        frame_records = run_tracker_record(
            model, img_dir, imgs,
            proposal_threshold=args.proposal_threshold,
            update_score_threshold=args.update_score_threshold,
            reuse_encoder_cache=args.reuse_encoder_cache,
            device=device,
        )

        for ev in birth_events:
            fp     = ev['frame_pos']
            m_box  = ev['mother_box']
            n_events_total += 1

            if m_box is None:
                continue

            obj_idx, match_iou = find_mother_track(frame_records, fp, m_box, device)
            if obj_idx is None:
                continue

            n_events_matched += 1
            traj = extract_score_trajectory(frame_records, obj_idx, fp, window)
            for offset, score in traj.items():
                if score is not None:
                    div_scores[offset].append(score)

            ctrl = extract_control_scores(frame_records, fp, obj_idx, window)
            for offset, score in ctrl:
                ctrl_scores[offset].append(score)

    # ── Report ────────────────────────────────────────────────────────────────
    filter_thresh = args.update_score_threshold

    print(f'\n{"="*65}')
    print(f'Division score trajectory  (split={args.split})')
    print(f'  GT birth events  : {n_events_total}')
    print(f'  Matched to model : {n_events_matched}  (IoU≥{IOU_MATCH_THRESH})')
    print(f'  Window           : ±{window} frames around birth frame T')
    print(f'  Filter threshold : {filter_thresh}')
    print(f'{"="*65}')
    print(f'{"Offset":>8}  {"Mother mean":>12}  {"Mother std":>10}  '
          f'{"Alive%":>7}  {"N":>5}  ||  '
          f'{"Ctrl mean":>10}  {"Ctrl std":>9}  {"N":>5}')
    print(f'{"-"*65}')

    csv_rows = []
    for offset in range(-window, window + 1):
        ds = div_scores.get(offset, [])
        cs = ctrl_scores.get(offset, [])
        label = f'T{offset:+d}' if offset != 0 else 'T (birth)'

        if ds:
            dm = np.mean(ds)
            dd = np.std(ds)
            alive_frac = np.mean([s >= filter_thresh for s in ds]) * 100
        else:
            dm = dd = alive_frac = float('nan')

        cm = np.mean(cs) if cs else float('nan')
        cd = np.std(cs)  if cs else float('nan')

        print(f'{label:>8}  {dm:>12.4f}  {dd:>10.4f}  '
              f'{alive_frac:>6.1f}%  {len(ds):>5}  ||  '
              f'{cm:>10.4f}  {cd:>9.4f}  {len(cs):>5}')

        csv_rows.append({
            'offset': offset,
            'div_mean': dm, 'div_std': dd, 'div_alive_pct': alive_frac, 'div_n': len(ds),
            'ctrl_mean': cm, 'ctrl_std': cd, 'ctrl_n': len(cs),
        })

    print(f'{"="*65}')
    print()
    print('Interpretation guide:')
    print('  • A sharp drop in Mother mean at T or T-1 → score drop is a reliable gate')
    print('  • Gradual drop over many frames → score drop triggers too early (ghost kills)')
    print('  • Ctrl mean stays high → score drop is specific to division / exit events')

    if args.out_csv:
        import csv
        with open(args.out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            w.writeheader()
            w.writerows(csv_rows)
        print(f'\nCSV saved to {args.out_csv}')


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
        'Division score diagnostic', parents=[get_args_parser()])
    parser.add_argument('--split',    default='val', choices=['train', 'val'])
    parser.add_argument('--window',   default=8, type=int,
                        help='Frames before/after birth to inspect (default 8)')
    parser.add_argument('--out_csv',  default=None,
                        help='Optional path to save results as CSV')
    parser.add_argument('--proposal_threshold',      default=0.05, type=float)
    parser.add_argument('--update_score_threshold',  default=0.3,  type=float)
    parser.add_argument('--miss_tolerance',          default=5,    type=int)
    # kept for apply_checkpoint_model_args compatibility
    parser.add_argument('--gap_close_frames',        default=5,    type=int)
    parser.add_argument('--gap_close_dist_factor',   default=1.5,  type=float)
    parser.add_argument('--div_score_thresh',        default=0.5,  type=float)
    parser.add_argument('--div_threshold',           default=0.5,  type=float)
    parser.add_argument('--max_div_dist_factor',     default=2.0,  type=float)

    if pre_args.config is not None:
        _apply_yaml_defaults(parser, pre_args.config)

    args = parser.parse_args(remaining)
    main(args)
