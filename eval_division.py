"""
Division-specific evaluation for SelfMOTR cell tracking.

For every GT division event, loads the GT and predicted masks at the
daughters' birth frame and checks — via pixel IoU — whether each daughter
is present in the prediction.

Reports per-sequence and aggregate statistics:
  - GT divisions found / missed
  - Fraction where BOTH daughters are detected
  - Fraction where only ONE is detected (D2 usually the missing one)
  - Whether each detected daughter carries a parent link in the prediction

Usage:
    conda run -n selfmotr python eval_division.py \\
        --gt_dir  /srv/home/chen/Cell-TRACTR/data/deepcell/CTC/val \\
        --res_dir /srv/home/chen/cell_motr/self/outputs/cell_deepcell_eval \\
        --iou_thr 0.3
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def parse_track_file(path: Path):
    """
    Parse a CTC track file (man_track.txt or res_track.txt).

    Returns
    -------
    tracks   : dict  label → {'begin': int, 'end': int, 'parent': int}
    children : dict  parent_label → [child_label, ...]
    """
    tracks   = {}
    children = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            label, begin, end, par = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            tracks[label] = {'begin': begin, 'end': end, 'parent': par}
            if par > 0:
                children[par].append(label)
    return tracks, dict(children)


def load_mask(path: Path):
    """Load a uint16 mask TIF; return None if missing."""
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return mask


def mask_iou(gt_mask, gt_label, pred_mask):
    """
    Compute IoU between a single GT label region and every predicted label.

    Returns a dict  pred_label → IoU  for all pred_labels that overlap.
    """
    gt_pixels = gt_mask == gt_label
    gt_area   = int(gt_pixels.sum())
    if gt_area == 0:
        return {}

    overlapping_labels = np.unique(pred_mask[gt_pixels])
    result = {}
    for pl in overlapping_labels:
        if pl == 0:
            continue
        pred_pixels  = pred_mask == pl
        intersection = int((gt_pixels & pred_pixels).sum())
        union        = int((gt_pixels | pred_pixels).sum())
        result[int(pl)] = intersection / union if union > 0 else 0.0
    return result


def best_match(gt_mask, gt_label, pred_mask, iou_thr):
    """
    Return (best_pred_label, best_iou) for the predicted label with the
    highest IoU against gt_label, or (None, 0.0) if nothing exceeds iou_thr.
    """
    ious = mask_iou(gt_mask, gt_label, pred_mask)
    if not ious:
        return None, 0.0
    best_pl, best_iou = max(ious.items(), key=lambda kv: kv[1])
    if best_iou >= iou_thr:
        return best_pl, best_iou
    return None, best_iou


# --------------------------------------------------------------------------- #
# Per-sequence evaluation                                                       #
# --------------------------------------------------------------------------- #

def evaluate_sequence(seq_key, gt_dir: Path, res_dir: Path, iou_thr: float):
    """
    Returns a list of per-division-event dicts:
      {
        'parent_gt'     : int,   GT parent label
        'birth_t'       : int,   frame where daughters first appear
        'n_gt_daughters': int,   number of GT daughters (should be 2)
        'd1_detected'   : bool,
        'd2_detected'   : bool,
        'd1_has_parent_link': bool,   pred track for D1 carries a parent link
        'd2_has_parent_link': bool,
        'd1_track_len'  : int or None,   length of matched pred track (frames)
        'd2_track_len'  : int or None,
        'd1_iou'        : float,
        'd2_iou'        : float,
      }
    """
    gt_tra_dir  = gt_dir   / f'{seq_key}_GT' / 'TRA'
    res_seq_dir = res_dir  / f'{seq_key}_RES'

    gt_track_file  = gt_tra_dir  / 'man_track.txt'
    res_track_file = res_seq_dir / 'res_track.txt'

    if not gt_track_file.exists():
        return []
    if not res_track_file.exists():
        print(f"  [{seq_key}] WARNING: no res_track.txt found, skipping")
        return []

    gt_tracks,  gt_children  = parse_track_file(gt_track_file)
    res_tracks, res_children = parse_track_file(res_track_file)

    events = []
    for par_label, child_labels in gt_children.items():
        n_daughters = len(child_labels)
        # Use the earlier birth frame (daughters typically start the same frame)
        birth_frames = [gt_tracks[c]['begin'] for c in child_labels if c in gt_tracks]
        if not birth_frames:
            continue
        birth_t = min(birth_frames)

        gt_mask_path  = gt_tra_dir  / f'man_track{birth_t:03d}.tif'
        res_mask_path = res_seq_dir / f'mask{birth_t:03d}.tif'

        gt_mask  = load_mask(gt_mask_path)
        res_mask = load_mask(res_mask_path)

        ev = {
            'parent_gt'          : par_label,
            'birth_t'            : birth_t,
            'n_gt_daughters'     : n_daughters,
            'd1_detected'        : False,
            'd2_detected'        : False,
            'd1_has_parent_link' : False,
            'd2_has_parent_link' : False,
            'd1_track_len'       : None,
            'd2_track_len'       : None,
            'd1_iou'             : 0.0,
            'd2_iou'             : 0.0,
        }

        if gt_mask is None or res_mask is None:
            events.append(ev)
            continue

        # Match each GT daughter to the best-overlapping predicted track.
        matched_pred_labels = set()
        for idx, gt_child in enumerate(child_labels[:2]):
            if gt_child not in gt_tracks:
                continue
            pred_label, iou = best_match(gt_mask, gt_child, res_mask, iou_thr)

            # Avoid assigning the same predicted track to both daughters.
            if pred_label in matched_pred_labels:
                pred_label, iou = None, iou

            key = 'd1' if idx == 0 else 'd2'
            ev[f'{key}_iou'] = iou
            if pred_label is not None:
                matched_pred_labels.add(pred_label)
                ev[f'{key}_detected'] = True
                ev[f'{key}_has_parent_link'] = (
                    res_tracks.get(pred_label, {}).get('parent', 0) > 0
                )
                if pred_label in res_tracks:
                    info = res_tracks[pred_label]
                    ev[f'{key}_track_len'] = info['end'] - info['begin'] + 1

        events.append(ev)

    return events


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser('Division-specific evaluation')
    parser.add_argument('--gt_dir',  default='/srv/home/chen/Cell-TRACTR/data/deepcell/CTC/val',
                        help='CTC ground-truth directory containing {seq}_GT/TRA/')
    parser.add_argument('--res_dir', default='/srv/home/chen/cell_motr/self/outputs/cell_deepcell_eval',
                        help='eval_cell.py output directory containing {seq}_RES/')
    parser.add_argument('--iou_thr', type=float, default=0.3,
                        help='Minimum IoU to count a daughter as detected (default 0.3)')
    args = parser.parse_args()

    gt_dir  = Path(args.gt_dir)
    res_dir = Path(args.res_dir)

    seq_keys = sorted(
        d.name.replace('_GT', '')
        for d in gt_dir.iterdir()
        if d.is_dir() and d.name.endswith('_GT')
    )

    all_events = []
    for seq_key in seq_keys:
        evs = evaluate_sequence(seq_key, gt_dir, res_dir, args.iou_thr)
        if evs:
            both   = sum(e['d1_detected'] and e['d2_detected'] for e in evs)
            only1  = sum(e['d1_detected'] and not e['d2_detected'] for e in evs)
            only2  = sum(not e['d1_detected'] and e['d2_detected'] for e in evs)
            neither= sum(not e['d1_detected'] and not e['d2_detected'] for e in evs)
            print(f"  seq {seq_key}: {len(evs):3d} GT divisions | "
                  f"both={both:3d}  only-D1={only1:3d}  only-D2={only2:3d}  neither={neither:3d}")
        all_events.extend(evs)

    if not all_events:
        print("No GT division events found.")
        return

    n   = len(all_events)
    d1  = sum(e['d1_detected'] for e in all_events)
    d2  = sum(e['d2_detected'] for e in all_events)
    both= sum(e['d1_detected'] and e['d2_detected'] for e in all_events)

    d1_link = sum(e['d1_has_parent_link'] for e in all_events if e['d1_detected'])
    d2_link = sum(e['d2_has_parent_link'] for e in all_events if e['d2_detected'])

    d1_lens = [e['d1_track_len'] for e in all_events if e['d1_track_len'] is not None]
    d2_lens = [e['d2_track_len'] for e in all_events if e['d2_track_len'] is not None]

    print()
    print("=" * 60)
    print(f"  GT division events  : {n}")
    print(f"  D1 detected         : {d1}/{n}  ({100*d1/n:.1f}%)")
    print(f"  D2 detected         : {d2}/{n}  ({100*d2/n:.1f}%)")
    print(f"  Both detected       : {both}/{n}  ({100*both/n:.1f}%)")
    if d1 > 0:
        print(f"  D1 has parent link  : {d1_link}/{d1}  ({100*d1_link/d1:.1f}%)")
    if d2 > 0:
        print(f"  D2 has parent link  : {d2_link}/{d2}  ({100*d2_link/d2:.1f}%)")
    if d1_lens:
        print(f"  D1 median track len : {np.median(d1_lens):.0f} frames")
    if d2_lens:
        print(f"  D2 median track len : {np.median(d2_lens):.0f} frames")
    print("=" * 60)
    print()
    print("Note: 'has parent link' = the matched predicted track carries a")
    print("parent pointer in res_track.txt (model correctly identified it")
    print("as a daughter, not just a coincidental detection at that location).")


if __name__ == '__main__':
    main()
