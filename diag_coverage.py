"""
Track coverage diagnostic for SelfMOTR.

For every GT track, measures what fraction of its frames are covered by a
predicted track (pixel IoU >= iou_thr), then classifies each track as:

  full    — covered in >= full_thr of its frames with no long gap
  dropout — detected at some point but has a run of > gap_frames consecutive
             missed frames (track was found then lost)
  late    — detected eventually but first seen > late_frames after GT begin
  partial — some detection but below full_thr, no long gap, not clearly late
  never   — 0 frames covered

Usage:
    conda run -n cell-tractr python diag_coverage.py
    conda run -n cell-tractr python diag_coverage.py \\
        --gt_dir  /srv/home/chen/Cell-TRACTR/data/moma/CTC/val \\
        --res_dir /srv/home/chen/cell_motr/self/outputs/cell_moma_eval \\
        --iou_thr 0.3 --full_thr 0.8 --gap_frames 5 --late_frames 5
"""

import argparse
import numpy as np
import cv2
from collections import defaultdict
from pathlib import Path


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def parse_track_file(path: Path):
    tracks = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            label, begin, end = int(parts[0]), int(parts[1]), int(parts[2])
            tracks[label] = (begin, end)
    return tracks


def load_mask(path: Path):
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def covered(gt_label: int, gt_mask, pred_mask, iou_thr: float) -> bool:
    """True if gt_label overlaps any predicted label at IoU >= iou_thr."""
    if gt_mask is None or pred_mask is None:
        return False
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    gt_px = gt_mask == gt_label
    if not gt_px.any():
        return False
    for pl in np.unique(pred_mask[gt_px]):
        if pl == 0:
            continue
        pred_px = pred_mask == pl
        inter = int((gt_px & pred_px).sum())
        union = int((gt_px | pred_px).sum())
        if union > 0 and inter / union >= iou_thr:
            return True
    return False


def classify(detected: list, gap_frames: int, late_frames: int, full_thr: float):
    """
    Returns (category, coverage, first_det_delay, max_gap).
    detected[i] = True/False for frame i relative to GT begin.
    """
    n = len(detected)
    coverage = sum(detected) / n if n > 0 else 0.0

    if coverage == 0.0:
        return 'never', 0.0, None, 0

    first_det = next(i for i, d in enumerate(detected) if d)
    max_gap, cur_gap = 0, 0
    for d in detected:
        cur_gap = 0 if d else cur_gap + 1
        max_gap = max(max_gap, cur_gap)

    if max_gap > gap_frames:
        cat = 'dropout'
    elif first_det > late_frames:
        cat = 'late'
    elif coverage >= full_thr:
        cat = 'full'
    else:
        cat = 'partial'

    return cat, coverage, first_det, max_gap


# --------------------------------------------------------------------------- #
# Per-sequence evaluation                                                       #
# --------------------------------------------------------------------------- #

def evaluate_sequence(seq_key, gt_dir: Path, res_dir: Path,
                       iou_thr, gap_frames, late_frames, full_thr):
    gt_tra  = gt_dir  / f'{seq_key}_GT' / 'TRA'
    res_seq = res_dir / f'{seq_key}_RES'

    if not (gt_tra / 'man_track.txt').exists():
        return []
    if not res_seq.exists():
        print(f'  [{seq_key}] WARNING: no RES dir, skipping')
        return []

    gt_tracks   = parse_track_file(gt_tra / 'man_track.txt')
    pred_tracks = parse_track_file(res_seq / 'res_track.txt') if (res_seq / 'res_track.txt').exists() else {}

    # Cache masks by frame (union of GT and pred frame ranges).
    gt_cache:   dict = {}
    pred_cache: dict = {}

    all_frames = set()
    for begin, end in gt_tracks.values():
        all_frames.update(range(begin, end + 1))
    for begin, end in pred_tracks.values():
        all_frames.update(range(begin, end + 1))
    for t in all_frames:
        gt_cache[t]   = load_mask(gt_tra  / f'man_track{t:03d}.tif')
        pred_cache[t] = load_mask(res_seq / f'mask{t:03d}.tif')

    # --- GT → pred (recall) ---
    gt_rows = []
    for label, (begin, end) in gt_tracks.items():
        detected = [
            covered(label, gt_cache[t], pred_cache[t], iou_thr)
            for t in range(begin, end + 1)
        ]
        cat, cov, first_det, max_gap = classify(detected, gap_frames, late_frames, full_thr)
        gt_rows.append({
            'seq':       seq_key,
            'label':     label,
            'n_frames':  end - begin + 1,
            'coverage':  cov,
            'category':  cat,
            'first_det': first_det,
            'max_gap':   max_gap,
        })

    # --- pred → GT (precision / false positives) ---
    # For each predicted track, count frames where it overlaps any GT cell.
    fp_rows = []
    for label, (begin, end) in pred_tracks.items():
        matched = [
            covered(label, pred_cache[t], gt_cache[t], iou_thr)
            for t in range(begin, end + 1)
        ]
        cov = sum(matched) / len(matched) if matched else 0.0
        fp_rows.append({
            'seq':      seq_key,
            'label':    label,
            'n_frames': end - begin + 1,
            'gt_coverage': cov,
            'spurious': cov == 0.0,
            'mostly_spurious': cov < 0.3,
        })

    return gt_rows, fp_rows


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser('Track coverage diagnostic')
    parser.add_argument('--gt_dir',     default='/srv/home/chen/Cell-TRACTR/data/deepcell/CTC/val')
    parser.add_argument('--res_dir',    default='/srv/home/chen/cell_motr/self/outputs/cell_deepcell_self_eval')
    parser.add_argument('--iou_thr',    type=float, default=0.3,
                        help='Min IoU to count a frame as covered')
    parser.add_argument('--full_thr',   type=float, default=0.8,
                        help='Min coverage fraction to count as "full"')
    parser.add_argument('--gap_frames', type=int,   default=5,
                        help='Consecutive missed frames that counts as a dropout gap')
    parser.add_argument('--late_frames',type=int,   default=5,
                        help='Frames after GT begin before detection = "late"')
    args = parser.parse_args()

    gt_dir  = Path(args.gt_dir)
    res_dir = Path(args.res_dir)

    seq_keys = sorted(
        d.name.replace('_GT', '')
        for d in gt_dir.iterdir()
        if d.is_dir() and d.name.endswith('_GT')
    )

    all_gt_rows = []
    all_fp_rows = []
    for seq in seq_keys:
        result = evaluate_sequence(seq, gt_dir, res_dir,
                                   args.iou_thr, args.gap_frames,
                                   args.late_frames, args.full_thr)
        if not result:
            continue
        gt_rows, fp_rows = result

        counts = defaultdict(int)
        for r in gt_rows:
            counts[r['category']] += 1
        total  = len(gt_rows)
        n_spur = sum(r['spurious'] for r in fp_rows)
        cats   = ['full', 'dropout', 'late', 'partial', 'never']
        parts  = '  '.join(f"{c}={counts[c]}({100*counts[c]//total}%)" for c in cats)
        print(f'  seq {seq}: GT={total:3d} pred={len(fp_rows):3d} spurious={n_spur} | {parts}')
        all_gt_rows.extend(gt_rows)
        all_fp_rows.extend(fp_rows)

    if not all_gt_rows:
        print('No GT tracks found.')
        return

    # ------------------------------------------------------------------ #
    # Recall summary                                                        #
    # ------------------------------------------------------------------ #
    n = len(all_gt_rows)
    counts = defaultdict(int)
    for r in all_gt_rows:
        counts[r['category']] += 1

    coverages  = [r['coverage']  for r in all_gt_rows]
    first_dets = [r['first_det'] for r in all_gt_rows if r['first_det'] is not None]
    max_gaps   = [r['max_gap']   for r in all_gt_rows if r['category'] == 'dropout']

    print()
    print('=' * 64)
    print(f'  GT track coverage  (iou≥{args.iou_thr}, full≥{args.full_thr:.0%},'
          f' gap>{args.gap_frames}fr, late>{args.late_frames}fr)')
    print('-' * 64)
    print(f'  Total GT tracks              : {n}')
    for cat, label in [
        ('full',    'Fully covered  (≥80%, no gap)'),
        ('dropout', 'Dropout        (detected, then lost)'),
        ('late',    'Late start     (>5 fr delay)'),
        ('partial', 'Partial        (some detection)'),
        ('never',   'Never detected (0% coverage)'),
    ]:
        c = counts[cat]
        print(f'  {label:38s}: {c:5d}  ({100*c/n:5.1f}%)')
    print('-' * 64)
    print(f'  Mean  coverage per track     : {np.mean(coverages):.1%}')
    print(f'  Median coverage per track    : {np.median(coverages):.1%}')
    if first_dets:
        print(f'  Median first-detection delay : {np.median(first_dets):.0f} frames')
    if max_gaps:
        print(f'  Median dropout gap length    : {np.median(max_gaps):.0f} frames')

    # Short-track vs long-track breakdown
    short = [r for r in all_gt_rows if r['n_frames'] <= 5]
    long  = [r for r in all_gt_rows if r['n_frames'] >  5]
    if short and long:
        print(f'  Short tracks (≤5 fr, n={len(short)}): '
              f'never={sum(r["category"]=="never" for r in short)}  '
              f'full={sum(r["category"]=="full" for r in short)}')
        print(f'  Long  tracks (>5 fr, n={len(long)}):  '
              f'never={sum(r["category"]=="never" for r in long)}  '
              f'full={sum(r["category"]=="full" for r in long)}')

    # ------------------------------------------------------------------ #
    # Precision / false positive summary                                    #
    # ------------------------------------------------------------------ #
    np_ = len(all_fp_rows)
    n_spurious      = sum(r['spurious']        for r in all_fp_rows)
    n_mostly_spur   = sum(r['mostly_spurious'] for r in all_fp_rows)
    gt_covs         = [r['gt_coverage'] for r in all_fp_rows]

    print()
    print('=' * 64)
    print('  Predicted track quality  (how much of each pred track matches GT)')
    print('-' * 64)
    print(f'  Total predicted tracks       : {np_}')
    print(f'  Fully spurious  (0% GT)      : {n_spurious:5d}  ({100*n_spurious/np_:5.1f}%)')
    print(f'  Mostly spurious (<30% GT)    : {n_mostly_spur:5d}  ({100*n_mostly_spur/np_:5.1f}%)')
    print(f'  Mean  GT coverage per pred   : {np.mean(gt_covs):.1%}')
    print(f'  Median GT coverage per pred  : {np.median(gt_covs):.1%}')

    # Spurious track length histogram
    spur_rows = [r for r in all_fp_rows if r['spurious']]
    if spur_rows:
        lengths = [r['n_frames'] for r in spur_rows]
        bins = [(1,1),(2,2),(3,5),(6,10),(11,20),(21,999)]
        print('-' * 64)
        print('  Spurious track length distribution:')
        for lo, hi in bins:
            n = sum(lo <= l <= hi for l in lengths)
            label = f'{lo}' if lo == hi else (f'{lo}-{hi}' if hi < 999 else f'{lo}+')
            print(f'    {label:>6} frames : {n:5d}  ({100*n/len(lengths):5.1f}%)')
        print(f'    mean length : {np.mean(lengths):.1f} frames')
    print('=' * 64)


if __name__ == '__main__':
    main()
