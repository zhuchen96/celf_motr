"""
Detection-only evaluation for the no-division SelfMOTR model.

For each frame the detection head (500 queries) is run independently — no
track queries, no temporal association.  Detected boxes are matched against
GT with IoU ≥ iou_threshold and precision/recall are reported.

Birth-frame analysis separately answers:
  "Were both daughter cells visible to the detector at their first frame?"

Usage:
    python eval_cell_detect_only.py \
        --config configs/CellTracking/deepcell/infer_no_div.yaml \
        [--split train|val] \
        [--proposal_threshold 0.05] \
        [--nms_threshold 0.5] \
        [--iou_threshold 0.5]
"""

# ---- model/dataset registry patches (same as eval_cell_no_div.py) --------
import models as _models
from models.motrv2_self_no_div import build as _build_no_div
from models.motrv2_self import RuntimeTrackerBase  # noqa: F401

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
# --------------------------------------------------------------------------

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torchvision.ops as tvops
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

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


@torch.no_grad()
def detect_frame(model, img_path: Path, proposal_threshold: float,
                 nms_threshold: float, device: torch.device):
    """
    Run detection head only on one image.

    Returns
    -------
    boxes_xyxy : Tensor [N, 4]  absolute pixel coords
    scores     : Tensor [N]
    img_size   : (H, W)
    """
    img_tensor, (H, W) = load_and_preprocess(img_path)
    img_tensor = img_tensor.to(device)

    proposals, *_ = model.inference_single_image_proposals_light_light(
        img_tensor, (H, W), score_threshold=proposal_threshold)
    # proposals: [N, 5]  cx cy w h score  (normalised)

    if proposals.shape[0] == 0:
        empty = torch.zeros(0, 4, device=device)
        return empty, torch.zeros(0, device=device), (H, W)

    boxes_norm = proposals[:, :4]       # cxcywh in [0,1]
    scores     = proposals[:, 4]

    scale = torch.tensor([W, H, W, H], dtype=torch.float32, device=device)
    boxes_xyxy = box_cxcywh_to_xyxy(boxes_norm) * scale

    # NMS to remove duplicates from the detection head
    keep = tvops.nms(boxes_xyxy, scores, nms_threshold)
    return boxes_xyxy[keep], scores[keep], (H, W)


def match_boxes(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor,
                iou_threshold: float):
    """
    Greedy matching: each GT can be claimed at most once.

    Returns
    -------
    matched_gt : set of GT indices that were matched
    matched_pred : set of pred indices that were matched
    """
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return set(), set()

    iou, _ = box_iou(pred_boxes, gt_boxes)   # [P, G]
    matched_gt = set()
    matched_pred = set()

    # sort by descending IoU so best pairs are claimed first
    flat = iou.flatten()
    order = flat.argsort(descending=True)
    for idx in order.tolist():
        p = idx // gt_boxes.shape[0]
        g = idx  % gt_boxes.shape[0]
        if iou[p, g].item() < iou_threshold:
            break
        if p in matched_pred or g in matched_gt:
            continue
        matched_pred.add(p)
        matched_gt.add(g)

    return matched_gt, matched_pred


def main(args):
    checkpoint = load_torch_checkpoint(args.resume, map_location='cpu',
                                       weights_only=False)
    args = apply_checkpoint_model_args(args, checkpoint, context='eval_cell_detect_only')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, _, _ = _models.build_model(args)
    model = load_model(model, args.resume)
    model.eval()
    model.to(device)

    mot_root = Path(args.mot_path)
    ann_file = mot_root / 'annotations' / args.split / 'anno.json'
    img_dir  = mot_root / args.split / 'img'

    with open(ann_file) as f:
        data = json.load(f)

    # --- build per-image GT lookup ---
    gt_by_img = defaultdict(list)   # image_id → list of [x1,y1,x2,y2]
    for ann in data['annotations']:
        if ann.get('empty', False):
            continue
        x, y, w, h = ann['bbox']
        gt_by_img[ann['image_id']].append({
            'box': [x, y, x + w, y + h],
            'track_id': int(ann['track_id']),
        })

    # --- build sequences ---
    seq_to_imgs = defaultdict(list)
    for img in data['images']:
        key = img.get('ctc_id', img.get('man_track_id', 'unknown'))
        seq_to_imgs[key].append(img)
    for key in seq_to_imgs:
        seq_to_imgs[key].sort(key=lambda x: x['id'])

    # --- per-frame accumulators ---
    total_tp = total_fp = total_fn = 0

    # birth-event accumulators
    birth_events = 0        # number of individual daughter cells
    birth_d_detected = 0   # daughters detected at birth frame
    birth_both = 0          # division events where BOTH daughters detected
    birth_div_total = 0     # division events (pairs)

    print(f"\nDetect-only eval  split={args.split}  "
          f"proposal_threshold={args.proposal_threshold}  "
          f"nms={args.nms_threshold}  iou_match={args.iou_threshold}\n")

    for seq_key in tqdm(sorted(seq_to_imgs.keys()), desc='sequences'):
        imgs = seq_to_imgs[seq_key]
        seen_tids: set = set()

        for frame_pos, img_meta in enumerate(imgs):
            img_id    = img_meta['id']
            img_path  = img_dir / img_meta['file_name']
            gt_entries = gt_by_img[img_id]

            if not gt_entries:
                seen_tids.update()
                continue

            # detect
            pred_boxes, pred_scores, (H, W) = detect_frame(
                model, img_path,
                proposal_threshold=args.proposal_threshold,
                nms_threshold=args.nms_threshold,
                device=device)

            # GT boxes → tensor
            gt_boxes_list = [e['box'] for e in gt_entries]
            gt_tids       = [e['track_id'] for e in gt_entries]
            gt_boxes = torch.tensor(gt_boxes_list, dtype=torch.float32, device=device)

            matched_gt, matched_pred = match_boxes(pred_boxes, gt_boxes,
                                                   args.iou_threshold)

            tp = len(matched_gt)
            fp = pred_boxes.shape[0] - len(matched_pred)
            fn = gt_boxes.shape[0] - len(matched_gt)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            # --- birth analysis ---
            if frame_pos > 0:
                current_tids = set(gt_tids)
                new_tids = current_tids - seen_tids
                if new_tids:
                    # map track_id → gt index in this frame
                    tid_to_gtidx = {tid: i for i, tid in enumerate(gt_tids)}
                    daughters_detected = [
                        tid_to_gtidx[tid] in matched_gt
                        for tid in new_tids
                        if tid in tid_to_gtidx
                    ]
                    n_daughters = len(daughters_detected)
                    n_detected  = sum(daughters_detected)
                    birth_events      += n_daughters
                    birth_d_detected  += n_detected
                    birth_div_total   += 1
                    if n_detected == n_daughters:
                        birth_both += 1

            seen_tids.update(gt_tids)

    # --- overall metrics ---
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print('=' * 60)
    print('OVERALL DETECTION (IoU ≥ {:.2f})'.format(args.iou_threshold))
    print(f'  TP={total_tp}  FP={total_fp}  FN={total_fn}')
    print(f'  Precision : {prec*100:.1f}%')
    print(f'  Recall    : {rec*100:.1f}%')
    print(f'  F1        : {f1*100:.1f}%')
    print()
    print('BIRTH-FRAME DETECTION')
    print(f'  Division events     : {birth_div_total}')
    print(f'  Daughter cells      : {birth_events}')
    print(f'  Daughters detected  : {birth_d_detected}/{birth_events}'
          f'  ({birth_d_detected/birth_events*100:.1f}%)' if birth_events else '')
    print(f'  Both detected       : {birth_both}/{birth_div_total}'
          f'  ({birth_both/birth_div_total*100:.1f}%)' if birth_div_total else '')
    print('=' * 60)


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
    pre.add_argument('--config', default="configs/CellTracking/deepcell/infer_no_div.yaml")
    pre_args, remaining = pre.parse_known_args()

    parser = argparse.ArgumentParser(
        'SelfMOTR detection-only eval', parents=[get_args_parser()])
    parser.add_argument('--split',              default='val',
                        choices=['train', 'val'])
    parser.add_argument('--proposal_threshold', default=0.05, type=float,
                        help='Min score to keep a detection query output')
    parser.add_argument('--nms_threshold',      default=0.5,  type=float,
                        help='IoU threshold for NMS on detector outputs')
    parser.add_argument('--iou_threshold',      default=0.5,  type=float,
                        help='IoU threshold for TP/FP/FN matching vs GT')
    # kept for apply_checkpoint_model_args compatibility
    parser.add_argument('--update_score_threshold', default=0.5, type=float)
    parser.add_argument('--miss_tolerance',         default=10,  type=int)
    parser.add_argument('--gap_close_frames',       default=0,   type=int)
    parser.add_argument('--gap_close_dist_factor',  default=1.5, type=float)
    parser.add_argument('--div_score_thresh',       default=0.5, type=float)
    parser.add_argument('--div_threshold',          default=0.5, type=float)
    parser.add_argument('--max_div_dist_factor',    default=2.0, type=float)

    if pre_args.config is not None:
        _apply_yaml_defaults(parser, pre_args.config)

    args = parser.parse_args(remaining)
    main(args)
