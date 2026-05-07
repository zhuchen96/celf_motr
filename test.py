"""
Standalone division-fire debugger for SelfMOTR.

Purpose:
    Check whether the model actually fires the division channel:
        sigmoid(pred_logits[:, num_classes])

This script does NOT read yaml config and does NOT add argparse arguments.
All parameters are hard-coded below.

Outputs:
    <OUTPUT_DIR>/division_frame_summary.csv
    <OUTPUT_DIR>/division_topk_queries.csv

Interpretation:
    1. max_div_tracked is low everywhere:
       division classifier did not learn / did not fire.

    2. max_div_all is high but max_div_tracked is low:
       division signal appears on detection/proposal queries, not active track queries.
       Then _spawn_daughter2_tracks will not fire because it requires obj_idxes >= 0.

    3. raw_spawn_candidates > 0 but guarded_spawn_candidates = 0:
       division logit fires, but cooldown or D2 distance/degenerate-box guard removes it.

    4. guarded_spawn_candidates > 0 but no D2 appears later:
       spawning happens, but D2 is killed by tracking/QIM/score filtering.
"""

import csv
import json
from collections import defaultdict
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
from util.misc import nested_tensor_from_tensor_list
from util.tool import apply_checkpoint_model_args, load_model, load_torch_checkpoint


# ============================================================
# Hard-coded parameters
# ============================================================

MOT_PATH = Path("/srv/home/chen/Cell-TRACTR/data/deepcell/COCO")
RESUME = Path("/srv/home/chen/cell_motr/self/outputs/cell_deepcell/checkpoint.pth")
OUTPUT_DIR = Path("/srv/home/chen/cell_motr/self/outputs/division_fire_debug")
GT_DIR = Path("/srv/home/chen/Cell-TRACTR/data/deepcell/CTC/val")

SPLIT = "val"

DEVICE = "cuda"

REUSE_ENCODER_CACHE = True

PROPOSAL_THRESHOLD = 0.05

# Normal cell filtering threshold used only for reporting active tracks
SCORE_THRESHOLD = 0.3

# RuntimeTrackerBase threshold
UPDATE_SCORE_THRESHOLD = 0.3

MISS_TOLERANCE = 5

# This is the actual division fire gate used by _spawn_daughter2_tracks
DIV_SCORE_THRESH = 0.5

# For debug statistics
DEBUG_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
TOPK = 10

# Set to None for all sequences, or e.g. 2 for a quick test
MAX_SEQUENCES = None


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ============================================================
# GT-conditioned division helpers
# ============================================================

GT_DIV_FIELDNAMES = [
    "seq", "div_t", "mother_label",
    # Mother query at div_t
    "mother_cell_score", "mother_div_score", "mother_query_iou",
    "pred_d1_cx", "pred_d1_cy", "pred_d1_w", "pred_d1_h",
    "pred_d2_cx", "pred_d2_cy", "pred_d2_w", "pred_d2_h",
    "pred_d2_valid", "pred_d2_dist_norm",
    # Predicted D2 accuracy vs GT daughters (computed at birth_t when GT is available)
    "pred_d2_to_gt_d1_iou", "pred_d2_to_gt_d2_iou",
    # GT daughter positions at birth_t
    "gt_d1_cx", "gt_d1_cy", "gt_d1_w", "gt_d1_h",
    "gt_d2_cx", "gt_d2_cy", "gt_d2_w", "gt_d2_h",
    # Best-matching queries at birth_t
    "d1_query_t1", "d1_cell_score_t1", "d1_iou_t1", "d1_obj_id_t1",
    "d2_query_t1", "d2_cell_score_t1", "d2_iou_t1", "d2_obj_id_t1",
    "two_daughters_detected",
]


def _label_bbox_normalized(tif_path: Path, label: int):
    """Return (cx, cy, w, h) in [0,1] for `label` in a uint16 mask TIF, or None."""
    if not tif_path.exists():
        return None
    mask = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    rows, cols = np.where(mask == label)
    if len(rows) == 0:
        return None
    H, W = mask.shape
    cx = (float(cols.min()) + float(cols.max()) + 1) / 2 / W
    cy = (float(rows.min()) + float(rows.max()) + 1) / 2 / H
    w  = (float(cols.max()) - float(cols.min()) + 1) / W
    h  = (float(rows.max()) - float(rows.min()) + 1) / H
    return (cx, cy, w, h)


def _iou_cxcywh(b1, b2):
    """IoU between two (cx, cy, w, h) boxes in the same coordinate space."""
    x1a = b1[0] - b1[2] / 2;  y1a = b1[1] - b1[3] / 2
    x2a = b1[0] + b1[2] / 2;  y2a = b1[1] + b1[3] / 2
    x1b = b2[0] - b2[2] / 2;  y1b = b2[1] - b2[3] / 2
    x2b = b2[0] + b2[2] / 2;  y2b = b2[1] + b2[3] / 2
    ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
    ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (x2a - x1a) * (y2a - y1a) + (x2b - x1b) * (y2b - y1b) - inter
    return inter / union if union > 0 else 0.0


def _load_gt_divisions(seq_key: str):
    """
    Parse man_track.txt and GT mask TIFs for one sequence.

    Returns a list of dicts, one per GT division event:
      div_t        : int   – last frame the mother exists (where division fires)
      birth_t      : int   – first frame daughters exist (= div_t + 1)
      mother_label : int
      child_labels : [int, int]
      mother_bbox  : (cx, cy, w, h) normalized at div_t, or None
      child_bboxes : [(cx, cy, w, h), (cx, cy, w, h)] normalized at birth_t
    """
    tra_dir = GT_DIR / f"{seq_key}_GT" / "TRA"
    track_file = tra_dir / "man_track.txt"
    if not track_file.exists():
        return []

    tracks = {}
    children_map = defaultdict(list)
    with open(track_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            label, begin, end, par = (int(parts[0]), int(parts[1]),
                                      int(parts[2]), int(parts[3]))
            tracks[label] = {"begin": begin, "end": end}
            if par > 0:
                children_map[par].append(label)

    events = []
    for par_label, child_labels in children_map.items():
        valid_children = [c for c in child_labels if c in tracks]
        if not valid_children:
            continue
        birth_t = min(tracks[c]["begin"] for c in valid_children)
        div_t   = birth_t - 1
        if div_t < 0:
            continue
        children2 = valid_children[:2]
        events.append({
            "div_t":        div_t,
            "birth_t":      birth_t,
            "mother_label": par_label,
            "child_labels": children2,
            "mother_bbox":  _label_bbox_normalized(
                                tra_dir / f"man_track{div_t:03d}.tif", par_label),
            "child_bboxes": [
                _label_bbox_normalized(
                    tra_dir / f"man_track{birth_t:03d}.tif", c)
                for c in children2
            ],
        })
    return events


@torch.no_grad()
def _analyze_mother_at_div_t(model, frame_res, current_tracks, event):
    """
    At div_t: find the tracked query that best matches the GT mother by box IoU.

    Returns a flat dict of mother stats to be merged into the CSV row, or None
    if the GT mother bbox is unavailable.
    """
    mother_bbox = event["mother_bbox"]
    if mother_bbox is None:
        return None

    pred_logits  = frame_res["pred_logits"][0]   # [N, C+1]
    pred_boxes8d = frame_res["pred_boxes"][0]    # [N, 8]
    obj_idxes    = current_tracks.obj_idxes
    is_tracked   = (obj_idxes >= 0).tolist()

    cell_scores = pred_logits[:, 0].sigmoid()
    div_scores  = pred_logits[:, model.num_classes].sigmoid()

    # Find tracked query with best IoU to GT mother.
    best_iou, best_idx = -1.0, -1
    for i in range(len(pred_boxes8d)):
        if not is_tracked[i]:
            continue
        iou = _iou_cxcywh(tuple(pred_boxes8d[i, :4].tolist()), mother_bbox)
        if iou > best_iou:
            best_iou, best_idx = iou, i

    if best_idx < 0:
        return None

    d1 = tuple(float(x) for x in pred_boxes8d[best_idx, :4].tolist())
    d2 = tuple(float(x) for x in pred_boxes8d[best_idx, 4:].tolist())

    m_diag   = (d1[2] ** 2 + d1[3] ** 2) ** 0.5
    d2_dist  = ((d2[0] - d1[0]) ** 2 + (d2[1] - d1[1]) ** 2) ** 0.5
    d2_valid = (sum(abs(x) for x in d2) > 1e-4) and (d2_dist <= m_diag * 4.0)

    return {
        "mother_cell_score": float(cell_scores[best_idx]),
        "mother_div_score":  float(div_scores[best_idx]),
        "mother_query_iou":  best_iou,
        "pred_d1_cx": d1[0], "pred_d1_cy": d1[1],
        "pred_d1_w":  d1[2], "pred_d1_h":  d1[3],
        "pred_d2_cx": d2[0], "pred_d2_cy": d2[1],
        "pred_d2_w":  d2[2], "pred_d2_h":  d2[3],
        "pred_d2_valid":     int(d2_valid),
        "pred_d2_dist_norm": d2_dist / m_diag if m_diag > 1e-6 else 0.0,
    }


@torch.no_grad()
def _analyze_daughters_at_birth_t(frame_res, current_tracks, event, mother_info):
    """
    At birth_t: match all queries to GT daughters by IoU; require distinct queries.

    Returns a flat dict of daughter stats to merge into the CSV row.
    """
    child_bboxes = event["child_bboxes"]
    if len(child_bboxes) < 2 or None in child_bboxes[:2]:
        # Fill with empty values so the CSV row is still complete.
        empty = {k: "" for k in [
            "pred_d2_to_gt_d1_iou", "pred_d2_to_gt_d2_iou",
            "gt_d1_cx", "gt_d1_cy", "gt_d1_w", "gt_d1_h",
            "gt_d2_cx", "gt_d2_cy", "gt_d2_w", "gt_d2_h",
            "d1_query_t1", "d1_cell_score_t1", "d1_iou_t1", "d1_obj_id_t1",
            "d2_query_t1", "d2_cell_score_t1", "d2_iou_t1", "d2_obj_id_t1",
            "two_daughters_detected",
        ]}
        return empty

    gt_d1, gt_d2 = child_bboxes[0], child_bboxes[1]

    pred_logits = frame_res["pred_logits"][0]       # [N, C+1]
    pred_boxes  = frame_res["pred_boxes"][0, :, :4] # [N, 4] cxcywh
    obj_idxes   = current_tracks.obj_idxes
    cell_scores = pred_logits[:, 0].sigmoid()

    N = len(pred_boxes)
    pred_list = [tuple(float(x) for x in pred_boxes[i].tolist()) for i in range(N)]

    ious_d1 = [_iou_cxcywh(p, gt_d1) for p in pred_list]
    ious_d2 = [_iou_cxcywh(p, gt_d2) for p in pred_list]

    best_d1_idx = int(np.argmax(ious_d1))
    # Exclude best D1's query index when searching for D2 match.
    ious_d2_excl = [v if i != best_d1_idx else -1.0 for i, v in enumerate(ious_d2)]
    best_d2_idx = int(np.argmax(ious_d2_excl))

    d1_cs  = float(cell_scores[best_d1_idx])
    d2_cs  = float(cell_scores[best_d2_idx])
    d1_iou = ious_d1[best_d1_idx]
    d2_iou = ious_d2[best_d2_idx]

    two_detected = (
        d1_cs  >= SCORE_THRESHOLD and d1_iou >= 0.1 and
        d2_cs  >= SCORE_THRESHOLD and d2_iou >= 0.1
    )

    # Accuracy of predicted D2 box (from div_t) vs GT daughters (now available).
    pred_d2 = (mother_info["pred_d2_cx"], mother_info["pred_d2_cy"],
               mother_info["pred_d2_w"], mother_info["pred_d2_h"])

    return {
        "pred_d2_to_gt_d1_iou": _iou_cxcywh(pred_d2, gt_d1),
        "pred_d2_to_gt_d2_iou": _iou_cxcywh(pred_d2, gt_d2),
        "gt_d1_cx": gt_d1[0], "gt_d1_cy": gt_d1[1],
        "gt_d1_w":  gt_d1[2], "gt_d1_h":  gt_d1[3],
        "gt_d2_cx": gt_d2[0], "gt_d2_cy": gt_d2[1],
        "gt_d2_w":  gt_d2[2], "gt_d2_h":  gt_d2[3],
        "d1_query_t1":      best_d1_idx,
        "d1_cell_score_t1": d1_cs,
        "d1_iou_t1":        d1_iou,
        "d1_obj_id_t1":     int(obj_idxes[best_d1_idx]),
        "d2_query_t1":      best_d2_idx,
        "d2_cell_score_t1": d2_cs,
        "d2_iou_t1":        d2_iou,
        "d2_obj_id_t1":     int(obj_idxes[best_d2_idx]),
        "two_daughters_detected": int(two_detected),
    }


def load_and_preprocess(img_path: Path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    tensor = TF.normalize(TF.to_tensor(img), MEAN, STD)
    return tensor.unsqueeze(0), (h, w)


def _box_diag_cxcywh(boxes):
    return (boxes[:, 2].pow(2) + boxes[:, 3].pow(2)).sqrt().clamp(min=1e-4)


def _build_args_from_checkpoint():
    """
    Use the original parser defaults and checkpoint model args,
    then overwrite runtime/debug params manually.
    """
    parser = get_args_parser()
    args = parser.parse_args([])

    args.resume = str(RESUME)
    args.mot_path = str(MOT_PATH)
    args.output_dir = str(OUTPUT_DIR)
    args.device = DEVICE

    checkpoint = load_torch_checkpoint(str(RESUME), map_location="cpu", weights_only=False)
    args = apply_checkpoint_model_args(args, checkpoint, context="division_debug")

    # Re-apply debug/runtime settings after checkpoint args,
    # so the checkpoint cannot silently overwrite these thresholds.
    args.resume = str(RESUME)
    args.mot_path = str(MOT_PATH)
    args.output_dir = str(OUTPUT_DIR)
    args.device = DEVICE

    args.split = SPLIT
    args.score_threshold = SCORE_THRESHOLD
    args.update_score_threshold = UPDATE_SCORE_THRESHOLD
    args.proposal_threshold = PROPOSAL_THRESHOLD
    args.miss_tolerance = MISS_TOLERANCE
    args.div_score_thresh = DIV_SCORE_THRESH
    args.reuse_encoder_cache = REUSE_ENCODER_CACHE

    return args, checkpoint


def _load_model():
    args, _ = _build_args_from_checkpoint()

    device = torch.device("cuda" if torch.cuda.is_available() and DEVICE == "cuda" else "cpu")

    model, _, _ = build_model(args)

    model.track_base = RuntimeTrackerBase(
        score_thresh=UPDATE_SCORE_THRESHOLD,
        filter_score_thresh=UPDATE_SCORE_THRESHOLD,
        miss_tolerance=MISS_TOLERANCE,
    )

    model = load_model(model, str(RESUME))
    model.eval()
    model.to(device)

    # Make sure model-side division threshold is exactly what we want.
    if hasattr(model, "div_score_thresh"):
        model.div_score_thresh = DIV_SCORE_THRESH

    print("=== Debug parameters ===")
    print("resume:", RESUME)
    print("mot_path:", MOT_PATH)
    print("split:", SPLIT)
    print("device:", device)
    print("proposal_threshold:", PROPOSAL_THRESHOLD)
    print("score_threshold:", SCORE_THRESHOLD)
    print("update_score_threshold:", UPDATE_SCORE_THRESHOLD)
    print("miss_tolerance:", MISS_TOLERANCE)
    print("div_score_thresh:", DIV_SCORE_THRESH)
    print("model.num_classes:", model.num_classes)
    print("model.div_score_thresh:", getattr(model, "div_score_thresh", None))
    print("track_base.score_thresh:", model.track_base.score_thresh)
    print("track_base.filter_score_thresh:", model.track_base.filter_score_thresh)
    print("========================")

    return model, args, device


def _load_sequences():
    ann_file = MOT_PATH / "annotations" / SPLIT / "anno.json"
    img_dir = MOT_PATH / SPLIT / "img"

    with open(ann_file) as f:
        data = json.load(f)

    seq_to_imgs = defaultdict(list)
    img_id_to_seq_frame = {}

    for img in data["images"]:
        seq_key = img.get("ctc_id", img.get("man_track_id", "unknown"))
        seq_to_imgs[seq_key].append(img)

    for seq_key in seq_to_imgs:
        seq_to_imgs[seq_key].sort(key=lambda x: x["frame_id"])
        for local_idx, img in enumerate(seq_to_imgs[seq_key]):
            img_id_to_seq_frame[img["id"]] = (seq_key, local_idx)

    # Try to count GT division annotations if the COCO json contains such keys.
    gt_div_count = defaultdict(int)
    div_keys = [
        "div_flag",
        "div_flags",
        "is_dividing",
        "division",
        "dividing",
        "has_division",
        "cell_division",
    ]

    found_div_key = False
    for ann in data.get("annotations", []):
        image_id = ann.get("image_id")
        if image_id not in img_id_to_seq_frame:
            continue

        is_div = False
        for k in div_keys:
            if k in ann:
                found_div_key = True
                try:
                    is_div = float(ann[k]) > 0.5
                except Exception:
                    is_div = bool(ann[k])
                if is_div:
                    break

        if is_div:
            seq_key, frame_idx = img_id_to_seq_frame[image_id]
            gt_div_count[(seq_key, frame_idx)] += 1

    if not found_div_key:
        print("Warning: no GT division key found in anno.json. gt_div_count will be 0.")

    return img_dir, seq_to_imgs, gt_div_count


@torch.no_grad()
def _manual_one_frame(model, img_tensor, ori_size, track_instances, device):
    """
    Run one frame manually so we can inspect raw pred_logits before spawning.

    This mirrors model.inference_single_image / inference_single_image_light_light,
    but exposes frame_res before _post_process_single_image().
    """
    img_tensor = img_tensor.to(device)

    if REUSE_ENCODER_CACHE:
        proposals, memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten = (
            model.inference_single_image_proposals_light_light(
                img_tensor,
                ori_size,
                score_threshold=PROPOSAL_THRESHOLD,
            )
        )

        if track_instances is None:
            current_tracks = model._generate_empty_tracks(proposals)
        else:
            current_tracks = Instances.cat([
                model._generate_empty_tracks(proposals),
                track_instances,
            ])

        nested_img = nested_tensor_from_tensor_list(img_tensor)

        frame_res = model._forward_single_image_light(
            nested_img,
            current_tracks,
            memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            mask_flatten,
            gtboxes=None,
        )

    else:
        proposals = model.inference_single_image_proposals(
            img_tensor,
            ori_size,
            score_threshold=PROPOSAL_THRESHOLD,
        )

        if track_instances is None:
            current_tracks = model._generate_empty_tracks(proposals)
        else:
            current_tracks = Instances.cat([
                model._generate_empty_tracks(proposals),
                track_instances,
            ])

        nested_img = nested_tensor_from_tensor_list(img_tensor)

        frame_res = model._forward_single_image(
            nested_img,
            track_instances=current_tracks,
            gtboxes=None,
        )

    return frame_res, current_tracks


def _analyze_raw_frame(model, frame_res, current_tracks):
    """
    Analyze division logits before _post_process_single_image().
    """
    pred_logits = frame_res["pred_logits"][0]
    pred_boxes_8d = frame_res["pred_boxes"][0]

    cell_scores = pred_logits[:, 0].sigmoid()

    if pred_logits.shape[-1] > model.num_classes:
        div_scores = pred_logits[:, model.num_classes].sigmoid()
    else:
        div_scores = torch.zeros_like(cell_scores)

    obj_idxes = current_tracks.obj_idxes
    is_tracked = obj_idxes >= 0
    is_active_tracked = is_tracked & (cell_scores >= SCORE_THRESHOLD)

    has_parent = current_tracks.has("parent_obj_id")
    if has_parent:
        parent_obj_id = current_tracks.parent_obj_id
        is_immune = parent_obj_id >= 0
    else:
        parent_obj_id = torch.full_like(obj_idxes, -1)
        is_immune = torch.zeros_like(is_tracked, dtype=torch.bool)

    raw_spawn_candidates = is_tracked & (div_scores >= DIV_SCORE_THRESH)

    # Same D2 validity guard as in _spawn_daughter2_tracks
    if pred_boxes_8d.shape[-1] >= 8:
        m_boxes = pred_boxes_8d[:, :4]
        d2_boxes = pred_boxes_8d[:, 4:]
        m_diag = _box_diag_cxcywh(m_boxes)
        d2_dist = ((d2_boxes[:, 0] - m_boxes[:, 0]).pow(2) +
                   (d2_boxes[:, 1] - m_boxes[:, 1]).pow(2)).sqrt()
        d2_valid = (d2_boxes.abs().sum(dim=1) > 1e-4) & (d2_dist <= m_diag * 4.0)
    else:
        d2_boxes = None
        d2_dist = torch.zeros_like(div_scores)
        d2_valid = torch.zeros_like(div_scores, dtype=torch.bool)

    guarded_spawn_candidates = raw_spawn_candidates & (~is_immune) & d2_valid

    def safe_max(x, mask=None):
        if mask is not None:
            x = x[mask]
        if len(x) == 0:
            return 0.0
        return float(x.max().item())

    stats = {
        "n_queries": int(len(div_scores)),
        "n_tracked_pre": int(is_tracked.sum().item()),
        "n_active_tracked_pre": int(is_active_tracked.sum().item()),
        "max_div_all": safe_max(div_scores),
        "max_div_tracked": safe_max(div_scores, is_tracked),
        "max_div_active_tracked": safe_max(div_scores, is_active_tracked),
        "raw_spawn_candidates": int(raw_spawn_candidates.sum().item()),
        "guarded_spawn_candidates": int(guarded_spawn_candidates.sum().item()),
        "immune_candidates": int((raw_spawn_candidates & is_immune).sum().item()),
        "d2_invalid_candidates": int((raw_spawn_candidates & (~d2_valid)).sum().item()),
    }

    for th in DEBUG_THRESHOLDS:
        key = str(th).replace(".", "_")
        stats[f"n_all_ge_{key}"] = int((div_scores >= th).sum().item())
        stats[f"n_tracked_ge_{key}"] = int((is_tracked & (div_scores >= th)).sum().item())
        stats[f"n_active_tracked_ge_{key}"] = int((is_active_tracked & (div_scores >= th)).sum().item())

    k = min(TOPK, len(div_scores))
    top_scores, top_idxes = torch.topk(div_scores, k=k)

    top_rows = []
    for rank, (score, qidx) in enumerate(zip(top_scores.tolist(), top_idxes.tolist()), start=1):
        row = {
            "rank": rank,
            "query_idx": int(qidx),
            "obj_id_pre": int(obj_idxes[qidx].item()),
            "parent_obj_id": int(parent_obj_id[qidx].item()),
            "cell_score": float(cell_scores[qidx].item()),
            "div_score": float(score),
            "is_tracked": int(is_tracked[qidx].item()),
            "is_active_tracked": int(is_active_tracked[qidx].item()),
            "raw_spawn_candidate": int(raw_spawn_candidates[qidx].item()),
            "is_immune": int(is_immune[qidx].item()),
            "d2_valid": int(d2_valid[qidx].item()),
            "guarded_spawn_candidate": int(guarded_spawn_candidates[qidx].item()),
            "d2_dist_norm": float(d2_dist[qidx].item()),
        }

        if d2_boxes is not None:
            row.update({
                "d2_cx": float(d2_boxes[qidx, 0].item()),
                "d2_cy": float(d2_boxes[qidx, 1].item()),
                "d2_w": float(d2_boxes[qidx, 2].item()),
                "d2_h": float(d2_boxes[qidx, 3].item()),
            })
        else:
            row.update({
                "d2_cx": 0.0,
                "d2_cy": 0.0,
                "d2_w": 0.0,
                "d2_h": 0.0,
            })

        top_rows.append(row)

    return stats, top_rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, args, device = _load_model()
    img_dir, seq_to_imgs, gt_div_count = _load_sequences()

    frame_summary_path = OUTPUT_DIR / "division_frame_summary.csv"
    topk_path          = OUTPUT_DIR / "division_topk_queries.csv"
    gt_div_path        = OUTPUT_DIR / "gt_division_analysis.csv"

    frame_fieldnames = [
        "seq",
        "frame_idx",
        "file_name",
        "gt_div_count",
        "n_queries",
        "n_tracked_pre",
        "n_active_tracked_pre",
        "max_div_all",
        "max_div_tracked",
        "max_div_active_tracked",
        "raw_spawn_candidates",
        "guarded_spawn_candidates",
        "immune_candidates",
        "d2_invalid_candidates",
    ]

    for th in DEBUG_THRESHOLDS:
        key = str(th).replace(".", "_")
        frame_fieldnames += [
            f"n_all_ge_{key}",
            f"n_tracked_ge_{key}",
            f"n_active_tracked_ge_{key}",
        ]

    topk_fieldnames = [
        "seq",
        "frame_idx",
        "file_name",
        "gt_div_count",
        "rank",
        "query_idx",
        "obj_id_pre",
        "parent_obj_id",
        "cell_score",
        "div_score",
        "is_tracked",
        "is_active_tracked",
        "raw_spawn_candidate",
        "is_immune",
        "d2_valid",
        "guarded_spawn_candidate",
        "d2_dist_norm",
        "d2_cx",
        "d2_cy",
        "d2_w",
        "d2_h",
    ]

    sequences = sorted(seq_to_imgs.keys())
    if MAX_SEQUENCES is not None:
        sequences = sequences[:MAX_SEQUENCES]

    with open(frame_summary_path, "w", newline="") as f_sum, \
         open(topk_path, "w", newline="") as f_top, \
         open(gt_div_path, "w", newline="") as f_gtdiv:

        summary_writer = csv.DictWriter(f_sum, fieldnames=frame_fieldnames)
        topk_writer    = csv.DictWriter(f_top, fieldnames=topk_fieldnames)
        gtdiv_writer   = csv.DictWriter(f_gtdiv, fieldnames=GT_DIV_FIELDNAMES)

        summary_writer.writeheader()
        topk_writer.writeheader()
        gtdiv_writer.writeheader()

        for seq_key in tqdm(sequences, desc="sequences"):
            print(f"\n=== Sequence {seq_key} ===")
            model.clear()
            track_instances = None

            imgs = seq_to_imgs[seq_key]
            frame_files = [img["file_name"] for img in imgs]

            seq_max_div = 0.0
            seq_max_tracked_div = 0.0
            seq_raw_fire_frames = 0
            seq_guarded_fire_frames = 0
            seq_gt_div_frames = 0

            # GT-conditioned division analysis bookkeeping.
            # div_events[t]  = GT events whose div_t == t
            # pending[t]     = (event, mother_info) to analyse at birth_t == t
            gt_div_events = _load_gt_divisions(seq_key)
            div_events = defaultdict(list)
            for ev in gt_div_events:
                div_events[ev["div_t"]].append(ev)
            pending = {}   # birth_t -> (event, mother_info)

            for frame_idx, fname in enumerate(tqdm(frame_files, desc=f"seq {seq_key}", leave=False)):
                img_tensor, ori_size = load_and_preprocess(img_dir / fname)

                frame_res, current_tracks = _manual_one_frame(
                    model,
                    img_tensor,
                    ori_size,
                    track_instances,
                    device,
                )

                stats, top_rows = _analyze_raw_frame(model, frame_res, current_tracks)

                gt_count = int(gt_div_count.get((seq_key, frame_idx), 0))
                if gt_count > 0:
                    seq_gt_div_frames += 1

                seq_max_div = max(seq_max_div, stats["max_div_all"])
                seq_max_tracked_div = max(seq_max_tracked_div, stats["max_div_tracked"])

                if stats["raw_spawn_candidates"] > 0:
                    seq_raw_fire_frames += 1
                if stats["guarded_spawn_candidates"] > 0:
                    seq_guarded_fire_frames += 1

                summary_row = {
                    "seq": seq_key,
                    "frame_idx": frame_idx,
                    "file_name": fname,
                    "gt_div_count": gt_count,
                    **stats,
                }
                summary_writer.writerow(summary_row)

                for r in top_rows:
                    topk_writer.writerow({
                        "seq": seq_key,
                        "frame_idx": frame_idx,
                        "file_name": fname,
                        "gt_div_count": gt_count,
                        **r,
                    })

                # ---- GT-conditioned analysis: capture mother state at div_t ----
                for ev in div_events.get(frame_idx, []):
                    mother_info = _analyze_mother_at_div_t(
                        model, frame_res, current_tracks, ev)
                    if mother_info is not None:
                        pending[ev["birth_t"]] = (ev, mother_info)

                # ---- GT-conditioned analysis: evaluate daughters at birth_t ----
                if frame_idx in pending:
                    ev, mother_info = pending.pop(frame_idx)
                    daughter_info = _analyze_daughters_at_birth_t(
                        frame_res, current_tracks, ev, mother_info)
                    gtdiv_writer.writerow({
                        "seq":          seq_key,
                        "div_t":        ev["div_t"],
                        "mother_label": ev["mother_label"],
                        **mother_info,
                        **daughter_info,
                    })

                # Continue normal inference state update after raw inspection.
                res = model._post_process_single_image(
                    frame_res,
                    current_tracks,
                    is_last=False,
                )
                track_instances = res["track_instances"]
                track_instances = model.post_process(track_instances, ori_size)

            print(
                f"seq {seq_key}: "
                f"max_div_all={seq_max_div:.4f}, "
                f"max_div_tracked={seq_max_tracked_div:.4f}, "
                f"raw_fire_frames={seq_raw_fire_frames}, "
                f"guarded_fire_frames={seq_guarded_fire_frames}, "
                f"gt_div_frames={seq_gt_div_frames}"
            )

    print("\nSaved:")
    print(frame_summary_path)
    print(topk_path)
    print(gt_div_path)

    print("\nHow to read:")
    print("1. max_div_tracked < 0.1 almost everywhere: division classifier is not firing on track queries.")
    print("2. max_div_all high but max_div_tracked low: division signal is on detection queries, not track queries.")
    print("3. raw_spawn_candidates > 0 but guarded_spawn_candidates = 0: D2 box/cooldown guard blocks spawning.")
    print("4. guarded_spawn_candidates > 0 but CTC has no D2: spawning happens, but D2 is killed after QIM/tracking.")


if __name__ == "__main__":
    main()