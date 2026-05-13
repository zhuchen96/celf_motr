"""
Dataset loader for Cell-TRACTR CTC/COCO format — no-division variant.

Identical to ctc_cell.py except every cell (including daughter cells born
at a division event) is treated as an independent track with its own obj_id.
Daughters appear as new detections at their first frame; the mother simply
disappears.  No div_flags / div_box2 fields are emitted.

This lets the model learn to detect newborn cells from scratch without any
division supervision.  Use with meta_arch=motrv2_self_no_div.
"""

import json
import re
import numpy as np
from collections import defaultdict
from pathlib import Path

import torch
import torch.utils.data
from PIL import Image

import datasets.transformsv2 as T
from models.structures import Instances


class CTCCellDatasetNoDiv:

    def __init__(self, args, image_set: str, transform):
        self.transform = transform
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.sampler_steps = args.sampler_steps
        self.lengths = args.sampler_lengths
        self.period_idx = 0
        self.current_epoch = 0

        root = Path(args.mot_path)
        ann_file = root / 'annotations' / image_set / 'anno.json'
        self.img_dir = root / image_set / 'img'

        assert ann_file.exists(), f'Annotation file not found: {ann_file}'
        assert self.img_dir.exists(), f'Image directory not found: {self.img_dir}'

        with open(ann_file) as f:
            data = json.load(f)

        self.images = {img['id']: img for img in data['images']}

        seq_to_ids = defaultdict(list)
        for img in data['images']:
            seq_key = img.get('ctc_id', img.get('man_track_id', 'default'))
            seq_to_ids[seq_key].append(img['id'])
        for seq in seq_to_ids:
            seq_to_ids[seq].sort()

        self.anns_by_image = defaultdict(list)
        for ann in data['annotations']:
            if ann.get('empty', False):
                continue
            x, y, w, h = ann['bbox']
            self.anns_by_image[ann['image_id']].append({
                'bbox': [x, y, x + w, y + h],
                'track_id': int(ann['track_id']),
            })

        self.sequences = []
        self.indices = []
        for seq_key, img_ids in sorted(seq_to_ids.items()):
            seq_idx = len(self.sequences)
            self.sequences.append((seq_key, img_ids))
            n = len(img_ids)
            for t in range(n - self.num_frames_per_batch + 1):
                self.indices.append((seq_idx, t))

        print(f"CTCCellDatasetNoDiv [{image_set}]: {len(self.sequences)} sequences, "
              f"{len(self.indices)} samples, "
              f"max_cells={data.get('max_num_of_cells', '?')}")
        print(f"sampler_steps={self.sampler_steps} lengths={self.lengths}")

        # Oversample clips that contain a birth event so the model sees
        # cells appearing from nowhere far more often than uniform sampling
        # would provide.  Only applied to training splits.
        if image_set == 'train':
            birth_oversample = getattr(args, 'birth_oversample', 5)
            if birth_oversample > 0:
                self._add_birth_oversample(birth_oversample)

    # ------------------------------------------------------------------
    # Epoch / curriculum support
    # ------------------------------------------------------------------

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps:
            for i, step in enumerate(self.sampler_steps):
                if epoch >= step:
                    self.period_idx = i + 1
            print(f"set epoch: epoch {epoch} period_idx={self.period_idx}")
            self.num_frames_per_batch = self.lengths[self.period_idx]

    def step_epoch(self):
        print(f"Dataset: epoch {self.current_epoch} finishes")
        self.set_epoch(self.current_epoch + 1)

    # ------------------------------------------------------------------
    # Birth-event oversampling
    # ------------------------------------------------------------------

    def _add_birth_oversample(self, oversample: int):
        """
        For every frame where one or more new track_ids appear mid-sequence,
        inject extra training clips so the model sees that "birth" event
        much more often than uniform sampling would provide.

        Two clip variants are added per event:
          • near:  birth frame is at clip position 1 — the model always sees
                   the frame before AND the birth frame, regardless of the
                   curriculum clip length.
          • far:   birth frame at the END of the max-length clip — the model
                   also sees a longer run-up (added at half weight so it
                   doesn't swamp the near variant).
        """
        max_frames = max(self.lengths)
        extra = []
        n_events = 0

        for seq_idx, (seq_key, img_ids) in enumerate(self.sequences):
            n = len(img_ids)
            seen_tids: set = set()
            for frame_pos, img_id in enumerate(img_ids):
                tids = {ann['track_id'] for ann in self.anns_by_image[img_id]}
                if frame_pos > 0 and (tids - seen_tids):
                    n_events += len(tids - seen_tids)

                    # near clip: birth is always at position 1 in the window
                    start_near = max(0, frame_pos - 1)
                    extra.extend([(seq_idx, start_near)] * oversample)

                    # far clip: birth at end of max-window (more run-up context)
                    start_far = max(0, frame_pos - max_frames + 1)
                    if start_far != start_near and start_far + max_frames <= n:
                        extra.extend([(seq_idx, start_far)] * max(1, oversample // 2))

                seen_tids.update(tids)

        self.indices.extend(extra)
        n_near = sum(1 for (_, s) in extra if True)  # total count already in extra
        print(f"  Birth oversampling ({oversample}×): {n_events} birth-cell appearances "
              f"→ +{len(extra)} extra clips  "
              f"(base {len(self.indices) - len(extra)} → total {len(self.indices)})")

    # ------------------------------------------------------------------
    # Core loading
    # ------------------------------------------------------------------

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt = Instances(tuple(img_shape))
        n_gt = len(targets['labels'])
        gt.boxes = targets['boxes'][:n_gt]
        gt.labels = targets['labels']
        gt.obj_ids = targets['obj_ids']
        return gt

    def _load_frame(self, seq_idx: int, frame_pos: int):
        """Load one frame. Every cell is an independent track — daughters
        appear as fresh cells with their own track_ids (no merging)."""
        seq_key, img_ids = self.sequences[seq_idx]
        image_id = img_ids[frame_pos]
        img_meta = self.images[image_id]

        img = Image.open(self.img_dir / img_meta['file_name']).convert('RGB')
        w, h = img.size   # noqa

        seq_offset = seq_idx * 100000
        anns = self.anns_by_image[image_id]

        boxes, labels, obj_ids = [], [], []
        for ann in anns:
            boxes.append(ann['bbox'])
            labels.append(0)
            obj_ids.append(ann['track_id'] + seq_offset)

        targets = {
            'dataset': 'CTC_cell_no_div',
            'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            'labels': torch.as_tensor(labels, dtype=torch.long),
            'obj_ids': torch.as_tensor(obj_ids, dtype=torch.float64),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.bool),
            'scores': torch.ones(len(boxes), dtype=torch.float32),
            'image_id': torch.tensor(image_id),
            'size': torch.as_tensor([h, w]),
            'orig_size': torch.as_tensor([h, w]),
        }
        return img, targets

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        seq_idx, start = self.indices[idx]
        _, img_ids = self.sequences[seq_idx]
        n_frames = len(img_ids)

        if self.sample_mode == 'fixed_interval':
            interval = self.sample_interval
        else:
            interval = np.random.randint(1, self.sample_interval + 1)

        frame_positions = [
            min(start + i * interval, n_frames - 1)
            for i in range(self.num_frames_per_batch)
        ]

        images, targets = zip(*[self._load_frame(seq_idx, fp) for fp in frame_positions])
        images, targets = list(images), list(targets)

        if self.transform is not None:
            images, targets = self.transform(images, targets)

        gt_instances = [
            self._targets_to_instances(t, img.shape[1:3])
            for img, t in zip(images, targets)
        ]
        return {
            'imgs': images,
            'gt_instances': gt_instances,
        }

    def __len__(self):
        return len(self.indices)


# ------------------------------------------------------------------
# Transforms (same augmentation as the div version)
# ------------------------------------------------------------------

def make_transforms_cell(image_set):
    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if image_set == 'train':
        return T.MotCompose([
            T.MotRandomHorizontalFlip(),
            T.MotRandomVerticalFlip(),
            T.MotRandomRotate90(),
            T.MOTHSV(),
            normalize,
        ])
    if image_set == 'val':
        return T.MotCompose([normalize])
    raise ValueError(f'unknown image_set: {image_set}')


# ------------------------------------------------------------------
# Builder (registered as 'e2e_cell_no_div')
# ------------------------------------------------------------------

def build(image_set, args):
    root = Path(args.mot_path)
    assert root.exists(), f'--mot_path {root} does not exist'
    transform = make_transforms_cell(image_set)
    return CTCCellDatasetNoDiv(args, image_set=image_set, transform=transform)
