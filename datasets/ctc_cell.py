"""
Dataset loader for Cell-TRACTR CTC/COCO format data, adapted to work with SelfMOTR.

Expected folder layout (identical to Cell-TRACTR's COCO output):
    <cell_data_path>/
        annotations/
            train/anno.json
            val/anno.json
        man_track/          ← optional; enables division supervision
            train/01.txt
            val/01.txt
        train/img/CTC_<seq>_frame_<t>.tif
        val/img/...

Generate this layout from raw CTC data using:
    Cell-TRACTR/scripts/create_coco_dataset_from_CTC.py

Division supervision
--------------------
When man_track/<split>/<seq>.txt is present, each frame annotation gains a
``div_flags`` tensor (float32, shape N) where entry i is 1.0 if cell i is
at its LAST frame before dividing (i.e. daughters appear in the next frame),
and 0.0 otherwise.  This feeds the ``div_ahead_embed`` head in MOTR.
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


class CTCCellDataset:

    def __init__(self, args, image_set: str, transform):
        self.transform = transform
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.sampler_steps = args.sampler_steps
        self.lengths = args.sampler_lengths
        self.period_idx = 0
        self.current_epoch = 0

        root = Path(args.mot_path)  # --mot_path points to the COCO root
        ann_file = root / 'annotations' / image_set / 'anno.json'
        self.img_dir = root / image_set / 'img'

        assert ann_file.exists(), f'Annotation file not found: {ann_file}'
        assert self.img_dir.exists(), f'Image directory not found: {self.img_dir}'

        with open(ann_file) as f:
            data = json.load(f)

        # image id → metadata
        self.images = {img['id']: img for img in data['images']}

        # Group image_ids by CTC sequence, ordered by image id (= frame order)
        seq_to_ids = defaultdict(list)
        for img in data['images']:
            seq_key = img.get('ctc_id', img.get('man_track_id', 'default'))
            seq_to_ids[seq_key].append(img['id'])
        for seq in seq_to_ids:
            seq_to_ids[seq].sort()

        # annotation lookup: image_id → list of {bbox_xyxy, track_id}
        self.anns_by_image = defaultdict(list)
        for ann in data['annotations']:
            if ann.get('empty', False):
                continue
            x, y, w, h = ann['bbox']
            self.anns_by_image[ann['image_id']].append({
                'bbox': [x, y, x + w, y + h],
                'track_id': int(ann['track_id']),
            })

        # Build (seq_index, start_frame_index) sampling table
        self.sequences = []
        self.indices = []
        for seq_key, img_ids in sorted(seq_to_ids.items()):
            seq_idx = len(self.sequences)
            self.sequences.append((seq_key, img_ids))
            n = len(img_ids)
            for t in range(n - self.num_frames_per_batch + 1):
                self.indices.append((seq_idx, t))

        # ------------------------------------------------------------------
        # Division supervision: load man_track/<split>/<seq>.txt
        # Format (CTC spec):  L  B  E  P
        #   L = cell label in GT mask, B = first frame, E = last frame,
        #   P = parent label (0 means no division origin)
        #
        # A cell C divides when its last frame (E) has exactly 2 daughters
        # in the file (two rows with P == C).  We flag C at frame E with
        # div_flag = 1, so the model can learn "this cell is about to split".
        # ------------------------------------------------------------------
        self.div_lookup: dict = {}   # (seq_key, cell_id, frame_nb) → True
        man_track_dir = root / 'man_track' / image_set
        n_dividing = 0
        if man_track_dir.exists():
            for seq_key, _ in self.sequences:
                mt_path = man_track_dir / f'{seq_key}.txt'
                if not mt_path.exists():
                    continue
                mt = np.loadtxt(mt_path, dtype=np.int32)
                if mt.ndim == 1:
                    mt = mt[None]   # single-row file
                # For each cell that has exactly 2 daughters, flag its last frame
                for row in mt:
                    cell_id, start_f, end_f, parent_id = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                    n_daughters = int((mt[:, 3] == cell_id).sum())
                    if n_daughters == 2:
                        self.div_lookup[(seq_key, cell_id, end_f)] = True
                        n_dividing += 1
        else:
            print(f"  [CTCCellDataset] man_track dir not found at {man_track_dir}; "
                  "division supervision disabled.")

        print(f"CTCCellDataset [{image_set}]: {len(self.sequences)} sequences, "
              f"{len(self.indices)} samples, "
              f"max_cells={data.get('max_num_of_cells', '?')}, "
              f"dividing_cells_flagged={n_dividing}")
        print(f"sampler_steps={self.sampler_steps} lengths={self.lengths}")

        # ------------------------------------------------------------------
        # Division-clip oversampling
        # Partition all (seq_idx, start) clips into two pools: those that
        # contain at least one dividing cell within the sampling window, and
        # those that do not.  _rebuild_indices() mixes them at div_ratio each
        # epoch so that division events are not lost in the noise.
        # ------------------------------------------------------------------
        self.div_ratio = getattr(args, 'div_ratio', 0.0)

        # Precompute which (seq_idx, frame_pos) frames have a dividing cell
        dividing_positions = set()
        for seq_idx, (seq_key, img_ids) in enumerate(self.sequences):
            for fp, img_id in enumerate(img_ids):
                img_meta = self.images[img_id]
                frame_nb = int(re.findall(r'\d+', img_meta['file_name'])[-1])
                for ann in self.anns_by_image[img_id]:
                    if self.div_lookup.get((seq_key, ann['track_id'], frame_nb), False):
                        dividing_positions.add((seq_idx, fp))
                        break   # one dividing cell is enough to flag the frame

        # A clip (seq_idx, start) is "div-containing" if any frame within the
        # maximum possible window [start, start + (max_len-1)*max_interval]
        # is a dividing frame.
        max_window = (max(self.lengths) - 1) * self.sample_interval
        self._div_indices    = []
        self._nondiv_indices = []
        for seq_idx, start in self.indices:
            n_frames = len(self.sequences[seq_idx][1])
            end = min(start + max_window, n_frames - 1)
            if any((seq_idx, fp) in dividing_positions for fp in range(start, end + 1)):
                self._div_indices.append((seq_idx, start))
            else:
                self._nondiv_indices.append((seq_idx, start))

        print(f"  div-containing clips: {len(self._div_indices)}, "
              f"non-div clips: {len(self._nondiv_indices)}, "
              f"div_ratio={self.div_ratio}")

        self._rebuild_indices()

    # ------------------------------------------------------------------
    # Epoch / curriculum support (mirrors DetMOTDetection interface)
    # ------------------------------------------------------------------

    def _rebuild_indices(self):
        """
        Rebuild self.indices from the div / non-div pools at the configured ratio.

        With div_ratio=0 (default) this is a no-op: all clips are used as-is.
        With div_ratio=r, exactly floor(total * r) clips are drawn from the
        div-containing pool (with replacement when the pool is smaller than
        needed) and the rest from the non-div pool.  Both pools are shuffled
        with an epoch-seeded RNG so the mix changes every epoch.
        """
        if self.div_ratio <= 0.0 or not self._div_indices:
            self.indices = self._nondiv_indices + self._div_indices
            return

        rng   = np.random.default_rng(self.current_epoch)
        total = len(self._div_indices) + len(self._nondiv_indices)
        n_div = int(total * self.div_ratio)
        n_non = total - n_div

        # Sample div clips (with replacement if the pool is smaller than needed)
        replace_div = len(self._div_indices) < n_div
        div_pick = rng.choice(len(self._div_indices), size=n_div,
                              replace=replace_div).tolist()
        sampled_div = [self._div_indices[i] for i in div_pick]

        # Sample non-div clips without replacement (cap at pool size)
        n_non = min(n_non, len(self._nondiv_indices))
        non_pick = rng.choice(len(self._nondiv_indices), size=n_non,
                              replace=False).tolist()
        sampled_non = [self._nondiv_indices[i] for i in non_pick]

        combined = sampled_div + sampled_non
        rng.shuffle(combined)
        self.indices = [tuple(x) for x in combined]

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps:
            for i, step in enumerate(self.sampler_steps):
                if epoch >= step:
                    self.period_idx = i + 1
            print(f"set epoch: epoch {epoch} period_idx={self.period_idx}")
            self.num_frames_per_batch = self.lengths[self.period_idx]
        self._rebuild_indices()

    def step_epoch(self):
        print(f"Dataset: epoch {self.current_epoch} finishes")
        self.set_epoch(self.current_epoch + 1)

    # ------------------------------------------------------------------
    # Core loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt = Instances(tuple(img_shape))
        n_gt = len(targets['labels'])
        gt.boxes = targets['boxes'][:n_gt]
        gt.labels = targets['labels']
        gt.obj_ids = targets['obj_ids']
        if 'div_flags' in targets:
            gt.div_flags = targets['div_flags'][:n_gt]
        return gt

    def _load_frame(self, seq_idx: int, frame_pos: int):
        """Load one frame and its annotations, including division flags."""
        seq_key, img_ids = self.sequences[seq_idx]
        image_id = img_ids[frame_pos]
        img_meta = self.images[image_id]

        # PIL handles .tif natively; convert to RGB for 3-channel input
        img = Image.open(self.img_dir / img_meta['file_name']).convert('RGB')
        w, h = img.size

        # Extract frame number from filename, e.g. CTC_01_frame_042.tif → 42
        frame_nb = int(re.findall(r'\d+', img_meta['file_name'])[-1])

        # Offset track IDs so they are globally unique across sequences
        seq_offset = seq_idx * 100000

        anns = self.anns_by_image[image_id]
        boxes, labels, obj_ids, div_flags = [], [], [], []
        for ann in anns:
            boxes.append(ann['bbox'])
            labels.append(0)
            obj_ids.append(ann['track_id'] + seq_offset)
            # div_flag = 1 if this cell is at its last frame before dividing
            div_flags.append(
                1.0 if self.div_lookup.get((seq_key, ann['track_id'], frame_nb), False) else 0.0
            )

        targets = {
            'dataset': 'CTC_cell',
            'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            'labels': torch.as_tensor(labels, dtype=torch.long),
            'obj_ids': torch.as_tensor(obj_ids, dtype=torch.float64),
            'div_flags': torch.as_tensor(div_flags, dtype=torch.float32),
            'iscrowd': torch.zeros(len(anns), dtype=torch.bool),
            'scores': torch.ones(len(anns), dtype=torch.float32),
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

        # Clamp to sequence bounds
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
# Transforms
# ------------------------------------------------------------------

def make_transforms_cell(image_set):
    """
    Minimal transforms for microscopy cells.

    No large-scale random resize (cells are already small and fixed-size).
    Horizontal flip + HSV jitter for train, bare normalize for val.
    """
    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if image_set == 'train':
        return T.MotCompose([
            T.MotRandomHorizontalFlip(),
            T.MOTHSV(),
            normalize,
        ])

    if image_set == 'val':
        return T.MotCompose([normalize])

    raise ValueError(f'unknown image_set: {image_set}')


# ------------------------------------------------------------------
# Builder (called by datasets/__init__.py)
# ------------------------------------------------------------------

def build(image_set, args):
    root = Path(args.mot_path)
    assert root.exists(), f'--mot_path {root} does not exist'
    transform = make_transforms_cell(image_set)
    return CTCCellDataset(args, image_set=image_set, transform=transform)
