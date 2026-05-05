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

Division supervision  (Cell-TRACTR design)
------------------------------------------
When man_track/<split>/<seq>.txt is present, the loader implements the same
temporal design as Cell-TRACTR: supervision happens at the daughters' FIRST
frame (t+1), not at the mother's last frame.

At frame t+1 the two daughters are merged into a SINGLE GT entry carrying the
mother's obj_id.  The propagated mother track query therefore matches it
automatically and is trained to predict both daughter positions simultaneously,
using current-frame image features rather than extrapolating into the future.

Per-cell GT fields produced by _load_frame
  ``boxes``           float32 [N,4]  xyxy pixel — daughter1 box for merged entries
  ``div_flags``       float32 [N]    1.0 for the merged division entry (at t+1)
  ``div_ahead_flags`` float32 [N]    1.0 for mothers about to divide (at t, for
                                     the optional div_ahead prediction head)
  ``div_box2``        float32 [N,4]  xyxy pixel box of daughter2 (zeros otherwise)

div_box2 is stored in the same coordinate system as boxes so it passes through
all geometric transforms (flip, rotate) transparently.  MotNormalize converts
it to normalised cxcywh alongside boxes.
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
        #   L = cell label, B = first frame, E = last frame, P = parent label
        #
        # Cell-TRACTR design: supervision is at the daughters' FIRST frame.
        # div_lookup  — flags the mother at her last frame (for div_ahead)
        # div_daughters — (seq_key, parent_id) → [d1_id, d2_id]
        # daughter_to_parent / daughter_first_frame — reverse lookups used in
        #   _load_frame to create merged GT entries at the daughters' birth frame
        # ------------------------------------------------------------------
        self.div_lookup: dict = {}        # (seq_key, cell_id, frame_nb) → True
        self.div_daughters: dict = {}     # (seq_key, parent_id) → [d1_id, d2_id]
        self.daughter_to_parent: dict = {}   # (seq_key, d_id) → parent_id
        self.daughter_first_frame: dict = {} # (seq_key, d_id) → first_frame_nb

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

                # Build start-frame lookup for this sequence
                cell_start_frame = {int(row[0]): int(row[1]) for row in mt}

                # Group children by parent
                parent_to_daughters: dict = defaultdict(list)
                for row in mt:
                    p = int(row[3])
                    if p > 0:
                        parent_to_daughters[p].append(int(row[0]))

                for row in mt:
                    cell_id = int(row[0])
                    end_f   = int(row[2])
                    daughters = parent_to_daughters.get(cell_id, [])
                    if len(daughters) == 2:
                        self.div_lookup[(seq_key, cell_id, end_f)] = True
                        self.div_daughters[(seq_key, cell_id)] = daughters
                        n_dividing += 1
                        for d_id in daughters:
                            self.daughter_to_parent[(seq_key, d_id)] = cell_id
                            self.daughter_first_frame[(seq_key, d_id)] = \
                                cell_start_frame.get(d_id, -1)
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
        # Flag both the mother's last frame (div_ahead) and the daughters'
        # birth frame (division detection) as division-containing positions.
        # ------------------------------------------------------------------
        self.div_ratio = getattr(args, 'div_ratio', 0.0)

        dividing_positions = set()
        for seq_idx, (seq_key, img_ids) in enumerate(self.sequences):
            for fp, img_id in enumerate(img_ids):
                img_meta = self.images[img_id]
                frame_nb = int(re.findall(r'\d+', img_meta['file_name'])[-1])
                for ann in self.anns_by_image[img_id]:
                    cell_id = ann['track_id']
                    # Mother at last frame before division
                    if self.div_lookup.get((seq_key, cell_id, frame_nb), False):
                        dividing_positions.add((seq_idx, fp))
                        break
                    # Daughter at birth frame
                    if (self.daughter_first_frame.get((seq_key, cell_id)) == frame_nb
                            and (seq_key, cell_id) in self.daughter_to_parent):
                        dividing_positions.add((seq_idx, fp))
                        break

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
    # Epoch / curriculum support
    # ------------------------------------------------------------------

    def _rebuild_indices(self):
        if self.div_ratio <= 0.0 or not self._div_indices:
            self.indices = self._nondiv_indices + self._div_indices
            return

        rng   = np.random.default_rng(self.current_epoch)
        total = len(self._div_indices) + len(self._nondiv_indices)
        n_div = int(total * self.div_ratio)
        n_non = total - n_div

        replace_div = len(self._div_indices) < n_div
        div_pick = rng.choice(len(self._div_indices), size=n_div,
                              replace=replace_div).tolist()
        sampled_div = [self._div_indices[i] for i in div_pick]

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
        if 'div_ahead_flags' in targets:
            gt.div_ahead_flags = targets['div_ahead_flags'][:n_gt]
        if 'div_box2' in targets:
            gt.div_box2 = targets['div_box2'][:n_gt]
        return gt

    def _load_frame(self, seq_idx: int, frame_pos: int):
        """Load one frame and its annotations (Cell-TRACTR division design).

        At a daughter pair's birth frame the two daughters are merged into one
        GT entry under the mother's obj_id (div_flags=1, div_box2=d2 xyxy).
        The mother at her last frame gets div_ahead_flags=1 (div_flags=0).
        All boxes are stored as xyxy pixel coordinates so they pass through
        geometric transforms unchanged; MotNormalize converts to cxcywh.
        """
        seq_key, img_ids = self.sequences[seq_idx]
        image_id = img_ids[frame_pos]
        img_meta = self.images[image_id]

        img = Image.open(self.img_dir / img_meta['file_name']).convert('RGB')
        w, h = img.size   # noqa: F841 (used implicitly via xyxy boxes)

        frame_nb = int(re.findall(r'\d+', img_meta['file_name'])[-1])
        seq_offset = seq_idx * 100000

        anns = self.anns_by_image[image_id]

        # ---- Step 1: identify daughters at their birth frame ----
        # For each pair (d1, d2) born at this frame, we create ONE merged GT
        # entry under the mother's obj_id.  Both daughter annotations are then
        # skipped when iterating regular cells.
        merged: dict = {}     # parent_id → {'d1': ann_or_None, 'd2': ann_or_None}
        daughter_ids: set = set()

        for ann in anns:
            cell_id = ann['track_id']
            if (self.daughter_first_frame.get((seq_key, cell_id)) == frame_nb
                    and (seq_key, cell_id) in self.daughter_to_parent):
                parent_id = self.daughter_to_parent[(seq_key, cell_id)]
                siblings  = self.div_daughters.get((seq_key, parent_id), [])
                if len(siblings) == 2:
                    d1_id, d2_id = siblings
                    if parent_id not in merged:
                        merged[parent_id] = {'d1': None, 'd2': None}
                    if cell_id == d1_id:
                        merged[parent_id]['d1'] = ann
                    else:
                        merged[parent_id]['d2'] = ann
                    daughter_ids.add(cell_id)

        # ---- Step 2: build GT tensors ----
        boxes, labels, obj_ids = [], [], []
        div_flags, div_ahead_flags, div_box2s = [], [], []

        # Merged division entries (daughters' birth frame)
        for parent_id, parts in merged.items():
            d1_ann, d2_ann = parts['d1'], parts['d2']
            if d1_ann is not None and d2_ann is not None:
                boxes.append(d1_ann['bbox'])    # daughter1 xyxy
                labels.append(0)
                obj_ids.append(parent_id + seq_offset)  # mother's obj_id
                div_flags.append(1.0)
                div_ahead_flags.append(0.0)
                div_box2s.append(torch.tensor(d2_ann['bbox'], dtype=torch.float32))
            else:
                # One daughter left the FOV; treat the remaining one as a single cell
                ann = d1_ann if d1_ann is not None else d2_ann
                boxes.append(ann['bbox'])
                labels.append(0)
                obj_ids.append(parent_id + seq_offset)
                div_flags.append(0.0)
                div_ahead_flags.append(0.0)
                div_box2s.append(torch.zeros(4, dtype=torch.float32))

        # Regular (non-daughter) cells
        for ann in anns:
            cell_id = ann['track_id']
            if cell_id in daughter_ids:
                continue
            is_div_ahead = self.div_lookup.get((seq_key, cell_id, frame_nb), False)
            boxes.append(ann['bbox'])
            labels.append(0)
            obj_ids.append(cell_id + seq_offset)
            div_flags.append(0.0)
            div_ahead_flags.append(1.0 if is_div_ahead else 0.0)
            div_box2s.append(torch.zeros(4, dtype=torch.float32))

        targets = {
            'dataset': 'CTC_cell',
            'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            'labels': torch.as_tensor(labels, dtype=torch.long),
            'obj_ids': torch.as_tensor(obj_ids, dtype=torch.float64),
            'div_flags': torch.as_tensor(div_flags, dtype=torch.float32),
            'div_ahead_flags': torch.as_tensor(div_ahead_flags, dtype=torch.float32),
            'div_box2': torch.stack(div_box2s) if div_box2s else torch.zeros(0, 4),
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
# Transforms
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
# Builder (called by datasets/__init__.py)
# ------------------------------------------------------------------

def build(image_set, args):
    root = Path(args.mot_path)
    assert root.exists(), f'--mot_path {root} does not exist'
    transform = make_transforms_cell(image_set)
    return CTCCellDataset(args, image_set=image_set, transform=transform)
