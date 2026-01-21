# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
import torchvision

from .coco import build as build_coco
from .detmot import build as build_e2e_mot
from .dance import build as build_e2e_dance
from .dancev2 import build as build_e2e_dancev2
from .dancev2_final import build as build_e2e_dancev2_final
from .static_detmot import build as build_e2e_static_mot
from .joint import build as build_e2e_joint
from .sportsmot import build as build_e2e_sportsmot
from .sportsmotv2 import build as build_e2e_sportsmotv2
from .bft import build as build_e2e_bft
from .wat import build as build_e2e_wat
from .torchvision_datasets import CocoDetection

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'e2e_joint':
        return build_e2e_joint(image_set, args)
    if args.dataset_file == 'e2e_static_mot':
        return build_e2e_static_mot(image_set, args)
    if args.dataset_file == 'e2e_mot':
        return build_e2e_mot(image_set, args)
    if args.dataset_file == 'e2e_dance':
        return build_e2e_dance(image_set, args)
    if args.dataset_file == 'e2e_dance_v2':
        return build_e2e_dancev2(image_set, args)
    if args.dataset_file == 'e2e_dance_v2_final':
        return build_e2e_dancev2_final(image_set, args)
    if args.dataset_file == 'e2e_sportsmot':
        return build_e2e_sportsmot(image_set, args)
    if args.dataset_file == 'e2e_bft':
        return build_e2e_bft(image_set, args)
    if args.dataset_file == 'e2e_wat':
        return build_e2e_wat(image_set, args)
    if args.dataset_file == 'e2e_sportsmot_v2':
        return build_e2e_sportsmotv2(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
