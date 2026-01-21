#from copy import deepcopy
#import json

import os
import numpy as np
import random
import argparse
import torchvision.transforms.functional as F
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw
from models import build_model
from util.tool import apply_checkpoint_model_args, load_model, load_torch_checkpoint
from train_dance import get_args_parser
from torch.nn.functional import interpolate
from typing import List
from util.evaluation import Evaluator
import motmetrics as mm
import shutil

from models.structures import Instances
from torch.utils.data import Dataset, DataLoader

"""
import debugpy

debugpy.listen(("0.0.0.0", 5678))  # listen for the debugger
print("⏳ Waiting for debugger attach...")
debugpy.wait_for_client()  # pause until debugger is attached
debugpy.breakpoint()  # this acts like a manual breakpoint
"""

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class Track(object):
    track_cnt = 0

    def __init__(self, box):
        self.box = box
        self.time_since_update = 0
        self.id = Track.track_cnt
        Track.track_cnt += 1
        self.miss = 0

    def miss_one_frame(self):
        self.miss += 1

    def clear_miss(self):
        self.miss = 0

    def update(self, box):
        self.box = box
        self.clear_miss()


class MOTR(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.active_trackers = {}
        self.inactive_trackers = {}
        self.disappeared_tracks = []

    def _remove_track(self, slot_id):
        self.inactive_trackers.pop(slot_id)
        self.disappeared_tracks.append(slot_id)

    def clear_disappeared_track(self):
        self.disappeared_tracks = []

    def update(self, dt_instances: Instances):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        dt_idxes = set(dt_instances.obj_idxes.tolist())
        track_idxes = set(self.active_trackers.keys()).union(set(self.inactive_trackers.keys()))
        matched_idxes = dt_idxes.intersection(track_idxes)

        unmatched_tracker = track_idxes - matched_idxes
        for track_id in unmatched_tracker:
            # miss in this frame, move to inactive_trackers.
            if track_id in self.active_trackers:
                self.inactive_trackers[track_id] = self.active_trackers.pop(track_id)
            self.inactive_trackers[track_id].miss_one_frame()
            if self.inactive_trackers[track_id].miss > 20:
                self._remove_track(track_id)

        for i in range(len(dt_instances)):
            idx = dt_instances.obj_idxes[i]
            bbox = np.concatenate([dt_instances.boxes[i], dt_instances.scores[i:i+1]], axis=-1)
            label = dt_instances.labels[i]
            if label == 0:
                # get a positive track.
                if idx in self.inactive_trackers:
                    # set state of track active.
                    self.active_trackers[idx] = self.inactive_trackers.pop(idx)
                if idx not in self.active_trackers:
                    # create a new track.
                    self.active_trackers[idx] = Track(idx)
                self.active_trackers[idx].update(bbox)
            elif label == 1:
                # get an occluded track.
                if idx in self.active_trackers:
                    # set state of track inactive.
                    self.inactive_trackers[idx] = self.active_trackers.pop(idx)
                if idx not in self.inactive_trackers:
                    # It's strange to obtain a new occluded track.
                    # TODO: think more rational disposal.
                    self.inactive_trackers[idx] = Track(idx)
                self.inactive_trackers[idx].miss_one_frame()
                if self.inactive_trackers[idx].miss > 20:
                    self._remove_track(idx)

        ret = []
        for i in range(len(dt_instances)):
            label = dt_instances.labels[i]
            if label == 0:
                id = dt_instances.obj_idxes[i]
                box_with_score = np.concatenate([dt_instances.boxes[i], dt_instances.scores[i:i+1]], axis=-1)
                ret.append(np.concatenate((box_with_score, [id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))

class ListImgDataset(Dataset):
    def __init__(self, mot_path, img_list) -> None:
        super().__init__()
        self.mot_path = mot_path
        self.img_list = img_list

        '''
        common settings
        '''
        self.img_height = 800
        self.img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_img_from_file(self, f_path):
        cur_img = cv2.imread(os.path.join(self.mot_path, f_path))
        assert cur_img is not None, f_path
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        return cur_img

    def init_img(self, img):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img = self.load_img_from_file(self.img_list[index])
        return self.init_img(img)


class Detector(object):
    def __init__(self, args, model, vid):
        self.args = args
        self.detr = model

        self.vid = vid
        self.seq_num = os.path.basename(vid)
        img_list = os.listdir(os.path.join(self.args.mot_path, vid, 'img1'))
        img_list = [os.path.join(vid, 'img1', i) for i in img_list if 'jpg' in i]

        self.img_list = sorted(img_list)
        self.img_len = len(self.img_list)
        self.tr_tracker = MOTR()

        self.predict_path = os.path.join(self.args.output_dir, args.exp_name)
        os.makedirs(self.predict_path, exist_ok=True)

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        keep &= dt_instances.obj_idxes >= 0
        return dt_instances[keep]
    
    @staticmethod
    def filter_dt_by_score_dict(dt_instances: dict, prob_threshold: float) -> dict:
        keep = (dt_instances['scores'] > prob_threshold).squeeze()
        filtered = {
            'scores': dt_instances['scores'][0][keep],
            'boxes': dt_instances['boxes'][0][keep],
        }
        return filtered

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]
    
    @staticmethod
    def filter_dt_by_area_dict(dt_instances: dict, area_threshold: float) -> dict:
        boxes = dt_instances['boxes']
        wh = boxes[:, 2:4] - boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold

        filtered = {
            'scores': dt_instances['scores'][keep],
            'boxes': dt_instances['boxes'][keep]
        }
        return filtered

    @staticmethod
    def write_results(txt_path, frame_id, bbox_xyxy, identities):
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
        with open(txt_path, 'a') as f:
            for xyxy, track_id in zip(bbox_xyxy, identities):
                if track_id < 0 or track_id is None:
                    continue
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                line = save_format.format(frame=int(frame_id), id=int(track_id), x1=x1, y1=y1, w=w, h=h)
                f.write(line)

    @staticmethod
    def write_results_new(txt_path, frame_id, bbox_xyxy, identities, scores):
        save_format = '{frame},{id},{x1},{y1},{w},{h},{sc},-1,-1,-1\n'
        with open(txt_path, 'a') as f:
            for xyxy, track_id, score in zip(bbox_xyxy, identities, scores):
                if track_id < 0 or track_id is None:
                    continue
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                line = save_format.format(frame=int(frame_id), id=int(track_id), x1=x1, y1=y1, w=w, h=h, sc=float(score))
                f.write(line)

    def eval_seq(self):
        data_root = os.path.join(self.args.mot_path, 'MOT15/images/train')
        result_filename = os.path.join(self.predict_path, 'gt.txt')
        evaluator = Evaluator(data_root, self.seq_num)
        accs = evaluator.eval_file(result_filename)
        return accs

    @staticmethod
    def visualize_img_with_bbox(img_path, img, dt_instances: Instances, ref_pts=None, gt_boxes=None):
        if dt_instances.has('scores'):
            img_show = draw_bboxes(img, np.concatenate([dt_instances.boxes, dt_instances.scores.reshape(-1, 1)], axis=-1), dt_instances.obj_idxes)
        else:
            img_show = draw_bboxes(img, dt_instances.boxes, dt_instances.obj_idxes)
        if ref_pts is not None:
            img_show = draw_points(img_show, ref_pts)
        if gt_boxes is not None:
            img_show = draw_bboxes(img_show, gt_boxes, identities=np.ones((len(gt_boxes), )) * -1)
        cv2.imwrite(img_path, img_show)

    def detect(self, prob_threshold=0.5, area_threshold=100, proposal_threshold=0.05, vis=False):
        total_dts = 0
        total_occlusion_dts = 0

        track_instances = None
        loader = DataLoader(ListImgDataset(self.args.mot_path, self.img_list), 1, num_workers=2)
        lines = []
        for i, data in enumerate(tqdm(loader)):
            cur_img, ori_img = [d[0] for d in data]
            cur_img = cur_img.cuda()

            # track_instances = None
            if track_instances is not None:
                track_instances.remove('boxes')
                track_instances.remove('labels')
            seq_h, seq_w, _ = ori_img.shape

            proposals = self.detr.inference_single_image_proposals(cur_img, (seq_h, seq_w), score_threshold=proposal_threshold)
            res = self.detr.inference_single_image_detector(cur_img, (seq_h, seq_w), proposals)
            #track_instances = res['track_instances']

            res = {
                'scores': res['scores'][-1],
                'boxes': res['boxes'][-1],
            }
            #dt_instances = track_instances.to(torch.device('cpu'))
            dt_instances = {
                k: v.to(torch.device('cpu')) for k, v in res.items()
            }

            # filter det instances by score.
            dt_instances = self.filter_dt_by_score_dict(dt_instances, prob_threshold)
            dt_instances = self.filter_dt_by_area_dict(dt_instances, area_threshold)

            nr_det = dt_instances['boxes'].shape[0]
            
            total_dts += nr_det

            det_identities = torch.arange(nr_det)

            self.write_results_new(txt_path=os.path.join(self.predict_path, f'{self.seq_num}.txt'),
                               frame_id=(i + 1),
                               bbox_xyxy=dt_instances['boxes'][:, :4],
                               identities=det_identities,
                               scores=dt_instances['scores'][:, 0])
            
        print("totally {} dts {} occlusion dts".format(total_dts, total_occlusion_dts))

class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.5, filter_score_thresh=0.5, miss_tolerance=20):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        device = track_instances.obj_idxes.device

        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        new_obj = (track_instances.obj_idxes == -1) & (track_instances.scores >= self.score_thresh)
        disappeared_obj = (track_instances.obj_idxes >= 0) & (track_instances.scores < self.filter_score_thresh)
        num_new_objs = new_obj.sum().item()

        track_instances.obj_idxes[new_obj] = self.max_obj_id + torch.arange(num_new_objs, device=device)
        self.max_obj_id += num_new_objs

        track_instances.disappear_time[disappeared_obj] += 1
        to_del = disappeared_obj & (track_instances.disappear_time >= self.miss_tolerance)
        track_instances.obj_idxes[to_del] = -1


if __name__ == '__main__':

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('--update_score_threshold', default=0.5, type=float)
    parser.add_argument('--miss_tolerance', default=20, type=int)
    parser.add_argument('--proposal_threshold', default=0.05, type=float)

    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    checkpoint = load_torch_checkpoint(args.resume, map_location='cpu', weights_only=False)
    args = apply_checkpoint_model_args(args, checkpoint, context='ablation/eval_detect_no_trackquery')

    # load model and weights
    detr, _, _ = build_model(args)
    detr.track_embed.score_thr = args.update_score_threshold
    detr.track_base = RuntimeTrackerBase(args.score_threshold, args.score_threshold, args.miss_tolerance)
    detr = load_model(detr, args.resume)
    detr.eval()
    detr = detr.cuda()

    sub_dir = 'DanceTrack/val'
    seq_nums = os.listdir(os.path.join(args.mot_path, sub_dir))
    if 'seqmap' in seq_nums:
        seq_nums.remove('seqmap')
    vids = [os.path.join(sub_dir, seq) for seq in seq_nums]

    rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
    ws = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
    vids = vids[rank::ws]

    for vid in vids:
        det = Detector(args, model=detr, vid=vid)
        det.detect(args.score_threshold, proposal_threshold=args.proposal_threshold)
