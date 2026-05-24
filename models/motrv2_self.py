# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
DETR model and criterion classes.
"""
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List

from util import box_ops, checkpointv2
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from models.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou

from .backbone import build_backbone
from .matcher import build_matcher
from .deformable_transformer_plusv2 import build_deforamble_transformer, pos2posemb
from .qimv2 import build as build_query_interaction_layer
from .deformable_detrv2 import SetCriterion, MLP, sigmoid_focal_loss
from .segmentation import MHAttentionMap, dice_loss


class DivisionProposalMLP(nn.Module):
    """Scores (mother_hs, proposal_hs, rel_pos) pairs for division proposal affinity.

    For each dividing mother query, computes a scalar logit for every detection
    proposal.  Trained with cross-entropy so the mother learns to attend to the
    proposal that best overlaps GT daughter-2, bridging training and inference
    (which picks the nearest proposal at division time).

    rel_pos encodes the spatial relationship between the mother and each proposal
    explicitly: (dx, dy, dist) normalised by the mother's box diagonal.  This
    lets the MLP learn geometric constraints (daughters appear nearby, roughly
    opposite each other) that are hard to read from hidden states alone —
    especially in dense scenes where many proposals have similar appearance.
    """
    def __init__(self, d_model: int):
        super().__init__()
        pos_hidden = max(16, d_model // 16)
        self.pos_proj = nn.Linear(3, pos_hidden)
        self.net = nn.Sequential(
            nn.Linear(d_model * 2 + pos_hidden, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, mother_hs: Tensor, proposal_hs: Tensor,
                rel_pos: Tensor) -> Tensor:
        """
        mother_hs:   [K, d]
        proposal_hs: [N, d]
        rel_pos:     [K, N, 3]  (dx, dy, dist) normalised by mother diagonal
        → logits:    [K, N]
        """
        K, d = mother_hs.shape
        N = proposal_hs.shape[0]
        m = mother_hs.unsqueeze(1).expand(K, N, d)
        p = proposal_hs.unsqueeze(0).expand(K, N, d)
        pos_feat = self.pos_proj(rel_pos)              # [K, N, pos_hidden]
        return self.net(torch.cat([m, p, pos_feat], dim=-1)).squeeze(-1)


class SimpleMaskHead(nn.Module):
    """Per-query instance mask prediction via cross-attention to encoder memory.

    Uses MHAttentionMap to compute per-query attention over the highest-resolution
    encoder feature map (level 0, 1/8 of the input image), applies a small conv
    to refine the nheads maps into one, then bilinearly upsamples to image size.
    """
    def __init__(self, d_model: int, nheads: int):
        super().__init__()
        self.attn = MHAttentionMap(d_model, d_model, nheads, dropout=0.0)
        self.refine = nn.Sequential(
            nn.Conv2d(nheads, nheads, 3, padding=1),
            nn.GroupNorm(min(8, nheads), nheads),
            nn.ReLU(inplace=True),
            nn.Conv2d(nheads, 1, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, hs: Tensor, memory_2d: Tensor) -> Tensor:
        """
        hs:        [B, N, D]      decoder hidden states
        memory_2d: [B, D, H0, W0] level-0 encoder spatial features
        Returns:   [B, N, H0, W0] mask logits (before sigmoid, before upsampling)
        """
        attn = self.attn(hs, memory_2d)          # [B, N, nheads, H0, W0]
        B, N, nh, H0, W0 = attn.shape
        x = attn.view(B * N, nh, H0, W0)         # merge batch & query
        x = self.refine(x)                         # [B*N, 1, H0, W0]
        return x.view(B, N, H0, W0)


class ClipMatcher(SetCriterion):
    def __init__(self, num_classes,
                        matcher,
                        weight_dict,
                        losses,
                        div_proposal_mlp=None,
                        score_consist_loss_coef=0.0,
                        mask_loss_coef=0.0,
                        div_pos_weight=5.0):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            div_proposal_mlp: optional DivisionProposalMLP for affinity loss (both daughters)
            score_consist_loss_coef: weight for score consistency loss across frames (0 = disabled)
        """
        super().__init__(num_classes, matcher, weight_dict, losses)
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss = True
        self.losses_dict = {}
        self._current_frame_idx = 0
        self.score_consist_loss_coef = score_consist_loss_coef
        self.mask_loss_coef = mask_loss_coef
        self.div_pos_weight = div_pos_weight
        # Store as a plain attribute (not an nn.Module child) so this criterion does
        # not double-register the MLP.  The MLP is owned by the MOTR model and is
        # included in model.parameters() / model.state_dict() from there.
        object.__setattr__(self, 'div_proposal_mlp', div_proposal_mlp)

    def initialize_for_single_clip(self, gt_instances: List[Instances]):
        self.gt_instances = gt_instances
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}
        self._prev_scores = {}  # obj_id (int) -> detached score from previous frame

    def _step(self):
        self._current_frame_idx += 1

    def calc_loss_for_track_scores(self, track_instances: Instances):
        frame_id = self._current_frame_idx - 1
        gt_instances = self.gt_instances[frame_id]
        outputs = {
            'pred_logits': track_instances.track_scores[None],
        }
        device = track_instances.track_scores.device

        num_tracks = len(track_instances)
        src_idx = torch.arange(num_tracks, dtype=torch.long, device=device)
        tgt_idx = track_instances.matched_gt_idxes  # -1 for FP tracks and disappeared tracks

        track_losses = self.get_loss('labels',
                                     outputs=outputs,
                                     gt_instances=[gt_instances],
                                     indices=[(src_idx, tgt_idx)],
                                     num_boxes=1)
        self.losses_dict.update(
            {'frame_{}_track_{}'.format(frame_id, key): value for key, value in
             track_losses.items()})

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def get_loss(self, loss, outputs, gt_instances, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, gt_instances, indices, num_boxes, **kwargs)

    def loss_boxes(self, outputs, gt_instances: List[Instances], indices: List[tuple], num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # We ignore the regression loss of the track-disappear slots.
        #TODO: Make this filter process more elegant.
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]

        target_boxes = torch.cat(
            [gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0) # size(16)
        mask = (target_obj_ids != -1)

        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[mask]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[mask])))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_labels(self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # Only use the first num_classes channels; the last channel is the division logit
        # which is supervised separately via BCE on div_flags.
        src_logits = outputs['pred_logits'][..., :self.num_classes]
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # The matched gt for disappear track query is set -1.
        labels = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            labels_per_img = torch.ones_like(J)
            # set labels of track-appear slots to 0.
            if len(gt_per_img) > 0:
                labels_per_img[J != -1] = gt_per_img.labels[J[J != -1]]
            labels.append(labels_per_img)
        target_classes_o = torch.cat(labels)
        target_classes[idx] = target_classes_o
        if self.focal_loss:
            gt_labels_target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[:, :, :-1]  # no loss for the last (background) class
            gt_labels_target = gt_labels_target.to(src_logits)
            loss_ce = sigmoid_focal_loss(src_logits.flatten(1),
                                             gt_labels_target.flatten(1),
                                             alpha=0.25,
                                             gamma=2,
                                             num_boxes=num_boxes, mean_in_dim1=False)
            loss_ce = loss_ce.sum()
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def match_for_single_frame(self, outputs: dict):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        n_proposals    = outputs_without_aux.get('n_proposals', 0)
        proposal_start = outputs_without_aux.get('proposal_start', 0)
        proposal_end   = outputs_without_aux.get('proposal_end', proposal_start + n_proposals)

        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.
        track_instances: Instances = outputs_without_aux['track_instances']
        pred_logits_i = track_instances.pred_logits  # predicted logits of i-th image.
        pred_boxes_i = track_instances.pred_boxes  # predicted boxes of i-th image.

        obj_idxes = gt_instances_i.obj_ids
        outputs_i = {
            'pred_logits': pred_logits_i.unsqueeze(0),
            'pred_boxes': pred_boxes_i.unsqueeze(0),
        }

        # step1. inherit and update the previous tracks.
        num_disappear_track = 0
        track_instances.matched_gt_idxes[:] = -1
        i, j = torch.where(track_instances.obj_idxes[:, None] == obj_idxes)
        track_instances.matched_gt_idxes[i] = j

        full_track_idxes = torch.arange(len(track_instances), dtype=torch.long, device=pred_logits_i.device)
        matched_track_idxes = (track_instances.obj_idxes >= 0)  # occu
        prev_matched_indices = torch.stack(
            [full_track_idxes[matched_track_idxes], track_instances.matched_gt_idxes[matched_track_idxes]], dim=1)

        # Score consistency loss: carry-over tracks that still have a GT match should
        # maintain stable confidence across frames.  We penalise the MSE between the
        # current frame's score and the (detached) score stored from the previous frame.
        # This teaches the model to commit to a confidence level rather than oscillating
        # around the threshold, which is the root cause of partial-coverage tracks.
        if self.score_consist_loss_coef > 0 and self._current_frame_idx > 0:
            # Always write the key so all ranks have the same keys in loss_dict
            # (prevent reduce_dict deadlock when some ranks have no carry-overs).
            loss_consist = pred_logits_i.sum() * 0.0
            if len(prev_matched_indices) > 0:
                still_gt = prev_matched_indices[:, 1] >= 0
                if still_gt.sum() > 0:
                    carry_src = prev_matched_indices[still_gt, 0]
                    cur_scores = pred_logits_i[carry_src, 0].sigmoid()
                    prev_score_vals, valid = [], []
                    for src in carry_src.tolist():
                        oid = int(track_instances.obj_idxes[src].item())
                        if oid in self._prev_scores:
                            prev_score_vals.append(self._prev_scores[oid])
                            valid.append(True)
                        else:
                            valid.append(False)
                    if any(valid):
                        valid_t = torch.tensor(valid, dtype=torch.bool, device=cur_scores.device)
                        prev_t = torch.stack(prev_score_vals).to(cur_scores.device)
                        loss_consist = F.mse_loss(cur_scores[valid_t], prev_t, reduction='mean')
            self.losses_dict[
                'frame_{}_loss_score_consist'.format(self._current_frame_idx)
            ] = loss_consist

        # Update stored scores for carry-over tracks with valid GT match.
        if len(prev_matched_indices) > 0:
            still_gt = prev_matched_indices[:, 1] >= 0
            if still_gt.sum() > 0:
                carry_src = prev_matched_indices[still_gt, 0]
                for src in carry_src.tolist():
                    oid = int(track_instances.obj_idxes[src].item())
                    self._prev_scores[oid] = pred_logits_i[src, 0].sigmoid().detach().cpu()

        # step2. select the unmatched slots.
        # note that the FP tracks whose obj_idxes are -2 will not be selected here.
        unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]

        # step3. select the untracked gt instances (new tracks).
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]

        tgt_state = torch.zeros(len(gt_instances_i), device=pred_logits_i.device)
        tgt_state[tgt_indexes] = 1
        untracked_tgt_indexes = torch.arange(len(gt_instances_i), device=pred_logits_i.device)[tgt_state == 0]
        # untracked_tgt_indexes = select_unmatched_indexes(tgt_indexes, len(gt_instances_i))
        untracked_gt_instances = gt_instances_i[untracked_tgt_indexes]

        def match_for_single_decoder_layer(unmatched_outputs, matcher):
            new_track_indices = matcher(unmatched_outputs,
                                             [untracked_gt_instances])  # list[tuple(src_idx, tgt_idx)]

            src_idx = new_track_indices[0][0]
            tgt_idx = new_track_indices[0][1]
            # concat src and tgt.
            new_matched_indices = torch.stack([unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]],
                                              dim=1).to(pred_logits_i.device)
            return new_matched_indices

        # step4. do matching between the unmatched slots and GTs.
        unmatched_outputs = {
            'pred_logits': track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0),
        }
        new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher)

        # step5. update obj_idxes according to the new matching result.
        track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[new_matched_indices[:, 1]].long()
        track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

        # step6. calculate iou.
        active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0)
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[track_instances.matched_gt_idxes[active_idxes]]
            active_track_boxes = box_ops.box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

        # step7. merge the unmatched pairs and the matched pairs.
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)

        # step8. calculate losses.
        self.num_samples += len(gt_instances_i) + num_disappear_track
        self.sample_device = pred_logits_i.device
        for loss in self.losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_i,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                                           num_boxes=1)
            self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_track_loss.items()})

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                unmatched_outputs_layer = {
                    'pred_logits': aux_outputs['pred_logits'][0, unmatched_track_idxes].unsqueeze(0),
                    'pred_boxes': aux_outputs['pred_boxes'][0, unmatched_track_idxes].unsqueeze(0),
                }
                new_matched_indices_layer = match_for_single_decoder_layer(unmatched_outputs_layer, self.matcher)
                matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices], dim=0)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss,
                                           aux_outputs,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])],
                                           num_boxes=1, )
                    self.losses_dict.update(
                        {'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
                         l_dict.items()})

        if 'ps_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['ps_outputs']):
                ar = torch.arange(len(gt_instances_i), device=obj_idxes.device)
                l_dict = self.get_loss('boxes',
                                        aux_outputs,
                                        gt_instances=[gt_instances_i],
                                        indices=[(ar, ar)],
                                        num_boxes=1, )
                self.losses_dict.update(
                    {'frame_{}_ps{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
                        l_dict.items()})

        # ------------------------------------------------------------------
        # Division losses (Cell-TRACTR style: 8-coord bbox_embed + flex_div)
        #
        # pred_logits[:, num_classes]: the division classification logit.
        #   At training: supervised by div_flags (BCE with pos_weight).
        #   At inference: gates daughter2 spawning in _spawn_daughter2_tracks.
        # pred_div_boxes (= pred_boxes[:, 4:]): daughter2 position, supervised
        #   at division frames only.
        #
        # flex_div: if the model fires division (logit sigmoid ≥ 0.5) on a cell
        #   that divides at the NEXT frame rather than the current frame, we
        #   accept the early prediction and supervise with the next frame's D2 box.
        #   This aligns training with the model's preferred division timing.
        # ------------------------------------------------------------------
        _FLEX_THRESH = 0.5
        has_div = gt_instances_i.has('div_flags')
        if has_div:
            valid_mask = matched_indices[:, 1] >= 0
            if valid_mask.sum() > 0:
                src_idx = matched_indices[valid_mask, 0]
                tgt_idx = matched_indices[valid_mask, 1]
                gt_div_flags = gt_instances_i.div_flags[tgt_idx].to(pred_logits_i)

                # --- flex_div: look one frame ahead for early-division predictions ---
                flex_div_flags = gt_div_flags.clone()
                flex_div_box2s = (gt_instances_i.div_box2[tgt_idx].to(pred_logits_i.device).clone()
                                  if gt_instances_i.has('div_box2') else None)
                frame_i = self._current_frame_idx
                if frame_i + 1 < len(self.gt_instances):
                    gt_next = self.gt_instances[frame_i + 1]
                    if gt_next.has('div_flags') and pred_logits_i.shape[-1] > self.num_classes:
                        for k in range(len(src_idx)):
                            if flex_div_flags[k] > 0.5:
                                continue  # already a true division frame
                            obj_id = gt_instances_i.obj_ids[tgt_idx[k]]
                            next_match = (gt_next.obj_ids == obj_id).nonzero(as_tuple=True)[0]
                            if len(next_match) == 0:
                                continue
                            next_t = next_match[0]
                            if gt_next.div_flags[next_t] < 0.5:
                                continue  # doesn't divide at t+1 either
                            # Cell divides at t+1 — check if model fires early
                            div_score_k = pred_logits_i[src_idx[k], self.num_classes].sigmoid()
                            if div_score_k >= _FLEX_THRESH:
                                flex_div_flags[k] = 1.0
                                if (flex_div_box2s is not None and
                                        gt_next.has('div_box2')):
                                    flex_div_box2s[k] = gt_next.div_box2[next_t].to(flex_div_box2s)

                # Division classification loss: pred_logits[:, num_classes] is the
                # primary division signal.  pos_weight handles ~2-5% positive rate.
                #
                # Balanced-negative sampling: keep ALL positives + at most 5× negatives.
                # Previously reduction='mean' over all K matched tracks caused the
                # per-event gradient to be 14× weaker in dense scenes (K=70) vs sparse
                # scenes (K=5), so the model never learned division for dense sequences.
                # With a fixed neg:pos ratio the gradient is density-independent.
                if pred_logits_i.shape[-1] > self.num_classes:
                    pred_div_logit = pred_logits_i[src_idx, self.num_classes]
                    pos_weight_cls = torch.tensor(self.div_pos_weight, device=pred_div_logit.device)
                    pos_mask = flex_div_flags > 0.5
                    n_pos_cls = int(pos_mask.sum().item())
                    if n_pos_cls > 0:
                        # Subsample negatives to at most 5 × n_pos for density-independent
                        # gradient.  This closes the 14× gap between K=5 and K=70 scenes.
                        neg_idx = (~pos_mask).nonzero(as_tuple=True)[0]
                        n_neg_keep = min(len(neg_idx), 5 * n_pos_cls)
                        if len(neg_idx) > n_neg_keep:
                            perm = torch.randperm(len(neg_idx), device=neg_idx.device)
                            neg_idx = neg_idx[perm[:n_neg_keep]]
                        pos_idx = pos_mask.nonzero(as_tuple=True)[0]
                        sample_idx = torch.cat([pos_idx, neg_idx])
                        loss_div_class = F.binary_cross_entropy_with_logits(
                            pred_div_logit[sample_idx], flex_div_flags[sample_idx],
                            pos_weight=pos_weight_cls, reduction='mean')
                    else:
                        # No divisions in this frame — suppress false positives with
                        # a small mean loss over all negatives.
                        loss_div_class = F.binary_cross_entropy_with_logits(
                            pred_div_logit, flex_div_flags, pos_weight=pos_weight_cls,
                            reduction='mean')
                    self.losses_dict[
                        'frame_{}_loss_div_class'.format(self._current_frame_idx)
                    ] = loss_div_class

                # Daughter2-box regression loss (L1, only at flex_div_flags=1 frames).
                # When a decoded proposal overlaps GT D2 with IoU > 0.3, snap the
                # regression target to that proposal box — this aligns training with
                # inference, which spawns D2 at the nearest proposal position.
                if gt_instances_i.has('div_box2') and track_instances.has('pred_div_boxes'):
                    div_cell_mask = flex_div_flags > 0.5
                    if div_cell_mask.sum() > 0:
                        pred_db = track_instances.pred_div_boxes[src_idx[div_cell_mask]]
                        gt_db = flex_div_box2s[div_cell_mask] if flex_div_box2s is not None else \
                                gt_instances_i.div_box2[tgt_idx[div_cell_mask]].to(pred_db)
                        valid_db = gt_db.abs().sum(dim=1) > 1e-6
                        if valid_db.sum() > 0:
                            reg_target = gt_db[valid_db]
                            if n_proposals > 0:
                                prop_boxes = track_instances.pred_boxes[proposal_start:proposal_end].detach()
                                iou_snap = pairwise_iou(
                                    Boxes(box_ops.box_cxcywh_to_xyxy(reg_target)),
                                    Boxes(box_ops.box_cxcywh_to_xyxy(prop_boxes)),
                                )  # [M, N_proposals]
                                best_iou_s, best_idx_s = iou_snap.max(dim=1)
                                snap_ok = best_iou_s > 0.3
                                if snap_ok.any():
                                    reg_target = reg_target.clone()
                                    reg_target[snap_ok] = prop_boxes[best_idx_s[snap_ok]]
                            loss_div_box = F.l1_loss(pred_db[valid_db], reg_target, reduction='mean')
                        else:
                            loss_div_box = track_instances.pred_div_boxes.sum() * 0.0
                    else:
                        loss_div_box = track_instances.pred_div_boxes.sum() * 0.0
                    self.losses_dict[
                        'frame_{}_loss_div_box'.format(self._current_frame_idx)
                    ] = loss_div_box

                # Division proposal affinity loss (single MLP, both daughters as positives).
                # The MLP scores cat(mother_hs, proposal_hs[j]) for every proposal j.
                # GT D1 (main box of merged entry) and GT D2 (div_box2) are both marked
                # as positives in a binary cross-entropy target — the network learns
                # "how daughter-like is this proposal" without distinguishing D1 from D2.
                # At inference the top-2 proposals become D1 and D2.
                if (self.div_proposal_mlp is not None and n_proposals > 0 and
                        gt_instances_i.has('div_box2')):
                    div_cell_mask = flex_div_flags > 0.5
                    if div_cell_mask.sum() > 0:
                        mother_src = src_idx[div_cell_mask]
                        gt_d2 = (flex_div_box2s[div_cell_mask] if flex_div_box2s is not None else
                                 gt_instances_i.div_box2[tgt_idx[div_cell_mask]].to(pred_logits_i))
                        gt_d1 = gt_instances_i.boxes[tgt_idx[div_cell_mask]].to(pred_logits_i)
                        valid = (gt_d2.abs().sum(dim=1) > 1e-6) & (gt_d1.abs().sum(dim=1) > 1e-6)
                        if valid.sum() > 0:
                            prop_boxes = track_instances.pred_boxes[proposal_start:proposal_end].detach()
                            iou_d1 = pairwise_iou(
                                Boxes(box_ops.box_cxcywh_to_xyxy(gt_d1[valid])),
                                Boxes(box_ops.box_cxcywh_to_xyxy(prop_boxes)),
                            )  # [K_valid, N_proposals]
                            iou_d2 = pairwise_iou(
                                Boxes(box_ops.box_cxcywh_to_xyxy(gt_d2[valid])),
                                Boxes(box_ops.box_cxcywh_to_xyxy(prop_boxes)),
                            )  # [K_valid, N_proposals]
                            best_iou_d1, best_idx_d1 = iou_d1.max(dim=1)
                            best_iou_d2, best_idx_d2 = iou_d2.max(dim=1)
                            # Both daughters must have a matching proposal (IoU > 0.3)
                            aff_ok = (best_iou_d1 > 0.3) & (best_iou_d2 > 0.3)
                            if aff_ok.sum() > 0:
                                K_aff    = aff_ok.sum().item()
                                m_hs_all = track_instances.output_embedding[mother_src[valid][aff_ok]]
                                p_hs_all = track_instances.output_embedding[proposal_start:proposal_end]
                                prop_boxes_all = track_instances.pred_boxes[proposal_start:proposal_end].detach()

                                # Filter proposals occupied by tracked non-dividing cells,
                                # matching the free-proposal restriction used at inference.
                                tracked_mask_tr = (track_instances.obj_idxes >= 0).clone()
                                tracked_mask_tr[mother_src[valid][aff_ok]] = False
                                if tracked_mask_tr.any():
                                    tr_xyxy   = box_ops.box_cxcywh_to_xyxy(
                                        track_instances.pred_boxes[tracked_mask_tr].clamp(0, 1).detach())
                                    pr_xyxy   = box_ops.box_cxcywh_to_xyxy(prop_boxes_all.clamp(0, 1))
                                    iou_occ_tr = pairwise_iou(Boxes(pr_xyxy), Boxes(tr_xyxy))
                                    free_mask_tr = iou_occ_tr.max(dim=1).values < 0.5
                                else:
                                    free_mask_tr = torch.ones(n_proposals, dtype=torch.bool,
                                                              device=prop_boxes_all.device)
                                free_idx_tr = free_mask_tr.nonzero(as_tuple=True)[0]

                                if len(free_idx_tr) >= 1:
                                    p_hs_free        = p_hs_all[free_idx_tr]
                                    free_prop_boxes_tr = prop_boxes_all[free_idx_tr]   # [N_free, 4]
                                    m_boxes_aff = track_instances.pred_boxes[
                                        mother_src[valid][aff_ok]].detach()            # [K_aff, 4]
                                    m_diag_aff  = (m_boxes_aff[:, 2].pow(2) +
                                                   m_boxes_aff[:, 3].pow(2)).sqrt().clamp(min=1e-4)
                                    dx_tr  = (free_prop_boxes_tr[None, :, 0] -
                                              m_boxes_aff[:, None, 0]) / m_diag_aff[:, None]
                                    dy_tr  = (free_prop_boxes_tr[None, :, 1] -
                                              m_boxes_aff[:, None, 1]) / m_diag_aff[:, None]
                                    rel_pos_tr = torch.stack(
                                        [dx_tr, dy_tr, (dx_tr.pow(2) + dy_tr.pow(2)).sqrt()],
                                        dim=-1)                                        # [K_aff, N_free, 3]
                                    logits_aff  = self.div_proposal_mlp(
                                        m_hs_all, p_hs_free, rel_pos_tr)              # [K_aff, N_free]
                                    # Remap GT targets to free-proposal indices
                                    d1_free = (free_idx_tr.unsqueeze(0) ==
                                               best_idx_d1[aff_ok].unsqueeze(1)).int().argmax(dim=1)
                                    d2_free = (free_idx_tr.unsqueeze(0) ==
                                               best_idx_d2[aff_ok].unsqueeze(1)).int().argmax(dim=1)
                                    # Only keep rows where both GT proposals survived the filter
                                    d1_in = free_mask_tr[best_idx_d1[aff_ok]]
                                    d2_in = free_mask_tr[best_idx_d2[aff_ok]]
                                    both_in = d1_in & d2_in
                                    if both_in.any():
                                        N_free = len(free_idx_tr)
                                        target_aff = torch.zeros(both_in.sum().item(), N_free,
                                                                 device=logits_aff.device)
                                        target_aff[torch.arange(both_in.sum().item()),
                                                   d1_free[both_in]] = 1.0
                                        target_aff[torch.arange(both_in.sum().item()),
                                                   d2_free[both_in]] = 1.0
                                        # pos_weight counters heavy imbalance: N_free positions,
                                        # only 2 positives per row (D1 and D2).
                                        pos_w_aff = torch.tensor(
                                            min(10.0, max(1.0, (N_free - 2) / 2.0)),
                                            device=logits_aff.device)
                                        loss_div_affinity = F.binary_cross_entropy_with_logits(
                                            logits_aff[both_in], target_aff,
                                            pos_weight=pos_w_aff)
                                        self.losses_dict[
                                            'frame_{}_loss_div_affinity'.format(self._current_frame_idx)
                                        ] = loss_div_affinity

                # Training-time D2 spawning (teacher-forced at flex_div frames).
                if gt_instances_i.has('div_box2'):
                    d2_mask = flex_div_flags > 0.5
                    if d2_mask.any():
                        gt_d2 = (flex_div_box2s[d2_mask] if flex_div_box2s is not None else
                                 gt_instances_i.div_box2[tgt_idx[d2_mask]].to(pred_logits_i.device))
                        valid_d2 = gt_d2.abs().sum(dim=1) > 1e-6
                        if valid_d2.any():
                            div_src = src_idx[d2_mask][valid_d2]
                            mothers = track_instances[div_src]
                            d2      = track_instances[div_src]
                            d2_pos  = gt_d2[valid_d2].clone()
                            # Snap D2 spawn to nearest proposal (IoU > 0.3) so the
                            # starting position matches what inference does via the MLP.
                            if n_proposals > 0:
                                spawn_prop = track_instances.pred_boxes[proposal_start:proposal_end].detach()
                                iou_spawn = pairwise_iou(
                                    Boxes(box_ops.box_cxcywh_to_xyxy(d2_pos)),
                                    Boxes(box_ops.box_cxcywh_to_xyxy(spawn_prop)),
                                )  # [M, N_proposals]
                                best_iou_sp, best_idx_sp = iou_spawn.max(dim=1)
                                spawn_ok = best_iou_sp > 0.3
                                if spawn_ok.any():
                                    d2_pos[spawn_ok] = spawn_prop[best_idx_sp[spawn_ok]]
                            d2.ref_pts    = d2_pos.clone()
                            d2.pred_boxes = d2_pos.clone()
                            d2.scores           = torch.ones_like(mothers.scores)
                            d2.iou              = torch.ones_like(mothers.iou)
                            d2.obj_idxes        = torch.full_like(mothers.obj_idxes, -1)
                            d2.disappear_time   = torch.zeros_like(mothers.disappear_time)
                            d2.matched_gt_idxes = torch.full_like(mothers.matched_gt_idxes, -1)
                            d2.parent_obj_id    = mothers.obj_idxes.clone()
                            d2.output_embedding = torch.zeros_like(mothers.output_embedding)
                            _qd = mothers.query_pos.shape[-1] // 4
                            d2.query_pos        = pos2posemb(d2.ref_pts, num_pos_feats=_qd)
                            d2.mem_bank         = torch.zeros_like(mothers.mem_bank)
                            d2.mem_padding_mask = torch.ones_like(mothers.mem_padding_mask)
                            # Snap D1 (mother's slot) to nearest D1 proposal, symmetric with D2.
                            # GT D1 is the main box of the merged entry at the division frame.
                            if n_proposals > 0:
                                gt_d1_boxes = gt_instances_i.boxes[tgt_idx[d2_mask][valid_d2]].to(pred_logits_i.device)
                                spawn_prop  = track_instances.pred_boxes[proposal_start:proposal_end].detach()
                                iou_d1 = pairwise_iou(
                                    Boxes(box_ops.box_cxcywh_to_xyxy(gt_d1_boxes)),
                                    Boxes(box_ops.box_cxcywh_to_xyxy(spawn_prop)),
                                )  # [M, N_proposals]
                                best_iou_d1, best_idx_d1 = iou_d1.max(dim=1)
                                d1_snap_ok = best_iou_d1 > 0.3
                                if d1_snap_ok.any():
                                    d1_pos = spawn_prop[best_idx_d1]
                                    snap_idxs = div_src[d1_snap_ok]
                                    snap_pos  = d1_pos[d1_snap_ok]
                                    # pred_boxes is a view of the decoder output; clone
                                    # before in-place write to satisfy autograd.
                                    track_instances.pred_boxes = track_instances.pred_boxes.clone()
                                    track_instances.ref_pts[snap_idxs]    = snap_pos
                                    track_instances.pred_boxes[snap_idxs] = snap_pos
                                    track_instances.query_pos[snap_idxs]  = pos2posemb(
                                        snap_pos, num_pos_feats=_qd)
                            track_instances  = Instances.cat([track_instances, d2])
                            if track_instances.has('parent_obj_id'):
                                track_instances.parent_obj_id[div_src] = mothers.obj_idxes.clone()
                            track_instances.obj_idxes[div_src] = -1
                            track_instances.scores[div_src] = torch.ones_like(mothers.scores)
                            track_instances.iou[div_src] = torch.ones_like(mothers.iou)
                            track_instances.disappear_time[div_src] = torch.zeros_like(mothers.disappear_time)
                            track_instances.matched_gt_idxes[div_src] = torch.full_like(mothers.matched_gt_idxes, -1)
            else:
                if track_instances.has('pred_div_boxes'):
                    self.losses_dict[
                        'frame_{}_loss_div_box'.format(self._current_frame_idx)
                    ] = track_instances.pred_div_boxes.sum() * 0.0
                if pred_logits_i.shape[-1] > self.num_classes:
                    self.losses_dict[
                        'frame_{}_loss_div_class'.format(self._current_frame_idx)
                    ] = pred_logits_i[:, self.num_classes].sum() * 0.0

        # reduce_dict (called in the engine) does a single NCCL allreduce on a
        # stacked tensor of ALL loss values.  Every rank must have exactly the
        # same keys.  loss_div_affinity is only written above when aff_ok fires,
        # which differs per rank/batch → size mismatch → NCCL deadlock.
        # Always write a zero-valued (but graph-connected) entry as fallback.
        if self.div_proposal_mlp is not None:
            key = 'frame_{}_loss_div_affinity'.format(self._current_frame_idx)
            if key not in self.losses_dict:
                self.losses_dict[key] = sum(
                    p.sum() * 0.0 for p in self.div_proposal_mlp.parameters()
                )

        # Mask loss: for each GT-matched carry-over or new query, penalise
        # sigmoid_focal + dice between predicted mask and GT binary mask.
        # The key is ALWAYS written so all ranks have the same keys in loss_dict,
        # preventing reduce_dict deadlocks when some ranks have no GT masks.
        if self.mask_loss_coef > 0 and 'pred_masks' in outputs_without_aux:
            pred_masks = outputs_without_aux['pred_masks']  # [1, N_queries, H, W]
            loss_m = pred_masks.sum() * 0.0  # zero sentinel keeps grad graph alive
            if gt_instances_i.has('masks'):
                valid = matched_indices[:, 1] >= 0
                if valid.sum() > 0:
                    src_idx = matched_indices[valid, 0]
                    tgt_idx = matched_indices[valid, 1]
                    src_masks = pred_masks[0, src_idx]          # [K, H, W]
                    tgt_masks = gt_instances_i.masks[tgt_idx].float()  # [K, H, W]
                    if tgt_masks.shape[-2:] != src_masks.shape[-2:]:
                        tgt_masks = F.interpolate(
                            tgt_masks.unsqueeze(1),
                            size=src_masks.shape[-2:],
                            mode='nearest').squeeze(1)
                    K = src_masks.shape[0]
                    loss_m = dice_loss(src_masks.flatten(1), tgt_masks.flatten(1), K)
                    loss_m = loss_m + sigmoid_focal_loss(
                        src_masks.flatten(1), tgt_masks.flatten(1),
                        num_boxes=K, alpha=0.25, gamma=2, mean_in_dim1=False)
            self.losses_dict[
                'frame_{}_loss_mask'.format(self._current_frame_idx)
            ] = loss_m

        self._step()
        return track_instances

    def forward(self, outputs, input_data: dict):
        # losses of each frame are calculated during the model's forwarding and are outputted by the model as outputs['losses_dict].
        losses = outputs.pop("losses_dict")
        num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in losses.items():
            losses[loss_name] /= num_samples
        return losses


class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.6, filter_score_thresh=0.5, miss_tolerance=10):
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


class TrackerPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, track_instances: Instances, target_size) -> Instances:
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits = track_instances.pred_logits
        out_bbox = track_instances.pred_boxes

        # prob = out_logits.sigmoid()
        scores = out_logits[..., 0].sigmoid()
        # scores, labels = prob.max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_size
        scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
        boxes = boxes * scale_fct[None, :]

        track_instances.boxes = boxes
        track_instances.scores = scores
        track_instances.labels = torch.full_like(scores, 0)
        # track_instances.remove('pred_logits')
        # track_instances.remove('pred_boxes')
        return track_instances


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MOTR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_queries_detect, num_feature_levels, criterion, track_embed,
                 aux_loss=True, with_box_refine=False, two_stage=False, memory_bank=None, use_checkpoint=False, query_denoise=0,
                 div_score_thresh=0.4, div_proposal_mlp=None, mask_head=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.num_queries_detect = num_queries_detect
        self.track_embed = track_embed
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 8, 3)
        self.num_feature_levels = num_feature_levels
        self.use_checkpoint = use_checkpoint
        self.query_denoise = query_denoise
        self.position = nn.Embedding(num_queries, 4)
        self.position_detect = nn.Embedding(num_queries_detect, 4)
        self.yolox_embed = nn.Embedding(1, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_detect = nn.Embedding(num_queries_detect, hidden_dim)
        if query_denoise:
            self.refine_embed = nn.Embedding(1, hidden_dim)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.shared_decoder = getattr(transformer.decoder, 'shared_decoder', False)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes + 1) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        nn.init.uniform_(self.position.weight.data, 0, 1)
        nn.init.uniform_(self.position_detect.weight.data, 0, 1)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            if self.shared_decoder:
                self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
                self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            else:
                self.class_embed = _get_clones(self.class_embed, num_pred)
                self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:4], -2.0)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[6:8], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:4], -2.0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[6:8], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        self.post_process = TrackerPostProcess()
        self.track_base = RuntimeTrackerBase()
        self.div_score_thresh = div_score_thresh
        self.criterion = criterion
        # Registered as a model submodule so its weights appear in model.parameters()
        # and model.state_dict() — critical for optimizer updates and checkpoint I/O.
        # ClipMatcher holds only a non-owning reference (via object.__setattr__).
        self.div_proposal_mlp = div_proposal_mlp  # nn.Module or None
        self.mask_head = mask_head                 # SimpleMaskHead or None
        self.memory_bank = memory_bank
        self.mem_bank_len = 0 if memory_bank is None else memory_bank.max_his_length

    def _generate_empty_tracks(self, proposals=None):
        track_instances = Instances((1, 1))
        num_queries, d_model = self.query_embed.weight.shape  # (300, 512)
        device = self.query_embed.weight.device
        if proposals is None:
            track_instances.ref_pts = self.position.weight
            track_instances.query_pos = self.query_embed.weight
        else:
            track_instances.ref_pts = torch.cat([self.position.weight, proposals[:, :4]])
            # proposals[:, :4] = cxcywh box; proposals[:, 4] = score (not used for pos emb).
            # pos2posemb expects [N, D_pos] and produces [N, D_pos * num_pos_feats].
            # With D_pos=4 and num_pos_feats=d_model//4 the output is [N, d_model]. ✓
            track_instances.query_pos = torch.cat([self.query_embed.weight, pos2posemb(proposals[:, :4], num_pos_feats=d_model // 4) + self.yolox_embed.weight])
        track_instances.output_embedding = torch.zeros((len(track_instances), d_model), device=device)
        track_instances.obj_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros((len(track_instances), ), dtype=torch.long, device=device)
        track_instances.iou = torch.ones((len(track_instances),), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros((len(track_instances), self.num_classes + 1), dtype=torch.float, device=device)

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros((len(track_instances), mem_bank_len, d_model), dtype=torch.float32, device=device)
        track_instances.mem_padding_mask = torch.ones((len(track_instances), mem_bank_len), dtype=torch.bool, device=device)
        track_instances.save_period = torch.zeros((len(track_instances), ), dtype=torch.float32, device=device)
        # Daughter-box prediction: normalised cxcywh for daughter2 (from bbox_embed's last 4 coords).
        track_instances.pred_div_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float32, device=device)
        # Parent track ID: set on the D2 daughter when spawned; -1 otherwise.
        track_instances.parent_obj_id = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)

        return track_instances.to(self.query_embed.weight.device)

    def clear(self):
        self.track_base.clear()
        self.division_log: list = []

    def _spawn_daughter2_tracks(self, track_instances: Instances,
                                n_proposals: int = 0,
                                proposal_start: int = 0,
                                proposal_end: int = 0) -> Instances:
        """At the division frame: for any active track whose division logit exceeds
        div_score_thresh, spawn D1 (reset mother) and D2 (new query).

        When DivisionProposalMLP is available and proposals are present, D2 is placed
        at the proposal selected by the MLP (matching the affinity loss used at training).
        Otherwise falls back to pred_div_boxes."""
        if not track_instances.has('pred_div_boxes'):
            return track_instances

        # Division signal: class_embed channel 1 (pred_logits[:, num_classes]).
        # This is the same logit supervised by loss_div_class during training.
        div_scores = torch.sigmoid(track_instances.pred_logits[:, self.num_classes])
        active_mask = track_instances.obj_idxes >= 0
        is_dividing = (div_scores >= self.div_score_thresh) & active_mask & (track_instances.scores >= self.track_base.filter_score_thresh)

        # Debug: print max div_score among active tracks to diagnose threshold issues.
        if active_mask.any():
            max_ds = div_scores[active_mask].max().item()
            n_above = (div_scores[active_mask] >= self.div_score_thresh).sum().item()
            if max_ds > 0.2 or n_above > 0:  # only print when non-trivial
                print(f'  [div-debug] n_active={active_mask.sum().item()} max_div_score={max_ds:.3f} n_above_thresh({self.div_score_thresh:.2f})={n_above}')

        # One-frame cooldown: just-born daughters and freshly-divided mothers both
        # carry parent_obj_id >= 0, preventing immediate re-division.
        if track_instances.has('parent_obj_id'):
            is_immune = track_instances.parent_obj_id >= 0
            is_dividing = is_dividing & ~is_immune
            track_instances.parent_obj_id[is_immune] = -1

        dividing_idxs = is_dividing.nonzero(as_tuple=True)[0]

        # Cap at 3 simultaneous divisions per frame to prevent OOM cascades.
        if len(dividing_idxs) > 3:
            topk = torch.topk(div_scores[dividing_idxs], k=3).indices
            dividing_idxs = dividing_idxs[topk]

        if len(dividing_idxs) == 0:
            return track_instances

        # --- Proposal-based D1 and D2 placement (single MLP, top-2 selection) ---
        # The MLP scores cat(mother_hs, proposal_hs) for every proposal.
        # It learns "how daughter-like is this proposal" without distinguishing D1/D2.
        # Top-1 proposal → D1 (mother's slot snapped to it).
        # Top-2 proposal → D2 (new spawned track).
        # If only one proposal scores high (large gap), only D2 is spawned (D1 inherits
        # mother's position as before). Fallback to pred_div_boxes when MLP unavailable.
        _mlp    = self.div_proposal_mlp
        use_mlp = _mlp is not None and n_proposals > 0 and proposal_end > proposal_start
        m_boxes = track_instances.pred_boxes[dividing_idxs]  # [K, 4] cxcywh
        m_diag  = (m_boxes[:, 2].pow(2) + m_boxes[:, 3].pow(2)).sqrt().clamp(min=1e-4)

        sel_d1_global = None
        sel_d2_global = None
        d1_pos        = None

        mlp_used = False
        if use_mlp:
            prop_hs    = track_instances.output_embedding[proposal_start:proposal_end]  # [N, d]
            mother_hs  = track_instances.output_embedding[dividing_idxs]               # [K, d]
            prop_boxes = track_instances.pred_boxes[proposal_start:proposal_end]        # [N, 4]

            # Restrict MLP to free proposals — those not already occupied by a tracked
            # non-dividing cell.  Daughters must appear at untracked positions; proposals
            # overlapping existing tracks are duplicates of those tracks, not daughters.
            tracked_mask = (track_instances.obj_idxes >= 0).clone()
            tracked_mask[dividing_idxs] = False   # dividing mothers are not "occupied"
            if tracked_mask.any():
                tracked_xyxy = box_ops.box_cxcywh_to_xyxy(
                    track_instances.pred_boxes[tracked_mask].clamp(0, 1))
                prop_xyxy = box_ops.box_cxcywh_to_xyxy(prop_boxes.clamp(0, 1))
                iou_occ   = pairwise_iou(Boxes(prop_xyxy), Boxes(tracked_xyxy))  # [N, N_tracked]
                free_mask = iou_occ.max(dim=1).values < 0.5
            else:
                free_mask = torch.ones(n_proposals, dtype=torch.bool, device=prop_boxes.device)

            free_idx = free_mask.nonzero(as_tuple=True)[0]  # indices into proposal slice

            if len(free_idx) >= 1:
                free_prop_hs    = prop_hs[free_idx]
                free_prop_boxes = prop_boxes[free_idx]          # [N_free, 4] cxcywh
                dx  = (free_prop_boxes[None, :, 0] - m_boxes[:, None, 0]) / m_diag[:, None]
                dy  = (free_prop_boxes[None, :, 1] - m_boxes[:, None, 1]) / m_diag[:, None]
                rel_pos_inf = torch.stack([dx, dy, (dx.pow(2) + dy.pow(2)).sqrt()], dim=-1)
                logits = _mlp(mother_hs, free_prop_hs, rel_pos_inf)  # [K, N_free]

                if len(free_idx) >= 2:
                    top2_result  = logits.topk(2, dim=1)
                    top2_scores  = top2_result.values.sigmoid()   # [K, 2]
                    top2_indices = top2_result.indices             # [K, 2] into free_idx
                    # Two daughters found when the 2nd-best score also clears the bar.
                    # Confident: top-1 → D1 (snap mother slot), top-2 → D2 (new track).
                    # Not confident: only one location found → top-1 → D2 (best proposal),
                    #                D1 inherits mother's current position (no snap).
                    d1_confident  = top2_scores[:, 1] > 0.3       # [K] per-mother
                    best_d1_local = free_idx[top2_indices[:, 0]]  # top-1; used only when d1_confident
                    best_d2_local = torch.where(
                        d1_confident,
                        free_idx[top2_indices[:, 1]],  # confident: D2 = top-2
                        free_idx[top2_indices[:, 0]]   # not confident: D2 = best (top-1)
                    )
                    # D2 MLP score: top-2 when two daughters found, top-1 otherwise.
                    d2_mlp_score = torch.where(d1_confident, top2_scores[:, 1], top2_scores[:, 0])
                else:
                    best_d2_local = free_idx[logits.argmax(dim=1)]
                    best_d1_local = None
                    d1_confident  = None
                    d2_mlp_score  = logits[:, 0].sigmoid()   # only one free proposal

                # Validate D2: non-zero position, within plausible distance, and MLP
                # score above minimum threshold.  If no proposal is confident enough,
                # skip spawning rather than placing D2 at an arbitrary weak location.
                # Hard distance cap: daughters appear within ~2 mother-diagonals,
                # capped at 0.15 normalised (~90 px on a 600-px image ≈ 1 cell diameter).
                max_spawn_dist = (m_diag * 2.0).clamp(max=0.15)
                sel_d2_boxes = prop_boxes[best_d2_local]
                sel_d2_dist  = ((sel_d2_boxes[:, 0] - m_boxes[:, 0]).pow(2) +
                                (sel_d2_boxes[:, 1] - m_boxes[:, 1]).pow(2)).sqrt()
                d2_valid      = ((sel_d2_boxes.abs().sum(dim=1) > 1e-4) &
                                 (sel_d2_dist <= max_spawn_dist) &
                                 (d2_mlp_score > 0.3))
                max_spawn_dist = max_spawn_dist[d2_valid]
                dividing_idxs = dividing_idxs[d2_valid]
                best_d2_local = best_d2_local[d2_valid]
                if best_d1_local is not None:
                    best_d1_local = best_d1_local[d2_valid]
                    d1_confident  = d1_confident[d2_valid]
                if len(dividing_idxs) == 0:
                    return track_instances

                sel_d2_global = proposal_start + best_d2_local
                d2_pos = prop_boxes[best_d2_local].clone()
                if best_d1_local is not None:
                    d1_pos = prop_boxes[best_d1_local].clone()
                    m_boxes_cur = track_instances.pred_boxes[dividing_idxs]
                    # Reject D1 snaps that are spatially implausible (same cap as D2).
                    d1_snap_dist = ((d1_pos[:, 0] - m_boxes_cur[:, 0]).pow(2) +
                                    (d1_pos[:, 1] - m_boxes_cur[:, 1]).pow(2)).sqrt()
                    if d1_confident is not None:
                        d1_confident = d1_confident & (d1_snap_dist <= max_spawn_dist)
                    # Fall back to mother's current position for low-confidence D1 snaps.
                    if d1_confident is not None and not d1_confident.all():
                        d1_pos[~d1_confident] = m_boxes_cur[~d1_confident]
                    # Only suppress proposals that were actually adopted as D1.
                    conf_d1 = d1_confident if d1_confident is not None else torch.ones(
                        len(best_d1_local), dtype=torch.bool, device=best_d1_local.device)
                    if conf_d1.any():
                        sel_d1_global = proposal_start + best_d1_local[conf_d1]
                mlp_used = True

        if not mlp_used:
            # Fallback: no free proposals or MLP not configured — use pred_div_boxes for D2.
            d2_boxes = track_instances.pred_div_boxes[dividing_idxs]
            d2_dist  = ((d2_boxes[:, 0] - m_boxes[:, 0]).pow(2) +
                        (d2_boxes[:, 1] - m_boxes[:, 1]).pow(2)).sqrt()
            max_spawn_dist = (m_diag * 2.0).clamp(max=0.15)
            d2_valid      = (d2_boxes.abs().sum(dim=1) > 1e-4) & (d2_dist <= max_spawn_dist)
            dividing_idxs = dividing_idxs[d2_valid]
            if len(dividing_idxs) == 0:
                return track_instances
            d2_pos = track_instances.pred_div_boxes[dividing_idxs].clone()

        mothers = track_instances[dividing_idxs]
        d2      = track_instances[dividing_idxs]   # copy — all fields from mother
        d2.ref_pts    = d2_pos
        d2.pred_boxes = d2_pos
        d2.obj_idxes        = torch.full_like(mothers.obj_idxes, -1)
        d2.disappear_time   = torch.zeros_like(mothers.disappear_time)
        d2.scores           = torch.ones_like(mothers.scores)
        d2.iou              = torch.ones_like(mothers.iou)
        d2.matched_gt_idxes = torch.full_like(mothers.matched_gt_idxes, -1)
        d2.parent_obj_id    = mothers.obj_idxes.clone()
        # D2 starts as a fresh spatial query: zero content forces it to rely on
        # ref_pts alone, preventing D2 from inheriting D1's QIM key and collapsing
        # onto D1 in the next decoder step.
        d2.output_embedding  = torch.zeros_like(mothers.output_embedding)
        _qd = self.query_embed.weight.shape[-1] // 4
        d2.query_pos         = pos2posemb(d2.ref_pts, num_pos_feats=_qd)
        d2.mem_bank          = torch.zeros_like(mothers.mem_bank)
        d2.mem_padding_mask  = torch.ones_like(mothers.mem_padding_mask)  # True = no memory

        # D1: reset mother slot and snap its position to the D1-selected proposal.
        track_instances.obj_idxes[dividing_idxs]        = -1
        track_instances.scores[dividing_idxs]           = torch.ones_like(mothers.scores)
        track_instances.iou[dividing_idxs]              = torch.ones_like(mothers.iou)
        track_instances.disappear_time[dividing_idxs]   = torch.zeros_like(mothers.disappear_time)
        track_instances.matched_gt_idxes[dividing_idxs] = torch.full_like(mothers.matched_gt_idxes, -1)
        track_instances.parent_obj_id[dividing_idxs]    = mothers.obj_idxes.clone()
        if d1_pos is not None:
            track_instances.ref_pts[dividing_idxs]    = d1_pos
            track_instances.pred_boxes[dividing_idxs] = d1_pos
            track_instances.query_pos[dividing_idxs]  = pos2posemb(d1_pos, num_pos_feats=_qd)

        for pid in mothers.obj_idxes.tolist():
            self.division_log.append({'parent_id': int(pid), 'd1_id': None, 'd2_id': None})

        track_instances = Instances.cat([track_instances, d2])

        # Suppress adopted proposals so RuntimeTrackerBase does not assign them
        # new IDs alongside the daughter queries spawned from them.
        if sel_d2_global is not None:
            track_instances.scores[sel_d2_global]    = 0.0
            track_instances.obj_idxes[sel_d2_global] = -2
        if sel_d1_global is not None:
            track_instances.scores[sel_d1_global]    = 0.0
            track_instances.obj_idxes[sel_d1_global] = -2

        return track_instances

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, }
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def _forward_single_image(self, samples, track_instances: Instances, gtboxes=None):
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if gtboxes is not None:
            n_dt = len(track_instances)
            ps_tgt = self.refine_embed.weight.expand(gtboxes.size(0), -1)
            query_embed = torch.cat([track_instances.query_pos, ps_tgt])
            ref_pts = torch.cat([track_instances.ref_pts, gtboxes])
            attn_mask = torch.zeros((len(ref_pts), len(ref_pts)), dtype=bool, device=ref_pts.device)
            attn_mask[:n_dt, n_dt:] = True
        else:
            query_embed = track_instances.query_pos
            ref_pts = track_instances.ref_pts
            attn_mask = None

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(srcs, masks, pos, query_embed, ref_pts=ref_pts,
                             mem_bank=track_instances.mem_bank, mem_bank_pad_mask=track_instances.mem_padding_mask, attn_mask=attn_mask)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp[..., :4] += reference
                tmp[..., 4:6] += reference[..., :2]  # D2 center offset from same ref
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                tmp[..., 4:6] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            # Aux outputs (intermediate layers) only need the first 4D for box matching/loss.
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord[..., :4])
        out['hs'] = hs[-1]
        return out

    def _forward_single_image_detector(self, samples, track_instances: Instances):
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embed = track_instances.query_pos
        ref_pts = track_instances.ref_pts

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embed, ref_pts=ref_pts, mem_bank=None, mem_bank_pad_mask=None, attn_mask=None)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp[..., :4] += reference
                tmp[..., 4:6] += reference[..., :2]  # D2 center offset from same ref
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                tmp[..., 4:6] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord[..., :4]}
        out['track_scores'] = out['pred_logits'][..., :1].sigmoid()
        return out

    def _forward_single_image_detect_self(self, samples):
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)


        query_embed = self.query_embed_detect.weight
        ref_pts = self.position_detect.weight
        attn_mask = None

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(srcs, masks, pos, query_embed, ref_pts=ref_pts, attn_mask=attn_mask)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp[..., :4] += reference
                tmp[..., 4:6] += reference[..., :2]  # D2 center offset from same ref
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                tmp[..., 4:6] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1, ..., :4]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord[..., :4])
        out['scores'] = out['pred_logits'][..., :1].sigmoid()
        return out

    def _forward_single_image_proposals(self, samples):
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)


        query_embed = self.query_embed_detect.weight
        ref_pts = self.position_detect.weight
        attn_mask = None

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(srcs, masks, pos, query_embed, ref_pts=ref_pts, attn_mask=attn_mask)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp[..., :4] += reference
                tmp[..., 4:6] += reference[..., :2]  # D2 center offset from same ref
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                tmp[..., 4:6] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord[..., :4]}
        #if self.aux_loss:
        #    out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        out['scores'] = out['pred_logits'][..., :1].sigmoid()
        return out

    def _forward_single_image_proposals_light(self, samples):
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)


        query_embed = self.query_embed_detect.weight
        ref_pts = self.position_detect.weight
        attn_mask = None

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten = \
            self.transformer.forward_light_proposal(srcs, masks, pos, query_embed, ref_pts=ref_pts, attn_mask=attn_mask)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp[..., :4] += reference
                tmp[..., 4:6] += reference[..., :2]  # D2 center offset from same ref
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                tmp[..., 4:6] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord[..., :4]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord[..., :4])
        out['scores'] = out['pred_logits'][..., :1].sigmoid()
        return out, memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten

    def _forward_single_image_light(self, samples, track_instances: Instances, memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten, gtboxes=None):
        """
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        """
        if gtboxes is not None:
            n_dt = len(track_instances)
            ps_tgt = self.refine_embed.weight.expand(gtboxes.size(0), -1)
            query_embed = torch.cat([track_instances.query_pos, ps_tgt])
            ref_pts = torch.cat([track_instances.ref_pts, gtboxes])
            attn_mask = torch.zeros((len(ref_pts), len(ref_pts)), dtype=bool, device=ref_pts.device)
            attn_mask[:n_dt, n_dt:] = True
        else:
            query_embed = track_instances.query_pos
            ref_pts = track_instances.ref_pts
            attn_mask = None

        srcs = None
        masks = None
        pos = None

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer.forward_light_tracking(srcs, masks, pos, query_embed, ref_pts,
                                memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten,
                                track_instances.mem_bank, track_instances.mem_padding_mask, attn_mask)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp[..., :4] += reference
                tmp[..., 4:6] += reference[..., :2]  # D2 center offset from same ref
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                tmp[..., 4:6] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            # Aux outputs (intermediate layers) only need the first 4D for box matching/loss.
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord[..., :4])
        out['hs'] = hs[-1]
        return out

    def _post_process_single_image(self, frame_res, track_instances, is_last):
        if self.query_denoise > 0:
            n_ins = len(track_instances)
            ps_logits = frame_res['pred_logits'][:, n_ins:]
            # pred_boxes is 8D; for ps targets only the first 4D (current cell position) is used
            ps_boxes = frame_res['pred_boxes'][:, n_ins:, :4]
            frame_res['hs'] = frame_res['hs'][:, :n_ins]
            frame_res['pred_logits'] = frame_res['pred_logits'][:, :n_ins]
            frame_res['pred_boxes'] = frame_res['pred_boxes'][:, :n_ins]
            ps_outputs = [{'pred_logits': ps_logits, 'pred_boxes': ps_boxes}]
            for aux_outputs in frame_res['aux_outputs']:
                ps_outputs.append({
                    'pred_logits': aux_outputs['pred_logits'][:, n_ins:],
                    'pred_boxes': aux_outputs['pred_boxes'][:, n_ins:, :4],
                })
                aux_outputs['pred_logits'] = aux_outputs['pred_logits'][:, :n_ins]
                aux_outputs['pred_boxes'] = aux_outputs['pred_boxes'][:, :n_ins]
            frame_res['ps_outputs'] = ps_outputs

        with torch.no_grad():
            track_scores = frame_res['pred_logits'][0, :, 0].sigmoid()

        track_instances.scores = track_scores
        track_instances.pred_logits = frame_res['pred_logits'][0]
        # pred_boxes from bbox_embed is 8D: first 4D = daughter1/current cell,
        # second 4D = daughter2. Split here so track_instances carries each separately.
        track_instances.pred_boxes = frame_res['pred_boxes'][0, :, :4]
        track_instances.pred_div_boxes = frame_res['pred_boxes'][0, :, 4:]
        track_instances.output_embedding = frame_res['hs'][0]
        # Mask prediction: cross-attend decoder hs to level-0 encoder memory.
        if self.mask_head is not None and 'encoder_memory' in frame_res:
            mem   = frame_res['encoder_memory']        # [1, ΣHiWi, D]
            ss    = frame_res['encoder_spatial_shapes'] # [L, 2] int tensor
            H0, W0 = int(ss[0, 0]), int(ss[0, 1])
            D = mem.shape[-1]
            mem_2d = mem[:, :H0 * W0, :].view(1, H0, W0, D).permute(0, 3, 1, 2)
            hs_q   = frame_res['hs']                   # [1, N_queries, D]
            m_logits = self.mask_head(hs_q, mem_2d)    # [1, N_queries, H0, W0]
            img_hw = frame_res.get('img_hw')
            if img_hw is not None:
                m_logits = F.interpolate(
                    m_logits[0].unsqueeze(1),          # [N, 1, H0, W0]
                    size=img_hw,
                    mode='bilinear', align_corners=False,
                ).squeeze(1).unsqueeze(0)              # [1, N, H, W]
            frame_res['pred_masks'] = m_logits
            track_instances.pred_masks = m_logits[0]  # [N, H, W] or [N, H0, W0]
        if self.training:
            frame_res['track_instances'] = track_instances
            track_instances = self.criterion.match_for_single_frame(frame_res)
        else:
            # Spawn daughters for any active track whose division logit exceeds threshold.
            n_prop = frame_res.get('n_proposals', 0)
            p_start = frame_res.get('proposal_start', self.num_queries)
            p_end   = frame_res.get('proposal_end', p_start + n_prop)
            track_instances = self._spawn_daughter2_tracks(track_instances, n_prop, p_start, p_end)

            # Record carry-over tracks (obj_idxes >= 0) AFTER daughter spawning.
            # Spawned mothers have already been reset to -1, so they are correctly
            # excluded from this mask and won't suppress their own re-detection (D1).
            was_tracked = (track_instances.obj_idxes >= 0).clone()

            self.track_base.update(track_instances)

            # NMS: suppress fresh detections that duplicate carry-over tracks.
            # At each frame, _generate_empty_tracks prepends num_queries + n_proposals
            # fresh slots (obj_idxes=-1). After update(), any fresh slot that scored
            # >= score_thresh gets a new ID. If that slot overlaps a carry-over track
            # (IoU > 0.65), it is a duplicate — reset it to -1 so QIMv2 drops it.
            #
            # We raise the IoU gate from 0.5 to 0.65 so that daughters (whose bounding
            # boxes partially overlap the mother's predicted box at IoU 0.3–0.6) are not
            # suppressed.  Very-tight duplicates (IoU > 0.65) are still removed.
            #
            # Additionally, carry-over tracks whose division score is significantly above
            # the per-frame median are excluded from being NMS suppressors.  When the
            # spawning mechanism did not fire (div_score below div_score_thresh), the
            # mother track is still alive and its pred_boxes may point at a daughter
            # position; excluding it from NMS lets the daughter proposal survive.
            newly_detected = (~was_tracked) & (track_instances.obj_idxes >= 0)
            if newly_detected.any() and was_tracked.any():
                # Build carry mask: exclude likely-dividing tracks (div_score outliers).
                carry_mask = was_tracked.clone()
                if (track_instances.has('pred_logits') and
                        track_instances.pred_logits.shape[-1] > self.num_classes):
                    div_s = torch.sigmoid(
                        track_instances.pred_logits[:, self.num_classes])
                    active_div = div_s[was_tracked]
                    if len(active_div) >= 4:
                        div_med = active_div.median()
                        div_std = active_div.std()
                        # Exclude tracks > median + 1.5·std (top ~7% in a normal dist).
                        carry_mask = was_tracked & (div_s <= div_med + 1.5 * div_std)
                        if not carry_mask.any():
                            carry_mask = was_tracked  # safety fallback

                new_boxes   = box_ops.box_cxcywh_to_xyxy(
                    track_instances.pred_boxes[newly_detected].clamp(0, 1))
                carry_boxes = box_ops.box_cxcywh_to_xyxy(
                    track_instances.pred_boxes[carry_mask].clamp(0, 1))
                iou_mat = pairwise_iou(Boxes(new_boxes), Boxes(carry_boxes))  # [Nnew, Ncarry]
                max_iou = iou_mat.max(dim=1).values
                suppress = max_iou > 0.65
                if suppress.any():
                    new_idxs = newly_detected.nonzero(as_tuple=True)[0]
                    track_instances.obj_idxes[new_idxs[suppress]] = -1
                    track_instances.scores[new_idxs[suppress]]    = 0.0
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
        tmp = {}
        tmp['track_instances'] = track_instances
        if not is_last:
            out_track_instances = self.track_embed(tmp)
            frame_res['track_instances'] = out_track_instances
        else:
            frame_res['track_instances'] = None
        return frame_res

    @torch.no_grad()
    def inference_single_image(self, img, ori_img_size, track_instances=None, proposals=None):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        if track_instances is None:
            track_instances = self._generate_empty_tracks(proposals)
        else:
            track_instances = Instances.cat([
                self._generate_empty_tracks(proposals),
                track_instances])
        res = self._forward_single_image(img,
                                         track_instances=track_instances)
        res['n_proposals']    = len(proposals) if proposals is not None else 0
        res['proposal_start'] = self.num_queries
        res['proposal_end']   = self.num_queries + res['n_proposals']
        res = self._post_process_single_image(res, track_instances, False)

        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size)
        ret = {'track_instances': track_instances}
        if 'ref_pts' in res:
            ref_pts = res['ref_pts']
            img_h, img_w = ori_img_size
            scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
            ref_pts = ref_pts * scale_fct[None]
            ret['ref_pts'] = ref_pts
        return ret
    
    @torch.no_grad()
    def inference_single_image_light_light(self, img, ori_img_size, track_instances, proposals, memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        if track_instances is None:
            track_instances = self._generate_empty_tracks(proposals)
        else:
            track_instances = Instances.cat([
                self._generate_empty_tracks(proposals),
                track_instances])
        res = self._forward_single_image_light(img, track_instances, memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten)
        res['n_proposals']    = len(proposals) if proposals is not None else 0
        res['proposal_start'] = self.num_queries
        res['proposal_end']   = self.num_queries + res['n_proposals']
        if self.mask_head is not None:
            res['encoder_memory']         = memory
            res['encoder_spatial_shapes'] = spatial_shapes
            res['img_hw']                 = tuple(img.tensors.shape[-2:])
        res = self._post_process_single_image(res, track_instances, False)

        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size)
        ret = {'track_instances': track_instances}
        if 'ref_pts' in res:
            ref_pts = res['ref_pts']
            img_h, img_w = ori_img_size
            scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
            ref_pts = ref_pts * scale_fct[None]
            ret['ref_pts'] = ref_pts
        return ret

    @torch.no_grad()
    def inference_single_image_detector(self, img, ori_img_size, proposals=None):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        track_instances = self._generate_empty_tracks(proposals)
        res = self._forward_single_image_detector(img, track_instances=track_instances)

        out_logits = res['pred_logits']
        out_bbox = res['pred_boxes']
        scores = res['track_scores']

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = ori_img_size
        scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
        boxes = boxes * scale_fct[None, :]

        ret = {
            'scores': scores,
            'boxes': boxes,
        }

        return ret
    
    @torch.no_grad()
    def inference_single_image_proposals(self, img, ori_img_size, score_threshold=0.05):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        track_instances = self._generate_empty_tracks()
    
        frame_res = self._forward_single_image_proposals(img)

        boxes = frame_res['pred_boxes'][-1, 0]    # shape [300, 4]
        scores = frame_res['scores'][-1, 0] # shape [300, 1]
        proposals_frame = torch.cat([boxes, scores], dim=1)
        proposals_frame = proposals_frame[proposals_frame[:, 4] > score_threshold]

        return proposals_frame
    
    @torch.no_grad()
    def inference_single_image_proposals_light_light(self, img, ori_img_size, score_threshold=0.05):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        track_instances = self._generate_empty_tracks()
    
        frame_res, memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten = self._forward_single_image_proposals_light(img)

        boxes = frame_res['pred_boxes'][-1, 0]    # shape [300, 4]
        scores = frame_res['scores'][-1, 0] # shape [300, 1]
        proposals_frame = torch.cat([boxes, scores], dim=1)
        proposals_frame = proposals_frame[proposals_frame[:, 4] > score_threshold]

        return proposals_frame, memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten
    
    @torch.no_grad()
    def inference_single_image_proposals_light(self, img, ori_img_size, score_threshold=0.05, layer=-1):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        track_instances = self._generate_empty_tracks()
    
        frame_res = self._forward_single_image_proposals(img)

        boxes = frame_res['pred_boxes'][layer, 0]    # shape [300, 4]
        scores = frame_res['scores'][layer, 0] # shape [300, 1]
        proposals_frame = torch.cat([boxes, scores], dim=1)
        proposals_frame = proposals_frame[proposals_frame[:, 4] > score_threshold]

        return proposals_frame

    def forward_detect_self_light(self, data: dict, score_threshold=0.5):
        """Detection pass that returns encoder features for reuse."""
        frames = data['imgs']  # list of Tensor.
        
        pred_logits_list = []
        pred_boxes_list = []
        proposals = []
        encoder_cache = []  # Store encoder outputs for each frame
        aux_outputs_list = None

        for frame_index, frame in enumerate(frames):
            frame.requires_grad = False
            
            frame = nested_tensor_from_tensor_list([frame])
            
            # Use _forward_single_image_proposals_light which returns encoder cache
            frame_res, memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten = \
                self._forward_single_image_proposals_light(frame)
            
            # Store encoder cache for this frame
            encoder_cache.append({
                'memory': memory,
                'spatial_shapes': spatial_shapes,
                'level_start_index': level_start_index,
                'valid_ratios': valid_ratios,
                'mask_flatten': mask_flatten,
            })

            # frame_res['pred_logits'] is [num_layers, batch, num_queries, num_classes]
            # frame_res['pred_boxes'] is [num_layers, batch, num_queries, 4]
            # Take the last layer (-1) for final outputs
            pred_logits_list.append(frame_res['pred_logits'][-1, 0])  # [300, 1]
            pred_boxes_list.append(frame_res['pred_boxes'][-1, 0])    # [300, 4]

            # Handle variable decoder depth (e.g. dec_layers=1 => no aux outputs).
            if aux_outputs_list is None:
                num_aux = len(frame_res.get('aux_outputs', []))
                aux_outputs_list = [
                    {'pred_logits': [], 'pred_boxes': []}
                    for _ in range(num_aux)
                ]
            for i, aux in enumerate(frame_res.get('aux_outputs', [])):
                aux_outputs_list[i]['pred_logits'].append(aux['pred_logits'][0])
                aux_outputs_list[i]['pred_boxes'].append(aux['pred_boxes'][0])

            boxes = frame_res['pred_boxes'][-1, 0]    # shape [300, 4]
            scores = frame_res['scores'][-1, 0]       # shape [300, 1]
            proposals_frame = torch.cat([boxes, scores], dim=1)
            proposals_frame = proposals_frame[proposals_frame[:, 4] > score_threshold]
            proposals.append(proposals_frame)

        outputs = {
            'pred_logits': torch.stack(pred_logits_list, dim=0),
            'pred_boxes': torch.stack(pred_boxes_list, dim=0),
            'aux_outputs': []
        }

        for aux in (aux_outputs_list or []):
            outputs['aux_outputs'].append({
                'pred_logits': torch.stack(aux['pred_logits'], dim=0),
                'pred_boxes': torch.stack(aux['pred_boxes'], dim=0),
            })

        return outputs, proposals, encoder_cache
    
    def forward_with_encoder_cache(self, data: dict, encoder_cache: list):
        """Tracking pass that reuses encoder features from detection pass."""
        if self.training:
            self.criterion.initialize_for_single_clip(data['gt_instances'])
        frames = data['imgs']
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
        }

        track_instances = None
        keys = list(self._generate_empty_tracks()._fields.keys())
        
        for frame_index, (frame, cache, proposals) in enumerate(zip(frames, encoder_cache, data['proposals'])):
            frame.requires_grad = False
            is_last = frame_index == len(frames) - 1

            if self.query_denoise > 0:
                gt = data['gt_instances'][frame_index]
                l_1 = l_2 = self.query_denoise
                gtboxes = gt.boxes.clone()
                _rs = torch.rand_like(gtboxes) * 2 - 1
                gtboxes[..., :2] += gtboxes[..., 2:] * _rs[..., :2] * l_1
                gtboxes[..., 2:] *= 1 + l_2 * _rs[..., 2:]
            else:
                gtboxes = None

            # Same logic as original forward()
            if track_instances is None:
                track_instances = self._generate_empty_tracks(proposals)
            else:
                track_instances = Instances.cat([
                    self._generate_empty_tracks(proposals),
                    track_instances])

            if self.use_checkpoint and frame_index < len(frames) - 1:
                def fn(frame, gtboxes, memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten, *args):
                    frame = nested_tensor_from_tensor_list([frame])
                    tmp = Instances((1, 1), **dict(zip(keys, args)))
                    frame_res = self._forward_single_image_light(
                        frame, tmp, memory, spatial_shapes, level_start_index,
                        valid_ratios, mask_flatten, gtboxes)
                    return (
                        frame_res['pred_logits'],
                        frame_res['pred_boxes'],  # 8D: [:4]=d1, [4:]=d2
                        frame_res['hs'],
                        *[aux['pred_logits'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_boxes'] for aux in frame_res['aux_outputs']]
                    )

                args = [
                    frame, gtboxes, cache['memory'], cache['spatial_shapes'],
                    cache['level_start_index'], cache['valid_ratios'], cache['mask_flatten'],
                    *[track_instances.get(k) for k in keys]
                ]
                params = tuple((p for p in self.parameters() if p.requires_grad))
                tmp = checkpointv2.CheckpointFunction.apply(fn, len(args), *args, *params)
                num_aux = (len(tmp) - 3) // 2
                frame_res = {
                    'pred_logits': tmp[0],
                    'pred_boxes': tmp[1],
                    'hs': tmp[2],
                    'aux_outputs': [{
                        'pred_logits': tmp[3 + i],
                        'pred_boxes': tmp[3 + num_aux + i],
                    } for i in range(num_aux)],
                }
            else:
                # Use cached encoder features - skip backbone and encoder.
                frame = nested_tensor_from_tensor_list([frame])
                frame_res = self._forward_single_image_light(
                    frame, track_instances,
                    cache['memory'], cache['spatial_shapes'],
                    cache['level_start_index'], cache['valid_ratios'],
                    cache['mask_flatten'],
                    gtboxes
                )
            frame_res['n_proposals']    = len(proposals) if proposals is not None else 0
            frame_res['proposal_start'] = self.num_queries
            frame_res['proposal_end']   = self.num_queries + frame_res['n_proposals']
            if self.mask_head is not None:
                frame_res['encoder_memory']         = cache['memory']
                frame_res['encoder_spatial_shapes'] = cache['spatial_shapes']
                _frame_nt = frame if isinstance(frame, NestedTensor) else nested_tensor_from_tensor_list([frame])
                frame_res['img_hw']                 = tuple(_frame_nt.tensors.shape[-2:])
            frame_res = self._post_process_single_image(frame_res, track_instances, is_last)

            track_instances = frame_res['track_instances']
            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])

        if not self.training:
            outputs['track_instances'] = track_instances
        else:
            outputs['losses_dict'] = self.criterion.losses_dict
        return outputs

    def forward_detect_self(self, data: dict, score_threshold=0.5):
        frames = data['imgs']  # list of Tensor.
        #outputs = {
        #    'pred_logits': [],
        #    'pred_boxes': [],
        #    'scores': [],
        #    'proposals': [],
        #}
        pred_logits_list = []
        pred_boxes_list = []
        proposals = []
        aux_outputs_list = None

        for frame_index, frame in enumerate(frames):
            frame.requires_grad = False
            
            frame = nested_tensor_from_tensor_list([frame])
            
            frame_res = self._forward_single_image_detect_self(frame)

            pred_logits_list.append(frame_res['pred_logits'][0])  # [300, 1]
            pred_boxes_list.append(frame_res['pred_boxes'][0])    # [300, 4]

            if aux_outputs_list is None:
                aux_outputs_list = [
                    {'pred_logits': [], 'pred_boxes': []}
                    for _ in range(len(frame_res.get('aux_outputs', [])))
                ]
            for i, aux in enumerate(frame_res.get('aux_outputs', [])):
                aux_outputs_list[i]['pred_logits'].append(aux['pred_logits'][0])
                aux_outputs_list[i]['pred_boxes'].append(aux['pred_boxes'][0])

            boxes = frame_res['pred_boxes'][0]    # shape [300, 4]
            scores = frame_res['scores'][0] # shape [300, 1]
            proposals_frame = torch.cat([boxes, scores], dim=1)
            proposals_frame = proposals_frame[proposals_frame[:, 4] > score_threshold]
            proposals.append(proposals_frame)

        outputs = {
            'pred_logits': torch.stack(pred_logits_list, dim=0),  # [num_frames, 300, 1]
            'pred_boxes': torch.stack(pred_boxes_list, dim=0),    # [num_frames, 300, 4]
            'aux_outputs': []
        }

        for aux in (aux_outputs_list or []):
            outputs['aux_outputs'].append({
                'pred_logits': torch.stack(aux['pred_logits'], dim=0),  # [num_frames, 300, 1]
                'pred_boxes': torch.stack(aux['pred_boxes'], dim=0),    # [num_frames, 300, 4]
            })

        return outputs, proposals

    def forward(self, data: dict):
        if self.training:
            self.criterion.initialize_for_single_clip(data['gt_instances'])
        frames = data['imgs']  # list of Tensor.
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
        }
        track_instances = None
        keys = list(self._generate_empty_tracks()._fields.keys())
        for frame_index, (frame, gt, proposals) in enumerate(zip(frames, data['gt_instances'], data['proposals'])):
            frame.requires_grad = False
            is_last = frame_index == len(frames) - 1

            if self.query_denoise > 0:
                l_1 = l_2 = self.query_denoise
                gtboxes = gt.boxes.clone()
                _rs = torch.rand_like(gtboxes) * 2 - 1
                gtboxes[..., :2] += gtboxes[..., 2:] * _rs[..., :2] * l_1
                gtboxes[..., 2:] *= 1 + l_2 * _rs[..., 2:]
            else:
                gtboxes = None

            if track_instances is None:
                track_instances = self._generate_empty_tracks(proposals)
            else:
                track_instances = Instances.cat([
                    self._generate_empty_tracks(proposals),
                    track_instances])

            if self.use_checkpoint and frame_index < len(frames) - 1:
                def fn(frame, gtboxes, *args):
                    frame = nested_tensor_from_tensor_list([frame])
                    tmp = Instances((1, 1), **dict(zip(keys, args)))
                    frame_res = self._forward_single_image(frame, tmp, gtboxes)
                    return (
                        frame_res['pred_logits'],
                        frame_res['pred_boxes'],  # 8D: [:4]=d1, [4:]=d2
                        frame_res['hs'],
                        *[aux['pred_logits'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_boxes'] for aux in frame_res['aux_outputs']]
                    )

                args = [frame, gtboxes] + [track_instances.get(k) for k in keys]
                params = tuple((p for p in self.parameters() if p.requires_grad))
                tmp = checkpointv2.CheckpointFunction.apply(fn, len(args), *args, *params)
                num_aux = (len(tmp) - 3) // 2
                frame_res = {
                    'pred_logits': tmp[0],
                    'pred_boxes': tmp[1],
                    'hs': tmp[2],
                    'aux_outputs': [{
                        'pred_logits': tmp[3+i],
                        'pred_boxes': tmp[3+num_aux+i],
                    } for i in range(num_aux)],
                }
            else:
                frame = nested_tensor_from_tensor_list([frame])
                frame_res = self._forward_single_image(frame, track_instances, gtboxes)
            frame_res['n_proposals']    = len(proposals) if proposals is not None else 0
            frame_res['proposal_start'] = self.num_queries
            frame_res['proposal_end']   = self.num_queries + frame_res['n_proposals']
            frame_res = self._post_process_single_image(frame_res, track_instances, is_last)

            track_instances = frame_res['track_instances']
            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])

        if not self.training:
            outputs['track_instances'] = track_instances
        else:
            outputs['losses_dict'] = self.criterion.losses_dict
        return outputs


def build(args):
    dataset_to_num_classes = {
        'coco': 91,
        'coco_panoptic': 250,
        'e2e_mot': 1,
        'e2e_dance': 1,
        'e2e_dance_v2': 1,
        'e2e_dance_v2_final': 1,
        'e2e_joint': 1,
        'e2e_static_mot': 1,
        'e2e_sportsmot': 1,
        'e2e_bft': 1,
        'e2e_wat': 1,
        'e2e_sportsmot_v2': 1,
        'e2e_cell': 1,
    }

    assert args.dataset_file in dataset_to_num_classes
    num_classes = dataset_to_num_classes[args.dataset_file]
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    d_model = transformer.d_model
    hidden_dim = args.dim_feedforward
    query_interaction_layer = build_query_interaction_layer(args, args.query_interaction_layer, d_model, hidden_dim, d_model*2)

    img_matcher = build_matcher(args)
    # +1 for the future frame that is always appended to every training clip
    # so the model sees daughters after a division event.
    num_frames_per_batch = max(args.sampler_lengths) + 1
    div_loss_coef = getattr(args, 'div_loss_coef', 0.0)
    div_affinity_loss_coef = getattr(args, 'div_affinity_loss_coef', 0.0)
    score_consist_loss_coef = getattr(args, 'score_consist_loss_coef', 0.0)
    mask_loss_coef = getattr(args, 'mask_loss_coef', 0.0)
    weight_dict = {}
    for i in range(num_frames_per_batch):
        weight_dict.update({"frame_{}_loss_ce".format(i): args.cls_loss_coef,
                            'frame_{}_loss_bbox'.format(i): args.bbox_loss_coef,
                            'frame_{}_loss_giou'.format(i): args.giou_loss_coef,
                            })
        if div_loss_coef > 0:
            weight_dict["frame_{}_loss_div_box".format(i)]   = div_loss_coef
            weight_dict["frame_{}_loss_div_class".format(i)] = div_loss_coef
        if div_affinity_loss_coef > 0:
            weight_dict["frame_{}_loss_div_affinity".format(i)] = div_affinity_loss_coef
        if score_consist_loss_coef > 0:
            weight_dict["frame_{}_loss_score_consist".format(i)] = score_consist_loss_coef
        if mask_loss_coef > 0:
            weight_dict["frame_{}_loss_mask".format(i)] = mask_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        for i in range(num_frames_per_batch):
            for j in range(args.dec_layers - 1):
                weight_dict.update({"frame_{}_aux{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_aux{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_aux{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    })
            for j in range(args.dec_layers):
                weight_dict.update({"frame_{}_ps{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_ps{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_ps{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    })
    if args.memory_bank_type is not None and len(args.memory_bank_type) > 0:
        memory_bank = build_memory_bank(args, d_model, hidden_dim, d_model * 2)
        for i in range(num_frames_per_batch):
            weight_dict.update({"frame_{}_track_loss_ce".format(i): args.cls_loss_coef})
    else:
        memory_bank = None
    losses = ['labels', 'boxes']
    # Create the MLP before model/criterion so the same object can be registered
    # as a model submodule (for optimizer + checkpoint) while criterion holds only
    # a non-owning reference (via object.__setattr__).
    div_proposal_mlp = DivisionProposalMLP(d_model) if div_affinity_loss_coef > 0 else None
    nheads = args.nheads
    mask_head = SimpleMaskHead(d_model, nheads) if getattr(args, 'masks', False) else None
    criterion = ClipMatcher(num_classes, matcher=img_matcher, weight_dict=weight_dict, losses=losses,
                            div_proposal_mlp=div_proposal_mlp,
                            score_consist_loss_coef=score_consist_loss_coef,
                            mask_loss_coef=mask_loss_coef,
                            div_pos_weight=getattr(args, 'div_pos_weight', 5.0))
    criterion.to(device)
    postprocessors = {}
    model = MOTR(
        backbone,
        transformer,
        track_embed=query_interaction_layer,
        num_feature_levels=args.num_feature_levels,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_queries_detect=args.num_queries_detect,
        aux_loss=args.aux_loss,
        criterion=criterion,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        memory_bank=memory_bank,
        use_checkpoint=args.use_checkpoint,
        query_denoise=args.query_denoise,
        div_score_thresh=getattr(args, 'div_score_thresh', 0.4),
        div_proposal_mlp=div_proposal_mlp,
        mask_head=mask_head,
    )
    return model, criterion, postprocessors
