# ------------------------------------------------------------------------
# SelfMOTR — No-Division variant
# Identical to motrv2_self.py except:
#   • bbox_embed outputs 4D (cell box only, no daughter-2 offset)
#   • class_embed outputs num_classes channels (no division logit)
#   • _generate_empty_tracks has no pred_div_boxes / parent_obj_id fields
#   • _post_process_single_image does no D2 split and no daughter spawning
#   • No _spawn_daughter2_tracks
#   • build() registers 'e2e_cell_no_div' dataset
# Everything else (QIM, transformer, loss) is unchanged.
# ------------------------------------------------------------------------

import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import List

from util import box_ops, checkpointv2
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from models.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou

from .backbone import build_backbone
from .matcher import build_matcher
from .deformable_transformer_plusv2 import build_deforamble_transformer, pos2posemb
from .qimv2 import build as build_query_interaction_layer
from .deformable_detrv2 import MLP

# Reuse the unchanged helpers from the div version.
from .motrv2_self import (
    ClipMatcher,
    TrackerPostProcess,
    RuntimeTrackerBase,
    _get_clones,
)


class MOTRNoDiv(nn.Module):
    """MOTRv2-Self without any division prediction or spawning."""

    def __init__(self, backbone, transformer, num_classes, num_queries,
                 num_queries_detect, num_feature_levels, criterion, track_embed,
                 aux_loss=True, with_box_refine=False, two_stage=False,
                 memory_bank=None, use_checkpoint=False, query_denoise=0):
        super().__init__()
        self.num_queries = num_queries
        self.num_queries_detect = num_queries_detect
        self.track_embed = track_embed
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes

        # No division logit: class_embed → num_classes channels only.
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        # No D2 box: bbox_embed → 4D (cx, cy, w, h) only.
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

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
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        nn.init.uniform_(self.position.weight.data, 0, 1)
        nn.init.uniform_(self.position_detect.weight.data, 0, 1)

        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            if self.shared_decoder:
                self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
                self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            else:
                self.class_embed = _get_clones(self.class_embed, num_pred)
                self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            # 4D bbox: only the wh bias [2:4] (no D2 coords)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:4], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:4], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        if two_stage:
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        self.post_process = TrackerPostProcess()
        self.track_base = RuntimeTrackerBase()
        self.criterion = criterion
        self.memory_bank = memory_bank
        self.mem_bank_len = 0 if memory_bank is None else memory_bank.max_his_length

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_empty_tracks(self, proposals=None):
        track_instances = Instances((1, 1))
        num_queries, d_model = self.query_embed.weight.shape
        device = self.query_embed.weight.device
        if proposals is None:
            track_instances.ref_pts = self.position.weight
            track_instances.query_pos = self.query_embed.weight
        else:
            track_instances.ref_pts = torch.cat([self.position.weight, proposals[:, :4]])
            track_instances.query_pos = torch.cat([
                self.query_embed.weight,
                pos2posemb(proposals[:, 4:], d_model) + self.yolox_embed.weight])
        track_instances.output_embedding = torch.zeros((len(track_instances), d_model), device=device)
        track_instances.obj_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros((len(track_instances),), dtype=torch.long, device=device)
        track_instances.iou = torch.ones((len(track_instances),), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)
        # num_classes channels only (no division logit)
        track_instances.pred_logits = torch.zeros((len(track_instances), self.num_classes), dtype=torch.float, device=device)
        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros((len(track_instances), mem_bank_len, d_model), dtype=torch.float32, device=device)
        track_instances.mem_padding_mask = torch.ones((len(track_instances), mem_bank_len), dtype=torch.bool, device=device)
        track_instances.save_period = torch.zeros((len(track_instances),), dtype=torch.float32, device=device)
        # No pred_div_boxes, no parent_obj_id
        return track_instances.to(self.query_embed.weight.device)

    def clear(self):
        self.track_base.clear()

    # ------------------------------------------------------------------
    # Auxiliary loss helper
    # ------------------------------------------------------------------

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    # ------------------------------------------------------------------
    # Forward helpers — backbone extraction
    # ------------------------------------------------------------------

    def _extract_features(self, samples):
        """Run backbone + input projections; return srcs, masks, pos."""
        features, pos = self.backbone(samples)
        srcs, masks = [], []
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
        return srcs, masks, pos

    def _decode_heads(self, hs, init_reference, inter_references):
        """Run class/bbox heads over all decoder layers; return stacked tensors."""
        outputs_classes, outputs_coords = [], []
        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coords.append(tmp.sigmoid())
            outputs_classes.append(outputs_class)
        return torch.stack(outputs_classes), torch.stack(outputs_coords)

    # ------------------------------------------------------------------
    # Single-frame forward passes
    # ------------------------------------------------------------------

    def _forward_single_image(self, samples, track_instances: Instances, gtboxes=None):
        srcs, masks, pos = self._extract_features(samples)

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

        hs, init_reference, inter_references, enc_out_cls, enc_out_coord = \
            self.transformer(srcs, masks, pos, query_embed, ref_pts=ref_pts,
                             mem_bank=track_instances.mem_bank,
                             mem_bank_pad_mask=track_instances.mem_padding_mask,
                             attn_mask=attn_mask)

        outputs_class, outputs_coord = self._decode_heads(hs, init_reference, inter_references)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        out['hs'] = hs[-1]
        return out

    def _forward_single_image_proposals(self, samples):
        srcs, masks, pos = self._extract_features(samples)
        query_embed = self.query_embed_detect.weight
        ref_pts = self.position_detect.weight

        hs, init_reference, inter_references, enc_out_cls, enc_out_coord = \
            self.transformer(srcs, masks, pos, query_embed, ref_pts=ref_pts, attn_mask=None)

        outputs_class, outputs_coord = self._decode_heads(hs, init_reference, inter_references)

        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        out['scores'] = out['pred_logits'][..., :1].sigmoid()
        return out

    def _forward_single_image_proposals_light(self, samples):
        srcs, masks, pos = self._extract_features(samples)
        query_embed = self.query_embed_detect.weight
        ref_pts = self.position_detect.weight

        hs, init_reference, inter_references, enc_out_cls, enc_out_coord, \
            memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten = \
            self.transformer.forward_light_proposal(srcs, masks, pos, query_embed,
                                                     ref_pts=ref_pts, attn_mask=None)

        outputs_class, outputs_coord = self._decode_heads(hs, init_reference, inter_references)

        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        out['scores'] = out['pred_logits'][..., :1].sigmoid()
        return out, memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten

    def _forward_single_image_light(self, samples, track_instances: Instances,
                                     memory, spatial_shapes, level_start_index,
                                     valid_ratios, mask_flatten, gtboxes=None):
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

        hs, init_reference, inter_references, enc_out_cls, enc_out_coord = \
            self.transformer.forward_light_tracking(
                None, None, None, query_embed, ref_pts,
                memory, spatial_shapes, level_start_index, valid_ratios,
                mask_flatten, track_instances.mem_bank,
                track_instances.mem_padding_mask, attn_mask)

        outputs_class, outputs_coord = self._decode_heads(hs, init_reference, inter_references)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        out['hs'] = hs[-1]
        return out

    # ------------------------------------------------------------------
    # Post-processing (no division spawning)
    # ------------------------------------------------------------------

    def _post_process_single_image(self, frame_res, track_instances, is_last):
        if self.query_denoise > 0:
            n_ins = len(track_instances)
            ps_logits = frame_res['pred_logits'][:, n_ins:]
            ps_boxes = frame_res['pred_boxes'][:, n_ins:]
            frame_res['hs'] = frame_res['hs'][:, :n_ins]
            frame_res['pred_logits'] = frame_res['pred_logits'][:, :n_ins]
            frame_res['pred_boxes'] = frame_res['pred_boxes'][:, :n_ins]
            ps_outputs = [{'pred_logits': ps_logits, 'pred_boxes': ps_boxes}]
            for aux_outputs in frame_res['aux_outputs']:
                ps_outputs.append({
                    'pred_logits': aux_outputs['pred_logits'][:, n_ins:],
                    'pred_boxes': aux_outputs['pred_boxes'][:, n_ins:],
                })
                aux_outputs['pred_logits'] = aux_outputs['pred_logits'][:, :n_ins]
                aux_outputs['pred_boxes'] = aux_outputs['pred_boxes'][:, :n_ins]
            frame_res['ps_outputs'] = ps_outputs

        with torch.no_grad():
            track_scores = frame_res['pred_logits'][0, :, 0].sigmoid()

        track_instances.scores = track_scores
        track_instances.pred_logits = frame_res['pred_logits'][0]
        track_instances.pred_boxes = frame_res['pred_boxes'][0]   # 4D, no D2 split
        track_instances.output_embedding = frame_res['hs'][0]

        if self.training:
            frame_res['track_instances'] = track_instances
            track_instances = self.criterion.match_for_single_frame(frame_res)
        else:
            # No daughter spawning — just update tracker IDs.
            self.track_base.update(track_instances)

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

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

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
        res = self._forward_single_image(img, track_instances=track_instances)
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
    def inference_single_image_light_light(self, img, ori_img_size, track_instances,
                                            proposals, memory, spatial_shapes,
                                            level_start_index, valid_ratios, mask_flatten):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        if track_instances is None:
            track_instances = self._generate_empty_tracks(proposals)
        else:
            track_instances = Instances.cat([
                self._generate_empty_tracks(proposals),
                track_instances])
        res = self._forward_single_image_light(img, track_instances, memory,
                                                spatial_shapes, level_start_index,
                                                valid_ratios, mask_flatten)
        res = self._post_process_single_image(res, track_instances, False)
        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size)
        ret = {'track_instances': track_instances}
        return ret

    @torch.no_grad()
    def inference_single_image_proposals(self, img, ori_img_size, score_threshold=0.05):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        frame_res = self._forward_single_image_proposals(img)
        boxes = frame_res['pred_boxes'][-1, 0]
        scores = frame_res['scores'][-1, 0]
        proposals_frame = torch.cat([boxes, scores], dim=1)
        proposals_frame = proposals_frame[proposals_frame[:, 4] > score_threshold]
        return proposals_frame

    @torch.no_grad()
    def inference_single_image_proposals_light_light(self, img, ori_img_size, score_threshold=0.05):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        frame_res, memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten = \
            self._forward_single_image_proposals_light(img)
        boxes = frame_res['pred_boxes'][-1, 0]
        scores = frame_res['scores'][-1, 0]
        proposals_frame = torch.cat([boxes, scores], dim=1)
        proposals_frame = proposals_frame[proposals_frame[:, 4] > score_threshold]
        return proposals_frame, memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten

    # ------------------------------------------------------------------
    # Training forward passes (mirrors motrv2_self.py)
    # ------------------------------------------------------------------

    def forward_detect_self_light(self, data: dict, score_threshold=0.5):
        frames = data['imgs']
        pred_logits_list, pred_boxes_list, proposals, encoder_cache = [], [], [], []
        aux_outputs_list = None

        for frame in frames:
            frame.requires_grad = False
            frame = nested_tensor_from_tensor_list([frame])
            frame_res, memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten = \
                self._forward_single_image_proposals_light(frame)

            encoder_cache.append({
                'memory': memory,
                'spatial_shapes': spatial_shapes,
                'level_start_index': level_start_index,
                'valid_ratios': valid_ratios,
                'mask_flatten': mask_flatten,
            })

            pred_logits_list.append(frame_res['pred_logits'][-1, 0])
            pred_boxes_list.append(frame_res['pred_boxes'][-1, 0])

            if aux_outputs_list is None:
                num_aux = len(frame_res.get('aux_outputs', []))
                aux_outputs_list = [{'pred_logits': [], 'pred_boxes': []} for _ in range(num_aux)]
            for i, aux in enumerate(frame_res.get('aux_outputs', [])):
                aux_outputs_list[i]['pred_logits'].append(aux['pred_logits'][0])
                aux_outputs_list[i]['pred_boxes'].append(aux['pred_boxes'][0])

            boxes = frame_res['pred_boxes'][-1, 0]
            scores = frame_res['scores'][-1, 0]
            proposals_frame = torch.cat([boxes, scores], dim=1)
            proposals_frame = proposals_frame[proposals_frame[:, 4] > score_threshold]
            proposals.append(proposals_frame)

        outputs = {
            'pred_logits': torch.stack(pred_logits_list, dim=0),
            'pred_boxes': torch.stack(pred_boxes_list, dim=0),
            'aux_outputs': [],
        }
        for aux in (aux_outputs_list or []):
            outputs['aux_outputs'].append({
                'pred_logits': torch.stack(aux['pred_logits'], dim=0),
                'pred_boxes': torch.stack(aux['pred_boxes'], dim=0),
            })
        return outputs, proposals, encoder_cache

    def forward_with_encoder_cache(self, data: dict, encoder_cache: list):
        if self.training:
            self.criterion.initialize_for_single_clip(data['gt_instances'])
        frames = data['imgs']
        outputs = {'pred_logits': [], 'pred_boxes': []}
        track_instances = None
        keys = list(self._generate_empty_tracks()._fields.keys())

        for frame_index, (frame, cache, proposals) in enumerate(
                zip(frames, encoder_cache, data['proposals'])):
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

            if track_instances is None:
                track_instances = self._generate_empty_tracks(proposals)
            else:
                track_instances = Instances.cat([
                    self._generate_empty_tracks(proposals),
                    track_instances])

            if self.use_checkpoint and frame_index < len(frames) - 1:
                def fn(frame, gtboxes, memory, spatial_shapes, level_start_index,
                       valid_ratios, mask_flatten, *args):
                    frame = nested_tensor_from_tensor_list([frame])
                    tmp = Instances((1, 1), **dict(zip(keys, args)))
                    frame_res = self._forward_single_image_light(
                        frame, tmp, memory, spatial_shapes, level_start_index,
                        valid_ratios, mask_flatten, gtboxes)
                    return (
                        frame_res['pred_logits'],
                        frame_res['pred_boxes'],
                        frame_res['hs'],
                        *[aux['pred_logits'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_boxes'] for aux in frame_res['aux_outputs']],
                    )
                args = [
                    frame, gtboxes, cache['memory'], cache['spatial_shapes'],
                    cache['level_start_index'], cache['valid_ratios'], cache['mask_flatten'],
                    *[track_instances.get(k) for k in keys],
                ]
                params = tuple(p for p in self.parameters() if p.requires_grad)
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
                frame = nested_tensor_from_tensor_list([frame])
                frame_res = self._forward_single_image_light(
                    frame, track_instances,
                    cache['memory'], cache['spatial_shapes'],
                    cache['level_start_index'], cache['valid_ratios'],
                    cache['mask_flatten'], gtboxes)

            frame_res = self._post_process_single_image(frame_res, track_instances, is_last)
            track_instances = frame_res['track_instances']
            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])

        if not self.training:
            outputs['track_instances'] = track_instances
        else:
            outputs['losses_dict'] = self.criterion.losses_dict
        return outputs

    def forward(self, data: dict):
        if self.training:
            self.criterion.initialize_for_single_clip(data['gt_instances'])
        frames = data['imgs']
        outputs = {'pred_logits': [], 'pred_boxes': []}
        track_instances = None
        keys = list(self._generate_empty_tracks()._fields.keys())

        for frame_index, (frame, gt, proposals) in enumerate(
                zip(frames, data['gt_instances'], data['proposals'])):
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
                        frame_res['pred_boxes'],
                        frame_res['hs'],
                        *[aux['pred_logits'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_boxes'] for aux in frame_res['aux_outputs']],
                    )
                args = [frame, gtboxes] + [track_instances.get(k) for k in keys]
                params = tuple(p for p in self.parameters() if p.requires_grad)
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
                frame = nested_tensor_from_tensor_list([frame])
                frame_res = self._forward_single_image(frame, track_instances, gtboxes)

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
        'e2e_cell': 1,
        'e2e_cell_no_div': 1,
    }
    assert args.dataset_file in dataset_to_num_classes, \
        f'motrv2_self_no_div: unsupported dataset {args.dataset_file!r}'
    num_classes = dataset_to_num_classes[args.dataset_file]
    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)
    d_model = transformer.d_model
    hidden_dim = args.dim_feedforward
    query_interaction_layer = build_query_interaction_layer(
        args, args.query_interaction_layer, d_model, hidden_dim, d_model * 2)

    img_matcher = build_matcher(args)
    num_frames_per_batch = max(args.sampler_lengths) + 1
    weight_dict = {}
    for i in range(num_frames_per_batch):
        weight_dict.update({
            f'frame_{i}_loss_ce': args.cls_loss_coef,
            f'frame_{i}_loss_bbox': args.bbox_loss_coef,
            f'frame_{i}_loss_giou': args.giou_loss_coef,
        })

    if args.aux_loss:
        for i in range(num_frames_per_batch):
            for j in range(args.dec_layers - 1):
                weight_dict.update({
                    f'frame_{i}_aux{j}_loss_ce': args.cls_loss_coef,
                    f'frame_{i}_aux{j}_loss_bbox': args.bbox_loss_coef,
                    f'frame_{i}_aux{j}_loss_giou': args.giou_loss_coef,
                })
            for j in range(args.dec_layers):
                weight_dict.update({
                    f'frame_{i}_ps{j}_loss_ce': args.cls_loss_coef,
                    f'frame_{i}_ps{j}_loss_bbox': args.bbox_loss_coef,
                    f'frame_{i}_ps{j}_loss_giou': args.giou_loss_coef,
                })

    losses = ['labels', 'boxes']
    criterion = ClipMatcher(num_classes, matcher=img_matcher,
                            weight_dict=weight_dict, losses=losses)
    criterion.to(device)

    model = MOTRNoDiv(
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
        memory_bank=None,
        use_checkpoint=args.use_checkpoint,
        query_denoise=args.query_denoise,
    )
    return model, criterion, {}
