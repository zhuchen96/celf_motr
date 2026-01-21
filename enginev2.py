import math
import os
import sys
from typing import Iterable
import copy

import torch
import util.misc as utils

from datasets.data_prefetcher import data_dict_to_cuda

from collections import defaultdict

def train_one_epoch_mot_self_proposal(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    criterion_detect: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    score_threshold: float,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    accum_iter: int = 1, lambda_detect: float = 1,
                    reuse_encoder_cache: bool = False):
    model.train()
    criterion.train()
    criterion_detect.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    iteration=0
    optimizer.zero_grad()
    #For logging:
    accumulated_loss = 0.0
    accumulated_loss_dict_scaled = defaultdict(float)
    accumulated_loss_print = 0.0
    accumulated_loss_dict_scaled_print = defaultdict(float)
    grad_total_norm_print = 0.0
    accumulated_loss_detect = 0.0
    accumulated_loss_dict_scaled_detect = defaultdict(float)
    accumulated_loss_print_detect = 0.0
    accumulated_loss_dict_scaled_print_detect = defaultdict(float)

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    # Handle both DDP-wrapped and unwrapped models
    model_module = getattr(model, 'module', model)

    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        iteration += 1
        data_dict = data_dict_to_cuda(data_dict, device)

        if reuse_encoder_cache:
            outputs_detector, proposals_detector, encoder_cache = model_module.forward_detect_self_light(
                data_dict, score_threshold=score_threshold)

            targets = data_dict['gt_instances']
            loss_dict_detect = criterion_detect.forward_detect(outputs_detector, targets)
            weight_dict_detect = criterion_detect.weight_dict

            for inst in data_dict['gt_instances']:
                if 'area' in inst._fields:
                    inst._fields.pop('area')

            data_dict['proposals'] = proposals_detector
            data_dict = data_dict_to_cuda(data_dict, device)
            outputs = model_module.forward_with_encoder_cache(data_dict, encoder_cache)
        else:
            # old implementation (inefficient) starting here
            outputs_detector, proposals_detector = model_module.forward_detect_self(data_dict, score_threshold=score_threshold)

            targets = data_dict['gt_instances']
            #targets = [t[0] for t in targets] <-- not used anymore at all
            loss_dict_detect = criterion_detect.forward_detect(outputs_detector, targets)
            weight_dict_detect = criterion_detect.weight_dict

            for inst in data_dict['gt_instances']:
                if 'area' in inst._fields:
                    inst._fields.pop('area')

            data_dict['proposals'] = proposals_detector
            #data_dict['proposals'] = [p.to(device) for p in outputs_detector['proposals']]

            data_dict = data_dict_to_cuda(data_dict, device)
            outputs = model(data_dict)
            # old implementation (inefficient) ending here
        

        loss_dict = criterion(outputs, data_dict)
        # print("iter {} after model".format(cnt-1))
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        losses_detect = sum(loss_dict_detect[k] * weight_dict_detect[k] for k in loss_dict_detect.keys() if k in weight_dict_detect)
        losses = losses + lambda_detect * losses_detect
        losses = losses / accum_iter


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_detect = utils.reduce_dict(loss_dict_detect)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_scaled_detect = {k: v * weight_dict_detect[k] * lambda_detect
                                    for k, v in loss_dict_reduced_detect.items() if k in weight_dict_detect}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        losses_reduced_scaled_detect = sum(loss_dict_reduced_scaled_detect.values())

        loss_value = losses_reduced_scaled.item()
        loss_value_detect = losses_reduced_scaled_detect.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        if not math.isfinite(loss_value_detect):
            print("Loss is {}, stopping training".format(loss_value_detect))
            print(loss_dict_reduced_detect)
            sys.exit(1)

        accumulated_loss += loss_value
        accumulated_loss_detect += loss_value_detect
        for k, v in loss_dict_reduced_scaled.items():
            accumulated_loss_dict_scaled[k] += (v.item() / accum_iter)
        for k, v in loss_dict_reduced_scaled_detect.items():
            accumulated_loss_dict_scaled_detect[k] += (v.item() / accum_iter)


        #optimizer.zero_grad()
        losses.backward()

        if (iteration % accum_iter == 0):
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

            grad_total_norm_print = grad_total_norm
            accumulated_loss_print = accumulated_loss / accum_iter
            accumulated_loss_dict_scaled_print = accumulated_loss_dict_scaled
            accumulated_loss = 0.0
            accumulated_loss_dict_scaled = defaultdict(float)
            accumulated_loss_print_detect = accumulated_loss_detect / accum_iter
            accumulated_loss_dict_scaled_print_detect = accumulated_loss_dict_scaled_detect
            accumulated_loss_detect = 0.0
            accumulated_loss_dict_scaled_detect = defaultdict(float)


        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        #metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        #metric_logger.update(loss=accumulated_loss_print, **{k: v for k, v in accumulated_loss_dict_scaled_print.items()})
        metric_logger.update(loss_overall=accumulated_loss_print+accumulated_loss_print_detect,
                             loss=accumulated_loss_print,
                             loss_detect=accumulated_loss_print_detect, 
                             **{
                                **{k: v for k, v in accumulated_loss_dict_scaled_print.items()},
                                **{k: v for k, v in accumulated_loss_dict_scaled_print_detect.items()}
                             })
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        #metric_logger.update(grad_norm=grad_total_norm)
        metric_logger.update(grad_norm=grad_total_norm_print)
        # gather the stats from all processes

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
