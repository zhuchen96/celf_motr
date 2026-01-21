# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
import numpy as np


def load_torch_checkpoint(path, map_location='cpu', weights_only=False):
    """Version-safe torch.load for checkpoints across old/new PyTorch."""
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        # Older PyTorch versions do not support weights_only argument.
        return torch.load(path, map_location=map_location)


def _infer_indexed_module_count(state_dict, prefix):
    ids = set()
    for key in state_dict.keys():
        if not key.startswith(prefix):
            continue
        parts = key.split('.')
        prefix_parts = prefix.rstrip('.').split('.')
        if len(parts) <= len(prefix_parts):
            continue
        idx = parts[len(prefix_parts)]
        if idx.isdigit():
            ids.add(int(idx))
    if ids:
        return max(ids) + 1
    return None


def infer_dec_layers_from_checkpoint(checkpoint):
    ckpt_args = checkpoint.get('args')
    if ckpt_args is not None and hasattr(ckpt_args, 'dec_layers'):
        return int(ckpt_args.dec_layers)

    state_dict = checkpoint.get('model', {})
    decoder_count = _infer_indexed_module_count(state_dict, 'transformer.decoder.layers.')
    if decoder_count is not None and decoder_count > 1:
        return decoder_count

    head_count = _infer_indexed_module_count(state_dict, 'bbox_embed.')
    if head_count is None:
        head_count = _infer_indexed_module_count(state_dict, 'class_embed.')
    if head_count is not None:
        two_stage = bool(getattr(ckpt_args, 'two_stage', False)) if ckpt_args is not None else False
        if two_stage and head_count > 0:
            return head_count - 1
        return head_count

    return decoder_count


def infer_shared_decoder_from_checkpoint(checkpoint):
    ckpt_args = checkpoint.get('args')
    if ckpt_args is not None and hasattr(ckpt_args, 'shared_decoder'):
        return bool(ckpt_args.shared_decoder)

    state_dict = checkpoint.get('model', {})
    decoder_count = _infer_indexed_module_count(state_dict, 'transformer.decoder.layers.')
    if decoder_count is None:
        return None
    if decoder_count > 1:
        return False

    head_count = _infer_indexed_module_count(state_dict, 'bbox_embed.')
    if head_count is None:
        head_count = _infer_indexed_module_count(state_dict, 'class_embed.')
    if head_count is not None and head_count > 1:
        return True
    return False


def apply_checkpoint_model_args(args, checkpoint, context='checkpoint'):
    ckpt_args = checkpoint.get('args')

    for name in ('with_box_refine', 'two_stage'):
        if ckpt_args is None or not hasattr(ckpt_args, name) or not hasattr(args, name):
            continue
        new_value = getattr(ckpt_args, name)
        old_value = getattr(args, name)
        if old_value != new_value:
            print(f"[{context}] Overriding --{name} from {old_value} to {new_value} based on checkpoint.")
            setattr(args, name, new_value)

    if hasattr(args, 'shared_decoder'):
        new_value = infer_shared_decoder_from_checkpoint(checkpoint)
        old_value = getattr(args, 'shared_decoder')
        if new_value is not None and old_value != new_value:
            print(f"[{context}] Overriding --shared_decoder from {old_value} to {new_value} based on checkpoint.")
            setattr(args, 'shared_decoder', new_value)

    if hasattr(args, 'dec_layers'):
        new_value = infer_dec_layers_from_checkpoint(checkpoint)
        old_value = getattr(args, 'dec_layers')
        if new_value is not None and old_value != new_value:
            print(f"[{context}] Overriding --dec_layers from {old_value} to {new_value} based on checkpoint.")
            setattr(args, 'dec_layers', new_value)

    return args


def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None):
    start_epoch = 0
    checkpoint = load_torch_checkpoint(
        model_path,
        map_location=lambda storage, loc: storage,
        weights_only=False,
    )
    print(f'loaded {model_path}')
    state_dict = checkpoint['model']
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                if 'class_embed' in k:
                    print("load class_embed: {} shape={}".format(k, state_dict[k].shape))
                    if model_state_dict[k].shape[0] == 1:
                        state_dict[k] = state_dict[k][1:2]
                    elif model_state_dict[k].shape[0] == 2:
                        state_dict[k] = state_dict[k][1:3]
                    elif model_state_dict[k].shape[0] == 3:
                        state_dict[k] = state_dict[k][1:4]
                    else:
                        raise NotImplementedError('invalid shape: {}'.format(model_state_dict[k].shape))
                    continue
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model

