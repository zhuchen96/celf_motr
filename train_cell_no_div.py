"""
Training script for the no-division variant of SelfMOTR.

Registers 'motrv2_self_no_div' in the model registry and 'e2e_cell_no_div'
in the dataset registry, then delegates to train_cell.main().

Usage (single GPU):
    python train_cell_no_div.py --config configs/CellTracking/deepcell/train_no_div.yaml

Multi-GPU:
    torchrun --nproc_per_node=2 train_cell_no_div.py \\
        --config configs/CellTracking/deepcell/train_no_div.yaml
"""

# ---- patch model registry before any other import touches models ----
import models as _models
from models.motrv2_self_no_div import build as _build_no_div

_orig_build_model = _models.build_model
def _patched_build_model(args):
    if getattr(args, 'meta_arch', '') == 'motrv2_self_no_div':
        return _build_no_div(args)
    return _orig_build_model(args)
_models.build_model = _patched_build_model

# ---- patch dataset registry ----
import datasets as _datasets
from datasets.ctc_cell_no_div import build as _build_no_div_dataset

_orig_build_dataset = _datasets.build_dataset
def _patched_build_dataset(image_set, args):
    if getattr(args, 'dataset_file', '') == 'e2e_cell_no_div':
        return _build_no_div_dataset(image_set, args)
    return _orig_build_dataset(image_set, args)
_datasets.build_dataset = _patched_build_dataset

# ---- delegate to the original training entry point ----
from train_cell import main, get_args_parser

if __name__ == '__main__':
    import argparse
    try:
        import yaml as _yaml
        _YAML_AVAILABLE = True
    except ImportError:
        _YAML_AVAILABLE = False

    def _apply_yaml_defaults(parser, yaml_path):
        if not _YAML_AVAILABLE:
            raise RuntimeError('PyYAML not installed')
        with open(yaml_path) as f:
            cfg = _yaml.safe_load(f)
        if cfg is None:
            return
        overrides = {k.replace('-', '_'): v for k, v in cfg.items() if v is not None}
        parser.set_defaults(**overrides)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    pre_args, remaining = pre.parse_known_args()

    parser = argparse.ArgumentParser('SelfMOTR no-div training', parents=[get_args_parser()])
    if pre_args.config is not None:
        _apply_yaml_defaults(parser, pre_args.config)

    args = parser.parse_args(remaining)
    from pathlib import Path
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
