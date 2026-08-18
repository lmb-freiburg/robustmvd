import os.path as osp
import sys
import argparse

import torch

from ..registry import register_model
from ..factory import prepare_custom_model
from rmvd.utils import get_path


@register_model(trainable=False)
def mvsanywhere_wrapped(
    pretrained=True, weights=None, train=False, num_gpus=1, **kwargs
):
    assert pretrained and (
        weights is None
    ), "Model supports only pretrained=True, weights=None."

    assert not train, "Model supports only train=False."

    # MVSAnywhere internally wraps the model in a nn.DataParallel over all visible CUDA devices, which moves the
    # model to a GPU regardless of the num_gpus setting. CPU usage (num_gpus=0) is therefore only possible if no
    # CUDA devices are visible, e.g. by setting the environment variable CUDA_VISIBLE_DEVICES to an empty string:
    assert num_gpus >= 1 or not torch.cuda.is_available(), \
        "Model supports num_gpus=0 only if no CUDA devices are visible, e.g. by setting CUDA_VISIBLE_DEVICES='' ."

    paths_file = osp.join(osp.dirname(osp.realpath(__file__)), 'paths.toml')
    repo_path = get_path(paths_file, "mvsanywhere", "root")
    repo_src_path = osp.join(repo_path, 'src')
    sys.path.insert(0, repo_src_path)

    import mvsanywhere.options as options
    from mvsanywhere.experiment_modules.rmvd_mvsa import MVSA_Wrapped

    config_file = osp.join(repo_path, 'configs/models/mvsanywhere_model.yaml')

    cl_args = {
        'name': 'mvsanywhere',
        'load_weights_from_checkpoint': osp.join(repo_path, 'pretrained_weights/mvsanywhere_hero.ckpt'),
    }
    cl_args = argparse.Namespace(**cl_args)

    option_handler = options.OptionsHandler()
    config_options = options.OptionsHandler.load_options_from_yaml(config_file)
    option_handler.merge_config_options(config_options)
    option_handler.config_filepaths = [config_file]
    option_handler.merge_cl_args(cl_args)
    opts = option_handler.options
    opts.gpus = num_gpus
    option_handler.options = opts

    # if no GPUs are available for us then, use the 32 bit on CPU
    if opts.gpus == 0:
        opts.precision = 32

    model = MVSA_Wrapped(opts, use_refinement=True)
    num_gpus = 0 if num_gpus == 0 else 1
    model = prepare_custom_model(model, num_gpus=num_gpus)  # intentionally not passing num_gpus>1, as MVSA_Wrapped anyways always wraps the model in DataParallel

    return model
