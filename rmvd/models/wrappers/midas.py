import os.path as osp

import torch
import torch.nn as nn
from torchvision.transforms import Compose
import numpy as np
import cv2

from ..registry import register_model
from ..helpers import build_model_with_cfg
from .utils import get_repo_path
from rmvd.utils import get_torch_model_device, to_numpy, to_torch, select_by_index, get_class


class MiDaS_Wrapped(nn.Module):
    def __init__(self, weights_name, midas_cls_name, norm_mean, norm_std, resize_method, model_kwargs=None):
        super().__init__()

        import sys
        repo_path = get_repo_path("midas")
        sys.path.insert(0, repo_path)
        midas_cls_name = f"midas.{midas_cls_name}"
        model_cls = get_class(midas_cls_name)
        from midas.transforms import Resize, NormalizeImage, PrepareForNet

        weights_path = osp.join(repo_path, "weights", weights_name)
        model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.model = model_cls(weights_path, **model_kwargs)

        net_w, net_h = 384, 384
        self.transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method=resize_method,
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=norm_mean, std=norm_std),
                PrepareForNet(),
            ]
        )

    def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
        device = get_torch_model_device(self)

        image_batch = select_by_index(images, keyview_idx)
        tmp_images = []
        for image in image_batch:
            image = image / 255.
            image = np.transpose(image, [1, 2, 0])
            image = self.transform({"image": image})["image"]
            tmp_images.append(image)
        image = np.stack(tmp_images)

        image = to_torch(image, device=device)

        sample = {'image': image}
        return sample

    def forward(self, image, **_):
        pred_invdepth = self.model(image)
        return pred_invdepth

    def output_adapter(self, model_output):
        pred_invdepth = model_output
        pred_invdepth = to_numpy(pred_invdepth)
        with np.errstate(divide='ignore', invalid='ignore'):
            pred_depth = 1 / pred_invdepth
        pred_depth = pred_depth[:, None]
        pred = {'depth': pred_depth}
        aux = {}
        return pred, aux


@register_model(trainable=False)
def midas_large_v2_1_wrapped(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    assert pretrained and (weights is None), "Model supports only pretrained=True, weights=None."
    
    cfg = {"weights_name": "midas_v21_384.pt", "midas_cls_name": "midas_net.MidasNet", "model_kwargs": {"non_negative": True},
           "norm_mean": [0.485, 0.456, 0.406], "norm_std": [0.229, 0.224, 0.225], "resize_method": "upper_bound"}
    
    model = build_model_with_cfg(model_cls=MiDaS_Wrapped, cfg=cfg, weights=None, train=train, num_gpus=num_gpus)
    return model


@register_model(trainable=False)
def midas_dpt_large_384_wrapped(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    assert pretrained and (weights is None), "Model supports only pretrained=True, weights=None."
    
    cfg = {"weights_name": "dpt_large_384.pt", "midas_cls_name": "dpt_depth.DPTDepthModel", "model_kwargs": {"non_negative": True, "backbone": "vitl16_384"},
           "norm_mean": [0.5, 0.5, 0.5], "norm_std": [0.5, 0.5, 0.5], "resize_method": "minimal"}
    
    model = build_model_with_cfg(model_cls=MiDaS_Wrapped, cfg=cfg, weights=None, train=train, num_gpus=num_gpus, )
    return model
