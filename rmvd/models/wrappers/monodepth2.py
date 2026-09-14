import os.path as osp

import torch
import torch.nn as nn
import numpy as np

from ..registry import register_model
from ..helpers import build_model_with_cfg
from .utils import get_repo_path
from rmvd.utils import get_torch_model_device, to_numpy, to_torch, select_by_index, logging
from rmvd.data.transforms import ResizeInputs


class Monodepth2_Wrapped(nn.Module):
    def __init__(self, model_name, trained_on_stereo, post_uncertainty=False):
        super().__init__()

        import sys
        repo_path = get_repo_path("monodepth2")
        sys.path.insert(0, repo_path)
        import networks

        self.encoder = networks.ResnetEncoder(18, False)
        self.decoder = networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        # load pretrained weights:
        encoder_weights_path = osp.join(repo_path, "models", model_name, "encoder.pth")
        decoder_weights_path = osp.join(repo_path, "models", model_name, "depth.pth")

        assert osp.isfile(encoder_weights_path) and osp.isfile(decoder_weights_path), \
            f"Model weights not found. Please download the {model_name} weights " \
            f"from https://github.com/nianticlabs/monodepth2 and extract it to" \
            f"{osp.join(repo_path, 'models')}"

        logging.info(f"Using model weights from {encoder_weights_path} and {decoder_weights_path}.")
        encoder_weights = torch.load(encoder_weights_path)
        filtered_encoder_weights = {k: v for k, v in encoder_weights.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_encoder_weights)
        decoder_weights = torch.load(decoder_weights_path)
        self.decoder.load_state_dict(decoder_weights)

        # extract the height and width of image that this model was trained with
        self.height = encoder_weights['height']
        self.width = encoder_weights['width']

        self.trained_on_stereo = trained_on_stereo
        self.post_uncertainty = post_uncertainty

    def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
        device = get_torch_model_device(self)

        image = select_by_index(images, keyview_idx)

        orig_ht, orig_wd = images[0].shape[-2:]
        if (orig_ht != self.height) or (orig_wd != self.width):
            resized = ResizeInputs(size=(self.height, self.width))({'images': [image]})
            image = resized['images'][0]

        image = image / 255
        image = to_torch(image, device=device)

        sample = {'image': image}
        return sample

    def forward(self, image, **_):
        if self.post_uncertainty:
            image_flipped = torch.flip(image, dims=[3])
            image = torch.cat([image, image_flipped], dim=0)
        
        features = self.encoder(image)
        outputs = self.decoder(features)
        disp = outputs[("disp", 0)]

        min_depth, max_depth = 0.1, 100
        min_disp, max_disp = 1 / max_depth, 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp

        if self.trained_on_stereo:
            stereo_scale_factor = 5.4
            scaled_disp = scaled_disp / stereo_scale_factor
            
        if self.post_uncertainty:
            scaled_disp_orig, scaled_disp_flipped = torch.split(scaled_disp, image.shape[0] // 2, dim=0)
            scaled_disp_flipped = torch.flip(scaled_disp_flipped, dims=[3])
            depth_uncertainty = torch.abs(scaled_disp_orig - scaled_disp_flipped)
            h, w = image.shape[-2:]
            l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
            l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, None, ...]
            r_mask = (l_mask[:, :, :, ::-1]).copy()
            l_mask = torch.from_numpy(l_mask).to(image.device)
            r_mask = torch.from_numpy(r_mask).to(image.device)
            scaled_disp = r_mask * scaled_disp_orig + l_mask * scaled_disp_flipped + (1.0 - l_mask - r_mask) * 0.5 * (scaled_disp_orig + scaled_disp_flipped)
        
        depth = 1 / (scaled_disp + 1e-9)

        pred = {'depth': depth}
        if self.post_uncertainty:
            pred ['depth_uncertainty'] = depth_uncertainty
            
        aux = {}
        return pred, aux

    def output_adapter(self, model_output):
        pred, aux = model_output
        return to_numpy(pred), to_numpy(aux)


@register_model(trainable=False)
def monodepth2_mono_stereo_wrapped(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    assert pretrained and (weights is None), "Model supports only pretrained=True, weights=None."
    cfg = {"model_name": "mono+stereo_640x192", "trained_on_stereo": True}
    model = build_model_with_cfg(model_cls=Monodepth2_Wrapped, cfg=cfg, weights=None, train=train, num_gpus=num_gpus)
    return model


@register_model(trainable=False)
def monodepth2_mono_stereo_1024x320_wrapped(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    assert pretrained and (weights is None), "Model supports only pretrained=True, weights=None."
    cfg = {"model_name": "mono+stereo_1024x320", "trained_on_stereo": True}
    model = build_model_with_cfg(model_cls=Monodepth2_Wrapped, cfg=cfg, weights=None, train=train, num_gpus=num_gpus)
    return model


@register_model(trainable=False)
def monodepth2_postuncertainty_mono_stereo_wrapped(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    assert pretrained and (weights is None), "Model supports only pretrained=True, weights=None."
    cfg = {"model_name": "mono+stereo_640x192", "trained_on_stereo": True}
    model = build_model_with_cfg(model_cls=Monodepth2_Wrapped, cfg=cfg, weights=None, train=train, num_gpus=num_gpus, post_uncertainty=True)
    return model
