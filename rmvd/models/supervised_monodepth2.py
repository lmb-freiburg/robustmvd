# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from collections import OrderedDict
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from skimage.transform import resize

from rmvd.utils import select_by_index, get_torch_model_device, to_torch, to_numpy
from .registry import register_model
from .helpers import build_model_with_cfg
from rmvd.data.transforms import UpscaleInputsToNextMultipleOf


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, imagenet_pretrained):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](imagenet_pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, x):
        self.features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
    
    
class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [F.interpolate(x, scale_factor=2, mode="nearest")]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[i] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs


class SupervisedMonodepth2(nn.Module):
    def __init__(self, num_layers, init_resnet_with_imagenet_pretrained, scales=range(4), num_output_channels=1, use_skips=True, fixed_input_resolution=None):
        super().__init__()

        self.encoder = ResnetEncoder(num_layers, init_resnet_with_imagenet_pretrained)
        num_ch_enc = self.encoder.num_ch_enc
        self.decoder = DepthDecoder(num_ch_enc, scales, num_output_channels, use_skips)
        
        self.fixed_input_resolution = fixed_input_resolution

    def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
        device = get_torch_model_device(self)

        image = select_by_index(images, keyview_idx)
        if self.fixed_input_resolution is not None:
            image = resize(image, list(image.shape[:-2]) + list(self.fixed_input_resolution))
        else:
            resized = UpscaleInputsToNextMultipleOf(32)({'images': [image]})
            image = resized['images'][0]

        image = image / 255
        image = to_torch(image, device=device)

        sample = {'images': [image], 'keyview_idx': 0}
        return sample

    def forward(self, images, keyview_idx, **_):

        image_key = select_by_index(images, keyview_idx)
        features = self.encoder(image_key)
        dec = self.decoder(features)
        
        aux = {
            'invdepths_all': [dec[key] for key in sorted(dec.keys(), reverse=True)]  # sorted from lowest to highest resolution
        }

        pred = {
            'depth': 1 / (aux['invdepths_all'][-1] + 1e-9),
        }

        return pred, aux

    def output_adapter(self, model_output):
        pred, aux = model_output
        return to_numpy(pred), to_numpy(aux)


@register_model(trainable=True)
def supervised_monodepth2(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    pretrained_weights = 'https://lmb.informatik.uni-freiburg.de/people/schroepp/weights/supervised_monodepth2.pt'
    weights = pretrained_weights if (pretrained and weights is None) else weights
    
    fixed_input_resolution = (kwargs["fixed_input_resolution"] if "fixed_input_resolution" in kwargs else None)
    
    model = build_model_with_cfg(model_cls=SupervisedMonodepth2, weights=weights, train=train, num_gpus=num_gpus, 
                                 num_layers=18, init_resnet_with_imagenet_pretrained=True,
                                 fixed_input_resolution=fixed_input_resolution)
    return model
