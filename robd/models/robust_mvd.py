import torch
import torch.nn as nn

from .registry import register_model
from .helpers import build_model_with_cfg
from .blocks.dispnet_context_encoder import DispnetContextEncoder
from .blocks.dispnet_encoder import DispnetEncoder
from .blocks.planesweep_corr import PlanesweepCorrelation
from .blocks.learned_fusion import LearnedFusion
from .blocks.dispnet_costvolume_encoder import DispnetCostvolumeEncoder
from .blocks.dispnet_decoder import DispnetDecoder

from robd.utils import get_torch_model_device


class RobustMVD(nn.Module):
    def __init__(self):

        super().__init__()

        self.encoder = DispnetEncoder()
        self.context_encoder = DispnetContextEncoder()
        self.corr_block = PlanesweepCorrelation()
        self.fusion_block = LearnedFusion()
        self.fusion_enc_block = DispnetCostvolumeEncoder()
        self.decoder = DispnetDecoder()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv3d) or isinstance(
                    m, nn.ConvTranspose3d):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, images, poses, intrinsics, keyview_idx, **_):

        keyview_idx = int(keyview_idx)  # TODO: make this batch compatible
        image_key = images[keyview_idx]  # TODO: handle this for batch size > 1
        images_source = [image for idx, image in enumerate(images) if idx != keyview_idx]
        intrinsics_key = intrinsics[keyview_idx]
        intrinsics_source = [intrinsic for idx, intrinsic in enumerate(intrinsics) if idx != keyview_idx]
        source_to_key_transforms = [pose for idx, pose in enumerate(poses) if idx != keyview_idx]

        all_enc_key, enc_key = self.encoder(image_key)
        enc_sources = [self.encoder(image_source)[1] for image_source in images_source]

        ctx = self.context_encoder(enc_key)

        corrs, masks = self.corr_block(feat_key=enc_key, intrinsics_key=intrinsics_key, feat_sources=enc_sources,
                                       source_to_key_transforms=source_to_key_transforms,
                                       intrinsics_sources=intrinsics_source)

        fused_corr, fused_mask = self.fusion_block(corrs=corrs, masks=masks)

        all_enc_fused, enc_fused = self.fusion_enc_block(corr=fused_corr, ctx=ctx)

        dec = self.decoder(enc_fused=enc_fused, all_enc={**all_enc_key, **all_enc_fused})

        pred = {
            'depth': 1 / (dec['invdepth'] + 1e-9),
            'depth_uncertainty': torch.exp(dec['invdepth_log_b']) / (dec['invdepth'] + 1e-9)
        }
        aux = dec

        return pred, aux

    def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
        device = get_torch_model_device(self)

        images = [image/255.0 - 0.4 for image in images]
        images = [torch.from_numpy(image).to(device) for image in images]
        keyview_idx = torch.from_numpy(keyview_idx).to(device)

        if poses is not None:
            poses = [torch.from_numpy(pose).to(device) for pose in poses]

        if intrinsics is not None:
            intrinsics = [torch.from_numpy(intrinsic).to(device) for intrinsic in intrinsics]

        if depth_range is not None:
            depth_range = [torch.Tensor(depth_range[0]).to(device), torch.Tensor(depth_range[1]).to(device)]

        sample = {
            'images': images,
            'keyview_idx': keyview_idx,
            'poses': poses,
            'intrinsics': intrinsics,
            'depth_range': depth_range,
        }
        return sample

    def output_adapter(self, model_output):
        pred_in, aux_in = model_output
        pred_out, aux_out = {}, {}

        for k, v in pred_in.items():
            pred_out[k] = v.detach().cpu().numpy()

        for k, v in aux_in.items():  # TODO: move this to a to_numpy function
            if isinstance(v, list):
                aux_out[k] = [ele.detach().cpu().numpy() for ele in v]
            else:
                aux_out[k] = v.detach().cpu().numpy()

        return pred_out, aux_out


@register_model
def robust_mvd(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    pretrained_weights = 'https://github.com/robustmvd/robustmvd/raw/master/weights/robustmvd.pt'
    weights = pretrained_weights if (pretrained and weights is None) else None
    model = build_model_with_cfg(model_cls=RobustMVD, weights=weights, train=train, num_gpus=num_gpus)
    return model
