import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import mae, pointwise_ae
from .registry import register_loss
from rmvd.utils import logging


class SingleScaleMAE(nn.Module):
    def __init__(self, model, weight_decay=1e-4, gt_interpolation="nearest", modality="invdepth", weight_by_sampling_interval=False, verbose=True):

        super().__init__()

        self.verbose = verbose

        if self.verbose:
            logging.info(f"Initializing {self.name} loss.")

        self.weight_decay = weight_decay
        self.gt_interpolation = gt_interpolation
        self.weight_by_sampling_interval = weight_by_sampling_interval

        self.modality = modality

        self.reg_params = self.get_regularization_parameters(model)

        if self.verbose:
            logging.info(f"\tWeight decay: {self.weight_decay}")
            logging.info(f"\tGT interpolation: {self.gt_interpolation}")
            logging.info(f"\tModality: {self.modality}")
            logging.info(f"Finished initializing {self.name} loss.")
            logging.info()

    @property
    def name(self):
        name = type(self).__name__
        return name

    def get_regularization_parameters(self, model):
        reg_params = []
        for name, param in model.named_parameters():
            if "pred" not in name and not name.endswith("bias") and not name.endswith(
                    "bn.weight") and param.requires_grad:
                reg_params.append((name, param))

        if self.verbose:
            logging.info(f"\tApplying regularization loss with weight decay {self.weight_decay} on:")
            for i, val in enumerate(reg_params):
                name, param = val
                logging.info(f"\t\t#{i} {name}: {param.shape} ({param.numel()})")

        return reg_params

    def forward(self, sample_inputs, sample_gt, pred, aux, iteration):

        sub_losses = {}
        pointwise_losses = {}

        gt = sample_gt[self.modality]
        gt_mask = gt > 0

        pred = aux[self.modality]
        
        if self.weight_by_sampling_interval:
            sampling_invdepths = aux["sampling_invdepths"]
            steps = sampling_invdepths.shape[1]
            max_depth = 1. / sampling_invdepths[:, 0:1, ...]  # shape [N, 1, H, W] or [N, 1, 1, 1]
            min_depth = 1. / sampling_invdepths[:, -1:, ...]  # shape [N, 1, H, W] or [N, 1, 1, 1]
            interval = (max_depth - min_depth) / (steps-1)
            loss_weight = 1 / interval
        else:
            loss_weight = 1

        total_reg_loss = 0

        with torch.no_grad():
            gt_resampled = F.interpolate(gt, size=pred.shape[-2:], mode=self.gt_interpolation)
            gt_mask_resampled = F.interpolate(gt_mask.float(), size=pred.shape[-2:], mode="nearest") == 1.0

        mae_loss = mae(gt=gt_resampled, pred=pred, mask=gt_mask_resampled, weight=loss_weight)
        pointwise_ae_loss = pointwise_ae(gt=gt_resampled, pred=pred, mask=gt_mask_resampled, weight=loss_weight)

        for _, param in self.reg_params:
            reg_loss = torch.sum(torch.mul(param, param)) / 2.0
            total_reg_loss += reg_loss
        total_reg_loss *= self.weight_decay

        total_loss = mae_loss + total_reg_loss

        sub_losses["00_mae"] = mae_loss
        sub_losses["01_reg"] = total_reg_loss
        pointwise_losses["0_ae"] = pointwise_ae_loss
        return total_loss, sub_losses, pointwise_losses
