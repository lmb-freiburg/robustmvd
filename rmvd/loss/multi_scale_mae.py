import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import mae, pointwise_ae
from .registry import register_loss
from rmvd.utils import logging


class MultiScaleMAE(nn.Module):
    def __init__(self, model, weight_decay=1e-4, gt_interpolation="nearest", modality="invdepth", verbose=True):

        super().__init__()

        self.verbose = verbose

        if self.verbose:
            logging.info(f"Initializing {self.name} loss.")

        self.weight_decay = weight_decay
        self.gt_interpolation = gt_interpolation

        self.loss_weights = [1 / 8, 1 / 4, 1 / 2, 1]
        self.loss_weights = [100 * 1050 * weight for weight in self.loss_weights]

        self.modality = modality

        self.reg_params = self.get_regularization_parameters(model)

        if self.verbose:
            logging.info(f"\tWeight decay: {self.weight_decay}")
            logging.info(f"\tGT interpolation: {self.gt_interpolation}")
            logging.info(f"\tModality: {self.modality}")
            logging.info(f"\tLoss weights: {self.loss_weights}")
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

        preds_all = aux[f"{self.modality}s_all"]

        total_mnll_loss = 0
        total_reg_loss = 0

        for level, pred in enumerate(preds_all):

            with torch.no_grad():
                gt_resampled = F.interpolate(gt, size=pred.shape[-2:], mode=self.gt_interpolation)
                gt_mask_resampled = F.interpolate(gt_mask.float(), size=pred.shape[-2:], mode="nearest") == 1.0

            loss = mae(gt=gt_resampled, pred=pred, mask=gt_mask_resampled, weight=self.loss_weights[level])
            pointwise_loss = pointwise_ae(gt=gt_resampled, pred=pred, mask=gt_mask_resampled,
                                            weight=self.loss_weights[level])

            sub_losses["02_mnll/level_%d" % level] = loss
            pointwise_losses["00_nll/level_%d" % level] = pointwise_loss

            total_mnll_loss += loss

        for name, param in self.reg_params:
            reg_loss = torch.sum(torch.mul(param, param)) / 2.0
            total_reg_loss += reg_loss
        total_reg_loss *= self.weight_decay

        total_loss = total_mnll_loss + total_reg_loss

        sub_losses["00_total_mnll"] = total_mnll_loss
        sub_losses["01_reg"] = total_reg_loss
        return total_loss, sub_losses, pointwise_losses


@register_loss
def supervised_monodepth2_loss(**kwargs):
    return MultiScaleMAE(weight_decay=0., gt_interpolation="nearest", modality="invdepth", **kwargs)
