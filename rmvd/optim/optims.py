import torch

from .registry import register_optimizer, register_scheduler


@register_optimizer
def adam(model, lr, **_):
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=lr)
    return optim


@register_optimizer
def adamw(model, lr, **_):
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=lr)
    return optim


@register_optimizer
def rmsprop(model, lr, **_):
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.RMSprop(params, lr=lr, alpha=0.9)
    return optim
    

@register_scheduler
def flownet_scheduler(optimizer, **_):
    lr_intervals = [300000, 400000, 500000]
    gamma = 0.5
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_intervals, gamma=gamma)
    return scheduler
    

@register_scheduler
def monodepth2_scheduler(optimizer, **_):
    lr_intervals = [48000]
    gamma = 0.1
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_intervals, gamma=gamma)
    return scheduler


@register_scheduler
def constant_lr_scheduler(optimizer, **_):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
