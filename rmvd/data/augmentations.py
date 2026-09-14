import torchvision.transforms
from .registry import register_augmentation
from .transforms import ResizeInputs, ResizeTargets, ColorJitter, Eraser, NormalizeImagesToMinMax, MaskDepthByMinMax, SpatialAugmentation, NormalizeIntrinsics, NormalizeImagesByShiftAndScale


@register_augmentation
def robust_mvd_augmentations_staticthings3d(**kwargs):
    transforms = [
        ColorJitter(saturation=(0, 2), contrast=(0.01, 8), brightness=(0.01, 2.0), hue=0.5),
        SpatialAugmentation(size=(384, 768), p=1.0),
        NormalizeImagesToMinMax(min_val=-0.4, max_val=0.6),
        NormalizeIntrinsics(),
        Eraser(bounds=[250, 500], p=0.6),
        MaskDepthByMinMax(min_depth=1/2.75, max_depth=1/0.009),
    ]
    transform = torchvision.transforms.Compose(transforms)
    return transform


@register_augmentation
def robust_mvd_augmentations_blendedmvs(**kwargs):
    transforms = [
        ColorJitter(saturation=(0, 2), contrast=(0.01, 8), brightness=(0.01, 2.0), hue=0.5),
        ResizeInputs(size=(384, 768), backend="cv2_bilinear"),
        ResizeTargets(size=(384, 768)),
        NormalizeImagesToMinMax(min_val=-0.4, max_val=0.6),
        NormalizeIntrinsics(),
        Eraser(bounds=[250, 500], p=0.6),
        # intentionally not masking depth
    ]
    transform = torchvision.transforms.Compose(transforms)
    return transform

@register_augmentation
def supervised_monodepth2_augmentations(**kwargs):
    transforms = [
        ColorJitter(saturation=(0.3, 1.5), contrast=(0.3, 1.5), brightness=(0.8, 1.2), hue=0.1),
        ResizeInputs(size=(192, 640)),
        ResizeTargets(size=(192, 640)),
        NormalizeImagesToMinMax(min_val=0., max_val=1.),
        NormalizeImagesByShiftAndScale(shift=[0.485, 0.456, 0.406], scale=[0.229, 0.224, 0.225]),
    ]
    transform = torchvision.transforms.Compose(transforms)
    return transform
