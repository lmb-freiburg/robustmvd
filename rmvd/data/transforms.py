import numpy as np
from skimage.transform import resize


class Resize:
    """Resize sample to given size (height, width).

    Args:
        size (tuple): tuple with output size (height, width).
        resize_target (bool, optional): If true, resize target data (e.g. depth map).
    """
    def __init__(
            self,
            size,
            resize_target=False,
            interpolation_order=1,
    ):
        self.__height = size[0]
        self.__width = size[1]

        self.__resize_target = resize_target
        self.__interpolation_order = interpolation_order

    def __call__(self, sample):
        image = sample["images"][0]
        orig_ht, orig_wd = image.shape[-2:]
        wd, ht = self.__width, self.__height

        # resize images
        images = sample["images"]
        images = [resize(image, list(image.shape[:-2]) + [ht, wd], order=self.__interpolation_order) for image in images]
        sample["images"] = images
        
        # resize intrinsics:
        if "intrinsics" in sample:
            scale_arr = np.array([[wd / orig_wd]*3, [ht / orig_ht]*3, [1.]*3], dtype=np.float32)  # 3, 3
            sample["intrinsics"] = [intrinsic * scale_arr for intrinsic in sample["intrinsics"]]

        sample["orig_width"] = orig_wd
        sample["orig_height"] = orig_ht

        if self.__resize_target:
            pass
            # TODO

        return sample
