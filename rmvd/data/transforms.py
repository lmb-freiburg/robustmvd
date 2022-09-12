import numpy as np
import cv2


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
            image_interpolation_method=cv2.INTER_LINEAR,
    ):
        self.__height = size[0]
        self.__width = size[1]

        self.__resize_target = resize_target
        self.__image_interpolation_method = image_interpolation_method

    def __call__(self, sample):
        image = sample["images"][0]
        orig_ht, orig_wd = image.shape[-2:]
        wd, ht = self.__width, self.__height

        # TODO: maybe use torch transform Resize or skimage transform resize intead of opencv resize

        # resize images
        images = sample["images"]
        images = [np.transpose(image, [1, 2, 0]) for image in images]  # H, W, 3
        images = [cv2.resize(image, (wd, ht), interpolation=self.__image_interpolation_method) for image in images]
        images = [np.transpose(image, [2, 0, 1]) for image in images]  # 3. H, W
        sample["images"] = images
        
        # resize intrinsics:
        orig_size_arr = np.array([[orig_wd]*3, [orig_ht]*3, [1.]*3], dtype=np.float32)  # 3, 3
        sample["intrinsics"] = [intrinsic / orig_size_arr for intrinsic in sample["intrinsics"]]  # TODO: * new size

        sample["orig_width"] = orig_wd
        sample["orig_height"] = orig_ht

        if self.__resize_target:
            pass
            # TODO

        return sample
