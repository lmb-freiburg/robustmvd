import os.path as osp
import re

import numpy as np
from PIL import Image

from .dataset import Dataset, Sample
from .registry import register_default_dataset
from .layouts import MVDUnstructuredDefaultLayout, AllImagesLayout


def readPFM(file):  # TODO: move to rmvd utils
    file = open(file, 'rb')

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    if data.ndim == 3:
        data = data.transpose(2, 0, 1)
    return data


def load_image(root, path):
    path = f"blended_images/{path:08d}_masked.jpg"
    img_path = osp.join(root, path)
    img = np.array(Image.open(img_path))
    img = img.transpose(2, 0, 1).astype(np.float32)  # 3,H,W ; dtype np.uint8
    return img


def load_pose(root, path):
    path = f"cams/{path:08d}_cam.txt"
    pose_path = osp.join(root, path)
    with open(pose_path) as pose_file:
        pose_lines = [x[:-1] for x in pose_file.readlines()][1:5]
        pose_eles = [float(x) for line in pose_lines for x in line.split()]
        pose = np.array([pose_eles[0:4], pose_eles[4:8], pose_eles[8:12], pose_eles[12:16], ], dtype=np.float32)
    return pose  # 4, 4


def load_intrinsics(root, path):
    path = f"cams/{path:08d}_cam.txt"
    pose_path = osp.join(root, path)
    with open(pose_path) as pose_file:
        intrinsic_lines = [x[:-1] for x in pose_file.readlines()][7:10]
        intrinsic_eles = [float(x) for line in intrinsic_lines for x in line.split()]
        intrinsic = np.array([intrinsic_eles[0:3], intrinsic_eles[3:6], intrinsic_eles[6:9], ], dtype=np.float32)
    return intrinsic  # 3, 3


def load_depth(root, path):
    path = f"rendered_depth_maps/{path:08d}.pfm"
    depth = readPFM(osp.join(root, path))
    depth = np.nan_to_num(depth, posinf=0., neginf=0., nan=0.)
    depth = np.expand_dims(depth, 0).astype(np.float32)  # 1HW
    return depth  # 1, H, W, np.float32


def load(key, root, val):
    if isinstance(val, list):
        return [load(key, root, v) for v in val]
    else:
        if key == 'images':
            return load_image(root, val)
        elif key == 'depth':
            return load_depth(root, val)
        elif key == 'intrinsics':
            return load_intrinsics(root, val)
        elif key == 'poses':
            return load_pose(root, val)
        else:
            return val


class BlendedMVSSample(Sample):

    def __init__(self, base, name):
        self.base = base
        self.name = name
        self.data = {}

    def load(self, root):

        base = osp.join(root, self.base)
        out_dict = {'_base': base, '_name': self.name}

        for key, val in self.data.items():
            out_dict[key] = load(key, base, val)

        return out_dict


# TODO: init on the fly instead of loading from file
@register_default_dataset
class BlendedMVSSeq4Train(Dataset):

    base_dataset = 'blendedmvs'
    split = 'seq4_train'
    dataset_type = 'mvd'

    def __init__(self, root=None, layouts=None, **kwargs):
        root = root if root is not None else self._get_path("blendedmvs", "root")

        default_layouts = [
            MVDUnstructuredDefaultLayout("default", num_views=5, max_views=5),
            AllImagesLayout("all_images", num_views=5),
        ]
        layouts = default_layouts + layouts if layouts is not None else default_layouts

        super().__init__(root=root, layouts=layouts, **kwargs)
