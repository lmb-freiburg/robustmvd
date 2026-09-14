import os
import os.path as osp

import numpy as np
from PIL import Image

from .dataset import Dataset, Sample
from .registry import register_folder_parser
from .layouts import AllImagesLayout


class RmvdSampleSample(Sample):
    def __init__(self, name, image_paths, intrinsics, poses):
        self.name = name
        self.image_paths = image_paths
        self.intrinsics = intrinsics
        self.poses = poses

    def load(self, root):
        images = [np.array(Image.open(osp.join(root, path)).convert('RGB'), dtype=np.float32).transpose(2, 0, 1)
                  for path in self.image_paths]
        return {'_base': root, '_name': self.name, 'images': images,
                'intrinsics': list(self.intrinsics), 'poses': list(self.poses), 'keyview_idx': 0}


@register_folder_parser
class RmvdSampleMVDParser(Dataset):
    
    folder_structure = 'rmvd_sample'
    data_format = 'mvd'

    def __init__(self, root, layouts=None, **kwargs):
        assert osp.isdir(osp.join(root, "key")) and osp.isdir(osp.join(root, "source")), \
            f"Path {root} does not contain a 'key' and a 'source' folder."

        num_views = 1 + len(os.listdir(osp.join(root, "source")))
        default_layouts = [AllImagesLayout("default", num_views=num_views)]
        layouts = default_layouts + layouts if layouts is not None else default_layouts

        super().__init__(root=root, layouts=layouts, **kwargs)

    def _init_samples(self):
        source_dir = osp.join(self.root, "source")
        view_dirs = [osp.join(self.root, "key")] + sorted([osp.join(source_dir, x) for x in os.listdir(source_dir)])

        sample = RmvdSampleSample(
            name=osp.basename(osp.abspath(self.root)),
            image_paths=[osp.relpath(osp.join(view_dir, "image.png"), self.root) for view_dir in view_dirs],
            intrinsics=[np.load(osp.join(view_dir, "K.npy")).astype(np.float32) for view_dir in view_dirs],
            poses=[np.load(osp.join(view_dir, "to_ref_transform.npy")).astype(np.float32) for view_dir in view_dirs],
        )
        self.samples.append(sample)
