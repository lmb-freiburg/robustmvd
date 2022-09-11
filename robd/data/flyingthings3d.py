import os.path as osp
import re

import numpy as np
from PIL import Image

from .dataset import Dataset, Sample
from .registry import register_dataset, register_default_dataset


def readPFM(file):
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


def load_image(root, cam, frame_num):
    filename = '{:04d}.png'.format(frame_num)
    cam = 'left' if cam == 'l' else 'right'
    img = np.array(Image.open(osp.join(root, "frames_cleanpass", cam, filename)))
    img = img.transpose([2, 0, 1]).astype(np.float32)
    return img  # 3, H, W, float32


def load_depth(root, cam, frame_num):
    filename = '{:04d}.pfm'.format(frame_num)
    cam = 'left' if cam == 'l' else 'right'
    disparity = readPFM(osp.join(root, "disparities", cam, filename))
    depth = 1050./disparity
    depth[(depth < 0.) | np.isinf(depth) | np.isnan(depth)] = 0.
    depth = np.expand_dims(depth, 0)  # 1HW
    return depth


def load_intrinsics(root, cam, frame_num):
    filename = '{:04d}.npy'.format(frame_num)
    cam = 'left' if cam == 'l' else 'right'
    return np.load(osp.join(root, "intrinsics", cam, filename))


def load_pose(root, cam, frame_num):
    filename = '{:04d}.npy'.format(frame_num)
    cam = 'left' if cam == 'l' else 'right'
    return np.load(osp.join(root, "poses", cam, filename))


def load(key, root, val):
    if isinstance(val, list):
        return [load(key, root, v) for v in val]
    else:
        if key == 'images':
            cam, frame_num = val
            return load_image(root, cam, frame_num)
        elif key == 'depth':
            cam, frame_num = val
            return load_depth(root, cam, frame_num)
        elif key == 'intrinsics':
            cam, frame_num = val
            return load_intrinsics(root, cam, frame_num)
        elif key == 'poses':
            cam, frame_num = val
            return load_pose(root, cam, frame_num)
        else:
            return val


class FlyingThings3DSample(Sample):

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


@register_default_dataset
class FlyingThings3DSeq4Train(Dataset):

    base_dataset = 'flyingthings3d'
    split = 'seq4_train'
    dataset_type = 'mvd'

    # TODO: create sample_list on the fly instead of loading from file?
    def __init__(self, root=None, **kwargs):
        root = root if root is not None else osp.join(self._get_path("flyingthings3d", "root"), "TRAIN")
        super().__init__(root=root, **kwargs)










# class FlyingThings3D(Dataset):
#     def __init__(self, sample_confs, version, split, type,
#                  sample_filter_fct=None, hard_samples_file=None, use_subset=False,
#                  root=None, scratch_paths=None, aug_fcts=None, to_torch=False,
#                  updates=None, update_strict=False, layouts=None, verbose=True):
#
#         self.version = version
#         self.split = split
#         self.type = type
#         root = root if root is not None else self._get_path("FlyingThings3D", self.version, self.split, "root")
#         scratch_paths = scratch_paths if scratch_paths is not None else self._get_path("FlyingThings3D", self.version, self.split, "scratch")
#
#         super().__init__(root=root, scratch_paths=scratch_paths, aug_fcts=aug_fcts, to_torch=to_torch,
#                          updates=updates, update_strict=update_strict, layouts=layouts, verbose=verbose,
#                          sample_confs=sample_confs, hard_samples_file=hard_samples_file,
#                          use_subset=use_subset, sample_filter_fct=sample_filter_fct)
#
#     def _get_scratch_available(self, root, scratch_paths):
#
#         scratch_paths = [scratch_paths] if not isinstance(scratch_paths, list) else scratch_paths
#
#         for scratch_path in scratch_paths:
#             if scratch_path is not None:
#                 if osp.isdir(scratch_path):
#                     return scratch_path
#
#         return False
#
#     def _get_sample_list_path(self):
#         lists_path = self._get_path("FlyingThings3D", self.version, self.split, "lists")
#         if lists_path is not None:
#             return osp.join(lists_path, "{}.{}.pickle".format(self.type, self.name))
#         else:
#             return None
#
#     def _init_samples(self, sample_confs=None, hard_samples_file=None, use_subset=False, sample_filter_fct=None):
#
#         sample_list_path = self._get_sample_list_path()
#
#         if sample_list_path is not None and osp.isfile(self._get_sample_list_path()):
#             self._init_samples_from_list(sample_list_path)
#
#         elif sample_confs is not None:
#             self._init_samples_from_confs(sample_confs, hard_samples_file, use_subset)
#
#             if sample_filter_fct is not None:
#                 self._filter_samples(sample_filter_fct)
#
#             self.write_samples()
#
#     def _init_samples_from_confs(self, sample_confs, hard_samples_file=None, use_subset=False):
#
#         if hard_samples_file is not None:
#             with open(hard_samples_file, 'r') as hard_samples_file:
#                 filtered_samples = hard_samples_file.readlines()
#                 filtered_samples = [x[:-1] for x in filtered_samples]
#                 filtered_samples = [x.split(':') for x in filtered_samples]
#         else:
#             filtered_samples = []
#
#         if use_subset:
#             filtered_samples += get_subset_filtered_frames()
#
#         sequences = sorted(glob(osp.join(self.root, '*/*[0-9]')))
#
#         for sample_conf in sample_confs:
#
#             for sequence in sequences:
#
#                 files = {}
#                 frame_nums = set()
#
#                 for key, conf in sample_conf.items():
#
#                     if not (isinstance(conf, DataConf) or isinstance(conf, list)):
#                         continue
#
#                     if isinstance(conf, DataConf):
#
#                         files[key] = {}
#
#                         for file in os.listdir(osp.join(sequence, conf.path)):
#                             frame_num = int(osp.splitext(file)[0])
#
#                             exclude_file = False
#                             for hard_sample in filtered_samples:
#                                 if hard_sample[0] in sequence and hard_sample[1] in file:
#                                     exclude_file = True
#
#                             if not exclude_file:
#                                 files[key][frame_num] = osp.join(conf.path, file)
#                                 frame_nums.add(frame_num)
#
#                     elif isinstance(conf, list):
#                         files[key] = []
#
#                         for conf_idx, conf in enumerate(conf):
#
#                             files[key].append({})
#
#                             for file in os.listdir(osp.join(sequence, conf.path)):
#                                 frame_num = int(osp.splitext(file)[0])
#
#                                 exclude_file = False
#                                 for hard_sample in filtered_samples:
#                                     if hard_sample[0] in sequence and hard_sample[1] in file:
#                                         exclude_file = True
#
#                                 if not exclude_file:
#                                     files[key][conf_idx][frame_num] = osp.join(conf.path, file)
#                                     frame_nums.add(frame_num)
#
#                 for frame_num in frame_nums:
#
#                     sample = FlyingThings3DSample(base=osp.relpath(sequence, self.root),
#                                                 name=osp.relpath(sequence, self.root) + "/key{:02d}".format(frame_num))
#
#                     sample_valid = True
#                     for key, conf in sample_conf.items():
#
#                         if not (isinstance(conf, DataConf) or isinstance(conf, list)):
#                             sample.info[key] = conf
#                             continue
#
#                         if isinstance(conf, DataConf):
#
#                             offset_num = frame_num + conf.offset
#                             if offset_num in files[key]:
#                                 sample.data[key] = files[key][offset_num]
#                             else:
#                                 sample_valid = False
#                                 break
#
#                         elif isinstance(conf, list):
#                             sample.data[key] = []
#
#                             for conf_idx, conf in enumerate(conf):
#
#                                 offset_num = frame_num + conf.offset
#                                 if offset_num in files[key][conf_idx]:
#                                     sample.data[key].append(files[key][conf_idx][offset_num])
#                                 else:
#                                     sample_valid = False
#                                     break
#
#                     if sample_valid:
#                         self.samples.append(sample)
#
#     def _filter_samples(self, sample_filter_fct):
#         filtered_samples = []
#
#         for idx, sample in enumerate(self.samples):
#             if idx % 1000 == 0:
#                 print("\tApplying filter to sample {} of {}.".format(idx, len(self.samples)))
#
#             if sample_filter_fct(self[idx]):
#                 filtered_samples.append(sample)
#             else:
#                 print("\tRemoved sample at index {} with name {}.".format(idx, sample['name']))
#
#         self.samples = filtered_samples
#
#     def _init_samples_from_list(self, sample_list_path):
#         if self.verbose:
#             print("\tInitializing samples from list at {}.".format(sample_list_path))
#         with open(sample_list_path, 'rb') as sample_list:
#             self.samples += pickle.load(sample_list)
#
#     def write_samples(self, path=None):
#         path = self._get_sample_list_path() if path is None else path
#         super().write_samples(path)
#
#
# def get_subset_filtered_frames():
#     subset_filtered_frames_path = osp.join(_get_path("FlyingThings3D", "orig", "train", "root"), "subset_filtered_frames.txt")
#
#     with open(subset_filtered_frames_path, 'r') as subset_filtered_frames_file:
#                 subset_filtered_frames = subset_filtered_frames_file.readlines()
#                 subset_filtered_frames = [x[:-1] for x in subset_filtered_frames]
#                 subset_filtered_frames = [x.split(':') for x in subset_filtered_frames]
#
#     return subset_filtered_frames
