import os
import os.path as osp
import random
import abc
import pickle
import time

import torch
import numpy as np

import rmvd.utils as utils
from rmvd.utils import logging
from .transforms import ResizeInputs, ResizeTargets
from .updates import Updates
from .layout import Layout
from .factory import create_augmentation


class Sample(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def load(self, root):
        return


class Dataset(torch.utils.data.Dataset, metaclass=abc.ABCMeta):

    def __init__(self, root, augmentations=None, input_size=None, input_resize_backend=None,
                 target_size=None, to_torch=False, updates=None, update_strict=False,
                 layouts=None, verbose=True, **kwargs):

        augmentations = [] if augmentations is None else augmentations
        augmentations = [augmentations] if not isinstance(augmentations, list) else augmentations
        self.verbose = verbose

        self.root = None
        self._init_root(root)

        if self.verbose:
            logging.info(f"Initializing dataset {self.name} from {self.root}")

        self.input_resize = ResizeInputs(size=input_size, backend=input_resize_backend) if input_size is not None else None
        self.target_resize = ResizeTargets(size=target_size) if target_size is not None else None
        self.augmentations = []
        self._augmentation_names = []  # name of each augmentation that was created from a name; None for the others
        self._init_augmentations(augmentations)
        self.to_torch = to_torch

        self._layouts = {}
        self._init_layouts(layouts)
        self.samples = []
        self._init_samples(**kwargs)  # passing kwargs to init_samples is used for subclasses that override _init_samples
        self._allowed_indices = []
        self.updates = []
        self._update_strict = update_strict
        self._init_updates(updates)

        if self.verbose:
            logging.info(f"\tNumber of samples: {len(self)}")
            if len(self.updates) > 0:
                logging.info(f"\tUpdates: {', '.join([update.name for update in self.updates])}")
            if self.input_resize is not None:
                logging.info(f"\tImage resolution (height, width): ({input_size[0]}, {input_size[1]})")
                logging.info(f"\tImage resize backend: {self.input_resize.backend}")
            if self.target_resize is not None:
                logging.info(f"\tTarget resolution (height, width): ({target_size[0]}, {target_size[1]})")
            logging.info(f"Finished initializing dataset {self.name}")
            logging.info()

    @property
    def name(self):
        if hasattr(self, "base_dataset"):
            name = self.base_dataset
            name = f"{name}.{self.split}" if hasattr(self, "split") else name
            name = f"{name}.{self.data_format}" if hasattr(self, "data_format") else name
        else:
            name = type(self).__name__
        return name

    @property
    def full_name(self):
        name = self.name
        for update in self.updates:
            name += f"+{update.name}"
        return name

    def _init_root(self, root):
        if isinstance(root, list):
            root = [path for path in root if osp.isdir(path)][0]
        self.root = os.fspath(root) if root is not None else None

    def _init_augmentations(self, augmentations):
        for augmentation in augmentations:
            if isinstance(augmentation, str):
                self._augmentation_names.append(augmentation)
                augmentation = create_augmentation(augmentation)
            else:
                self._augmentation_names.append(None)
            self.augmentations.append(augmentation)

    def _init_samples(self):
        self._init_samples_from_list()
        
    def _init_samples_from_list(self):
        sample_list_path = _get_sample_list_path(self.name)
        if self.verbose:
            logging.info("\tInitializing samples from list at {}".format(sample_list_path))
        with open(sample_list_path, 'rb') as sample_list:
            self.samples = pickle.load(sample_list)

    def _write_samples_list(self, path=None):
        path = _get_sample_list_path(self.name) if path is None else path
        if osp.isdir(osp.split(path)[0]):
            if self.verbose:
                logging.info(f"Writing sample list to {path}")
            with open(path, 'wb') as file:
                pickle.dump(self.samples, file)
        elif self.verbose:
            logging.info(f"Could not write sample list to {path}")

    def _init_updates(self, updates):

        if updates is not None:
            for update in updates:
                if isinstance(update, Updates):
                    update = update
                elif isinstance(update, str):
                    update = Updates.from_file(path=update, verbose=False)
                self.updates.append(update)

        if self._update_strict:
            self._allowed_indices = [i for i in range(len(self.samples)) if all([i in update for update in self.updates])]
        else:
            self._allowed_indices = list(range(len(self.samples)))

    def _init_layouts(self, layouts):
        if layouts is not None:
            for layout in layouts:
                layout = layout if isinstance(layout, Layout) else Layout.from_file(layout)
                self.add_layout(layout)

    def add_layout(self, layout):
        self._layouts[layout.name.lower()] = layout

    def get_layout_names(self):
        return list(self._layouts.keys())

    def get_layout(self, layout_name=None):
        layout_name = layout_name if layout_name is not None else 'default'
        return self._layouts[layout_name.lower()]

    def __len__(self):
        return len(self._allowed_indices)

    def __getitem__(self, index):

        index = self._allowed_indices[index]
        sample = self.samples[index]

        sample_dict = sample.load(root=self.root)
        sample_dict['_index'] = index
        sample_dict['_dataset'] = self.full_name

        _preprocess_sample(sample_dict)

        for update in self.updates:
            update.apply_update(sample_dict, index=index)

        for augmentation in self.augmentations:
            augmentation(sample_dict)

        if self.input_resize is not None:
            self.input_resize(sample_dict)
        if self.target_resize is not None:
            self.target_resize(sample_dict)

        if self.to_torch:
            sample_dict = utils.torch_collate([sample_dict])

        return sample_dict

    def __str__(self):
        return self.name

    def _get_paths_file(self):
        local_paths_file = osp.join(osp.dirname(osp.realpath(__file__)), 'paths.toml')
        home_paths_file = osp.join(osp.expanduser('~'), 'rmvd_data_paths.toml')
        return utils.get_paths_file(local_paths_file, home_paths_file, "rmvd/data/README.md")

    def _get_path(self, *keys):
        paths_file = self._get_paths_file()
        return utils.get_path(paths_file, *keys)

    def _get_root_from_paths_file(self, *keys):
        paths_file = self._get_paths_file()
        keys = (*keys, "root")
        keys_string = ".".join(keys)
        dataset_keys_string = ".".join(keys[:-1])

        root = utils.get_path(paths_file, *keys)

        if root is None:
            raise ValueError(f"No root path for the dataset {dataset_keys_string} in the paths file {paths_file}. Please add a "
                            f"[{dataset_keys_string}] section with a root entry, as specified in the rmvd/data/README.md file.")

        if isinstance(root, list):
            existing_roots = [path for path in root if osp.isdir(path)]
            if not existing_roots:
                raise NotADirectoryError(f"None of the root paths of the dataset {dataset_keys_string} is an existing directory. The root paths are: " \
                                         f"{' '.join([str(path) for path in root])}")
            root = existing_roots[0]

        if not osp.isdir(root):
            raise NotADirectoryError(f"The root path {root} of the dataset {dataset_keys_string} from the paths file "
                                    f"{paths_file} is not an existing directory. Please check the paths file and the "
                                    f"setup instructions in the rmvd/data/README.md file.")

        return root

    @classmethod
    def init_as_loader(cls, batch_size=1, shuffle=False, num_workers=0, collate_fn=None, pin_memory=False,
                       drop_last=False, worker_init_fn=None, indices=None, **kwargs):
        kwargs.pop("to_torch", None)  # create dataset with to_torch=False to avoid adding batch dimension twice
        dataset = cls(**kwargs)
        dataset = torch.utils.data.Subset(dataset, indices) if indices is not None else dataset
        return dataset.get_loader(batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,
                                  num_workers=num_workers, collate_fn=collate_fn,
                                  drop_last=drop_last, worker_init_fn=worker_init_fn)

    def get_loader(self, batch_size=1, shuffle=False, num_workers=0, collate_fn=None, pin_memory=False,
                   drop_last=False, worker_init_fn=None, indices=None):
        dataset = torch.utils.data.Subset(self, indices) if indices is not None else self
        dataset.to_torch = False  # create dataset with to_torch=False to avoid adding batch dimension twice
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,
                                                 num_workers=num_workers, collate_fn=collate_fn, drop_last=drop_last,
                                                 worker_init_fn=worker_init_fn)
        return dataloader

    def timeit(self, num_batches=100, batch_size=1, num_workers=0):
        start = time.time()

        loader = self.get_loader(batch_size=batch_size, num_workers=num_workers)
        for idx, data_blob in enumerate(loader):
            if idx >= num_batches-1:
                break

        end = time.time()
        logging.info("Total time for loading {} batches: %1.4fs.".format(num_batches) % (end - start))
        logging.info("Mean time per batch: %1.4fs." % ((end - start)/num_batches))

    def write_config(self, path, extra_updates=None, extra_layouts=None, **overrides):
        """Write a config file that describes this dataset and can be used to re-create it.

        The dataset can be re-created from the written config file with create_dataset_from_cfg.

        Args:
            path (str): Path of the config file that will be written.
            extra_updates (list): Paths of updates that will be appended to the updates of this dataset.
            extra_layouts (list): Paths of layouts that will be appended to the layouts of this dataset.

        Keyword Args:
            **overrides: Config entries that should deviate from the config of this dataset, e.g. input_size=None
                to write a config for the dataset without the input resize. Only entries that are part of the
                config are allowed.

        Augmentations that were not created from a name and updates and layouts that were not created from a file
        are not included in the written config. 
        """

        def abspath(p):
            return osp.abspath(osp.expanduser(os.fspath(p))) if p is not None else None

        def as_paths(paths):
            paths = [] if paths is None else paths
            paths = paths if isinstance(paths, list) else [paths]
            return [abspath(path) for path in paths]

        input_size = self.input_resize.size if self.input_resize is not None else None
        target_size = self.target_resize.size if self.target_resize is not None else None
        input_resize_backend = self.input_resize.backend if self.input_resize is not None else None

        augmentation_names = []
        for augmentation_name in self._augmentation_names:
            if augmentation_name is None:
                logging.warning(f"Warning: an augmentation of the dataset {self.name} was not created from a name "
                                f"and is therefore not included in the written dataset config.")
            else:
                augmentation_names.append(augmentation_name)

        update_paths = []
        for update in self.updates:
            if update.path is None:
                logging.warning(f"Warning: the update {update.name} of the dataset {self.name} was not created from "
                                f"a file and is therefore not included in the written dataset config.")
            else:
                update_paths.append(abspath(update.path))

        layout_paths = [abspath(layout.path) for layout in self._layouts.values() if layout.path is not None]

        config = {'dataset_cls_name': utils.get_full_class_name(self),
                  'root': abspath(self.root),
                  'augmentations': augmentation_names,
                  'input_size': input_size,
                  'input_resize_backend': input_resize_backend,
                  'target_size': target_size,
                  'updates': update_paths,
                  'update_strict': self._update_strict,
                  'layouts': layout_paths}
        # intentionally not writing to_torch to the config

        unknown_overrides = [key for key in overrides if key not in config]
        assert not unknown_overrides, \
            f"Unknown config entries {', '.join(sorted(unknown_overrides))} for the dataset {self.name}. " \
            f"Available config entries are: {', '.join(sorted(config.keys()))}."
        config.update(overrides)

        config['updates'] = as_paths(config['updates']) + as_paths(extra_updates)
        config['layouts'] = as_paths(config['layouts']) + as_paths(extra_layouts)

        with open(path, 'wb') as file:
            pickle.dump(config, file)


def _get_sample_list_path(name):
    sample_lists_path = osp.join(osp.dirname(osp.realpath(__file__)), 'sample_lists')
    return osp.join(sample_lists_path, "{}.pickle".format(name))


def _preprocess_sample(sample):
    assert not ('depth' in sample and 'invdepth' in sample)  # we want a unique source of depth information

    if 'depth' in sample:
        with np.errstate(divide='ignore', invalid='ignore'):
            sample['depth'] = sample['depth'].astype(np.float32)
            sample['depth'][sample['depth'] <= 0] = 0
            sample['depth'][~np.isfinite(sample['depth'])] = 0
            sample['invdepth'] = np.nan_to_num(1 / sample['depth'], copy=False, nan=0, posinf=0, neginf=0)

    elif 'invdepth' in sample:
        sample['invdepth'] = sample['invdepth'].astype(np.float32)
        sample['invdepth'][sample['invdepth'] <= 0] = 0
        sample['invdepth'][~np.isfinite(sample['invdepth'])] = 0
        sample['depth'] = np.nan_to_num(1 / sample['invdepth'], copy=False, nan=0, posinf=0, neginf=0)

    if 'depth_range' not in sample and 'depth' in sample:
        sample['depth_range'] = utils.compute_depth_range(depth=sample['depth'])

    key_idx = sample['keyview_idx'] if 'keyview_idx' in sample else 0
    
    if 'poses' in sample:
        key_to_ref_transform = sample['poses'][key_idx]
        ref_to_key_transform = utils.invert_transform(key_to_ref_transform)
        for idx, to_ref_transform in enumerate(sample['poses']):
            to_key_transform = np.dot(to_ref_transform, ref_to_key_transform)
            sample['poses'][idx] = to_key_transform
