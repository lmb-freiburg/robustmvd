import os.path as osp
import pickle
from glob import glob

from .registry import _get_dataset, has_dataset, _get_folder_parser, list_datasets, has_folder_parser, list_folder_parsers
from .registry import _get_augmentation, has_augmentation, list_augmentations, \
    _get_batch_augmentation, has_batch_augmentation, list_batch_augmentations
from .compound_dataset import CompoundDataset
import rmvd.utils as utils
from rmvd.utils import logging


def create_dataset(dataset_name_or_path, data_format=None, split=None, folder_parser_name=None, **kwargs):
    """Create a dataset by its name or from a dataset config file.

    Args:
        dataset_name_or_path (str): The name of the dataset to create or the path to a folder (or a dataset config file).
            If it is the name of a dataset, it can optionally contain the data_format and split in the format
            base_dataset_name.split.data_format.
        data_format (str): Data format of the dataset to create.
            Can optionally be provided within the dataset_name_or_path.   
            Ignored when dataset_name_or_path is a path to a dataset config file.
        split (str): The split of the dataset to create. Only used when dataset_name_or_path is a dataset name.
            Can optionally be provided within the dataset_name_or_path.
        folder_parser_name (str): The name of the parser to parse the folder at the specified path into a dataset.
            Only used when dataset_name_or_path is a path to a folder.
            The parser name consists of the folder structure name. It can optionally contain the data_format 
            in the format folder_structure.data_format, which is required if the data_format is not provided separately.

    Keyword Args:
        **kwargs: Arguments for the dataset.
    """
    if has_dataset(dataset_name=dataset_name_or_path, data_format=data_format, split=split):
        if folder_parser_name is not None:
            logging.warning(f"Warning: folder parser {folder_parser_name} was ignored when creating the dataset {dataset_name_or_path}.")
        return create_dataset_from_registry(dataset_name=dataset_name_or_path, data_format=data_format, split=split, **kwargs)

    elif not osp.exists(dataset_name_or_path):
        assert False, \
            f'The requested dataset "{dataset_name_or_path}" (data_format={data_format}, split={split}) does not exist ' \
            f'and "{dataset_name_or_path}" is not an existing path. ' \
            f'Available datasets are: {" ".join(list_datasets())}'

    else:   
        dataset_path = dataset_name_or_path  # it is not a dataset name but a dataset path

        if has_dataset_cfg(dataset_path):
            ignored_kwargs = [name for name, value in kwargs.items() if value is not None]
            if len(ignored_kwargs) > 0:
                logging.warning(f"Warning: arguments {', '.join(ignored_kwargs)} were ignored when creating the dataset {dataset_path}.")
            if data_format is not None:
                logging.warning(f"Warning: data_format {data_format} was ignored when creating the dataset {dataset_path}.")
            if split is not None:
                logging.warning(f"Warning: split {split} was ignored when creating the dataset {dataset_path}.")
            if folder_parser_name is not None:
                logging.warning(f"Warning: folder parser {folder_parser_name} was ignored when creating the dataset {dataset_path}.")
            return create_dataset_from_cfg(path=dataset_path)

        else:
            assert osp.isdir(dataset_path), \
                f'The requested dataset path "{dataset_path}" is a file instead of a directory. ' \
                f'To create a dataset from a config file, the file must be named "dataset.cfg". ' \
                f'To create a dataset from a folder, the path must point to an existing directory.'
            assert has_folder_parser(folder_parser_name=folder_parser_name, data_format=data_format), \
                f'The requested folder parser "{folder_parser_name}" (data_format={data_format}) does not exist. ' \
                f'Available folder parsers are: {" ".join(list_folder_parsers())}'
            if split is not None:
                logging.warning(f"Warning: split {split} was ignored when creating the dataset {dataset_path}.")
            return create_dataset_from_folder(path=dataset_path, folder_parser_name=folder_parser_name, data_format=data_format, **kwargs)


def create_dataset_from_registry(dataset_name, data_format=None, split=None, **kwargs):
    assert has_dataset(dataset_name=dataset_name, data_format=data_format, split=split), \
        f'The requested dataset "{dataset_name}" (data_format={data_format}, split={split}) does not exist. ' \
        f'Available datasets are: {" ".join(list_datasets())}'
    dataset_cls = _get_dataset(dataset_name=dataset_name, data_format=data_format, split=split)
    dataset = dataset_cls(**kwargs)
    return dataset


def has_dataset_cfg(path):
    if osp.basename(path) == 'dataset.cfg':
        return osp.isfile(path)
    return osp.isdir(path) and len(glob(f'{path}/**/dataset.cfg', recursive=True)) > 0


def create_dataset_from_cfg(path, more_updates=None, more_layouts=None, verbose=None):
    if not osp.split(path)[1] == 'dataset.cfg':
        paths = glob(f'{path}/**/dataset.cfg', recursive=True)
        assert len(paths) > 0, f"No dataset.cfg file found in {path} or its subdirectories."
        path = paths[0]
    else:
        assert osp.isfile(path), f'The requested dataset config file "{path}" is not an existing file.'
    with open(path, 'rb') as file:
        config = pickle.load(file)

    if more_updates is not None:
        more_updates = [more_updates] if not isinstance(more_updates, list) else more_updates
        updates = config['updates'] if 'updates' in config else []
        updates += more_updates
        config['updates'] = updates

    if more_layouts is not None:
        more_layouts = [more_layouts] if not isinstance(more_layouts, list) else more_layouts
        layouts = config['layouts'] if 'layouts' in config else []
        layouts += more_layouts
        config['layouts'] = layouts

    if verbose is not None:
        config['verbose'] = verbose

    dataset_cls_name = config['dataset_cls_name']
    del config['dataset_cls_name']
    dataset_cls = utils.get_class(dataset_cls_name)
    return dataset_cls(**config)


def create_dataset_from_folder(path, folder_parser_name, data_format=None, **kwargs):
    assert has_folder_parser(folder_parser_name=folder_parser_name, data_format=data_format), \
        f'The requested folder parser "{folder_parser_name}" (data_format={data_format}) does not exist. ' \
        f'Available folder parsers are: {" ".join(list_folder_parsers())}'
    assert osp.isdir(path), f'The requested path "{path}" is not an existing directory.'
    parser_cls = _get_folder_parser(folder_parser_name=folder_parser_name, data_format=data_format)
    dataset = parser_cls(root=path, **kwargs)
    return dataset


def create_compound_dataset(datasets, common_keys=None):
    """Create a compound dataset.

    Args:
        datasets (list): List of datasets to be combined.
        common_keys (list): List of keys that need to be present in all datasets and will be used to filter the data
            in each sample.
    """
    compound_dataset = CompoundDataset(datasets=datasets, common_keys=common_keys)
    return compound_dataset


def create_augmentation(augmentation_name, **kwargs):
    assert has_augmentation(augmentation_name), f'The requested augmentation function "{augmentation_name}" does not exist. Available augmentation functions are: {" ".join(list_augmentations())}'
    augmentation_fct = _get_augmentation(augmentation_name)
    augmentation = augmentation_fct(**kwargs)
    return augmentation


def create_batch_augmentation(batch_augmentation_name, **kwargs):
    assert has_batch_augmentation(batch_augmentation_name), f'The requested batch augmentation function "{batch_augmentation_name}" does not exist. Available batch augmentation functions are: {" ".join(list_batch_augmentations())}'
    batch_augmentation_fct = _get_batch_augmentation(batch_augmentation_name)
    batch_augmentation = batch_augmentation_fct(**kwargs)
    return batch_augmentation
