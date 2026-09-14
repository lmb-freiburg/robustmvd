"""Dataset Registry.

Maintains a registry of datasets.
Based on the model registry from the timm package ( https://github.com/rwightman/pytorch-image-models ).
"""


_datasets = {}  # keys are (base_dataset, data_format, split)
_default_splits = {}
_folder_parsers = {}  # keys are (folder_structure, data_format)
_aug_fcts = {}
_batch_aug_fcts = {}


def register_dataset(dataset_cls):
    """Register a dataset."""
    base_dataset = dataset_cls.base_dataset.lower()
    data_format = dataset_cls.data_format.lower()
    split = dataset_cls.split.lower()
    assert (base_dataset, data_format, split) not in _datasets, \
        f"Dataset {str((base_dataset, data_format, split))} is already registered."
    _datasets[(base_dataset, data_format, split)] = dataset_cls
    return dataset_cls


def register_default_dataset(dataset_cls):
    """Register the default split of a dataset."""
    register_dataset(dataset_cls)

    base_dataset = dataset_cls.base_dataset.lower()
    data_format = dataset_cls.data_format.lower()
    split = dataset_cls.split.lower()
    assert (base_dataset, data_format) not in _default_splits, \
        f"Dataset {str((base_dataset, data_format))} has already a default split."
    _default_splits[(base_dataset, data_format)] = split
    return dataset_cls


def list_datasets(dataset_name=None, data_format=None, split=None, no_data_format=False, no_split=False):
    """Get list of available datasets."""
    base_dataset = None
    if dataset_name is not None:
        base_dataset, data_format, split = _split_dataset_name(dataset_name=dataset_name, data_format=data_format, split=split, resolve_default_split=False)
    datasets = _filter_datasets(base_dataset=base_dataset, data_format=data_format, split=split)
    datasets = [_build_dataset_name(*k, no_data_format=no_data_format, no_split=no_split) for k in datasets]
    return list(sorted(datasets))


def _list_base_datasets(data_format=None, split=None):
    """Get list of all base datasets."""
    datasets = _filter_datasets(data_format=data_format, split=split)
    base_datasets = sorted(list(set([k[0] for k in datasets])))
    return base_datasets


def list_splits(base_dataset=None, data_format=None):
    """Get list of all dataset splits."""
    datasets = _filter_datasets(base_dataset=base_dataset, data_format=data_format)
    splits = sorted(list(set([k[2] for k in datasets])))
    return splits


def list_data_formats(base_dataset=None, split=None, folder_structure=None):
    """Get list of all data formats."""
    assert not (folder_structure is not None and (base_dataset is not None or split is not None))

    if base_dataset is not None or split is not None:
        keys = _filter_datasets(base_dataset=base_dataset, split=split)
    elif folder_structure is not None:
        keys = _filter_folder_parsers(folder_structure=folder_structure)
    else:
        keys = _datasets.keys() | _folder_parsers.keys()

    data_formats = sorted(list(set([k[1] for k in keys])))
    return data_formats


def _filter_datasets(base_dataset=None, data_format=None, split=None):
    """Get filtered list of available datasets."""
    base_dataset = base_dataset.lower() if base_dataset is not None else None
    data_format = data_format.lower() if data_format is not None else None
    split = split.lower() if split is not None else None

    datasets = _datasets.keys()
    datasets = [k for k in datasets if base_dataset is None or k[0] == base_dataset]
    datasets = [k for k in datasets if data_format is None or k[1] == data_format]
    datasets = [k for k in datasets if split is None or k[2] == split]

    return datasets


def _split_dataset_name(dataset_name=None, data_format=None, split=None, resolve_default_split=True):
    dataset_name = str(dataset_name).lower() if dataset_name else None
    data_format = data_format.lower() if data_format else None
    split = split.lower() if split else None

    base_dataset = None
    if dataset_name is not None:
        s = dataset_name.split(".")
        assert len(s) <= 3 and all(s), f"{dataset_name} is not a valid dataset name."

        if len(s) == 3 or s[-1] in list_data_formats():
            if data_format is not None:
                assert s[-1] == data_format, "The given dataset name conflicts with the given data format."
            else:
                data_format = s[-1]
            s = s[:-1]

        if len(s) == 2:
            encoded_split = s[1]
            if split is not None:
                assert split == encoded_split, \
                    "The given dataset name conflicts with the given split."
            split = encoded_split
            s.pop()

        base_dataset = s[0] if s else None

    if resolve_default_split and split is None and base_dataset is not None and data_format is not None and (base_dataset, data_format) in _default_splits:
        split = _default_splits[(base_dataset, data_format)]

    return base_dataset, data_format, split


def _build_dataset_name(dataset_name=None, data_format=None, split=None, no_data_format=False, no_split=False, resolve_default_split=True):
    base_dataset, data_format, split = _split_dataset_name(dataset_name=dataset_name,
                                                           data_format=data_format,
                                                           split=split,
                                                           resolve_default_split=resolve_default_split)

    s = [base_dataset] if base_dataset is not None else []
    s = s + [split] if (split is not None and not no_split) else s
    s = s + [data_format] if (data_format is not None and not no_data_format) else s
    dataset_name = '.'.join(s)
    return dataset_name


def has_dataset(dataset_name, data_format=None, split=None):
    """Check if dataset is registered."""
    try:
        base_dataset, data_format, split = _split_dataset_name(dataset_name=dataset_name,
                                                                data_format=data_format,
                                                                split=split)
    except AssertionError:
        return False

    return (base_dataset, data_format, split) in _datasets


def _get_dataset(dataset_name, data_format=None, split=None):
    """Get dataset entrypoint by name."""
    base_dataset, data_format, split = _split_dataset_name(dataset_name=dataset_name,
                                                            data_format=data_format,
                                                            split=split)
    if base_dataset is None or data_format is None or split is None:
        raise ValueError("Incomplete dataset specification.")
    return _datasets[(base_dataset, data_format, split)]


def register_folder_parser(dataparser_cls):
    """Register a folder parser."""
    folder_structure = dataparser_cls.folder_structure.lower()
    data_format = dataparser_cls.data_format.lower()
    assert (folder_structure, data_format) not in _folder_parsers, \
        f"Folder parser {str((folder_structure, data_format))} is already registered."
    _folder_parsers[(folder_structure, data_format)] = dataparser_cls
    return dataparser_cls


def list_folder_parsers(folder_parser_name=None, data_format=None, no_data_format=False):
    """Get list of available folder parsers."""
    folder_structure = None
    if folder_parser_name is not None:
        folder_structure, data_format = _split_folder_parser_name(folder_parser_name=folder_parser_name, data_format=data_format)
    folder_parsers = _filter_folder_parsers(folder_structure=folder_structure, data_format=data_format)
    folder_parsers = [_build_folder_parser_name(*k, no_data_format=no_data_format) for k in folder_parsers]
    return list(sorted(folder_parsers))


def _list_folder_structures(data_format=None):
    """Get list of all folder structures."""
    folder_parsers = _filter_folder_parsers(data_format=data_format)
    folder_structures = sorted(list(set([k[0] for k in folder_parsers])))
    return folder_structures


def _filter_folder_parsers(folder_structure=None, data_format=None):
    """Get filtered list of available folder parsers."""
    folder_structure = folder_structure.lower() if folder_structure is not None else None
    data_format = data_format.lower() if data_format is not None else None

    folder_parsers = _folder_parsers.keys()
    folder_parsers = [k for k in folder_parsers if folder_structure is None or k[0] == folder_structure]
    folder_parsers = [k for k in folder_parsers if data_format is None or k[1] == data_format]

    return folder_parsers


def _split_folder_parser_name(folder_parser_name=None, data_format=None):
    folder_parser_name = str(folder_parser_name).lower() if folder_parser_name else None
    data_format = data_format.lower() if data_format else None

    folder_structure = None
    if folder_parser_name is not None:
        s = folder_parser_name.split(".")
        assert len(s) <= 2 and all(s), f"{folder_parser_name} is not a valid folder parser name."

        if len(s) == 2:
            if data_format is not None:
                assert s[-1] == data_format, "The given folder parser name conflicts with the given data format."
            else:
                data_format = s[-1]
            s = s[:-1]

        folder_structure = s[0] if s else None
    return folder_structure, data_format


def _build_folder_parser_name(folder_parser_name=None, data_format=None, no_data_format=False):
    folder_structure, data_format = _split_folder_parser_name(folder_parser_name=folder_parser_name,
                                                              data_format=data_format)

    s = [folder_structure] if folder_structure is not None else []
    s = s + [data_format] if (data_format is not None and not no_data_format) else s
    folder_parser_name = '.'.join(s)
    return folder_parser_name


def has_folder_parser(folder_parser_name, data_format=None):
    """Check if folder parser is registered."""
    try:
        folder_structure, data_format = _split_folder_parser_name(folder_parser_name=folder_parser_name, data_format=data_format)
    except AssertionError:
        return False

    return (folder_structure, data_format) in _folder_parsers


def _get_folder_parser(folder_parser_name, data_format=None):
    """Get folder parser entrypoint by name."""
    folder_structure, data_format = _split_folder_parser_name(folder_parser_name=folder_parser_name, data_format=data_format)
    
    return _folder_parsers[(folder_structure, data_format)]


def register_augmentation(augmentation_factory):
    """Register augmentation by name."""
    aug_fct_name = augmentation_factory.__name__
    assert aug_fct_name not in _aug_fcts, f'Augmentation "{aug_fct_name}" is already registered.'
    _aug_fcts[aug_fct_name] = augmentation_factory
    return augmentation_factory


def list_augmentations():
    """Get list of all augmentations."""
    return list(sorted(_aug_fcts.keys()))


def has_augmentation(augmentation_name):
    """Check if augmentation is registered."""
    return augmentation_name in _aug_fcts


def _get_augmentation(augmentation_name):
    """Get augmentation entrypoint by name."""
    return _aug_fcts[augmentation_name]


def register_batch_augmentation(batch_augmentation_factory):
    """Register batch augmentation by name."""
    batch_aug_fct_name = batch_augmentation_factory.__name__
    assert batch_aug_fct_name not in _batch_aug_fcts, f'Batch augmentation "{batch_aug_fct_name}" is already registered.'
    _batch_aug_fcts[batch_aug_fct_name] = batch_augmentation_factory
    return batch_augmentation_factory


def list_batch_augmentations():
    """Get list of all batch augmentations."""
    return list(sorted(_batch_aug_fcts.keys()))


def has_batch_augmentation(batch_augmentation_name):
    """Check if batch augmentation is registered."""
    return batch_augmentation_name in _batch_aug_fcts


def _get_batch_augmentation(batch_augmentation_name):
    """Get batch augmentation entrypoint by name."""
    return _batch_aug_fcts[batch_augmentation_name]
