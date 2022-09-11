from .registry import get_dataset
from .compound_dataset import CompoundDataset


def create_dataset(dataset_name, dataset_type=None, split=None, **kwargs):
    """Create a dataset.

    Args:
        dataset_name (str): The name of the dataset to create. Can optionally contain the dataset_type and split in the
            format base_dataset_name.split.dataset_type.
        dataset_type (str): The type of the dataset to create. Can optionally be provided within the dataset_name.
        split (str): The split of the dataset to create. Can optionally be provided within the dataset_name.

    Keyword Args:
        **kwargs: Arguments for the dataset.
    """
    dataset_cls = get_dataset(dataset_name=dataset_name, dataset_type=dataset_type, split=split)
    dataset = dataset_cls(**kwargs)
    return dataset


def create_dataloader(dataset_name, dataset_type=None, split=None, batch_size=1, shuffle=False, num_workers=0,
                      collate_fn=None, pin_memory=False, drop_last=False, worker_init_fn=None, indices=None, **kwargs):
    """Create a dataloader.

    Args:
        dataset_name (str): The name of the dataset to create. Can optionally contain the dataset_type and split in the
            format base_dataset_name.split.dataset_type.
        dataset_type (str): The type of the dataset to create. Can optionally be provided within the dataset_name.
        split (str): The split of the dataset to create. Can optionally be provided within the dataset_name.

    Keyword Args:
        **kwargs: Arguments for the dataset.
    """
    dataset_cls = get_dataset(dataset_name=dataset_name, dataset_type=dataset_type, split=split)
    dataloader = dataset_cls.init_as_loader(batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,
                                            num_workers=num_workers, collate_fn=collate_fn, drop_last=drop_last,
                                            worker_init_fn=worker_init_fn, indices=indices, **kwargs)
    return dataloader


def create_compound_dataset(datasets, common_keys=None):
    """Create a compound dataset.

    Args:
        datasets (list): List of datasets to be combined.
        common_keys (list): List of keys that need to be present in all datasets and will be used to filter the data
            in each sample.
    """
    compound_dataset = CompoundDataset(datasets=datasets, common_keys=common_keys)
    return compound_dataset


def create_compound_dataloader(datasets, common_keys=None, batch_size=1, shuffle=False, num_workers=0, collate_fn=None,
                               pin_memory=False, drop_last=False, worker_init_fn=None, indices=None):
    dataloader = CompoundDataset.init_as_loader(datasets=datasets, common_keys=common_keys, batch_size=batch_size,
                                                shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers,
                                                collate_fn=collate_fn, drop_last=drop_last,
                                                worker_init_fn=worker_init_fn, indices=indices)
    return dataloader


# TODO: create dataset from cfg file
