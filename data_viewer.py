#!/usr/bin/env python3

import argparse
import os.path as osp

from rmvd import create_dataset, run_viewer, list_datasets, list_augmentations, list_folder_parsers, has_folder_parser
from rmvd.data import has_dataset_cfg


def get_dataset_arg(args):
    # The dataset source group is mutually exclusive and required, so exactly one of these is not None.
    if args.dataset is not None:
        return args.dataset, None
    if args.folder is not None:
        return args.folder[0], args.folder[1]
    return args.dataset_cfg, None


def validate_args(args, parser):
    # The factory functions also validate the args, but the goal is to do it before the data_viewer() function starts.
    # Some checks are already done at parser level. The ones that can not be done at parser level, are done here.
    dataset_name_or_path, folder_parser_name = get_dataset_arg(args)

    if args.dataset_cfg is not None and not has_dataset_cfg(args.dataset_cfg):
        parser.error(f"No dataset.cfg file found in dataset config path: {args.dataset_cfg}")

    if folder_parser_name is not None and not osp.isdir(dataset_name_or_path):
        parser.error(f"Folder path does not exist or is not a directory: {dataset_name_or_path}")

    if folder_parser_name is not None and not has_folder_parser(folder_parser_name):
        parser.error(f"Unknown folder parser: {folder_parser_name}. Available folder parsers are: "
                     f"{' '.join(list_folder_parsers())}")


def data_viewer(args):
    dataset_name_or_path, folder_parser_name = get_dataset_arg(args)
    dataset = create_dataset(dataset_name_or_path=dataset_name_or_path, data_format=None,
                             augmentations=args.augmentations, folder_parser_name=folder_parser_name)

    if args.layout is None:
        layout_names = dataset.get_layout_names()
        if any([layout.startswith("eval") for layout in layout_names]):
            layout = [layout for layout in layout_names if layout.startswith("eval")][0]
        else:
            layout = "default"
    else:
        layout = args.layout

    run_viewer(dataset, layout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument('--dataset', help=f"Registered dataset. Available datasets are: {' '.join(list_datasets())}",
                               choices=list_datasets(), metavar='DATASET')
    dataset_group.add_argument('--folder', nargs=2, metavar=('PATH', 'FOLDER_PARSER'),
                               help=f"Folder dataset specified as a path and folder parser. Available folder parsers are: "
                                    f"{' '.join(list_folder_parsers())}")
    dataset_group.add_argument('--dataset_cfg', metavar='PATH',
                               help="Dataset config file or a directory containing a dataset.cfg file.")

    parser.add_argument('--layout', help="Data viewer layout. If not specified, an evaluation layout of the dataset is used, if the dataset has one, and the default layout otherwise.")
    parser.add_argument('--augmentations', nargs='*', choices=list_augmentations(), metavar='AUGMENTATION',
                        help=f"Data augmentations. Options are: {' '.join(list_augmentations())}")
    args = parser.parse_args()
    validate_args(args, parser)
    data_viewer(args)
