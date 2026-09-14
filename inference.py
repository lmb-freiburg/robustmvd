#!/usr/bin/env python3

import argparse
import sys
import os.path as osp
import os

import torch

from rmvd import create_model, list_models, create_dataset, list_datasets, create_inference, list_inference_types, list_folder_parsers, has_folder_parser, has_dataset
from rmvd.data import has_dataset_cfg, list_resize_backends
from rmvd.utils import set_random_seed, logging


def get_dataset_arg(args):
    # The dataset source group is mutually exclusive and required, so exactly one of these is not None.
    if args.dataset is not None:
        return args.dataset, None
    if args.folder is not None:
        return args.folder[0], args.folder[1]
    return args.dataset_cfg, None


def validate_args(args, parser):
    # The factory functions also validate the args, but the goal is to do it before the inference() function starts and creates folders etc.
    # Some checks are already done at parser level. The ones that can not be done at parser level, are done here.
    dataset_name_or_path, folder_parser_name = get_dataset_arg(args)

    if args.dataset_cfg is not None and not has_dataset_cfg(args.dataset_cfg):
        parser.error(f"No dataset.cfg file found in dataset config path: {args.dataset_cfg}")

    if folder_parser_name is not None and not osp.isdir(dataset_name_or_path):
        parser.error(f"Folder path does not exist or is not a directory: {dataset_name_or_path}")

    if folder_parser_name is not None and not has_folder_parser(folder_parser_name, data_format=args.inference_type):
        parser.error(f"Unknown folder parser for the {args.inference_type} inference type: {folder_parser_name}. "
                     f"Available folder parsers are: "
                     f"{' '.join(list_folder_parsers(data_format=args.inference_type))}")


@torch.no_grad()
def inference(args):
    
    set_random_seed(args.seed)

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    log_file_path = osp.join(out_dir, "log.txt")
    logging.add_log_file(log_file_path, flush_line=True)

    with open(osp.join(out_dir, "cmd.txt"), 'a') as f:
        f.write("python " + " ".join(sys.argv) + "\n")

    dataset_name_or_path, folder_parser_name = get_dataset_arg(args)
    logging.info()
    logging.info(f"Running inference with the model {args.model} on the dataset {dataset_name_or_path} in the {args.inference_type} inference type.\n")
    
    dataset = create_dataset(dataset_name_or_path=dataset_name_or_path, data_format=args.inference_type,
                             folder_parser_name=folder_parser_name, input_size=args.input_size,
                             input_resize_backend=args.input_resize_backend)
    model = create_model(name=args.model, weights=args.weights, train=False, num_gpus=(0 if args.cpu else 1))

    if args.disable_clipping:
        clip_pred = False
    elif args.clipping is not None:
        clip_pred = args.clipping
    else:
        clip_pred = True

    inference_runner = create_inference(inference_type=args.inference_type,
                                        inputs=args.inputs,
                                        max_source_views=args.max_source_views,
                                        clip_pred_depth=clip_pred,
                                        alignment=args.alignment,
                                        out_dir=out_dir,
                                        num_workers=args.num_workers,
                                        verbose=True,)

    samples = args.num_samples if args.num_samples is not None else args.samples

    inference_runner.run_on_dataset(dataset=dataset, model=model, samples=samples)
    
    logging.remove_log_file(log_file_path)


if __name__ == '__main__':
    # Parse inference type first:
    inference_type_parser = argparse.ArgumentParser(add_help=False)
    inference_type_parser.add_argument('--inference_type', choices=list_inference_types())
    selected_args, _ = inference_type_parser.parse_known_args()
    inference_type = selected_args.inference_type

    if inference_type is None:
        inference_type_parser = argparse.ArgumentParser(description='Select an inference type, then run "python inference.py --inference_type INFERENCE_TYPE --help" for its full help.')
        inference_type_parser.add_argument('--inference_type', choices=list_inference_types(), metavar='INFERENCE_TYPE', required=True,
                                           help=f"Inference setting. Options are: {' '.join(list_inference_types())}")
        if '-h' in sys.argv or '--help' in sys.argv:
            inference_type_parser.print_help()
            sys.exit()
        inference_type_parser.error(f"the following arguments are required: --inference_type. Options are: {' '.join(list_inference_types())}")

    # Parse other arguments second:
    available_datasets = list_datasets(data_format=inference_type)
    available_dataset_choices = sorted(set(
        available_datasets
        + list_datasets(data_format=inference_type, no_data_format=True)
        + [n for n in list_datasets(data_format=inference_type, no_split=True, no_data_format=True)
           if has_dataset(n, data_format=inference_type)]
    ))
    available_folder_parsers = list_folder_parsers(data_format=inference_type)

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help="Path to folder for output data.", required=True)
    parser.add_argument('--cpu', action='store_true', help="Run inference on CPU. Default: run on a single GPU. To select a specific GPU, set the CUDA_VISIBLE_DEVICES environment variable.")
    parser.add_argument('--seed', type=int, help="Random seed. Default: 42.", default=42)
    parser.add_argument('--num_workers', type=int, help="Number of workers for data loading. Default: 4.", default=4)
    
    parser.add_argument('--inference_type', help=f"Inference setting. Options are: {' '.join(list_inference_types())}",
                        choices=list_inference_types(), metavar='INFERENCE_TYPE', required=True)
    
    parser.add_argument('--model', help=f"Model for inference. Available models are: {' '.join(list_models())}",
                        choices=list_models(), metavar='MODEL', required=True)
    parser.add_argument('--weights', help="Path to weights of the model. Optional. If None, default weights are used.")
    parser.add_argument('--inputs', nargs='*', choices=['intrinsics', 'poses', 'depth_range'], metavar='INPUT',
                        help=f"Model inputs. Images are always provided to the model. "
                             f"It is possible to specify multiple additional inputs, "
                             f"e.g. --inputs intrinsics poses. "
                             f"Options for additional model inputs are: intrinsics, poses, depth_range.",
                        type=str)

    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument('--dataset', help=f"Registered dataset. Available datasets are: {' '.join(available_datasets)}",
                               choices=available_dataset_choices, metavar='DATASET')
    dataset_group.add_argument('--folder', nargs=2, metavar=('PATH', 'FOLDER_PARSER'),
                               help=f"Folder dataset specified as a path and folder parser. Available folder parsers are: "
                                    f"{' '.join(available_folder_parsers)}")
    dataset_group.add_argument('--dataset_cfg', metavar='PATH',
                               help="Dataset config file or a directory containing a dataset.cfg file.")
    parser.add_argument('--input_size', type=int, nargs=2, help="Input image size in the format (height, width). "
                                                                "If not provided, scales images up to the nearest size "
                                                                "that works with the model.")
    parser.add_argument('--input_resize_backend', choices=list_resize_backends(), metavar='BACKEND',
                        default=None,
                        help=f"Backend that is used to resize the input images to the specified input size. "
                             f"Only used if an input size is specified. "
                             f"Options are: {' '.join(list_resize_backends())}. If not specified, the default resizing backend is used.")

    parser.add_argument('--max_source_views', type=int, help='Maximum number of source views to feed to the model. Default: use all available source views.')

    samples_group = parser.add_mutually_exclusive_group()
    samples_group.add_argument('--num_samples', type=int,
                               help='Number of samples to run inference on. Mutually exclusive with --samples. Default: run inference on all samples.')
    samples_group.add_argument('--samples', type=int, nargs='*',
                               help='Indices of samples to run inference on. Mutually exclusive with --num_samples.')

    parser.add_argument('--clipping', type=float, nargs=2,
                        help="Clipping range for depth values in the format (min_depth, max_depth) in meters. "
                             "If omitted, the inference default is used unless --disable_clipping is set.")
    parser.add_argument('--disable_clipping', action='store_true',
                        help="Do not clip predicted depth values. Takes precedence over --clipping if both are provided.")
    
    parser.add_argument('--alignment', choices=['median', 'pose_scale_mean', 'pose_scale_median', 'scale_shift_least_squares_invdepth'],
                        help="Alignment between predicted and ground truth depths. Default: no alignment")

    args = parser.parse_args()
    validate_args(args, parser)
    inference(args)
