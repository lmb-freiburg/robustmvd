#!/usr/bin/env python3

import argparse
import sys
import os
import os.path as osp

import torch

from rmvd import create_model, list_models, create_dataset, list_datasets, list_folder_parsers, has_folder_parser, has_dataset, create_evaluation, list_evaluations
from rmvd.data import has_dataset_cfg, list_resize_backends
from rmvd.utils import set_random_seed, writer, logging


def get_dataset_arg(args):
    # args.dataset, args.folder, and args.dataset_cfg are mutually exclusive, so only one of them can be not None.
    if getattr(args, 'dataset', None) is not None:
        return args.dataset, None
    if getattr(args, 'folder', None) is not None:
        return args.folder[0], args.folder[1]
    if getattr(args, 'dataset_cfg', None) is not None:
        return args.dataset_cfg, None
    return None


def validate_args(args, parser):
    # The factory functions also validate the args, but the goal is to do it before the eval() function starts and creates folders etc.
    # Some checks are already done at parser level. The ones that can not be done at parser level, are done here.
    dataset_arg = get_dataset_arg(args)

    if args.eval_type == "robustmvd":
        return

    elif dataset_arg is None:
        parser.error("No dataset specified. Use --dataset, --folder, or --dataset_cfg.")

    dataset_name_or_path, folder_parser_name = dataset_arg

    if args.dataset_cfg is not None and not has_dataset_cfg(args.dataset_cfg):
        parser.error(f"No dataset.cfg file found in dataset config path: {args.dataset_cfg}")

    if folder_parser_name is not None and not osp.isdir(dataset_name_or_path):
        parser.error(f"Folder path does not exist or is not a directory: {dataset_name_or_path}")

    if folder_parser_name is not None and not has_folder_parser(folder_parser_name, data_format=args.eval_type):
        parser.error(f"Unknown folder parser for the {args.eval_type} evaluation type: {folder_parser_name}. "
                     f"Available folder parsers are: "
                     f"{' '.join(list_folder_parsers(data_format=args.eval_type))}")


@torch.no_grad()
def eval(args):
    
    set_random_seed(args.seed)

    log_dir = args.log_dir if args.log_dir is not None else args.output
    tensorboard_log_dir = osp.join(log_dir, "tensorboard_logs")
    wandb_log_dir = osp.join(log_dir, "wandb_logs")
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    os.makedirs(wandb_log_dir, exist_ok=True)
    writer.setup_writers(log_tensorboard=not args.no_tensorboard, 
                         log_wandb=args.wandb, 
                         tensorboard_logs_dir=tensorboard_log_dir, 
                         wandb_logs_dir=wandb_log_dir,
                         exp_id=args.exp_id,
                         comment=args.comment,)
    log_file_path = osp.join(args.output, "log.txt")
    logging.add_log_file(log_file_path, flush_line=True)

    with open(osp.join(args.output, "cmd.txt"), 'a') as f:
        f.write("python " + " ".join(sys.argv) + "\n")

    if args.eval_type != "robustmvd":
        dataset_name_or_path, folder_parser_name = get_dataset_arg(args)
        logging.info()
        logging.info(f"Evaluating {args.model} model on dataset {dataset_name_or_path} in the {args.eval_type} evaluation setting.\n")
        dataset = create_dataset(dataset_name_or_path=dataset_name_or_path, data_format=args.eval_type,
                                 folder_parser_name=folder_parser_name, input_size=args.input_size,
                                 input_resize_backend=args.input_resize_backend)

    else:
        logging.info()
        logging.info(f"Evaluating {args.model} model on the Robust Multi-view Depth Benchmark.\n")
        dataset = None

    model = create_model(name=args.model, weights=args.weights, train=False, num_gpus=(0 if args.cpu else 1))

    if args.eval_type == "robustmvd":
        clip_pred = True  # The robustmvd benchmark is defined to always clip predicted depth values to the range (0.1m, 100m) for evaluation.
    elif args.disable_clipping:
        clip_pred = False
    elif args.clipping is not None:
        clip_pred = args.clipping
    else:
        clip_pred = True

    evaluation = create_evaluation(evaluation_type=args.eval_type,
                                   out_dir=args.output,
                                   inputs=args.inputs,
                                   alignment=args.alignment,
                                   view_ordering=args.view_ordering,
                                   min_source_views=args.min_source_views,
                                   max_source_views=args.max_source_views,
                                   eval_uncertainty=args.eval_uncertainty,
                                   clip_pred_depth=clip_pred,
                                   num_workers=args.num_workers)

    samples = args.num_samples if args.num_samples is not None else args.samples
    qualitatives = args.qualitatives if args.qualitatives is not None else args.num_qualitatives

    evaluation(dataset=dataset, model=model, samples=samples, qualitatives=qualitatives,
               eth3d_size=getattr(args, 'eth3d_size', None), kitti_size=getattr(args, 'kitti_size', None),
               dtu_size=getattr(args, 'dtu_size', None), scannet_size=getattr(args, 'scannet_size', None),
               tanks_and_temples_size=getattr(args, 'tanks_and_temples_size', None),
               input_resize_backend=args.input_resize_backend,
               eval_name=args.eval_name, finished_iterations=args.finished_iterations,)
    
    logging.remove_log_file(log_file_path)


if __name__ == '__main__':
    # Parse evaluation type first:
    evaluation_type_parser = argparse.ArgumentParser(add_help=False)
    evaluation_type_parser.add_argument('--eval_type', choices=list_evaluations())
    selected_args, _ = evaluation_type_parser.parse_known_args()
    evaluation_type = selected_args.eval_type

    if evaluation_type is None:
        evaluation_type_parser = argparse.ArgumentParser(description='Select an evaluation type, then run "python eval.py --eval_type EVAL_TYPE --help" for its full help.')
        evaluation_type_parser.add_argument('--eval_type', choices=list_evaluations(), metavar='EVAL_TYPE', required=True,
                                            help=f"Evaluation setting. Options are: {' '.join(list_evaluations())}")
        if '-h' in sys.argv or '--help' in sys.argv:
            evaluation_type_parser.print_help()
            sys.exit()
        evaluation_type_parser.error(f"the following arguments are required: --eval_type. Options are: {' '.join(list_evaluations())}")

    # Parse other arguments second:
    if evaluation_type != "robustmvd":
        available_datasets = list_datasets(data_format=evaluation_type)
        available_dataset_choices = sorted(set(
            available_datasets
            + list_datasets(data_format=evaluation_type, no_data_format=True)
            + [n for n in list_datasets(data_format=evaluation_type, no_split=True, no_data_format=True)
               if has_dataset(n, data_format=evaluation_type)]
        ))
        available_folder_parsers = list_folder_parsers(data_format=evaluation_type)

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help="Path to folder for output data.", required=True)
    parser.add_argument('--cpu', action='store_true', help="Evaluate on CPU. Default: evaluate on a single GPU. To select a specific GPU, set the CUDA_VISIBLE_DEVICES environment variable.")
    parser.add_argument('--seed', type=int, help="Random seed. Default: 42.", default=42)
    parser.add_argument('--num_workers', type=int, help="Number of workers for data loading. Default: 4.", default=4)
    
    parser.add_argument('--eval_type', help=f"Evaluation setting. Options are: {' '.join(list_evaluations())}",
                        choices=list_evaluations(), metavar='EVAL_TYPE', required=True)
    parser.add_argument('--eval_name', help="Name of the evaluation. Optional.", type=str)
    
    parser.add_argument('--model', help=f"Model for evaluation. Available models are: {' '.join(list_models())}",
                        choices=list_models(), metavar='MODEL', required=True)
    parser.add_argument('--finished_iterations', type=int, help="Number of iterations that the model was trained. Only used for logging. Optional.")
    parser.add_argument('--weights', help="Path to weights of the model. Optional. If None, default weights are used.")
    parser.add_argument('--inputs', nargs='*', choices=['intrinsics', 'poses', 'depth_range'], metavar='INPUT',
                        help=f"Model inputs. Images are always provided to the model. "
                             f"It is possible to specify multiple additional inputs, "
                             f"e.g. --inputs intrinsics poses. "
                             f"Options for additional model inputs are: intrinsics, poses, depth_range.",
                        type=str)

    # data arguments for non-"robustmvd" evaluations:
    if evaluation_type != "robustmvd":
        dataset_group = parser.add_mutually_exclusive_group()
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

    # data arguments for the "robustmvd" evaluation:
    if evaluation_type == "robustmvd":
        parser.add_argument('--eth3d_size', type=int, nargs=2,
                            help="Input image size on ETH3D in the format (height, width). "
                                 "If not provided, scales images up to the nearest size that works with the model.")
        parser.add_argument('--kitti_size', type=int, nargs=2,
                            help="Input image size on KITTI in the format (height, width). "
                                 "If not provided, scales images up to the nearest size that works with the model.")
        parser.add_argument('--dtu_size', type=int, nargs=2,
                            help="Input image size on DTU in the format (height, width). "
                                 "If not provided, scales images up to the nearest size that works with the model.")
        parser.add_argument('--scannet_size', type=int, nargs=2,
                            help="Input image size on ScanNet in the format (height, width). "
                                 "If not provided, scales images up to the nearest size that works with the model.")
        parser.add_argument('--tanks_and_temples_size', type=int, nargs=2,
                            help="Input image size on Tanks and Temples in the format (height, width). "
                                 "If not provided, scales images up to the nearest size that works with the model.")

    parser.add_argument('--input_resize_backend', choices=list_resize_backends(), metavar='BACKEND',
                        default=None,
                        help=f"Backend that is used to resize the input images to the specified input size. "
                             f"Only used if an input size is specified. "
                             f"Options are: {' '.join(list_resize_backends())}. If not specified, the default resizing backend is used.")

    samples_group = parser.add_mutually_exclusive_group()
    samples_group.add_argument('--num_samples', type=int,
                               help='Number of samples to be evaluated. Mutually exclusive with --samples. Default: evaluate all.')
    samples_group.add_argument('--samples', type=int, nargs='*',
                               help='Indices of samples to be evaluated. Mutually exclusive with --num_samples.')

    parser.add_argument('--max_source_views', type=int, help='Maximum number of source views to use for evaluation. '
                                                             'Default: use all available source views.')
    parser.add_argument('--min_source_views', type=int, default=1,
                        help='Minimum number of source views to use for evaluation. Default: 1.')
    parser.add_argument('--view_ordering', choices=['quasi-optimal', 'nearest'], default="quasi-optimal",
                        help="Source view ordering. Options are: quasi-optimal (default), nearest.")
    parser.add_argument('--alignment', choices=['median', 'pose_scale_mean', 'pose_scale_median', 'scale_shift_least_squares_invdepth'],
                        help="Alignment between predicted and ground truth depths. Default: no alignment")

    qualitatives_group = parser.add_mutually_exclusive_group()
    qualitatives_group.add_argument('--num_qualitatives', type=int, default=10,
                                    help='Number of qualitatives to be output. Negative values output all qualitatives. '
                                         'Mutually exclusive with --qualitatives. Default: 10.')
    qualitatives_group.add_argument('--qualitatives', type=int, nargs='*',
                                    help='Indices of samples where qualitatives should be output. Mutually exclusive with --num_qualitatives.')

    parser.add_argument('--eval_uncertainty', action='store_true', help='Evaluate predicted depth uncertainty.')
    if evaluation_type != "robustmvd":
        parser.add_argument('--clipping', type=float, nargs=2,
                            help="Clipping range for depth values in the format (min_depth, max_depth) in meters. "
                                 "If omitted, the evaluation default is used unless --disable_clipping is set.")
        parser.add_argument('--disable_clipping', action='store_true',
                            help="Do not clip predicted depth values. Takes precedence over --clipping if both are provided.")
    
    parser.add_argument('--log_dir', help="Path to folder for tensorboard and wandb logs. Optional. Default: use output dir.")
    parser.add_argument('--no_tensorboard', action='store_true', help='Do not log to tensorboard. Default: do log.')
    parser.add_argument('--wandb', action='store_true', help='Log to weights and biases. Default: Do not log.')
    parser.add_argument('--exp_id', type=str, help="Experiment ID. Used for wandb logging.")
    parser.add_argument('--comment', type=str, help="Comment for the experiment. Used for wandb logging.")

    args = parser.parse_args()
    validate_args(args, parser)
    eval(args)
