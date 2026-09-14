#!/usr/bin/env python3

import argparse
import sys
import os
import os.path as osp

from rmvd import create_model, list_models, create_dataset, create_compound_dataset, list_datasets, list_folder_parsers, has_folder_parser, has_dataset, create_training, list_trainings, create_optimizer, list_optimizers, create_scheduler, list_schedulers, create_loss, list_losses, list_augmentations, list_batch_augmentations
from rmvd.data import has_dataset_cfg, list_resize_backends
from rmvd.utils import set_random_seed, writer, logging


def get_dataset_args(args):
    registered_datasets = [(dataset_name, None) for dataset_name in (args.dataset or [])]
    dataset_cfgs = [(path, None) for path in (args.dataset_cfg or [])]
    folders = [(path, folder_parser_name) for path, folder_parser_name in (args.folder or [])]
    return registered_datasets + folders + dataset_cfgs


def validate_args(args, parser):
    # The factory functions also validate the args, but the goal is to do it before the train() function starts and creates folders etc.
    # Some checks are already done at parser level. The ones that can not be done at parser level, are done here.

    dataset_args = get_dataset_args(args)

    if len(dataset_args) == 0:
        parser.error("No dataset specified. Use --dataset, --dataset_cfg, --folder.")

    invalid_dataset_cfg_paths = [path for path in (args.dataset_cfg or []) if not has_dataset_cfg(path)]
    if invalid_dataset_cfg_paths:
        parser.error(f"No dataset.cfg file found in dataset config path(s): {' '.join(invalid_dataset_cfg_paths)}")

    missing_folder_paths = [path for path, folder_parser_name in dataset_args if folder_parser_name is not None and not osp.isdir(path)]
    if missing_folder_paths:
        parser.error(f"Folder path(s) do not exist or are not directories: {' '.join(missing_folder_paths)}")

    unknown_folder_parsers = [folder_parser_name for _, folder_parser_name in dataset_args if folder_parser_name is not None and not has_folder_parser(folder_parser_name, data_format=args.training_type)]
    if unknown_folder_parsers:
        parser.error(f"Unknown folder parser(s) for the {args.training_type} training type: "
                     f"{' '.join(unknown_folder_parsers)}. Available folder parsers are: "
                     f"{' '.join(list_folder_parsers(data_format=args.training_type))}")

    if args.augmentations_per_dataset is not None and len(args.augmentations_per_dataset) != len(dataset_args):
        parser.error(f"There must be as many --augmentations_per_dataset arguments "
                     f"(currently: {len(args.augmentations_per_dataset)}) as --dataset, --folder and --dataset_cfg arguments "
                     f"combined (currently: {len(dataset_args)}).")


def train(args):
    
    set_random_seed(args.seed)
    
    out_dir = args.output
    tensorboard_logs_dir = osp.join(out_dir, "tensorboard_logs")
    wandb_logs_dir = osp.join(out_dir, "wandb_logs")
    log_file_path = osp.join(out_dir, "log.txt")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tensorboard_logs_dir, exist_ok=True)
    os.makedirs(wandb_logs_dir, exist_ok=True)
    writer.setup_writers(log_tensorboard=not args.no_tensorboard, 
                         log_wandb=args.wandb, 
                         max_iterations=args.max_iterations,
                         tensorboard_logs_dir=tensorboard_logs_dir, 
                         wandb_logs_dir=wandb_logs_dir,
                         exp_id=args.exp_id,
                         comment=args.comment,)
    logging.add_log_file(log_file_path, flush_line=True)

    with open(osp.join(out_dir, "cmd.txt"), 'a') as f:
        f.write("python " + " ".join(sys.argv) + "\n")

    dataset_args = get_dataset_args(args)
    dataset_names = [dataset_name_or_path for dataset_name_or_path, _ in dataset_args]
        
    logging.info()
    logging.info(f"Training {args.model} model on the dataset {'+'.join(dataset_names)} in the {args.training_type} training setting.\n")

    datasets = []
    for dataset_idx, (dataset_name_or_path, folder_parser_name) in enumerate(dataset_args):
        augmentation = args.augmentations_per_dataset[dataset_idx] if args.augmentations_per_dataset is not None else args.augmentations
        dataset = create_dataset(dataset_name_or_path=dataset_name_or_path, data_format=args.training_type, folder_parser_name=folder_parser_name,
                                 input_size=args.input_size, input_resize_backend=args.input_resize_backend,
                                 target_size=args.target_size, augmentations=augmentation)
        datasets.append(dataset)
    dataset = datasets[0] if len(datasets) == 1 else create_compound_dataset(datasets=datasets)

    model = create_model(name=args.model, pretrained=False, weights=args.weights, train=True, num_gpus=args.num_gpus)

    optimizer = create_optimizer(name=args.optimizer, model=model, lr=args.lr)
    scheduler = create_scheduler(name=args.scheduler, optimizer=optimizer)

    loss = create_loss(name=args.loss, model=model)

    training = create_training(training_type=args.training_type,
                               out_dir=out_dir,
                               model=model,
                               dataset=dataset,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               loss=loss,
                               batch_size=args.batch_size,
                               max_iterations=args.max_iterations,
                               inputs=args.inputs,
                               batch_augmentations=args.batch_augmentations,
                               grad_clip_max_norm=args.grad_clip_max_norm,
                               num_workers=args.num_workers,
                               log_interval=args.log_interval,
                               log_full_batch=args.log_full_batch,
                               verbose=True,)

    training()
    
    logging.remove_log_file(log_file_path)


if __name__ == '__main__':
    # Parse training type first:
    training_type_parser = argparse.ArgumentParser(add_help=False)
    training_type_parser.add_argument('--training_type', choices=list_trainings())
    selected_args, _ = training_type_parser.parse_known_args()
    training_type = selected_args.training_type

    if training_type is None:
        training_type_parser = argparse.ArgumentParser(description='Select a training type, then run "python train.py --training_type TRAINING_TYPE --help" for its full help.')
        training_type_parser.add_argument('--training_type', choices=list_trainings(), metavar='TRAINING_TYPE', required=True,
                                          help=f"Training setting. Options are: {' '.join(list_trainings())}")
        if '-h' in sys.argv or '--help' in sys.argv:
            training_type_parser.print_help()
            sys.exit()
        training_type_parser.error(f"the following arguments are required: --training_type. Options are: {' '.join(list_trainings())}")

    # Parse other arguments second:
    available_datasets = list_datasets(data_format=training_type)
    available_dataset_choices = sorted(set(
        available_datasets
        + list_datasets(data_format=training_type, no_data_format=True)
        + [n for n in list_datasets(data_format=training_type, no_split=True, no_data_format=True)
           if has_dataset(n, data_format=training_type)]
    ))
    available_folder_parsers = list_folder_parsers(data_format=training_type)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help="Path to folder for output data.", required=True)
    parser.add_argument('--num_gpus', type=int, help="Number of GPUs. 0 means use CPU. Default: use 1 GPU.", default=1)
    parser.add_argument('--seed', type=int, help="Random seed. Default: 42.", default=42)
    parser.add_argument('--num_workers', type=int, help="Number of workers for data loading. Default: 8.", default=8)
    
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size.")
    parser.add_argument('--max_iterations', type=int, required=True, help="Maximum number of iterations to train.")
    parser.add_argument('--training_type', help=f"Training setting. Options are: {' '.join(list_trainings())}",
                        choices=list_trainings(), metavar='TRAINING_TYPE', required=True)

    parser.add_argument('--model', help=f"Model for training. Available models "
                                        f"are: {' '.join(list_models(trainable_only=True))}",
                        choices=list_models(trainable_only=True), metavar='MODEL', required=True)
    parser.add_argument('--weights', help="Path to weights of the model. Optional. "
                                          "If None, default weight initialization of the model is used.")

    parser.add_argument('--dataset', help=f"Registered dataset. Available datasets are: {' '.join(available_datasets)}",
                        choices=available_dataset_choices, metavar='DATASET', action='append')
    parser.add_argument('--dataset_cfg', action='append', metavar='PATH',
                        help="Dataset config file or a directory containing a dataset.cfg file. Can be specified multiple times.")
    parser.add_argument('--folder', nargs=2, action='append', metavar=('PATH', 'FOLDER_PARSER'),
                        help=f"Folder dataset specified as a path and folder parser. Available folder parsers are: "
                             f"{' '.join(available_folder_parsers)}")
    parser.add_argument('--input_size', type=int, nargs=2, help="Input data size in the format (height, width). If not provided, uses the original size.")
    parser.add_argument('--input_resize_backend', choices=list_resize_backends(), metavar='BACKEND',
                        default=None,
                        help=f"Backend that is used to resize the input images to the specified input size. "
                             f"Only used if an input size is specified. "
                             f"Options are: {' '.join(list_resize_backends())}. If not specified, the default resizing backend is used.")
    parser.add_argument('--target_size', type=int, nargs=2, help="Target data size in the format (height, width). If not provided, uses the original size.")

    augmentations_group = parser.add_mutually_exclusive_group()
    augmentations_group.add_argument('--augmentations', nargs='*', choices=list_augmentations(), metavar='AUGMENTATION',
                                     help=f"Data augmentations. Mutually exclusive with --augmentations_per_dataset. "
                                          f"Options are: {' '.join(list_augmentations())}")
    augmentations_group.add_argument('--augmentations_per_dataset', nargs='*', action='append',
                                     choices=list_augmentations(), metavar='AUGMENTATION',
                                     help=f"Data augmentations per dataset in case training is done on multiple datasets "
                                          f"and the datasets should have different augmentations. Ordered by --dataset arguments followed by "
                                          f"--folder arguments followed by --dataset_cfg arguments. "
                                          f"Mutually exclusive with --augmentations. Options are: "
                                          f"{' '.join(list_augmentations())}")
    parser.add_argument('--batch_augmentations', nargs='*', choices=list_batch_augmentations(), metavar='BATCH_AUGMENTATION',
                        help=f"Data augmentations that are applied to the whole batch. Options are: {' '.join(list_batch_augmentations())}")

    parser.add_argument('--optimizer', help=f"Optimizer. Options are: {' '.join(list_optimizers())}",
                        choices=list_optimizers(), metavar='OPTIMIZER', required=True)
    parser.add_argument('--lr', type=float, help="Learning rate. Default: 1e-4", default=1e-4)
    parser.add_argument('--grad_clip_max_norm', type=float, help="Maximum norm of gradient. Default: do not clip gradient.")

    parser.add_argument('--scheduler', help=f"Scheduler. Options are: {' '.join(list_schedulers())}",
                        choices=list_schedulers(), metavar='SCHEDULER', required=True)

    parser.add_argument('--loss', help=f"Loss. Options are: {' '.join(list_losses())}",
                        choices=list_losses(), metavar='LOSS', required=True)

    parser.add_argument('--inputs', nargs='*', choices=['intrinsics', 'poses', 'depth_range'], metavar='INPUT',
                        help=f"Model inputs. Images are always provided to the model. "
                             f"It is possible to specify multiple additional inputs, "
                             f"e.g. --inputs intrinsics poses. "
                             f"Options for additional model inputs are: intrinsics, poses, depth_range.",
                        type=str)

    parser.add_argument('--log_interval', type=int, help="Log interval in iterations. Default: 5000.", default=5000)
    parser.add_argument('--log_full_batch', action='store_true', help='Write all samples in batch to log. '
                                                                      'Default: only log first sample.')
    parser.add_argument('--no_tensorboard', action='store_true', help='Do not log to tensorboard. Default: do log.')
    parser.add_argument('--wandb', action='store_true', help='Log to weights and biases. Default: Do not log.')
    parser.add_argument('--exp_id', type=str, help="Experiment ID. Used for wandb logging.")
    parser.add_argument('--comment', type=str, help="Comment for the experiment. Used for wandb logging.")

    args = parser.parse_args()
    validate_args(args, parser)
    train(args)
