#!/usr/bin/env python3

import argparse
import sys
import os.path as osp

import torch

from rmvd import create_model, list_models, create_dataset, list_datasets, create_training, list_trainings, create_optimizer, list_optimizers, create_scheduler, list_schedulers, create_loss, list_losses


def train(args):

    if args.model is None:
        print(f"No model specified via --model. Available models are: {', '.join(list_models(trainable_only=True))}")
        return

    if args.training_type is None:
        print(f"No training type specified via --training_type. Available training types are: {', '.join(list_trainings())}")
        return

    if args.dataset is None:  # or dataset not available
        datasets = list_datasets(dataset_type=args.training_type, no_dataset_type=True)
        print(f"No dataset specified via --dataset. Available datasets are: {', '.join(datasets)}")
        return

    if args.optimizer is None:
        print(f"No optimizer specified via --optimizer. Available optimizers are: {', '.join(list_optimizers())}")
        return

    if args.scheduler is None:
        print(f"No scheduler specified. Available schedulers are: {', '.join(list_schedulers())}")
        return
    
    if args.loss is None:
        print(f"No loss specified. Available losses are: {', '.join(list_losses())}")
        return

    print()
    print(f"Training {args.model} model on dataset {args.dataset} in the {args.training_type} training setting.\n")
    dataset = create_dataset(dataset_name_or_path=args.dataset, dataset_type=args.training_type,
                             input_size=args.input_size)  # TODO: augmentations etc.

    model = create_model(name=args.model, pretrained=False, weights=args.weights, train=True, num_gpus=args.num_gpus)

    optimizer = create_optimizer(name=args.optimizer, model=model, lr=args.lr)
    scheduler = create_scheduler(name=args.scheduler, optimizer=optimizer)

    loss = create_loss(name=args.loss, model=model)

    training = create_training(training_type=args.training_type,
                               out_dir=args.output,
                               model=model,
                               dataset=dataset,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               loss=loss,
                               batch_size=args.batch_size,
                               max_iterations=args.max_iterations,
                               inputs=args.inputs,
                               num_workers=args.num_workers,
                               log_tensorboard=not args.no_tensorboard,
                               log_wandb=args.wandb,
                               log_full_batch=args.log_full_batch,
                               verbose=True,)

    with open(osp.join(args.output, "cmd.txt"), 'a') as f:
        f.write("python " + " ".join(sys.argv) + "\n")

    training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help="Path to folder for output data.")
    parser.add_argument('--num_gpus', type=int, help="Number of GPUs. 0 means use CPU. Default: use 1 GPU.", default=1)
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size.")
    parser.add_argument('--max_iterations', type=int, required=True, help="Maximum number of iterations to train.")
    parser.add_argument('--num_workers', type=int, help="Number of workers for data loading. Default: 8.", default=8)
    parser.add_argument('--training_type', help=f"Training setting. Options are: {', '.join(list_trainings())}")

    parser.add_argument('--model', help=f"Model for evaluation. Available models "
                                        f"are: {', '.join(list_models(trainable_only=True))}")
    parser.add_argument('--weights', help="Path to weights of the model. Optional. "
                                          "If None, default weight initialization of the model is used.")

    parser.add_argument('--dataset', help=f"Dataset. Available datasets are: {', '.join(list_datasets())}")
    parser.add_argument('--input_size', type=int, nargs=2, help="Input image size in the format (height, width). "
                                                                "If not provided, scales images up to the nearest size "
                                                                "that works with the model.")

    parser.add_argument('--optimizer', help=f"Optimizer. Options are: {', '.join(list_optimizers())}")
    parser.add_argument('--lr', type=float, help="Learning rate. Default: 1e-4", default=1e-4)

    parser.add_argument('--scheduler', help=f"Scheduler. Options are: {', '.join(list_schedulers())}")

    parser.add_argument('--loss', help=f"Loss. Options are: {', '.join(list_losses())}")

    parser.add_argument('--inputs', nargs='*',
                        help=f"Model inputs. Images are always provided to the model. "
                             f"It is possible to specify multiple additional inputs, "
                             f"e.g. --inputs intrinsics poses. "
                             f"Options for additional model inputs are: intrinsics, poses, depth_range.",
                        type=str)

    parser.add_argument('--log_full_batch', action='store_true', help='Write all samples in batch to log. '
                                                                      'Default: only log first sample.')
    parser.add_argument('--no_tensorboard', action='store_true', help='Do not log to tensorboard. Default: do log.')
    parser.add_argument('--wandb', action='store_true', help='Log to weights and biases. Default: Do not log.')

    args = parser.parse_args()

    train(args)
