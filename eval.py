import argparse
import sys
import os.path as osp

import torch

from robd import create_model, list_models, create_dataset, list_datasets, create_evaluation, list_evaluations


@torch.no_grad()
def eval(args):

    if args.model is None:
        print(f"No model specified. Available models are: {', '.join(list_models())}")
        return

    if args.setting is None:
        print(f"No setting specified. Available evaluation settings are: {', '.join(list_evaluations())}")
        return

    if args.dataset is None:  # or dataset not available
        datasets = list_datasets(dataset_type=args.setting, no_dataset_type=True)
        print(f"No dataset specified. Available datasets are: {', '.join(datasets)}")
        return

    print(f"Evaluating {args.model} model on dataset {args.dataset} in the {args.setting} setting.\n")

    inputs = args.inputs if args.inputs is not None else ["images"]

    model = create_model(name=args.model, weights=args.weights, train=False, num_gpus=args.num_gpus)
    dataset = create_dataset(dataset_name=args.dataset, dataset_type=args.setting, input_size=(args.input_height, args.input_width))
    eval = create_evaluation(evaluation_type=args.setting,
                             out_dir=args.output,
                             inputs=inputs,
                             eval_uncertainty=args.eval_uncertainty)
    # TODO: how to deal with that different evaluations will accept different input params?

    with open(osp.join(args.output, "cmd.txt"), 'w') as f:
        f.write("python " + " ".join(sys.argv))

    samples = args.num_samples if args.num_samples is not None else args.samples
    qualitatives = args.qualitatives if args.qualitatives is not None else args.num_qualitatives

    eval(dataset=dataset, model=model, samples=samples, qualitatives=qualitatives)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help=f"Model for evaluation. Available models are: {', '.join(list_models())}")
    parser.add_argument('--weights', help="Path to weights of the model. Optional. If None, default weights are used.")
    parser.add_argument('--num_gpus', type=int, help="Number of GPUs. 0 means use CPU. Default: use 1 GPU.", default=1)
    parser.add_argument('--dataset', help=f"Dataset. Available datasets are: {', '.join(list_datasets())}")
    parser.add_argument('--setting', help=f"Evaluation setting. Options are: {', '.join(list_evaluations())}")
    parser.add_argument('--input', nargs='*',
                        help=f"Model inputs. Images are always provided to the model. "
                             f"It is possible to specify multiple additional inputs, "
                             f"e.g. --input intrinsics --input poses. "
                             f"Options for additional model inputs are: intrinsics, poses, depth_range.",
                        type=str, dest='inputs')
    parser.add_argument('--output', help="Path to folder for output data.")

    parser.add_argument('--num_samples', type=int, help='Number of samples to be evaluated. Default: evaluate all.')
    parser.add_argument('--sample', dest='samples', type=int, action='append',
                        help='Index of sample that should be evaluated. Ignored if num_samples is used. '
                             'Default: evaluate all.')

    parser.add_argument('--num_qualitatives', type=int, default=10,
                        help='Number of qualitatives to be output. Negative values output all qualitatives. '
                             'Ignored if --qualitative is used. Default: 10.')
    parser.add_argument('--qualitative', dest='qualitatives', type=int, action='append',
                        help='Index of sample where qualitatives should be output.')

    parser.add_argument('--eval_uncertainty', action='store_true', help='Evaluate predicted depth uncertainty.')

    parser.add_argument('--input_width', type=int, help="Input image width.")  # TODO: add text what happens if not provided
    parser.add_argument('--input_height', type=int, help="Input image height.")
    args = parser.parse_args()

    eval(args)
