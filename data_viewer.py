#!/usr/bin/env python3

import argparse
from rmvd import list_datasets, create_dataset, run_viewer


def data_viewer(args):
    dataset = create_dataset(args.data)
    run_viewer(dataset, args.layout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help=f"Data to be viewed. Available datasets are: {', '.join(list_datasets())}")
    parser.add_argument('--layout', help=f"Data viewer layout. If not specified, the default layout is used.")
    args = parser.parse_args()
    data_viewer(args)
