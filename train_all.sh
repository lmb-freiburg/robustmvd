#!/bin/bash

set -e

usage()
{
   echo "Usage: $0 -o [OPTIONAL] out_base -g [OPTIONAL] gpu_list"
   echo -e "\t-o Path to output base directory. Optional. Default: /tmp/rmvd_train"
   echo -e "\t-g List of space-separated gpu numbers to launch train on (e.g. 0 2 4 5). Optional. Default: 0"
   echo -e "\tNote: the order of the arguments is important"
   exit 1 # Exit program after printing help
}

while getopts "og" opt; do
    case "${opt}" in
        o )
          OUT_BASE=${OPTARG}
          ;;
        g )
          gpu_list=${OPTARG}
          ;;
        ? ) usage ;;
    esac
done

if [ -z ${OUT_BASE} ]; then
    OUT_BASE=/tmp/rmvd_train
fi
echo Output base directory: ${OUT_BASE}

shift $((OPTIND-1))
GPU_IDX=("$@")
if [ -z ${GPU_IDX} ]; then
    GPU_IDX=(0)
fi
GPU_IDX_STR=$(printf ",%s" "${GPU_IDX[@]}")
GPU_IDX_STR=${GPU_IDX_STR:1}
NUM_GPUS=${#GPU_IDX[@]}
echo Using ${NUM_GPUS} GPUs with indices: ${GPU_IDX[@]}
export CUDA_VISIBLE_DEVICES=${GPU_IDX_STR}

echo

# robust_mvd model:
python train.py --training_type mvd --output ${OUT_BASE}/robust_mvd --num_gpus ${NUM_GPUS} --batch_size 4 --max_iterations 600000 --model robust_mvd --inputs poses intrinsics --optimizer adam --lr 1e-4 --scheduler flownet_scheduler --loss MultiScaleUniLaplace --dataset flyingthings3d.seq4_train.mvd --input_size 384 768
