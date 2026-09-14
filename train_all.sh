#!/bin/bash

set -e

usage()
{
   echo "Usage: $0 -o [OPTIONAL] out_base -w [OPTIONAL] -g [OPTIONAL] num_gpus"
   echo -e "\t-o Path to output base directory. Optional. Default: /tmp/rmvd_train"
   echo -e "\t-w Enable wandb logging. Default: Do not enable wandb logging"
   echo -e "\t-g Number of GPUs to train on. 0 means train on CPU. Optional. Default: 1. Note: training uses the first num_gpus visible devices. To select specific GPUs, set CUDA_VISIBLE_DEVICES before calling this script."
   exit 1 # Exit program after printing help
}

NUM_GPUS=1

while getopts "o:wg:" opt; do
    case "${opt}" in
        o )
          OUT_BASE=${OPTARG}
          ;;
        w )
          enable_wandb="--wandb"
          ;;
        g )
          NUM_GPUS=${OPTARG}
          ;;
        ? ) usage ;;
    esac
done

if ! [[ ${NUM_GPUS} =~ ^[0-9]+$ ]]; then
    echo "Error: -g expects a number of GPUs, got: ${NUM_GPUS}" >&2
    usage
fi

if [ -z "${OUT_BASE}" ]; then
    OUT_BASE=/tmp/rmvd_train
fi
echo Output base directory: ${OUT_BASE}

if [ ${NUM_GPUS} -eq 0 ]; then
    `cpu`="--cpu"  # eval.py has a --cpu flag instead of a number of GPUs, as evaluation always uses a single device
    echo Training on CPU
else
    echo "Training on ${NUM_GPUS} GPU(s) (with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(not set)})"
fi

echo

# robust_mvd model:
export MAX_ITER=600000
python train.py --training_type mvd --output ${OUT_BASE}/robust_mvd --num_gpus ${NUM_GPUS} --batch_size 4 --max_iterations $MAX_ITER --model robust_mvd --inputs poses intrinsics --optimizer adam --lr 1e-4 --grad_clip_max_norm 5 --scheduler flownet_scheduler --loss robust_mvd_loss --dataset staticthings3d.robust_mvd.mvd --dataset blendedmvs.robust_mvd.mvd --augmentations_per_dataset robust_mvd_augmentations_staticthings3d --augmentations_per_dataset robust_mvd_augmentations_blendedmvs --batch_augmentations robust_mvd_batch_augmentations --seed 42 ${enable_wandb}
python eval.py --eval_type robustmvd --model robust_mvd --weights ${OUT_BASE}/robust_mvd/weights_only_checkpoints_dir/snapshot-iter-$(printf %09d "$MAX_ITER").pt --inputs poses intrinsics --output ${OUT_BASE}/robust_mvd/eval/snapshot-iter-$(printf %09d "$MAX_ITER").pt --finished_iterations $MAX_ITER --log_dir ${OUT_BASE}/robust_mvd ${cpu} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280 --input_resize_backend cv2_bilinear --eval_uncertainty --seed 42 ${enable_wandb}

# robust_mvd_5M model:
export MAX_ITER=5000000
python train.py --training_type mvd --output ${OUT_BASE}/robust_mvd_5M --num_gpus ${NUM_GPUS} --batch_size 4 --max_iterations $MAX_ITER --model robust_mvd --inputs poses intrinsics --optimizer adam --lr 1e-4 --grad_clip_max_norm 5 --scheduler flownet_scheduler --loss robust_mvd_loss --dataset staticthings3d.robust_mvd.mvd --dataset blendedmvs.robust_mvd.mvd --augmentations_per_dataset robust_mvd_augmentations_staticthings3d --augmentations_per_dataset robust_mvd_augmentations_blendedmvs --batch_augmentations robust_mvd_batch_augmentations --seed 42 ${enable_wandb}
python eval.py --eval_type robustmvd --model robust_mvd --weights ${OUT_BASE}/robust_mvd_5M/weights_only_checkpoints_dir/snapshot-iter-$(printf %09d "$MAX_ITER").pt --inputs poses intrinsics --output ${OUT_BASE}/robust_mvd_5M/eval/snapshot-iter-$(printf %09d "$MAX_ITER").pt --finished_iterations $MAX_ITER --log_dir ${OUT_BASE}/robust_mvd_5M ${cpu} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280 --input_resize_backend cv2_bilinear --eval_uncertainty --seed 42 ${enable_wandb}

# supervised_monodepth2 model:
export MAX_ITER=64000
python train.py --training_type mvd --output ${OUT_BASE}/supervised_monodepth2 --num_gpus ${NUM_GPUS} --batch_size 12 --max_iterations $MAX_ITER --model supervised_monodepth2 --optimizer adam --lr 1e-4 --scheduler monodepth2_scheduler --loss supervised_monodepth2_loss --dataset kitti.eigen_dense_depth_train.mvd --augmentations supervised_monodepth2_augmentations --seed 42 ${enable_wandb}
python eval.py --eval_type robustmvd --model supervised_monodepth2 --weights ${OUT_BASE}/supervised_monodepth2/weights_only_checkpoints_dir/snapshot-iter-$(printf %09d "$MAX_ITER").pt --output ${OUT_BASE}/supervised_monodepth2/eval/snapshot-iter-$(printf %09d "$MAX_ITER").pt --finished_iterations $MAX_ITER --log_dir ${OUT_BASE}/supervised_monodepth2 ${cpu} --eth3d_size 192 640 --kitti_size 192 640 --dtu_size 192 640 --scannet_size 192 640 --tanks_and_temples_size 192 640 --max_source_views 0 --alignment median --seed 42 ${enable_wandb}  # the model is applied at its 192x640 training resolution on all datasets, analogous to the monodepth2_*_wrapped models, which internally resize to their training resolution
python eval.py --eval_type mvd --dataset kitti.eigen_dense_depth_test.mvd --model supervised_monodepth2 --weights ${OUT_BASE}/supervised_monodepth2/weights_only_checkpoints_dir/snapshot-iter-$(printf %09d "$MAX_ITER").pt --output ${OUT_BASE}/supervised_monodepth2/eval/snapshot-iter-$(printf %09d "$MAX_ITER").pt/kitti.eigen_dense_depth_test.mvd --finished_iterations $MAX_ITER --log_dir ${OUT_BASE}/supervised_monodepth2 ${cpu} --input_size 192 640 --clipping 1e-3 80 --max_source_views 0 --seed 42 ${enable_wandb}
