#!/bin/bash

usage()
{
   echo "Usage: $0 -o [OPTIONAL] out_base -g [OPTIONAL] gpu_list"
   echo -e "\t-o Path to output base directory. Optional. Default: /tmp/eval_benchmark"
   echo -e "\t-g List of space-separated gpu numbers to launch train on (e.g. 0 2 4 5). Optional. Default: 1"
   exit 1 # Exit program after printing help
}

while getopts "o:g" opt; do
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
    OUT_BASE=/tmp/eval_benchmark
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

export export CUDA_VISIBLE_DEVICES=${GPU_IDX_STR}

# robust_mvd model:
python eval.py --eval_type robustmvd --model robust_mvd --inputs poses intrinsics --output ${OUT_BASE}/robust_mvd --num_gpus ${NUM_GPUS} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280

# robust_mvd_5M model:
python eval.py --eval_type robustmvd --model robust_mvd_5M --inputs poses intrinsics --output ${OUT_BASE}/robust_mvd_5M --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280

# monodepth2_mono_stereo_1024x320_wrapped model:
python eval.py --eval_type robustmvd --model monodepth2_mono_stereo_1024x320_wrapped --output ${OUT_BASE}/monodepth2_mono_stereo_1024x320_wrapped --max_source_views 0 --alignment median

# monodepth2_mono_stereo_640x192_wrapped model:
python eval.py --eval_type robustmvd --model monodepth2_mono_stereo_640x192_wrapped --output ${OUT_BASE}/monodepth2_mono_stereo_640x192_wrapped --max_source_views 0 --alignment median

# mvsnet_pl_wrapped model:
python eval.py --eval_type robustmvd --model mvsnet_pl_wrapped --inputs poses intrinsics depth_range --output ${OUT_BASE}/mvsnet_pl_wrapped/known_depth_range --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280
python eval.py --eval_type robustmvd --model mvsnet_pl_wrapped --inputs poses intrinsics --output ${OUT_BASE}/mvsnet_pl_wrapped/unknown_depth_range --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280

# midas_big_v2_1_wrapped model:
python eval.py --eval_type robustmvd --model midas_big_v2_1_wrapped --output ${OUT_BASE}/midas_big_v2_1_wrapped --max_source_views 0 --alignment least_squares_scale_shift
