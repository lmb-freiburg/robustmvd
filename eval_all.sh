#!/bin/bash

set -e

usage()
{
   echo "Usage: $0 -o [OPTIONAL] out_base -n [OPTIONAL] num_samples -u [OPTIONAL] -c [OPTIONAL]"
   echo -e "\t-o Path to output base directory. Optional. Default: /tmp/rmvd_eval"
   echo -e "\t-n Number of samples to evaluate per dataset. Optional. Default: Evaluate all samples"
   echo -e "\t-u Evaluate Uncertainty. Default: Do not evaluate uncertainty"
   echo -e "\t-c Run on CPU. Default: Run on a single GPU. Note: evaluation always uses a single device. To select a specific GPU, set CUDA_VISIBLE_DEVICES before calling this script."
   exit 1 # Exit program after printing help
}

while getopts "o:n:uc" opt; do
    case "${opt}" in
        o )
          OUT_BASE=${OPTARG}
          ;;
        n )
          num_samples="--num_samples ${OPTARG}"
          ;;
        u )
          eval_uncertainty="--eval_uncertainty"
          ;;
        c )
          cpu="--cpu"
          ;;
        ? ) usage ;;
    esac
done

if [ -z "${OUT_BASE}" ]; then
    OUT_BASE=/tmp/rmvd_eval
fi
echo Output base directory: ${OUT_BASE}

if [ -n "${cpu}" ]; then
    echo Running on CPU
else
    echo "Running on 1 GPU (with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(not set)})"
fi
echo

# robust_mvd model:
conda activate rmvd
# evaluate with --input_resize_backend cv2_bilinear as this gives better results than the default input_resize_backend and was also used for the results from the RMVD paper:
python eval.py --eval_type robustmvd --model robust_mvd --inputs poses intrinsics --output ${OUT_BASE}/robust_mvd/robustmvd_benchmark ${cpu} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280 --input_resize_backend cv2_bilinear ${num_samples} ${eval_uncertainty}
python eval.py --eval_type mvd --dataset eth3d.mvsanywhere --model robust_mvd --inputs poses intrinsics --output ${OUT_BASE}/robust_mvd/robustmvd_benchmark/eth3d.mvsanywhere.mvd ${cpu} --input_size 768 1152 --input_resize_backend cv2_bilinear ${num_samples} ${eval_uncertainty}
# additionally evaluate on eth3d.mvsanywhere.mvd with the default input_resize_backend to reproduce the results from MVSAnywhere:
python eval.py --eval_type mvd --dataset eth3d.mvsanywhere --model robust_mvd --inputs poses intrinsics --output ${OUT_BASE}/robust_mvd/reproduce/robustmvd_benchmark/eth3d.mvsanywhere.mvd ${cpu} --input_size 768 1152 ${num_samples} ${eval_uncertainty}
# note: it is not necessary to evaluate on dtu.robustmvd_paper.mvd to reproduce the results from the RMVD paper, as the model anyways does not use the depth range as input
conda deactivate

# robust_mvd_5M model:
conda activate rmvd
python eval.py --eval_type robustmvd --model robust_mvd_5M --inputs poses intrinsics --output ${OUT_BASE}/robust_mvd_5M/robustmvd_benchmark ${cpu} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280 --input_resize_backend cv2_bilinear ${num_samples} ${eval_uncertainty}
python eval.py --eval_type mvd --dataset eth3d.mvsanywhere --model robust_mvd_5M --inputs poses intrinsics --output ${OUT_BASE}/robust_mvd_5M/robustmvd_benchmark/eth3d.mvsanywhere.mvd ${cpu} --input_size 768 1152 --input_resize_backend cv2_bilinear ${num_samples} ${eval_uncertainty}
conda deactivate

# supervised_monodepth2 model:
conda activate rmvd
python eval.py --eval_type robustmvd --model supervised_monodepth2 --output ${OUT_BASE}/supervised_monodepth2/robustmvd_benchmark ${cpu} --eth3d_size 192 640 --kitti_size 192 640 --dtu_size 192 640 --scannet_size 192 640 --tanks_and_temples_size 192 640 --max_source_views 0 --alignment median ${num_samples}  # the model is applied at its 192x640 training resolution on all datasets, analogous to the monodepth2_*_wrapped models, which internally resize to their training resolution
python eval.py --eval_type mvd --dataset eth3d.mvsanywhere --model supervised_monodepth2 --output ${OUT_BASE}/supervised_monodepth2/robustmvd_benchmark/eth3d.mvsanywhere.mvd ${cpu} --input_size 192 640 --max_source_views 0 --alignment median ${num_samples}
python eval.py --eval_type mvd --dataset kitti.eigen_dense_depth_test.mvd --model supervised_monodepth2 --output ${OUT_BASE}/supervised_monodepth2/misc/kitti.eigen_dense_depth_test.mvd ${cpu} --input_size 192 640 --clipping 1e-3 80 --max_source_views 0 ${num_samples}
conda deactivate

# monodepth2_mono_stereo_wrapped model:
conda activate rmvd
python eval.py --eval_type robustmvd --model monodepth2_mono_stereo_wrapped --output ${OUT_BASE}/monodepth2_mono_stereo_wrapped/robustmvd_benchmark ${cpu} --max_source_views 0 --alignment median ${num_samples}
python eval.py --eval_type mvd --dataset eth3d.mvsanywhere --model monodepth2_mono_stereo_wrapped --output ${OUT_BASE}/monodepth2_mono_stereo_wrapped/robustmvd_benchmark/eth3d.mvsanywhere.mvd ${cpu} --max_source_views 0 --alignment median ${num_samples}
python eval.py --eval_type mvd --dataset kitti.eigen_dense_depth_test.mvd --model monodepth2_mono_stereo_wrapped --output ${OUT_BASE}/monodepth2_mono_stereo_wrapped/reproduce/kitti.eigen_dense_depth_test.mvd/median_alignment ${cpu} --max_source_views 0 --clipping 1e-3 80 --alignment median ${num_samples}
python eval.py --eval_type mvd --dataset kitti.eigen_dense_depth_test.mvd --model monodepth2_mono_stereo_wrapped --output ${OUT_BASE}/monodepth2_mono_stereo_wrapped/misc/kitti.eigen_dense_depth_test.mvd/no_alignment ${cpu} --max_source_views 0 --clipping 1e-3 80 ${num_samples}
conda deactivate

# monodepth2_mono_stereo_1024x320_wrapped model:
conda activate rmvd
python eval.py --eval_type robustmvd --model monodepth2_mono_stereo_1024x320_wrapped --output ${OUT_BASE}/monodepth2_mono_stereo_1024x320_wrapped/robustmvd_benchmark ${cpu} --max_source_views 0 --alignment median ${num_samples}
python eval.py --eval_type mvd --dataset eth3d.mvsanywhere --model monodepth2_mono_stereo_1024x320_wrapped --output ${OUT_BASE}/monodepth2_mono_stereo_1024x320_wrapped/robustmvd_benchmark/eth3d.mvsanywhere.mvd ${cpu} --max_source_views 0 --alignment median ${num_samples}
python eval.py --eval_type mvd --dataset kitti.eigen_dense_depth_test.mvd --model monodepth2_mono_stereo_1024x320_wrapped --output ${OUT_BASE}/monodepth2_mono_stereo_1024x320_wrapped/misc/kitti.eigen_dense_depth_test.mvd/median_alignment ${cpu} --max_source_views 0 --clipping 1e-3 80 --alignment median ${num_samples}
python eval.py --eval_type mvd --dataset kitti.eigen_dense_depth_test.mvd --model monodepth2_mono_stereo_1024x320_wrapped --output ${OUT_BASE}/monodepth2_mono_stereo_1024x320_wrapped/misc/kitti.eigen_dense_depth_test.mvd/no_alignment ${cpu} --max_source_views 0 --clipping 1e-3 80 ${num_samples}
conda deactivate

# monodepth2_postuncertainty_mono_stereo_wrapped model:
conda activate rmvd
python eval.py --eval_type robustmvd --model monodepth2_postuncertainty_mono_stereo_wrapped --output ${OUT_BASE}/monodepth2_postuncertainty_mono_stereo_wrapped/robustmvd_benchmark ${cpu} --max_source_views 0 --alignment median ${num_samples} ${eval_uncertainty}
python eval.py --eval_type mvd --dataset eth3d.mvsanywhere --model monodepth2_postuncertainty_mono_stereo_wrapped --output ${OUT_BASE}/monodepth2_postuncertainty_mono_stereo_wrapped/robustmvd_benchmark/eth3d.mvsanywhere.mvd ${cpu} --max_source_views 0 --alignment median ${num_samples} ${eval_uncertainty}
python eval.py --eval_type mvd --dataset kitti.eigen_dense_depth_test.mvd --model monodepth2_postuncertainty_mono_stereo_wrapped --output ${OUT_BASE}/monodepth2_postuncertainty_mono_stereo_wrapped/reproduce/kitti.eigen_dense_depth_test.mvd ${cpu} --max_source_views 0 --clipping 1e-3 80 ${num_samples} --eval_uncertainty  # intentionally using --eval_uncertainty, because this is the focus of this re-evaluation
conda deactivate

# mvsnet_pl_wrapped model:
conda activate rmvd-mvsnet_pl
# evaluate with --input_resize_backend cv2_bilinear and additionally on dtu.robustmvd_paper.mvd to reproduce the results from the RMVD paper:
python eval.py --eval_type robustmvd --model mvsnet_pl_wrapped --inputs poses intrinsics depth_range --output ${OUT_BASE}/mvsnet_pl_wrapped/reproduce/robustmvd_benchmark/known_depth_range --eval_name known_depth_range_reproduce ${cpu} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280 --input_resize_backend cv2_bilinear ${num_samples} ${eval_uncertainty}
python eval.py --eval_type mvd --dataset dtu.robustmvd_paper --model mvsnet_pl_wrapped --inputs poses intrinsics depth_range --output ${OUT_BASE}/mvsnet_pl_wrapped/reproduce/robustmvd_benchmark/known_depth_range/dtu.robustmvd_paper.mvd --eval_name known_depth_range_reproduce ${cpu} --input_size 896 1216 --input_resize_backend cv2_bilinear ${num_samples} ${eval_uncertainty}
python eval.py --eval_type robustmvd --model mvsnet_pl_wrapped --inputs poses intrinsics --output ${OUT_BASE}/mvsnet_pl_wrapped/reproduce/robustmvd_benchmark/unknown_depth_range --eval_name unknown_depth_range_reproduce ${cpu} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280 --input_resize_backend cv2_bilinear ${num_samples} ${eval_uncertainty}
# note: it is not necessary to evaluate on dtu.robustmvd_paper.mvd for unknown_depth_range, as dtu.robustmvd.mvd and dtu.robustmvd_paper.mvd anyways just differ in the depth range
# evaluate without --input_resize_backend cv2_bilinear, as this gives better results (especially on ETH3D):
python eval.py --eval_type robustmvd --model mvsnet_pl_wrapped --inputs poses intrinsics depth_range --output ${OUT_BASE}/mvsnet_pl_wrapped/robustmvd_benchmark/known_depth_range --eval_name known_depth_range ${cpu} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280 ${num_samples} ${eval_uncertainty}
python eval.py --eval_type mvd --dataset eth3d.mvsanywhere --model mvsnet_pl_wrapped --inputs poses intrinsics depth_range --output ${OUT_BASE}/mvsnet_pl_wrapped/robustmvd_benchmark/known_depth_range/eth3d.mvsanywhere.mvd --eval_name known_depth_range ${cpu} --input_size 768 1152 ${num_samples} ${eval_uncertainty}
python eval.py --eval_type robustmvd --model mvsnet_pl_wrapped --inputs poses intrinsics --output ${OUT_BASE}/mvsnet_pl_wrapped/robustmvd_benchmark/unknown_depth_range --eval_name unknown_depth_range ${cpu} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280 ${num_samples} ${eval_uncertainty}
python eval.py --eval_type mvd --dataset eth3d.mvsanywhere --model mvsnet_pl_wrapped --inputs poses intrinsics --output ${OUT_BASE}/mvsnet_pl_wrapped/robustmvd_benchmark/unknown_depth_range/eth3d.mvsanywhere.mvd --eval_name unknown_depth_range ${cpu} --input_size 768 1152 ${num_samples} ${eval_uncertainty}
conda deactivate

# midas_large_v2_1_wrapped model:
conda activate rmvd-midas
python eval.py --eval_type robustmvd --model midas_large_v2_1_wrapped --output ${OUT_BASE}/midas_large_v2_1_wrapped/robustmvd_benchmark ${cpu} --max_source_views 0 --alignment scale_shift_least_squares_invdepth ${num_samples}
python eval.py --eval_type mvd --dataset eth3d.mvsanywhere --model midas_large_v2_1_wrapped --output ${OUT_BASE}/midas_large_v2_1_wrapped/robustmvd_benchmark/eth3d.mvsanywhere.mvd ${cpu} --max_source_views 0 --alignment scale_shift_least_squares_invdepth ${num_samples}
conda deactivate

# midas_dpt_large_384_wrapped model:
conda activate rmvd-midas
python eval.py --eval_type robustmvd --model midas_dpt_large_384_wrapped --output ${OUT_BASE}/midas_dpt_large_384_wrapped/robustmvd_benchmark ${cpu} --max_source_views 0 --alignment scale_shift_least_squares_invdepth ${num_samples}
python eval.py --eval_type mvd --dataset eth3d.mvsanywhere --model midas_dpt_large_384_wrapped --output ${OUT_BASE}/midas_dpt_large_384_wrapped/robustmvd_benchmark/eth3d.mvsanywhere.mvd ${cpu} --max_source_views 0 --alignment scale_shift_least_squares_invdepth ${num_samples}
conda deactivate

# vis_mvsnet_wrapped model:
conda activate rmvd
# evaluate with --input_resize_backend cv2_bilinear and additionally on dtu.robustmvd_paper.mvd to reproduce the results from the RMVD paper:
python eval.py --eval_type robustmvd --model vis_mvsnet_wrapped --inputs poses intrinsics depth_range --output ${OUT_BASE}/vis_mvsnet_wrapped/reproduce/robustmvd_benchmark/known_depth_range --eval_name known_depth_range_reproduce ${cpu} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280 --input_resize_backend cv2_bilinear ${num_samples} ${eval_uncertainty}
python eval.py --eval_type mvd --dataset dtu.robustmvd_paper --model vis_mvsnet_wrapped --inputs poses intrinsics depth_range --output ${OUT_BASE}/vis_mvsnet_wrapped/reproduce/robustmvd_benchmark/known_depth_range/dtu.robustmvd_paper.mvd --eval_name known_depth_range_reproduce ${cpu} --input_size 896 1216 --input_resize_backend cv2_bilinear ${num_samples} ${eval_uncertainty}
python eval.py --eval_type robustmvd --model vis_mvsnet_wrapped --inputs poses intrinsics --output ${OUT_BASE}/vis_mvsnet_wrapped/reproduce/robustmvd_benchmark/unknown_depth_range --eval_name unknown_depth_range_reproduce ${cpu} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280 --input_resize_backend cv2_bilinear ${num_samples} ${eval_uncertainty}
# note: it is not necessary to evaluate on dtu.robustmvd_paper.mvd for unknown_depth_range, as dtu.robustmvd.mvd and dtu.robustmvd_paper.mvd anyways just differ in the depth range
# evaluate without --input_resize_backend cv2_bilinear, as this gives better results (especially on ETH3D):
python eval.py --eval_type robustmvd --model vis_mvsnet_wrapped --inputs poses intrinsics depth_range --output ${OUT_BASE}/vis_mvsnet_wrapped/robustmvd_benchmark/known_depth_range --eval_name known_depth_range ${cpu} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280 ${num_samples} ${eval_uncertainty}
python eval.py --eval_type mvd --dataset eth3d.mvsanywhere --model vis_mvsnet_wrapped --inputs poses intrinsics depth_range --output ${OUT_BASE}/vis_mvsnet_wrapped/robustmvd_benchmark/known_depth_range/eth3d.mvsanywhere.mvd --eval_name known_depth_range ${cpu} --input_size 768 1152 ${num_samples} ${eval_uncertainty}
python eval.py --eval_type robustmvd --model vis_mvsnet_wrapped --inputs poses intrinsics --output ${OUT_BASE}/vis_mvsnet_wrapped/robustmvd_benchmark/unknown_depth_range --eval_name unknown_depth_range ${cpu} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280 ${num_samples} ${eval_uncertainty}
python eval.py --eval_type mvd --dataset eth3d.mvsanywhere --model vis_mvsnet_wrapped --inputs poses intrinsics --output ${OUT_BASE}/vis_mvsnet_wrapped/robustmvd_benchmark/unknown_depth_range/eth3d.mvsanywhere.mvd --eval_name unknown_depth_range ${cpu} --input_size 768 1152 ${num_samples} ${eval_uncertainty}
conda deactivate

# cvp_mvsnet_wrapped model:
conda activate rmvd
# evaluate with a fixed number of 6 source views, with --input_resize_backend cv2_bilinear and additionally on dtu.robustmvd_paper.mvd to reproduce the results from the RMVD paper:
python eval.py --eval_type robustmvd --model cvp_mvsnet_wrapped --inputs poses intrinsics depth_range --output ${OUT_BASE}/cvp_mvsnet_wrapped/reproduce/robustmvd_benchmark/known_depth_range --eval_name known_depth_range_reproduce --min_source_views 6 --max_source_views 6 --view_ordering nearest ${cpu} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 1184 1600 --scannet_size 448 640 --tanks_and_temples_size 1088 1952 --input_resize_backend cv2_bilinear ${num_samples} ${eval_uncertainty}
python eval.py --eval_type mvd --dataset dtu.robustmvd_paper --model cvp_mvsnet_wrapped --inputs poses intrinsics depth_range --output ${OUT_BASE}/cvp_mvsnet_wrapped/reproduce/robustmvd_benchmark/known_depth_range/dtu.robustmvd_paper.mvd --eval_name known_depth_range_reproduce --min_source_views 6 --max_source_views 6 --view_ordering nearest ${cpu} --input_size 1184 1600 --input_resize_backend cv2_bilinear ${num_samples} ${eval_uncertainty}
python eval.py --eval_type robustmvd --model cvp_mvsnet_wrapped --inputs poses intrinsics --output ${OUT_BASE}/cvp_mvsnet_wrapped/reproduce/robustmvd_benchmark/unknown_depth_range --eval_name unknown_depth_range_reproduce --min_source_views 6 --max_source_views 6 --view_ordering nearest ${cpu} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 1184 1600 --scannet_size 448 640 --tanks_and_temples_size 1088 1952 --input_resize_backend cv2_bilinear ${num_samples} ${eval_uncertainty}
# note: it is not necessary to evaluate on dtu.robustmvd_paper.mvd for unknown_depth_range, as dtu.robustmvd.mvd and dtu.robustmvd_paper.mvd anyways just differ in the depth range
# evaluate with varying numbers of source views and without --input_resize_backend cv2_bilinear, as this gives better results (especially due to the varying number of source views):
python eval.py --eval_type robustmvd --model cvp_mvsnet_wrapped --inputs poses intrinsics depth_range --output ${OUT_BASE}/cvp_mvsnet_wrapped/robustmvd_benchmark/known_depth_range --eval_name known_depth_range --min_source_views 2 --view_ordering nearest ${cpu} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 1184 1600 --scannet_size 448 640 --tanks_and_temples_size 1088 1952 ${num_samples} ${eval_uncertainty}
python eval.py --eval_type mvd --dataset eth3d.mvsanywhere --model cvp_mvsnet_wrapped --inputs poses intrinsics depth_range --output ${OUT_BASE}/cvp_mvsnet_wrapped/robustmvd_benchmark/known_depth_range/eth3d.mvsanywhere.mvd --eval_name known_depth_range --min_source_views 2 --view_ordering nearest ${cpu} --input_size 768 1152 ${num_samples} ${eval_uncertainty}
python eval.py --eval_type robustmvd --model cvp_mvsnet_wrapped --inputs poses intrinsics --output ${OUT_BASE}/cvp_mvsnet_wrapped/robustmvd_benchmark/unknown_depth_range --eval_name unknown_depth_range --min_source_views 2 --view_ordering nearest ${cpu} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 1184 1600 --scannet_size 448 640 --tanks_and_temples_size 1088 1952 ${num_samples} ${eval_uncertainty}
python eval.py --eval_type mvd --dataset eth3d.mvsanywhere --model cvp_mvsnet_wrapped --inputs poses intrinsics --output ${OUT_BASE}/cvp_mvsnet_wrapped/robustmvd_benchmark/unknown_depth_range/eth3d.mvsanywhere.mvd --eval_name unknown_depth_range --min_source_views 2 --view_ordering nearest ${cpu} --input_size 768 1152 ${num_samples} ${eval_uncertainty}
conda deactivate

# patchmatchnet_wrapped model:
conda activate rmvd
# the results from the RMVD paper were produced with --input_resize_backend cv2_bilinear, but for this model it does not make a difference (as it anyways runs at high resolution), therefore we skip it here:
python eval.py --eval_type robustmvd --model patchmatchnet_wrapped --inputs poses intrinsics depth_range --output ${OUT_BASE}/patchmatchnet_wrapped/robustmvd_benchmark/known_depth_range --eval_name known_depth_range ${cpu} --eth3d_size 1792 2688 --kitti_size 376 1280 --dtu_size 1200 1600 --scannet_size 480 640 --tanks_and_temples_size 1140 2048 ${num_samples} ${eval_uncertainty}
python eval.py --eval_type mvd --dataset eth3d.mvsanywhere --model patchmatchnet_wrapped --inputs poses intrinsics depth_range --output ${OUT_BASE}/patchmatchnet_wrapped/robustmvd_benchmark/known_depth_range/eth3d.mvsanywhere.mvd --eval_name known_depth_range ${cpu} --input_size 1792 2688 ${num_samples} ${eval_uncertainty}
python eval.py --eval_type robustmvd --model patchmatchnet_wrapped --inputs poses intrinsics --output ${OUT_BASE}/patchmatchnet_wrapped/robustmvd_benchmark/unknown_depth_range --eval_name unknown_depth_range ${cpu} --eth3d_size 1792 2688 --kitti_size 376 1280 --dtu_size 1200 1600 --scannet_size 480 640 --tanks_and_temples_size 1140 2048 ${num_samples} ${eval_uncertainty}
python eval.py --eval_type mvd --dataset eth3d.mvsanywhere --model patchmatchnet_wrapped --inputs poses intrinsics --output ${OUT_BASE}/patchmatchnet_wrapped/robustmvd_benchmark/unknown_depth_range/eth3d.mvsanywhere.mvd --eval_name unknown_depth_range ${cpu} --input_size 1792 2688 ${num_samples} ${eval_uncertainty}
# additionally evaluate on dtu.robustmvd_paper.mvd to reproduce the results from the RMVD paper:
python eval.py --eval_type mvd --dataset dtu.robustmvd_paper --model patchmatchnet_wrapped --inputs poses intrinsics depth_range --output ${OUT_BASE}/patchmatchnet_wrapped/robustmvd_benchmark/known_depth_range/dtu.robustmvd_paper.mvd --eval_name known_depth_range ${cpu} --input_size 1200 1600 ${num_samples} ${eval_uncertainty}
# note: it is not necessary to evaluate on dtu.robustmvd_paper.mvd for unknown_depth_range, as dtu.robustmvd.mvd and dtu.robustmvd_paper.mvd anyways just differ in the depth range
conda deactivate

# mvsanywhere_wrapped model:
conda activate mvsanywhere
python eval.py --eval_type robustmvd --model mvsanywhere_wrapped --inputs poses intrinsics --output ${OUT_BASE}/mvsanywhere_wrapped/robustmvd_benchmark/unknown_depth_range --eval_name unknown_depth_range ${cpu} --eth3d_size 480 640 --kitti_size 480 1280 --dtu_size 480 640 --scannet_size 480 640 --tanks_and_temples_size 480 640 ${num_samples}
python eval.py --eval_type mvd --dataset eth3d.mvsanywhere --model mvsanywhere_wrapped --inputs poses intrinsics --output ${OUT_BASE}/mvsanywhere_wrapped/robustmvd_benchmark/unknown_depth_range/eth3d.mvsanywhere.mvd --eval_name unknown_depth_range ${cpu} --input_size 480 640 ${num_samples}
python eval.py --eval_type robustmvd --model mvsanywhere_wrapped --inputs poses intrinsics depth_range --output ${OUT_BASE}/mvsanywhere_wrapped/robustmvd_benchmark/known_depth_range --eval_name known_depth_range ${cpu} --eth3d_size 480 640 --kitti_size 480 1280 --dtu_size 480 640 --scannet_size 480 640 --tanks_and_temples_size 480 640 ${num_samples}
python eval.py --eval_type mvd --dataset eth3d.mvsanywhere --model mvsanywhere_wrapped --inputs poses intrinsics depth_range --output ${OUT_BASE}/mvsanywhere_wrapped/robustmvd_benchmark/known_depth_range/eth3d.mvsanywhere.mvd --eval_name known_depth_range ${cpu} --input_size 480 640 ${num_samples}
# additionally evaluate the unknown_depth_range setting with --max_source_views 7 to reproduce the results from MVSAnywhere:
python eval.py --eval_type robustmvd --model mvsanywhere_wrapped --inputs poses intrinsics --output ${OUT_BASE}/mvsanywhere_wrapped/reproduce/robustmvd_benchmark/unknown_depth_range --eval_name unknown_depth_range_reproduce --max_source_views 7 ${cpu} --eth3d_size 480 640 --kitti_size 480 1280 --dtu_size 480 640 --scannet_size 480 640 --tanks_and_temples_size 480 640 ${num_samples}
python eval.py --eval_type mvd --dataset eth3d.mvsanywhere --model mvsanywhere_wrapped --inputs poses intrinsics --output ${OUT_BASE}/mvsanywhere_wrapped/reproduce/robustmvd_benchmark/unknown_depth_range/eth3d.mvsanywhere.mvd --eval_name unknown_depth_range_reproduce --max_source_views 7 ${cpu} --input_size 480 640 ${num_samples}
conda deactivate
