#!/bin/bash

python eval.py --model robust_mvd --eval_type robustmvd --inputs poses intrinsics --output /tmp/eval_benchmark --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280
python eval.py --model robust_mvd_5M --eval_type robustmvd --inputs poses intrinsics --output /tmp/eval_benchmark --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280
python eval.py --model monodepth2_mono_stereo_1024x320_wrapped --eval_type robustmvd --output /tmp/eval_benchmark --max_source_views 0 --alignment median
python eval.py --model monodepth2_mono_stereo_640x192_wrapped --eval_type robustmvd --output /tmp/eval_benchmark --max_source_views 0 --alignment median