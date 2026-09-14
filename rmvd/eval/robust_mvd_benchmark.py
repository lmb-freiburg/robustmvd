import os
import os.path as osp
from typing import List, Optional, Tuple, Union
import gc

import torch
import pandas as pd

from .multi_view_depth_evaluation import MultiViewDepthEvaluation
from rmvd import create_dataset
from rmvd.utils import prepend_level, logging


class RobustMultiViewDepthBenchmark:
    """Robust Multi-view depth benchmark.

    Evaluates a model on the Robust Multi-view Depth Benchmark.
    Supports multiple input modalities and an optional alignment between predicted and GT depths.
    Can additionally evaluate the estimated uncertainty of the model if available.

    A typical MVS model would set
        inputs=["images", "intrinsics", "poses", "depth_range"], alignment=None.

    A typical depth-from-video model would set
        inputs=["images", "intrinsics"], alignment="median".

    Args:
        out_dir: Directory where results will be written. If None, results are not written to disk.
        inputs: List of input modalities that are supplied to the algorithm.
            Can include: ["images", "intrinsics", "poses", "depth_range"].
            Default: ["images"]
        alignment: Alignment between predicted and ground truth depths. Options are None, "median",
            "pose_scale_mean", "pose_scale_median", "scale_shift_least_squares_invdepth".
            None evaluates predictions without any alignment.
            "median" scales predicted depth maps with the ratio of medians of predicted and ground truth depth maps.
            "pose_scale_*" scales predicted depth maps based on the ratio of the scales of the predicted and ground truth poses.
            "scale_shift_least_squares_invdepth" scales and shifts predicted depth maps to minimize
            the squared error between predicted and ground truth inverse depths.
            Default: None.
        max_source_views: Maximum number of source views to be considered. None means all available source views are
            considered. Default: None.
        min_source_views. Minimum number of source views provided to the model.
            If max_source_views is not None, is set to min(min_source_views, max_source_views). Default: 1.
        view_ordering: Ordering of source views during the evaluation.
            Options are "quasi-optimal" and "nearest". Default: "quasi-optimal".
            "quasi-optimal": evaluate predicted depth maps for all (keyview, sourceview) pairs.
                Order source views according to the prediction accuracy. Increase source view set based on
                the obtained ordering and re-evaluate for each additional source view.
                Log results based on the number of source views. Log best results as overall results.
            "nearest": evaluate predicted depth maps for increasing number of source views. Increase source
                view set based on the ordering of views in the sample, i.e. based on the distance between source
                view indices and the keyview index. Log results based on the number of source views.
                Log best results as overall results.
        eval_uncertainty: Evaluate predicted uncertainty (pred_depth_uncertainty) if available.
            Increases evaluation time.
        sparse_pred: Predicted depth is sparse. Invalid predictions are indicated by 0 values and ignored in
            the evaluation. Defaults to True.
        num_workers: Number of workers for the dataloader. Default: 4.
        verbose: Print evaluation details.
    """
    def __init__(self,
                 out_dir: Optional[str] = None,
                 inputs: Optional[List[str]] = None,
                 alignment: Optional[str] = None,
                 max_source_views: Optional[int] = None,
                 min_source_views: int = 1,
                 view_ordering: Optional[str] = "quasi-optimal",
                 eval_uncertainty: bool = True,
                 sparse_pred: bool = False,
                 num_workers: int = 4,
                 verbose: bool = True,
                 **_
                 ):

        self.verbose = verbose

        self.out_dir = out_dir
        self.log_file_path = osp.join(self.out_dir, "log.txt") if self.out_dir is not None else None
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            logging.add_log_file(self.log_file_path, flush_line=True)

        if self.verbose:
            logging.info(f"Initializing evaluation {self.name}.")

        self.inputs = list(set(inputs + ["images"])) if inputs is not None else ["images"]
        self.alignment = alignment
        self.max_source_views = max_source_views
        self.min_source_views = min_source_views if max_source_views is None else min(min_source_views, max_source_views)
        self.view_ordering = view_ordering if (self.max_source_views is None) or (self.max_source_views > 0) else None
        self.eval_uncertainty = eval_uncertainty
        self.sparse_pred = sparse_pred
        self.num_workers = num_workers

        if self.verbose:
            logging.info(self)
            logging.info(f"Finished initializing evaluation {self.name}.")
            logging.info()
            
    def __del__(self):
        if self.log_file_path is not None:
            logging.remove_log_file(self.log_file_path)

    @property
    def name(self):
        return type(self).__name__

    def __str__(self):
        ret = f"{self.name} with settings:"
        ret += f"\n\tInputs: {self.inputs}"
        ret += f"\n\tAlignment: {self.alignment}"
        ret += f"\n\tMax source views: {self.max_source_views}"
        ret += f"\n\tMin source views: {self.min_source_views}"
        ret += f"\n\tEvaluate uncertainty: {self.eval_uncertainty}"
        ret += f"\n\tPredicted depth is sparse: {self.sparse_pred}"
        if self.out_dir is not None:
            ret += f"\n\tOutput directory: {self.out_dir}"
        else:
            ret += "\n\tOutput directory: None. Results will not be written to disk!"
        return ret

    @torch.no_grad()
    def __call__(self,
                 model,
                 eth3d_size: Optional[Tuple[int, int]] = None,
                 kitti_size: Optional[Tuple[int, int]] = None,
                 dtu_size: Optional[Tuple[int, int]] = None,
                 scannet_size: Optional[Tuple[int, int]] = None,
                 tanks_and_temples_size: Optional[Tuple[int, int]] = None,
                 input_resize_backend: Optional[str] = None,
                 samples: Optional[Union[int, List[int]]] = None,
                 qualitatives: Optional[Union[int, List[int]]] = 2,
                 eval_name: Optional[str] = None,
                 finished_iterations: Optional[int] = None,
                **_):
        """Run Robust Multi-view Depth Benchmark evaluation for a model.

        Args:
            model: Class or function that applies the model to a sample.
            eth3d_size: Input image size on ETH3D in the format (height, width).
                If not provided, scales images up to the nearest size that works with the model.
            kitti_size: Input image size on KITTI in the format (height, width).
                If not provided, scales images up to the nearest size that works with the model.
            dtu_size: Input image size on DTU in the format (height, width).
                If not provided, scales images up to the nearest size that works with the model.
            scannet_size: Input image size on ScanNet in the format (height, width).
                If not provided, scales images up to the nearest size that works with the model.
            tanks_and_temples_size: Input image size on Tanks and Temples in the format (height, width).
                If not provided, scales images up to the nearest size that works with the model.
            input_resize_backend: Backend that is used to resize the input images to the specified input size.
                Only used for datasets where an input size is specified. Default: skimage_bilinear_antialias.
            samples: Integer that indicates the number of samples that should be evaluated or list that indicates
                the indices of samples that should be evaluated. None evaluates all samples.
            qualitatives: Integer that indicates the number of qualitatives that should be logged or list that indicates
                the indices of samples for which qualitatives should be logged. -1 logs qualitatives for all samples.
            eval_name: Name of the experiment. Optional.
            finished_iterations: Number of iterations that the model has been trained for. Optional.

        Returns:
            Results of the evaluation.
        """

        datasets = [("kitti.robustmvd.mvd", kitti_size),
                    ("dtu.robustmvd.mvd", dtu_size),
                    ("scannet.robustmvd.mvd", scannet_size),
                    ("tanks_and_temples.robustmvd.mvd", tanks_and_temples_size),
                    ("eth3d.robustmvd.mvd", eth3d_size),]

        results = []

        for dataset_name, input_size in datasets:
            logging.info(f"Running evaluation on {dataset_name}.")

            if self.out_dir is not None:
                out_dir = osp.join(self.out_dir, dataset_name)
                os.makedirs(out_dir, exist_ok=True)
            else:
                out_dir = None

            eval = MultiViewDepthEvaluation(out_dir=out_dir, inputs=self.inputs, alignment=self.alignment,
                                            view_ordering=self.view_ordering, max_source_views=self.max_source_views,
                                            min_source_views=self.min_source_views,
                                            eval_uncertainty=self.eval_uncertainty, clip_pred_depth=True,
                                            sparse_pred=self.sparse_pred, num_workers=self.num_workers,
                                            verbose=self.verbose)

            dataset = create_dataset(dataset_name_or_path=dataset_name, data_format="mvd", input_size=input_size,
                                     input_resize_backend=input_resize_backend)
            result = eval(dataset=dataset, model=model, samples=samples, qualitatives=qualitatives, burn_in_samples=3,
                          eval_name=eval_name, finished_iterations=finished_iterations)
            result = prepend_level(result, "dataset", dataset_name, axis=1)
            results.append(result)
            logging.info()
            del eval ; gc.collect()

        results = pd.concat(results, axis=1)
        self._output_results(results, self.out_dir)
        return results

    def _output_results(self, results, out_dir):
        num_source_view_results = results.drop('best', axis=1, level=1).mean()
        results = results.loc[:, (slice(None), 'best')].droplevel(level=1, axis=1).mean()

        if self.verbose:
            logging.info()
            logging.info("Robust MVD Benchmark Results:")
            logging.info(results)

        if out_dir is not None:

            if self.verbose:
                logging.info(f"Writing Robust MVD Benchmark results to {out_dir}.")

            results.to_csv(osp.join(out_dir, "results.csv"))
            results.to_pickle(osp.join(out_dir, "results.pickle"))
            num_source_view_results.to_csv(osp.join(out_dir, "num_source_view_results.csv"))
            num_source_view_results.to_pickle(osp.join(out_dir, "num_source_view_results.pickle"))
