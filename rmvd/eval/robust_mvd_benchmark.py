import os
import os.path as osp
from typing import Optional, Sequence, Tuple, Union

import torch

from .multi_view_depth_evaluation import MultiViewDepthEvaluation
from rmvd import create_dataset


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
        alignment: Alignment between predicted and ground truth depths. Options are None, "median", "translation".
            None evaluates predictions without any alignment.
            "median" scales predicted depth maps with the ratio of medians of predicted and ground truth depth maps.
            "translation" scales predicted depth maps with the ratio of the predicted and ground truth translation.
        max_source_views: Maximum number of source views to be considered in case view_ordering is
            "quasi-optimal" or "nearest". None means all available source views are considered.
        eval_uncertainty: Evaluate predicted uncertainty (pred_depth_uncertainty) if available.
            Increases evaluation time.
        sparse_pred: Predicted depth is sparse. Invalid predictions are indicated by 0 values and ignored in
            the evaluation. Defaults to True.
        verbose: Print evaluation details.
    """
    def __init__(self,
                 out_dir: Optional[str] = None,
                 inputs: Sequence[str] = None,
                 alignment: Optional[str] = None,
                 max_source_views: Optional[int] = None,
                 eval_uncertainty: bool = True,
                 sparse_pred: bool = False,
                 verbose: bool = True,
                 ):

        self.verbose = verbose

        if self.verbose:
            print(f"Initializing evaluation {self.name}.")

        self.out_dir = out_dir
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)

        self.inputs = inputs
        self.alignment = alignment
        self.max_source_views = max_source_views
        self.eval_uncertainty = eval_uncertainty
        self.sparse_pred = sparse_pred

        if self.verbose:
            print(self)
            print(f"Finished initializing evaluation {self.name}.")
            print()

    @property
    def name(self):
        return type(self).__name__

    def __str__(self):
        ret = f"{self.name} with settings:"
        ret += f"\n\tInputs: {self.inputs}"
        ret += f"\n\tAlignment: {self.alignment}"
        ret += f"\n\tMax source views: {self.max_source_views}"
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
                 qualitatives: Union[int, Sequence[int]] = 2,
                 **_):
        """Run benchmark evaluation for a model.

        Args:
            model: Class or function that applies the model to a sample.
            qualitatives: Integer that indicates the number of qualitatives that should be logged or list that indicates
                the indices of samples for which qualitatives should be logged. -1 logs qualitatives for all samples.

        Returns:
            Results of the evaluation.
        """
        if self.out_dir is not None:
            model_dir = osp.join(self.out_dir, model.name)
            os.makedirs(model_dir, exist_ok=True)

        datasets = [("eth3d.robustmvd.mvd", eth3d_size), ("kitti.robustmvd.mvd", kitti_size),
                    ("dtu.robustmvd.mvd", dtu_size)]

        for dataset_name, input_size in datasets:
            print(f"Running evaluation on {dataset_name}.")

            if self.out_dir is not None:
                out_dir = osp.join(model_dir, dataset_name)
                os.makedirs(out_dir, exist_ok=True)
            else:
                out_dir = None

            eval = MultiViewDepthEvaluation(out_dir=out_dir, inputs=self.inputs, alignment=self.alignment,
                                            view_ordering="quasi-optimal", max_source_views=self.max_source_views,
                                            eval_uncertainty=self.eval_uncertainty, clip_pred_depth=True,
                                            sparse_pred=self.sparse_pred, verbose=self.verbose)

            dataset = create_dataset(dataset_name=dataset_name, dataset_type="mvd", input_size=input_size)
            eval(dataset=dataset, model=model, qualitatives=qualitatives, burn_in_samples=3)
        # TODO: add return value
