import os
import os.path as osp
import time
from copy import deepcopy
from typing import Optional, Sequence, Tuple, Union
import warnings

import torch
import skimage.transform
import numpy as np
import pandas as pd

from robd.utils import numpy_collate
from .metrics import m_rel_ae, pointwise_rel_ae, thresh_inliers, sparsification


# TODO: add tensorboard logging


class MultiViewDepthEvaluation:
    """Multi-view depth evaluation.

    Can be applied to a dataset and model to evaluate the depth estimation performance of the model on the dataset.
    Supports multiple evaluation settings, including MVS and depth-from-video.
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
        view_ordering: Ordering of source views during the evaluation.
            Options are "quasi-optimal", "nearest" and None. Default: "quasi-optimal".
            None: supply all source views to the model and evaluate predicted depth map.
            "quasi-optimal": evaluate predicted depth maps for all (keyview, sourceview) pairs.
                Order source views according to the prediction accuracy. Increase source view set based on
                the obtained ordering and re-evaluate for each additional source view.
                Log results based on the number of source views. Log best results as overall results.
            "nearest": evaluate predicted depth maps for increasing number of source views. Increase source
                view set based on the ordering of views in the sample, i.e. based on the distance between source
                view indices and the keyview index. Log results based on the number of source views.
                Log best results as overall results.
        max_source_views: Maximum number of source views to be considered in case view_ordering is
            "quasi-optimal" or "nearest". None means all available source views are considered.
        eval_uncertainty: Evaluate predicted uncertainty (pred_depth_uncertainty) if available.
            Increases evaluation time.
        clip_pred_depth: Clip model predictions before evaluation to a reasonable range. This makes sense to reduce
            the effect of unreasonable outliers. Defaults to True.
        sparse_pred: Predicted depth is sparse. Invalid predictions are indicated by 0 values and ignored in
            the evaluation. Defaults to True.
        verbose: Print evaluation details.
    """
    def __init__(self,
                 out_dir: Optional[str]=None,
                 inputs: Sequence[str]=None,
                 alignment: Optional[str]=None,
                 view_ordering: str = "quasi-optimal",
                 max_source_views: Optional[int]=None,
                 eval_uncertainty: bool=True,
                 clip_pred_depth: Union[bool, Tuple[float, float], None] = (0.1, 100),
                 sparse_pred: bool=False,
                 verbose: bool=True,
                 ):

        self.verbose = verbose

        if self.verbose:
            print(f"Initializing evaluation {self.name}.")

        self.out_dir = out_dir
        if self.out_dir is not None:
            self.results_dir = osp.join(self.out_dir, "results")
            self.sample_results_dir = osp.join(self.results_dir, "per_sample")
            self.qualitatives_dir = osp.join(self.out_dir, "qualitatives")
            os.makedirs(self.out_dir, exist_ok=True)
            os.makedirs(self.results_dir, exist_ok=True)
            os.makedirs(self.sample_results_dir, exist_ok=True)
            os.makedirs(self.qualitatives_dir, exist_ok=True)

        self.inputs = list(set(inputs + ["images"])) if inputs is not None else ["images"]
        self.alignment = alignment
        self.view_ordering = view_ordering
        self.max_source_views = max_source_views
        self.eval_uncertainty = eval_uncertainty
        self.clip_pred_depth = clip_pred_depth
        self.sparse_pred = sparse_pred

        # will be set/used in __call__:
        self.dataset = None
        self.dataloader = None
        self.model = None
        self.sample_indices = None
        self.qualitative_indices = None
        self.burn_in_samples = None
        self.cur_sample_num = 0
        self.cur_sample_idx = 0
        self.results = None
        self.sparsification_curves = None

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
        ret += f"\n\tView ordering: {self.view_ordering}"
        ret += f"\n\tMax source views: {self.max_source_views}"
        ret += f"\n\tEvaluate uncertainty: {self.eval_uncertainty}"
        ret += f"\n\tClip predicted depth: {self.clip_pred_depth}"
        ret += f"\n\tPredicted depth is sparse: {self.sparse_pred}"
        if self.out_dir is not None:
            ret += f"\n\tOutput directory: {self.out_dir}"
        else:
            ret += "\n\tOutput directory: None. Results will not be written to disk!"
        return ret

    def __call__(self,
                 dataset,
                 model,
                 input_adapter_fct=None,
                 output_adapter_fct=None,
                 samples: Optional[Union[int, Sequence[int]]]=None,
                 qualitatives: Union[int, Sequence[int]]=10,
                 burn_in_samples: int=3,):
        """Run depth evaluation for a dataset and model.

        Args:
            dataset: Dataset for the evaluation.
            model: Class or function that applies the model to a sample.
            input_adapter_fct: Function that adapts sample format of the evaluation to the format required by the model.
                Will be excluded from the model runtime measurements. Optional.
            output_adapter_fct: Function that adapts output of the model to the format of the evaluation.
                Will be excluded from the model runtime measurements. Optional.
            samples: Integer that indicates the number of samples that should be evaluated or list that indicates
                the indices of samples that should be evaluated. None evaluates all samples.
            qualitatives: Integer that indicates the number of qualitatives that should be logged or list that indicates
                the indices of samples for which qualitatives should be logged. -1 logs qualitatives for all samples.
            burn_in_samples: Number of samples that will not be considered for runtime/memory measurements.
                Defaults to 3.

        Returns:
            Results of the evaluation.
        """
        self._init_evaluation(dataset=dataset, model=model, samples=samples, qualitatives=qualitatives,
                              burn_in_samples=burn_in_samples)
        results = self._evaluate()
        self._output_results()
        self._reset_evaluation()
        return results

    def _init_evaluation(self,
                         dataset,
                         model,
                         samples=None,
                         qualitatives=10,
                         burn_in_samples=3,):
        self.dataset = dataset
        self.model = model
        self._init_sample_indices(samples=samples)
        self._init_qualitative_indices(qualitatives=qualitatives)
        self._init_results()
        self.burn_in_samples = burn_in_samples
        self.dataloader = self.dataset.get_loader(batch_size=1, indices=self.sample_indices, num_workers=0,
                                                  collate_fn=numpy_collate)
        # we intentionally fix a batch_size=1, to ensure comparable runtimes
        # TODO: write artifacts, e.g. layout, ..

    def _init_sample_indices(self, samples):
        if isinstance(samples, list):
            self.sample_indices = samples
            if self.verbose:
                print(f"Evaluating samples with indices: {self.sample_indices}.")
        elif isinstance(samples, int) and samples > 0:
            step_size = len(self.dataset) / samples  # <=1
            self.sample_indices = [int(i*step_size) for i in range(samples)]
            if self.verbose:
                print(f"Evaluating samples with indices: {self.sample_indices}.")
        else:
            self.sample_indices = list(range(len(self.dataset)))

    def _init_qualitative_indices(self, qualitatives=None):
        if qualitatives is None:
            self.qualitative_indices = []
        if qualitatives < 0:
            self.qualitative_indices = self.sample_indices
        elif isinstance(qualitatives, list):
            self.qualitative_indices = qualitatives
        elif isinstance(qualitatives, int):
            step_size = len(self.sample_indices) / qualitatives  # <=1
            self.qualitative_indices = list(set([self.sample_indices[int(i*step_size)] for i in range(qualitatives)]))

    def _evaluate(self):
        eval_start = time.time()
        for sample_num, (sample_idx, sample) in enumerate(zip(self.sample_indices, self.dataloader)):
            self.cur_sample_num = sample_num
            self.cur_sample_idx = sample_idx

            if self.verbose:
                print(f"Processing sample {self.cur_sample_num+1} / {len(self.sample_indices)} "
                      f"(index: {self.cur_sample_idx}):")

            # prepare sample:
            should_qualitative = (self.cur_sample_idx in self.qualitative_indices) and (self.out_dir is not None)
            keyview_idx = int(sample['keyview_idx'])
            sample_inputs, sample_gt = self._inputs_and_gt_from_sample(sample)

            # get evaluation order:
            ordered_source_indices = self._get_source_view_ordering(sample_inputs=sample_inputs, sample_gt=sample_gt)
            max_source_views = min(len(ordered_source_indices), self.max_source_views) \
                if self.max_source_views is not None else len(ordered_source_indices)
            min_source_views = 1 if self.view_ordering is not None else max_source_views

            best_metrics = None
            best_num_source_views = np.nan
            best_pred = None

            for num_source_views in range(min_source_views, max_source_views+1):
                cur_source_indices = ordered_source_indices[:num_source_views]
                cur_view_indices = list(sorted([keyview_idx] + cur_source_indices))
                cur_keyview_idx = cur_view_indices.index(keyview_idx)

                if self.verbose:
                    print(f"\tEvaluating with {num_source_views} / {len(ordered_source_indices)} source views:")
                    print(f"\t\tSource view indices: {cur_source_indices}.")

                self._reset_memory_stats()

                # construct current input sample:
                cur_sample_gt = deepcopy(sample_gt)
                cur_sample_inputs = deepcopy(sample_inputs)
                if "images" in self.inputs:
                    cur_sample_inputs['images'] = [cur_sample_inputs['images'][i] for i in cur_view_indices]
                if "poses" in self.inputs:
                    cur_sample_inputs['poses'] = [cur_sample_inputs['poses'][i] for i in cur_view_indices]
                if "intrinsics" in self.inputs:
                    cur_sample_inputs['intrinsics'] = [cur_sample_inputs['intrinsics'][i] for i in cur_view_indices]
                cur_sample_inputs['keyview_idx'] = np.array([cur_keyview_idx])
                # depth_range is not changed

                # run model:
                pred, runtimes, gpu_mem = self._run_model(cur_sample_inputs)
                self._postprocess_sample_and_output(cur_sample_inputs, cur_sample_gt, pred)

                # compute and log metrics:
                metrics = self._compute_metrics(sample_inputs=cur_sample_inputs,
                                                sample_gt=cur_sample_gt,
                                                pred=pred)
                metrics.update(runtimes)
                metrics.update(gpu_mem)
                self._log_metrics(metrics, num_source_views)

                if self.verbose:
                    print(f"\t\tAbsrel={metrics['absrel']}.")

                if np.isfinite(metrics['absrel']) and \
                        (best_metrics is None or metrics['absrel'] < best_metrics['absrel']):
                    best_metrics = metrics
                    best_num_source_views = num_source_views
                    best_pred = pred

            # compute and log qualitatives:
            if should_qualitative:
                qualitatives = self._compute_qualitatives(sample_inputs=sample_inputs, sample_gt=sample_gt,
                                                          pred=best_pred)
                self._log_qualitatives(qualitatives)

            # log best metrics for this sample:
            self._log_metrics(best_metrics, 'best')
            self._log_metrics({'num_views': best_num_source_views}, 'best')

            if self.eval_uncertainty:
                uncertainty_metrics = self._compute_uncertainty_metrics(sample_inputs=cur_sample_inputs,
                                                                        sample_gt=cur_sample_gt,
                                                                        pred=best_pred)
                self._log_metrics(uncertainty_metrics, 'best')
                # TODO reorder qualitatives, logging and uncertainty evaluation; maybe first put uncertainty metrics into best_metrics

            if self.verbose:
                print(f"Test sample #{self.cur_sample_idx} has AbsRel={best_metrics['absrel']} "
                      f"with {best_num_source_views} source views.\n")

        eval_end = time.time()
        print(f"Evaluation took {eval_end - eval_start} seconds.")

        return self.results

    def _compute_qualitatives(self, sample_inputs, sample_gt, pred):

        gt_depth = sample_gt['depth'][0, 0]

        pred_depth = pred['depth'][0, 0]
        pred_invdepth = pred['invdepth'][0, 0]
        pred_mask = pred_depth != 0 if self.sparse_pred else np.ones_like(pred_depth, dtype=bool)

        pointwise_absrel = pointwise_rel_ae(gt=gt_depth, pred=pred_depth, mask=pred_mask)

        qualitatives = {'pointwise_absrel': pointwise_absrel,
                        'pred_depth': pred_depth,
                        'pred_invdepth': pred_invdepth}

        if self.eval_uncertainty:
            qualitatives['pred_depth_uncertainty'] = pred['depth_uncertainty'][0, 0]

        return qualitatives

    def _log_qualitatives(self, qualitatives):

        for qualitative_name, qualitative in qualitatives.items():
            out_path = osp.join(self.qualitatives_dir, f'{self.cur_sample_idx:07d}-{qualitative_name}.npy')
            np.save(out_path, qualitative)

            # TODO self.add_update(sample_num, qualitative_name, out_path, is_info=False)

    # def _add_update(self, sample_num, update_name, update_value, is_info=False):
    #     update_name = update_name if not is_info else "_" + update_name
    #
    #     if sample_num in self.cur_updates:
    #         self.cur_updates[sample_num][update_name] = update_value
    #     elif not is_info:  # we dont create an update for a sample num when there is only info data
    #         self.cur_updates[sample_num] = {update_name: update_value}
    #
    # def _output_dataset_cfg(self):
    #
    #     with open(self.test_dataset_updates_path, 'wb') as dataset_udpates_file:
    #         pickle.dump(self.cur_updates, dataset_udpates_file)
    #
    #     dataset_layout = self.cur_test.get_layout()
    #     dataset_layout.write(self.test_dataset_layout_path)
    #
    #     dataset_cls_name = SCHEDULE.cur_test.dataset_cls_name
    #     self.cur_test_dataset.write_config(path=self.test_dataset_cfg_path, dataset_cls_name=dataset_cls_name,
    #                                        updates=[self.test_dataset_updates_path], update_strict=True,
    #                                        layouts=[self.test_dataset_layout_path])

    def _reset_evaluation(self):
        self.dataset = None
        self.dataloader = None
        self.model = None
        self.sample_indices = None
        self.qualitative_indices = None
        self.burn_in_samples = None
        self.cur_sample_idx = 0
        self.cur_sample_num = 0
        self.results = None
        self.sparsification_curves = None

    def _init_results(self):
        self.results = pd.DataFrame()
        self.results.index.name = 'sample_idx'
        self.results.columns.name = 'metric'
        self.results = pd.concat({1: self.results}, axis=1, names=["num_views"])

        if self.eval_uncertainty:
            x = np.linspace(0, 0.99, 100)
            columns = pd.Index(x, name="frac_removed")
            index = pd.MultiIndex.from_tuples([], names=("sample_idx", "curve"))
            self.sparsification_curves = pd.DataFrame(columns=columns, index=index)

    def _get_source_view_ordering(self, sample_inputs, sample_gt):
        if self.view_ordering == 'quasi-optimal':
            return self._get_quasi_optimal_source_view_ordering(sample_inputs=sample_inputs, sample_gt=sample_gt)
        else:
            return self._get_nearest_source_view_ordering(sample_inputs=sample_inputs, sample_gt=sample_gt)

    def _get_nearest_source_view_ordering(self, sample_inputs, sample_gt):
        keyview_idx = int(sample_inputs['keyview_idx'])
        source_indices = [idx for idx in range(len(sample_inputs['images'])) if idx != keyview_idx]
        source_view_ordering = sorted(source_indices, key=lambda x: np.abs(x-keyview_idx))
        return source_view_ordering

    def _get_quasi_optimal_source_view_ordering(self, sample_inputs, sample_gt):
        keyview_idx = int(sample_inputs['keyview_idx'])
        source_indices = [idx for idx in range(len(sample_inputs['images'])) if idx != keyview_idx]
        source_scores = {}

        for source_idx in source_indices:
            # construct temporary sample with a single source view:
            cur_sample_inputs = deepcopy(sample_inputs)
            cur_sample_gt = deepcopy(sample_gt)
            cur_sample_inputs['images'] = [cur_sample_inputs['images'][keyview_idx],
                                           cur_sample_inputs['images'][source_idx]]
            cur_sample_inputs['poses'] = [cur_sample_inputs['poses'][keyview_idx],
                                          cur_sample_inputs['poses'][source_idx]]
            cur_sample_inputs['intrinsics'] = [cur_sample_inputs['intrinsics'][keyview_idx],
                                               cur_sample_inputs['intrinsics'][source_idx]]
            cur_sample_inputs['keyview_idx'] = np.array([0])

            # run model:
            pred, _, _ = self._run_model(cur_sample_inputs)
            self._postprocess_sample_and_output(cur_sample_inputs, cur_sample_gt, pred)

            # compute absrel for using current source view:
            metrics = self._compute_metrics(sample_inputs=cur_sample_inputs,
                                            sample_gt=cur_sample_gt, pred=pred)
            source_scores[source_idx] = metrics['absrel']

        source_view_ordering = sorted(source_scores, key=source_scores.get)
        return source_view_ordering

    def _reset_memory_stats(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

    def _inputs_and_gt_from_sample(self, sample):
        is_input = lambda key: key in self.inputs or key == "keyview_idx"
        sample_inputs = {key: val for key, val in sample.items() if is_input(key)}
        sample_gt = {key: val for key, val in sample.items() if not is_input(key)}
        return sample_inputs, sample_gt

    def _postprocess_sample_and_output(self, sample_inputs, sample_gt, pred):

        gt_depth = sample_gt['depth']

        pred_depth = pred['depth']
        pred_depth = skimage.transform.resize(pred_depth, gt_depth.shape, order=0, anti_aliasing=False)

        pred_mask = pred_depth != 0 if self.sparse_pred else np.ones_like(pred_depth, dtype=bool)
        gt_mask = gt_depth > 0

        if self.alignment == "median":
            mask = gt_mask & pred_mask
            ratio = np.median(gt_depth[mask]) / np.median(pred_depth[mask])

            if mask.any() and np.isfinite(ratio):
                pred_depth = pred_depth * ratio
            else:
                ratio = 0

            pred['scaling_factor'] = ratio

        if self.clip_pred_depth:
            pred_depth = np.clip(pred_depth, 0.1, 100) * pred_mask

        with np.errstate(divide='ignore', invalid='ignore'):
            pred_invdepth = np.nan_to_num(1 / pred_depth, nan=0, posinf=0, neginf=0)

        if 'depth_uncertainty' in pred:
            pred_depth_uncertainty = pred['depth_uncertainty']
            pred_depth_uncertainty = skimage.transform.resize(pred_depth_uncertainty, gt_depth.shape, order=0,
                                                              anti_aliasing=False)
            pred['depth_uncertainty'] = pred_depth_uncertainty

        pred['depth'] = pred_depth
        pred['invdepth'] = pred_invdepth

    def _run_model(self, sample_inputs):
        start_model_and_io = time.time()

        if hasattr(self.model, "input_adapter"):
            sample_inputs = self.model.input_adapter(**sample_inputs)

        start_model = time.time()
        pred = self.model(**sample_inputs)
        end_model = time.time()

        if hasattr(self.model, "output_adapter"):
            pred, _ = self.model.output_adapter(pred)

        end_model_and_io = time.time()

        time_and_mem_valid = self.cur_sample_num >= self.burn_in_samples
        runtime_model_in_sec = end_model - start_model if time_and_mem_valid else np.nan
        runtime_model_and_io_in_sec = end_model_and_io - start_model_and_io if time_and_mem_valid else np.nan
        runtimes = {'runtime_model_in_sec': runtime_model_in_sec,
                    'runtime_model_in_msec': 1000 * runtime_model_in_sec,
                    'runtime_model_and_io_in_sec': runtime_model_and_io_in_sec,
                    'runtime_model_and_io_in_msec': 1000 * runtime_model_and_io_in_sec,}

        gpu_mem_alloc = int(torch.cuda.max_memory_allocated() / 1024 / 1024) if time_and_mem_valid else np.nan
        gpu_mem_reserved = int(torch.cuda.max_memory_reserved() / 1024 / 1024) if time_and_mem_valid else np.nan
        gpu_mem = {'gpu_mem_alloc_in_mib': gpu_mem_alloc, 'gpu_mem_alloc_in_mib': gpu_mem_reserved}

        return pred, runtimes, gpu_mem

    def _compute_metrics(self, sample_inputs, sample_gt, pred):

        gt_depth = sample_gt['depth'][0, 0]
        pred_depth = pred['depth'][0, 0]
        eval_mask = pred_depth != 0 if self.sparse_pred else np.ones_like(pred_depth, dtype=bool)

        absrel = m_rel_ae(gt=gt_depth, pred=pred_depth, mask=eval_mask, output_scaling_factor=100.0)
        inliers103 = thresh_inliers(gt=gt_depth, pred=pred_depth, thresh=1.03, mask=eval_mask, output_scaling_factor=100.0)

        metrics = {'absrel': absrel, 'inliers103': inliers103,}

        if self.alignment == "median":
            scale = pred['scaling_factor']
            metrics['scaling_factor'] = scale if scale != 0 else np.nan

        metrics['pred_depth_density'] = np.sum(eval_mask) / eval_mask.size * 100

        return metrics

    def _log_metrics(self, metrics, sum_source_views):

        for metric, val in metrics.items():
            self.results.loc[self.cur_sample_idx, (sum_source_views, metric)] = val

    def _compute_uncertainty_metrics(self, sample_inputs, sample_gt, pred):
        if self.verbose:
            print("\tComputing uncertainty metrics:")

        gt_depth = sample_gt['depth'][0, 0]

        pred_depth = pred['depth'][0, 0]
        pred_depth_uncertainty = pred['depth_uncertainty'][0, 0]
        pred_mask = pred_depth != 0 if self.sparse_pred else np.ones_like(pred_depth, dtype=bool)

        oracle_uncertainty = pointwise_rel_ae(gt=gt_depth, pred=pred_depth, mask=pred_mask)

        sparsification_oracle = sparsification(gt=gt_depth, pred=pred_depth, uncertainty=oracle_uncertainty,
                                               mask=pred_mask, show_pbar=self.verbose,
                                               pbar_desc="            Oracle sparsification")
        sparsification_pred = sparsification(gt=gt_depth, pred=pred_depth, uncertainty=pred_depth_uncertainty,
                                             mask=pred_mask, show_pbar=self.verbose,
                                             pbar_desc="            Prediction sparsification")
        sparsification_errors = sparsification_pred - sparsification_oracle
        ause = sparsification_errors.sum(skipna=False) / 100
        ause = ause if np.isfinite(ause) else np.nan

        self.sparsification_curves.loc[(self.cur_sample_idx, "oracle"), :] = sparsification_oracle
        self.sparsification_curves.loc[(self.cur_sample_idx, "pred"), :] = sparsification_pred
        self.sparsification_curves.loc[(self.cur_sample_idx, "error"), :] = sparsification_errors

        if self.verbose:
            print(f"\t\t\tAUSE={ause}.")

        return {'ause': ause}

    def _output_results(self):

        results_per_sample = self.results['best']
        results = results_per_sample.mean()

        num_source_view_results_per_sample = self.results.drop('best', axis=1, level=0)
        num_source_view_results = num_source_view_results_per_sample.mean()

        if self.verbose:
            print()
            print("Results:")
            print(results)

        if self.out_dir is not None:
            results_per_sample.to_pickle(osp.join(self.sample_results_dir, "results.pickle"))
            results_per_sample.to_csv(osp.join(self.sample_results_dir, "results.csv"))
            results.to_pickle(osp.join(self.results_dir, "results.pickle"))
            results.to_csv(osp.join(self.results_dir, "results.csv"))

            num_source_view_results_per_sample.to_csv(osp.join(self.sample_results_dir, "num_source_view_results.csv"))
            num_source_view_results_per_sample.to_pickle(osp.join(self.sample_results_dir, "num_source_view_results.pickle"))
            num_source_view_results.to_csv(osp.join(self.results_dir, "num_source_view_results.csv"))
            num_source_view_results.to_pickle(osp.join(self.results_dir, "num_source_view_results.pickle"))

            if self.eval_uncertainty:
                sample_sparsification_curves = self.sparsification_curves
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    sparsification_curves = sample_sparsification_curves.mean(axis=0, level=1)

                sparsification_curves.to_pickle(osp.join(self.results_dir, "sparsification_curves.pickle"))
                sparsification_curves.to_csv(osp.join(self.results_dir, "sparsification_curves.csv"))
                sample_sparsification_curves.to_pickle(osp.join(self.sample_results_dir, "sparsification_curves.pickle"))
                sample_sparsification_curves.to_csv(osp.join(self.sample_results_dir, "sparsification_curves.csv"))
