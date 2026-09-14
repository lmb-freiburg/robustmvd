import os
import os.path as osp
from copy import deepcopy
from typing import List, Optional, Tuple, Union
import pickle

import torch
import skimage.transform
import numpy as np
import tqdm

from rmvd.utils import numpy_collate, vis, select_by_index, exclude_index, writer, logging, trans_from_transform, invert_transform
from rmvd.data.layout import Layout, Visualization
from rmvd.data.updates import Update


class MultiViewDepthInference:
    """Multi-view depth inference.

    Can be applied to a dataset and model to run multi-view depth inference on all samples in the dataset.
    Supports multiple input modalities.

    A typical MVS model would set
        inputs=["images", "intrinsics", "poses", "depth_range"].

    A typical depth-from-video model would set
        inputs=["images", "intrinsics"].

    A typical depth-from-single-view model would set
        inputs=["images"], max_source_views=0.
        
    An optional alignment between predicted and GT depths is supported (however, some alignments require the dataset to have ground truth depth available).

    Args:
        inputs: List of input modalities that are supplied to the model.
            Can include: ["images", "intrinsics", "poses", "depth_range"].
            Default: ["images"]
        max_source_views: Maximum umber of source views to be fed to the model. Default: None (means all source views).
        clip_pred_depth: Clip model predictions before evaluation to a reasonable range. This makes sense to reduce
            the effect of unreasonable outliers. Defaults to True.
        alignment: Alignment between predicted and ground truth depths. Options are None, "median",
            "pose_scale_mean", "pose_scale_median", "scale_shift_least_squares_invdepth".
            None outputs predictions without any alignment.
            "median" scales predicted depth maps with the ratio of medians of predicted and ground truth depth maps.
            "pose_scale_*" scales predicted depth maps based on the ratio of the scales of the predicted and ground truth poses.
            "scale_shift_least_squares_invdepth" scales and shifts predicted depth maps to minimize
            the squared error between predicted and ground truth inverse depths.
            Default: None.
        sparse_pred: Predicted depth is sparse. Defaults to False.
        out_dir: Directory where results will be written. Only used when running inference on a whole dataset. If None, results are not written to disk.
        num_workers: Number of workers for the dataloader. Default: 4.
        verbose: Print inference details.
    """
    def __init__(self,
                 inputs: Optional[List[str]] = None,
                 max_source_views: Optional[int] = None,
                 clip_pred_depth: Union[bool, Tuple[float, float], List[float]] = True,
                 alignment: Optional[str] = None,
                 sparse_pred: bool = False,
                 out_dir: Optional[str] = None,
                 num_workers: int = 4,
                 verbose: bool = False,
                 **_
                 ):

        self.verbose = verbose

        if self.verbose:
            logging.info(f"Initializing inference {self.name}.")
        
        self.out_dir = out_dir
        self.log_file_path = osp.join(self.out_dir, "log.txt") if self.out_dir is not None else None
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            logging.add_log_file(self.log_file_path, flush_line=True)

        self.inputs = list(set(inputs + ["images"])) if inputs is not None else ["images"]
        self.max_source_views = max_source_views
        self.clip_pred_depth = clip_pred_depth
        assert alignment in [None, "median", "scale_shift_least_squares_invdepth", "pose_scale_mean", "pose_scale_median"]
        self.alignment = alignment
        self.sparse_pred = sparse_pred
        self.num_workers = num_workers

        # will be set/used in __call__ / run_on_dataset:
        self.dataset = None
        self.dataloader = None
        self.model = None
        self.sample_indices = None
        self.cur_sample_num = 0
        self.cur_sample_idx = 0
        self.dataset_updates = None
        self.pred_has_uncertainty = None

        if self.verbose:
            logging.info(self)
            logging.info(f"Finished initializing inference {self.name}.")
            logging.info()
            
    def __del__(self):
        if self.log_file_path is not None and logging is not None:
            logging.remove_log_file(self.log_file_path)

    @property
    def name(self):
        return type(self).__name__

    def __str__(self):
        ret = f"{self.name} with settings:"
        ret += f"\n\tInputs: {self.inputs}"
        ret += f"\n\tMax source views: {'All' if self.max_source_views is None else self.max_source_views}"
        ret += f"\n\tClip predicted depth: {self.clip_pred_depth}"
        if self.alignment is not None:
            ret += f"\n\tAlignment: {self.alignment}"
        if self.sparse_pred:
            ret += f"\n\tPredicted depth is sparse: {self.sparse_pred}"
        if self.out_dir is not None:
            ret += f"\n\tOutput directory: {self.out_dir}"
        return ret
    
    @torch.no_grad()
    def __call__(self,
                 model,
                 sample,
                 **_):
        if self.out_dir is not None:
            logging.warning("Results will not be written to disk, only returned.")
        
        self.model = model
        pred = self._run_inference_on_sample(sample=sample)
        self.model = None
        return pred
        
    def _run_inference_on_sample(self, sample):        
        # prepare sample:
        keyview_idx = int(sample['keyview_idx'].item())
        sample_inputs, sample_gt = self._inputs_and_gt_from_sample(sample)
        
        ordered_source_indices = self._get_nearest_source_view_ordering(sample_inputs=sample_inputs)
        num_source_views = min(len(ordered_source_indices), self.max_source_views) if self.max_source_views is not None else len(ordered_source_indices)
        source_view_indices = ordered_source_indices[:num_source_views]
        view_indices = list(sorted([keyview_idx] + source_view_indices))
        sample_inputs = filter_views_in_sample(sample=sample_inputs, indices_to_keep=view_indices)
        sample_gt = filter_views_in_sample(sample=sample_gt, indices_to_keep=view_indices)

        # run model:
        pred, _ = self.model.run(**sample_inputs)
        self._postprocess_sample_and_output(sample_inputs, sample_gt, pred)
        return pred

    @torch.no_grad()
    def run_on_dataset(self,
                       model,
                       dataset,
                       samples: Optional[Union[int, List[int]]] = None,
                       **_):
        """Run inference for a dataset and model.

        Args:
            dataset: Dataset to run inference on.
            model: Class or function that applies the model to a sample.
            samples: Integer that indicates the number of samples to run inference or list that indicates
                the indices of samples to run inference. None runs inference all samples.
        """
        assert self.out_dir is not None, "Output directory must be set to run inference on a dataset."
        
        self._init_inference(dataset=dataset, model=model, samples=samples)
        self._run_inference_on_dataset()
        self._output_dataset_cfg()
        self._reset_inference()

    def _init_inference(self,
                        dataset,
                        model,
                        samples=None,):
        self.dataset = dataset
        self.model = model
        self._init_sample_indices(samples=samples)
        self.dataloader = self.dataset.get_loader(batch_size=1, indices=self.sample_indices, num_workers=self.num_workers, collate_fn=numpy_collate)
        self.dataset_updates = {}
        self.pred_has_uncertainty = False

    def _init_sample_indices(self, samples):
        if isinstance(samples, list):
            self.sample_indices = samples
            if self.verbose:
                logging.info(f"Running inference on samples with indices: {self.sample_indices}.")
        elif isinstance(samples, int) and samples > 0:
            step_size = len(self.dataset) / samples  # <=1
            self.sample_indices = [int(i*step_size) for i in range(samples)]
            if self.verbose:
                logging.info(f"Running inference on samples with indices: {self.sample_indices}.")
        else:
            self.sample_indices = list(range(len(self.dataset)))

    def _run_inference_on_dataset(self):
        for sample_num, (sample_idx, sample) in enumerate(zip(self.sample_indices, self.dataloader)):
            self.cur_sample_num = sample_num
            self.cur_sample_idx = sample_idx

            if self.verbose:
                logging.info(f"Processing sample {self.cur_sample_num+1} / {len(self.sample_indices)} "
                             f"(index: {self.cur_sample_idx})")

            pred = self._run_inference_on_sample(sample=sample)
            
            if "depth_uncertainty" in pred:
                self.pred_has_uncertainty = True
                
            qualitatives = self._get_qualitatives(pred=pred)
            self._log_qualitatives(qualitatives)

    def _get_qualitatives(self, pred):

        pred_depth = pred['depth'][0]
        pred_invdepth = pred['invdepth'][0]

        qualitatives = {'pred_depth': pred_depth,
                        'pred_invdepth': pred_invdepth}

        if 'depth_uncertainty' in pred:
            qualitatives['pred_depth_uncertainty'] = pred['depth_uncertainty'][0]

        return qualitatives

    def _log_qualitatives(self, qualitatives):

        for qualitative_name, qualitative in qualitatives.items():
            out_path = osp.join(self.out_dir, f'{self.cur_sample_idx:07d}-{qualitative_name}')
            npy_path = out_path + '.npy'
            png_path = out_path + '.png'
            np.save(npy_path, qualitative)
            vis(qualitative).save(png_path)
            self._add_dataset_update({qualitative_name: npy_path})

    def _add_dataset_update(self, update_dict):
        if self.cur_sample_idx not in self.dataset_updates:
            self.dataset_updates[self.cur_sample_idx] = MultiViewDepthInferenceUpdate()

        self.dataset_updates[self.cur_sample_idx].update_dict.update(update_dict)

    def _get_nearest_source_view_ordering(self, sample_inputs):
        keyview_idx = int(sample_inputs['keyview_idx'].item())
        source_indices = [idx for idx in range(len(sample_inputs['images'])) if idx != keyview_idx]
        source_view_ordering = sorted(source_indices, key=lambda x: np.abs(x-keyview_idx))
        return source_view_ordering

    def _inputs_and_gt_from_sample(self, sample):
        is_input = lambda key: key in self.inputs or key == "keyview_idx"
        is_gt = lambda key: key not in self.inputs or key == "keyview_idx"
        sample_inputs = {key: val for key, val in sample.items() if is_input(key)}
        sample_gt = {key: val for key, val in sample.items() if is_gt(key)}
        return sample_inputs, sample_gt

    def _postprocess_sample_and_output(self, sample_inputs, sample_gt, pred):

        pred_depth = pred['depth']
        pred_mask = pred_depth != 0 if self.sparse_pred else np.ones_like(pred_depth, dtype=bool)

        if self.alignment == "median":
            assert 'depth' in sample_gt, "Ground truth depth is required for median alignment."
            
            gt_depth = sample_gt['depth']
            resized_pred_depth = skimage.transform.resize(pred_depth, gt_depth.shape, order=0, anti_aliasing=False)

            resized_pred_mask = resized_pred_depth != 0 if self.sparse_pred else np.ones_like(resized_pred_depth, dtype=bool)
            gt_mask = gt_depth > 0
            mask = gt_mask & resized_pred_mask
            
            ratio = np.median(gt_depth[mask]) / np.median(resized_pred_depth[mask])

            if mask.any() and np.isfinite(ratio):
                pred_depth = pred_depth * ratio
            else:
                ratio = np.nan

            pred['scaling_factor'] = ratio

        elif self.alignment == 'scale_shift_least_squares_invdepth':
            assert 'depth' in sample_gt, "Ground truth depth is required for least squares alignment."
            
            gt_depth = sample_gt['depth']
            resized_pred_depth = skimage.transform.resize(pred_depth, gt_depth.shape, order=0, anti_aliasing=False)

            resized_pred_mask = resized_pred_depth != 0 if self.sparse_pred else np.ones_like(resized_pred_depth, dtype=bool)
            gt_mask = gt_depth > 0
            mask = gt_mask & resized_pred_mask
            
            with np.errstate(divide='ignore', invalid='ignore'):
                pred_invdepth = np.nan_to_num(1 / pred_depth, nan=0, posinf=0, neginf=0)
                resized_pred_invdepth = np.nan_to_num(1 / resized_pred_depth, nan=0, posinf=0, neginf=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                gt_invdepth = np.nan_to_num(1 / gt_depth, nan=0, posinf=0, neginf=0)

            if mask.any():
                masked_gt_invdepth = (gt_invdepth[mask]).astype(np.float64)
                masked_resized_pred_invdepth = (resized_pred_invdepth[mask]).astype(np.float64)

                # system matrix: A = [[a_00, a_01], [a_10, a_11]]
                a_00 = np.sum(masked_resized_pred_invdepth * masked_resized_pred_invdepth)
                a_01 = np.sum(masked_resized_pred_invdepth)
                a_11 = np.sum(mask.astype(np.float64))

                # right hand side: b = [b_0, b_1]
                b_0 = np.sum(masked_gt_invdepth * masked_resized_pred_invdepth)
                b_1 = np.sum(masked_gt_invdepth)

                det = a_00 * a_11 - a_01 * a_01
                valid = det > 0

                if valid:
                    scale = ((a_11 * b_0 - a_01 * b_1) / det).astype(np.float32)
                    shift = ((-a_01 * b_0 + a_00 * b_1) / det).astype(np.float32)
                else:
                    scale = np.nan
                    shift = np.nan

            else:
                scale = np.nan
                shift = np.nan

            pred_invdepth = scale * pred_invdepth + shift
            with np.errstate(divide='ignore', invalid='ignore'):
                pred_depth = np.nan_to_num(1 / pred_invdepth, nan=0, posinf=0, neginf=0)
            del pred_invdepth
            del gt_invdepth
            pred['least_squares_scale'] = scale
            pred['least_squares_shift'] = shift
            
        elif self.alignment in ["pose_scale_mean", "pose_scale_median"]:
            assert "poses" not in sample_inputs, f"{self.alignment} is only allowed if poses are not provided as input to the model."
            assert "poses" in sample_gt, f"{self.alignment} is only possible if poses are available in the ground truth."
            assert "poses" in pred, f"{self.alignment} is only possible if poses are available in the predictions."

            keyview_idx = sample_inputs['keyview_idx']
            gt_poses = sample_gt['poses']
            pred_poses = pred['poses']
            assert gt_poses[0].shape[0] == 1, "Alignment is implemented only for batch size 1 (np.linalg.norm is applied over whole array and not over each batch separately)."

            # Make all poses relative to the keyview:
            gt_key_to_ref_transform = select_by_index(gt_poses, keyview_idx)
            gt_ref_to_key_transform = invert_transform(gt_key_to_ref_transform)
            gt_poses = [gt_view_to_ref_transform @ gt_ref_to_key_transform for gt_view_to_ref_transform in gt_poses]

            pred_key_to_ref_transform = select_by_index(pred_poses, keyview_idx)
            pred_ref_to_key_transform = invert_transform(pred_key_to_ref_transform)
            pred_poses = [pred_view_to_ref_transform @ pred_ref_to_key_transform for pred_view_to_ref_transform in pred_poses]

            # Get the magnitudes of the translations of the source views relative to the keyview:
            gt_source_view_to_key_transforms = exclude_index(gt_poses, keyview_idx)
            pred_source_view_to_key_transforms = exclude_index(pred_poses, keyview_idx)
            gt_source_view_to_key_translations = [trans_from_transform(transform) for transform in gt_source_view_to_key_transforms]
            pred_source_view_to_key_translations = [trans_from_transform(transform) for transform in pred_source_view_to_key_transforms]
            gt_source_view_to_key_translation_mags = [np.linalg.norm(translation) for translation in gt_source_view_to_key_translations]
            pred_source_view_to_key_translation_mags = [np.linalg.norm(translation) for translation in pred_source_view_to_key_translations]

            # Compute the scaling factor based on the ratios between GT and predicted translation magnitudes:
            translation_mag_ratios = np.array(gt_source_view_to_key_translation_mags) / np.array(pred_source_view_to_key_translation_mags)
            translation_mag_ratios = translation_mag_ratios[np.isfinite(translation_mag_ratios)]
            if self.alignment == 'pose_scale_mean':
                ratio = np.mean(translation_mag_ratios)
            elif self.alignment == 'pose_scale_median':
                ratio = np.median(translation_mag_ratios)

            if np.isfinite(ratio):
                pred_depth = pred_depth * ratio
            else:
                ratio = np.nan

            pred['scaling_factor'] = ratio

        if isinstance(self.clip_pred_depth, tuple) or isinstance(self.clip_pred_depth, list):
            pred_depth = np.clip(pred_depth, self.clip_pred_depth[0], self.clip_pred_depth[1]) * pred_mask
        elif self.clip_pred_depth:
            pred_depth = np.clip(pred_depth, 0.1, 100) * pred_mask

        with np.errstate(divide='ignore', invalid='ignore'):
            pred_invdepth = np.nan_to_num(1 / pred_depth, nan=0, posinf=0, neginf=0)

        pred['depth'] = pred_depth
        pred['invdepth'] = pred_invdepth
            
    def _output_dataset_cfg(self):
        update_name = self.model.name if hasattr(self.model, "name") else "dataset_update"
        dataset_updates_path = osp.join(self.out_dir, f"{update_name}.pickle")
        dataset_layout_path = osp.join(self.out_dir, "layout.pickle")
        dataset_cfg_path = osp.join(self.out_dir, "dataset.cfg")
        with open(dataset_updates_path, 'wb') as dataset_udpates_file:
            pickle.dump(self.dataset_updates, dataset_udpates_file)

        dataset_layout = self._get_layout()
        dataset_layout.write(dataset_layout_path)

        self.dataset.write_config(path=dataset_cfg_path,
                                  extra_updates=[dataset_updates_path], extra_layouts=[dataset_layout_path], update_strict=True,
                                  augmentations=None, target_size=None, input_size=None, input_resize_backend=None)

    def _get_layout(self):
        layout = Layout(name="inference_mvd")

        def load_key_img(sample_dict):
            from itypes.vizdata.image import ImageVisualizationData
            key_img = sample_dict['images'][sample_dict['keyview_idx']].transpose(1, 2, 0).astype(np.uint8)
            key_img = ImageVisualizationData(key_img)
            return {'data': key_img}

        key_img_visualization = Visualization(col=0, row=0, visualization_type="image", load_fct=load_key_img, name="Key Image")
        layout.visualizations.append(key_img_visualization)

        def load_pred_depth(sample_dict):
            from itypes.vizdata.float import FloatVisualizationData
            depth = sample_dict['pred_depth'].transpose(1, 2, 0)
            depth = FloatVisualizationData(depth)
            return {'data': depth}

        pred_depth_visualization = Visualization(col=0, row=1, visualization_type="float", load_fct=load_pred_depth, name="Predicted Depth")
        layout.visualizations.append(pred_depth_visualization)

        def load_pred_invdepth(sample_dict):
            from itypes.vizdata.float import FloatVisualizationData
            invdepth = sample_dict['pred_invdepth'].transpose(1, 2, 0)
            invdepth = FloatVisualizationData(invdepth)
            return {'data': invdepth}

        pred_invdepth_visualization = Visualization(col=1, row=1, visualization_type="float", load_fct=load_pred_invdepth, name="Predicted Inverse Depth")
        layout.visualizations.append(pred_invdepth_visualization)

        if self.pred_has_uncertainty:
            def load_pred_depth_uncertainty(sample_dict):
                from itypes.vizdata.float import FloatVisualizationData
                if 'pred_depth_uncertainty' in sample_dict:
                    uncertainty = sample_dict['pred_depth_uncertainty'].transpose(1, 2, 0)
                else:
                    uncertainty = np.ones_like(sample_dict['pred_depth']).transpose(1, 2, 0)
                uncertainty = FloatVisualizationData(uncertainty)
                return {'data': uncertainty}

            pred_depth_uncertainty_visualization = Visualization(col=3, row=1, visualization_type="float", load_fct=load_pred_depth_uncertainty, name="Predicted Depth Uncertainty")
            layout.visualizations.append(pred_depth_uncertainty_visualization)

        return layout

    def _reset_inference(self):
        self.dataset = None
        self.dataloader = None
        self.model = None
        self.sample_indices = None
        self.cur_sample_idx = 0
        self.cur_sample_num = 0
        self.dataset_updates = None
        self.pred_has_uncertainty = None


def filter_views_in_sample(sample, indices_to_keep):
    sample = deepcopy(sample)
    keyview_idx = int(sample['keyview_idx'].item())
    assert keyview_idx in indices_to_keep, "Keyview must not be filtered out."
    keyview_idx = indices_to_keep.index(keyview_idx)

    if "images" in sample:
        sample['images'] = [select_by_index(sample['images'], i) for i in indices_to_keep]
    if "poses" in sample:
        sample['poses'] = [select_by_index(sample['poses'], i) for i in indices_to_keep]
    if "intrinsics" in sample:
        sample['intrinsics'] = [select_by_index(sample['intrinsics'], i) for i in indices_to_keep]
    sample['keyview_idx'] = np.array([keyview_idx])

    return sample


class MultiViewDepthInferenceUpdate(Update):
    def __init__(self):
        self.update_dict = {}

    def load(self, orig_sample_dict, root=None):
        out_dict = {}
        for key, val in self.update_dict.items():
            if isinstance(val, str):
                if osp.isfile(val):
                    val = np.load(val)
            out_dict[key] = val
        return out_dict
