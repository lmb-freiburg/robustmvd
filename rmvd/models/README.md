# Models

`rmvd` contains implementations of depth estimation models. The following provides an overview of the available models
and describes the usage of these models.

---

## Overview
`rmvd` contains two types of model implementations:
### 1. Native models:
Models that are (re-)implemented natively within the `rmvd` framework. 
These models can be used for training and inference/evaluation.

### 2. Wrapped models:
Model wrappers around existing implementations. These models can only be used for inference/evaluation.
Wrapped models are indicated by names that end with `_wrapped`.
The setup of these models is usually a bit more involved, as it is required to download the original implementation
(e.g. cloning the respective repository from GitHub). Usually the setup requires the follwing steps:
- clone the original repository to a local directory
- specify the path to the local directory in the `paths.toml` file
- install required model-specific dependencies

---

## The `paths.toml` file

The `paths.toml` file contains paths to local directories of wrapped models. It can be located at:
- `rmvd/models/wrappers/paths.toml` (prioritized)
- `~/rmvd_model_paths.toml` (useful when working on the `rmvd` framework and using `git`)

To use wrapped models, it is required to create such a `paths.toml` file at one of the above locations.
The file should contain the paths to the local directories of the wrapped models. The format is as follows:
```toml
[monodepth2]
    root = '/tmp/monodepth2'

...
```

A template `paths.toml` file is provided at `rmvd/models/wrappers/_paths.toml`. Further details
can be found in the respective model descriptions below.

---

## Available native models

The following provides an overview of all available native models. 

To train these models, use the commands from the `train_all.sh` script. The weights that are used when these 
models are used with `pretrained=True`, are the weights that are obtained after training with the `train_all.sh` script.

To evaluate these models with the provided pretrained weights, use the commands from the `eval_all.sh` script.

### `robust_mvd`
This is the Robust MVD Baseline Model presented in the publication 
"A Benchmark and a Baseline for Robust Multi-view Depth Estimation" by Schröppel et al.

### `robust_mvd_5M`
This is the Robust MVD Baseline Model presented in the publication 
"A Benchmark and a Baseline for Robust Multi-view Depth Estimation" by Schröppel et al., but trained for 5M iterations instead
of the 600k iterations in the paper. The longer training slightly improves results.

### `supervised_monodepth2`
This is a variant of the Monodepth2 model presented in the publication 
"Digging into Self-Supervised Monocular Depth Estimation" by Godard et al. 
In particular, this variant differs in that it is trained with supervision from GT depth maps.

Pretrained weights are provided, which were produced with the training command from the `train_all.sh` script. Training was done with `192x640` resolution and evaluation on the `kitti.eigen_dense_depth_test.mvd` dataset gives an Abs Rel of `9.2%` (for the eval command, see `eval_all.sh` or equivalently `train_all.sh`).

## Available wrapped models

The following provides an overview of all available wrapped models including their respective setup instructions.

To evaluate these models, use the commands from the `eval_all.sh` script.

### `monodepth2_mono_stereo_wrapped`
This is the "Monodepth2 MS-trained" model presented in the publication 
"Digging into Self-Supervised Monocular Depth Estimation" by Godard et al. 
The model is wrapped around the original implementation from <https://github.com/nianticlabs/monodepth2>, where it is 
indicated as `mono+stereo_640x192`.

#### Setup:
From the directory of this `README` file, execute the script `wrappers/scripts/setup_monodepth2.sh` and specify the local
directory to clone the original repository:
```bash
./wrappers/scripts/setup_monodepth2.sh /path/to/monodepth2
```

Then specify the local directory `/path/to/monodepth2` in the `paths.toml` file as described [above](#the-pathstoml-file).

It is not necessary to install additional dependencies.

#### Misc:
The model is applied at a fixed input size of `width=640` and `height=192`. It therefore does not make sense to load
data at a specific downsampled resolution. Thus, don't use the `input_size` parameters of `Dataset` classes and of the
`eval.py` and `inference.py` scripts, when using this model.

#### Reproducing the results of the original publication:
In the original publication, the model is reported to have an Abs Rel of `0.080` on the KITTI Eigen split with the 
improved ground truth depths (see Table 7 in the paper). In the codebase of the original implementation, this result 
can be reproduced via:
```python
python evaluate_depth.py --eval_mono --load_weights_folder path/to/monostereo_640x192/weights  --eval_split eigen_benchmark
```
(Note that it seems like an error that the model is evaluated with the `--eval_mono` flag.
However, it gives results that perfectly match those in the paper, whereas using the `--eval_stereo` flag gives slightly worse results.)

Within `rmvd`, this result can be reproduced as follows:
```python
python eval.py --output /tmp/eval_output --model monodepth2_mono_stereo_wrapped --dataset kitti.eigen_dense_depth_test.mvd --eval_type mvd --max_source_views 0 --clipping 1e-3 80 --alignment median
```
This command gives an Abs Rel of `8.11%` (i.e. `0.0811`). The difference to the original result is because a) the downsampling of the input images and upsampling of predicted depth maps is done slightly differently, and b) RMVD loads the original PNG images, whereas the original codebase loads JPEGs.

### `monodepth2_mono_stereo_1024x320_wrapped`
This is the "Monodepth2 (1024x320) MS-trained" model presented in the publication 
"Digging into Self-Supervised Monocular Depth Estimation" by Godard et al. 
The model is wrapped around the original implementation from <https://github.com/nianticlabs/monodepth2>, where it is 
indicated as `mono+stereo_1024x320`.

#### Setup:
Same as for the `monodepth2_mono_stereo_wrapped` model.

#### Misc:
The model is applied at a fixed input size of `width=1024` and `height=320`. It therefore does not make sense to load
data at a specific downsampled resolution. Thus, don't use the `input_size` parameters of `Dataset` classes and of the
`eval.py` and `inference.py` scripts, when using this model.

### `monodepth2_postuncertainty_mono_stereo_wrapped`
This is the "Monodepth2-Post MS-trained" model presented in the publication 
"On the Uncertainty of Self-supervised Monocular Depth Estimation" by Poggi et al. 
The model is wrapped around the implementation from <https://github.com/nianticlabs/monodepth2>, where it is 
indicated as `mono+stereo_640x192`. It uses the "Post" uncertainty estimation (via image flipping) as described by Poggi et al.

#### Setup:
Same as for the `monodepth2_mono_stereo_wrapped` model.

#### Misc:
The model is applied at a fixed input size of `width=640` and `height=192`. It therefore does not make sense to load
data at a specific downsampled resolution. Thus, don't use the `input_size` parameters of `Dataset` classes and of the
`eval.py` and `inference.py` scripts, when using this model.

#### Reproducing the results of the original publication:
In the original publication from Poggi et al., the model is reported to have an Abs Rel of `0.082` and AUSE of `0.036`
on the KITTI Eigen split with the improved ground truth depths (see Table 3 in the paper). 

Within `rmvd`, this result can be reproduced as follows:
```python
python eval.py --output /tmp/eval_output --model monodepth2_postuncertainty_mono_stereo_wrapped --dataset kitti.eigen_dense_depth_test.mvd --eval_type mvd --max_source_views 0 --clipping 1e-3 80 --eval_uncertainty
```
This command gives an Abs Rel of `8.31%` (i.e. `0.0831`) and an unnormalized AUSE of `0.038`. The difference to the original result is because a) the downsampling of the input images and upsampling of predicted depth and uncertainty maps is done slightly differently, b) RMVD loads the original PNG images, whereas the original codebase loads JPEGs, and c) the uncertainty maps are stored in a different data type and the AUSE metric is implemented slightly differently.

### `mvsnet_pl_wrapped`
This is the MVSNet model presented in the publication "MVSNet: Depth Inference for Unstructured Multi-view Stereo" by Yao et al. 
The model is wrapped around the unofficial implementation from <https://github.com/kwea123/MVSNet_pl>.

#### Setup:
From the directory of this `README` file, execute the script `wrappers/scripts/setup_mvsnet_pl.sh` and specify the local
directory to clone the original repository:
```bash
./wrappers/scripts/setup_mvsnet_pl.sh /path/to/mvsnet_pl
```

Then specify the local directory `/path/to/mvsnet_pl` in the `paths.toml` file as described [above](#the-pathstoml-file).

It is required to install additional dependencies. Set up a new environment `rmvd-mvsnet_pl` using the same setup instructions as for the `rmvd` environment as described in [the RMVD setup instructions](../../README.md#setup). Then install the following additional dependencies in the `rmvd-mvsnet_pl` environment:
```bash
# potentially helpful: export CC=gcc-13 CXX=g++-13
pip install setuptools_scm
pip install --no-build-isolation git+https://github.com/mapillary/inplace_abn.git@v1.0.11
pip install kornia
```

#### Misc:
If no depth range is provided, a default depth range from `0.2` to `100` is used.

### `midas_large_v2_1_wrapped`
This is the "MiDaS v2.1 Large 384" model based on the publication "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer" by Ranftl et al. 
It differs from the "MiDaS Mix 5" model in the publication by being trained on 5 additional datasets. Named "MiDaS Mix 6" in the DPT publication "Vision Transformers for Dense Prediction" by Ranftl et al.
The model is wrapped around the original implementation from <https://github.com/isl-org/MiDaS>, where it is indicated as `v2.1 Large 384 / midas_v21_384`.

#### Setup:
From the directory of this `README` file, execute the script `wrappers/scripts/setup_midas.sh` and specify the local directory to clone the original repository:
```bash
./wrappers/scripts/setup_midas.sh /path/to/midas
```

Then specify the local directory `/path/to/midas` in the `paths.toml` file as described [above](#the-pathstoml-file).

It is required to install additional dependencies. Set up a new environment `rmvd-midas` using the same setup instructions as for the `rmvd` environment as described in [the RMVD setup instructions](../../README.md#setup). Then install the following additional dependencies in the `rmvd-midas` environment:
```bash
pip install timm
```

#### Misc:
The wrapper applies the model at a fixed input size. It therefore does not make sense to load data at a specific downsampled resolution. Thus, don't use the `input_size` parameters of `Dataset` classes and of the `eval.py` and `inference.py` scripts, when using this model.

### `midas_dpt_large_384_wrapped`
This is the "DPT-Large Mix 6" model presented in the publication "Vision Transformers for Dense Prediction" by Ranftl et al.
The model is wrapped around the original implementation from <https://github.com/isl-org/MiDaS>, where it is indicated as `v3.0 DPTL-384`.

#### Setup:
Same as for the `midas_large_v2_1_wrapped` model.

#### Misc:
The wrapper applies the model at a fixed input size. It therefore does not make sense to load data at a specific downsampled resolution. Thus, don't use the `input_size` parameters of `Dataset` classes and of the `eval.py` and `inference.py` scripts, when using this model.

### `vis_mvsnet_wrapped`
This is the Vis-MVSNet model presented in the publication "Visibility-aware Multi-view Stereo Network" by Zhang et al.
The model is wrapped around the original implementation from <https://github.com/jzhangbs/Vis-MVSNet>.

#### Setup:
From the directory of this `README` file, execute the script `wrappers/scripts/setup_vis_mvsnet.sh` and specify the local
directory to clone the original repository:
```bash
./wrappers/scripts/setup_vis_mvsnet.sh /path/to/vis_mvsnet
```

Then specify the local directory `/path/to/vis_mvsnet` in the `paths.toml` file as described [above](#the-pathstoml-file).

It is not necessary to install additional dependencies.

#### Misc:
If no depth range is provided, a default depth range from `0.2` to `100` is used.

### `cvp_mvsnet_wrapped`
This is the CVP-MVSNet model presented in the publication "Cost Volume Pyramid Based Depth Inference for Multi-View Stereo" by Yang et al.
The model is wrapped around the original implementation from <https://github.com/JiayuYANG/CVP-MVSNet>.

#### Setup:
From the directory of this `README` file, execute the script `wrappers/scripts/setup_cvp_mvsnet.sh` and specify the local directory to clone the original repository:
```bash
./wrappers/scripts/setup_cvp_mvsnet.sh /path/to/cvp_mvsnet
```

Then specify the local directory `/path/to/cvp_mvsnet` in the `paths.toml` file as described [above](#the-pathstoml-file).

It is not necessary to install additional dependencies.

#### Misc:
With the original implementation, the number of calculated depth hypotheses is sometimes too small. We apply a small patch (see `cvp_mvsnet.patch` file) to fix this.

Further, the implementation does not support running the model with a single source view. It is therefore not possible to evaluate the model with the `quasi-optimal` view selection, but only with the `nearest` view selection strategy.

If no depth range is provided, a default depth range from `0.2` to `100` is used.

### `patchmatchnet_wrapped`
This is the PatchmatchNet model presented in the publication "PatchmatchNet: Learned Multi-View Patchmatch Stereo" by Wang et al.
The model is wrapped around the original implementation from <https://github.com/FangjinhuaWang/PatchmatchNet>.

#### Setup:
From the directory of this `README` file, execute the script `wrappers/scripts/setup_patchmatchnet.sh` and specify the local directory to clone the original repository:
```bash
./wrappers/scripts/setup_patchmatchnet.sh /path/to/patchmatchnet
```

Then specify the local directory `/path/to/patchmatchnet` in the `paths.toml` file as described [above](#the-pathstoml-file).

It is not necessary to install additional dependencies.

#### Misc:
If no depth range is provided, a default depth range from `0.2` to `100` is used.

### `mvsanywhere_wrapped`
This is the MVSAnywhere model as presented in the publication "MVSAnywhere: Zero-Shot Multi-View Stereo" by Izquierdo et al. 
The model is wrapped around the original implementation <https://github.com/nianticlabs/mvsanywhere>. It is trained on a variety of datasets and targets a very similar setting as the RMVD benchmark, namely multi view depth estimation independent of the target data, including generalization across domains and scales (hence "Anywhere"). 

#### Setup:
From the directory of this `README` file, execute the script `wrappers/scripts/setup_mvsanywhere.sh` and specify the local
directory to clone the original repository:
```bash
./wrappers/scripts/setup_mvsanywhere.sh /path/to/mvsanywhere
```

Then specify the local directory `/path/to/mvsanywhere` in the `paths.toml` file as described [above](#the-pathstoml-file).

To fulfill the dependencies required by MVSAnywhere, it is easiest to set up a new virtual environment named `mvsanywhere` following the instructions in the MVSAnywhere repository. In the simplest case, just `cd` to `/path/to/mvsanywhere` and execute `mamba env create -f environment.yml -n mvsanywhere`. No additional dependencies need to be installed for RMVD, as the MVSAnywhere repository already fulfills all dependencies for RMVD. 

#### Misc:
If no depth range is provided, the model estimates the depth range itself, based on a heuristic that uses the camera baselines and intrinsics, instead of using a fixed default depth range.

#### Reproducing the results of the original publication:

##### RMVD benchmark
In the original publication, the model is reported to have the following results on the RMVD benchmark (see Table 2 in the MVSAnywhere paper):

| Metric | KITTI | ScanNet | ETH3D | DTU | T&T | Average |
|---|---|---|---|---|---|---|
| Abs Rel in % | 3.2 | 3.7 | 3.2 | 1.3 | 2.1 | 2.7 |
| Inliers in % | 68.8 | 62.9 | 68.0 | 95.0 | 90.5 | 77.0 |

Within `rmvd`, this result can be reproduced as follows:
```python
python eval.py --eval_type robustmvd --model mvsanywhere_wrapped --inputs poses intrinsics --output /tmp/mvsanywhere_wrapped/unknown_depth_range --eval_name unknown_depth_range --max_source_views 7 --eth3d_size 480 640 --kitti_size 480 1280 --dtu_size 480 640 --scannet_size 480 640 --tanks_and_temples_size 480 640
```
This command gives:

| Metric | KITTI | ScanNet | ETH3D | DTU | T&T | Average |
|---|---|---|---|---|---|---|
| Abs Rel in % | 3.24 | 3.79 | 3.18 | 1.27 | 2.12 | 2.72 |
| Inliers in % | 68.80 | 62.20 | 67.72 | 94.98 | 90.45 | 76.83 |

##### Undistorted ETH3D split
In the original publication, the model is reported to have absrel=1.27, inliers=93.24 on the proposed undistorted
ETH3D split (`eth3d.mvsanywhere.mvd`) (see Table 3 in the MVSAnywhere paper).

Within `rmvd`, this result can be reproduced as follows:
```python
python eval.py --eval_type mvd --dataset eth3d.mvsanywhere --model mvsanywhere_wrapped --inputs poses intrinsics --output /tmp/mvsanywhere_wrapped/eth3d.mvsanywhere.mvd --max_source_views 7 --input_size 480 640
```
This command gives absrel=1.27, inliers=93.18.

---

## Usage

All models can be used with the same interface. The following describes the usage of the models.

### Initialization

To initialize a model, use the `create_model` function:
```python
from rmvd import create_model
model_name = "robust_mvd"  # available models: see above (e.g. "monodepth2_mono_stereo_1024x320_wrapped", etc.)
model = create_model(model_name, pretrained=True, weights=None, train=False, num_gpus=1)  # optional: model-specific parameters
```

#### Weights

If `pretrained` is set to True, the default pretrained weights for the model will be used. 
Alternatively, custom weights can be loaded by providing the path to the weights with the `weights` parameter. 
The `weights` parameter overrides the `pretrained` parameter.

#### Train mode

If `train` is set to True, the model is created in training mode.

#### GPU usage

If `num_gpus` is `>0`, the model will be executed on the GPU.

### Inference
The interface to do inference with the model is:
```python
pred, aux = model.run(images=images, keyview_idx=keyview_idx, poses=poses, intrinsics=intrinsics, 
                      depth_range=depth_range)  # alternatively: run(**sample)
```

#### Inputs
The inputs can be:
- numpy arrays with a prepended batch dimension  (e.g. images are `N3HW` and of type `np.ndarray`)
- numpy arrays without a batch dimension (e.g. images are `3HW` and of type `np.ndarray`)

The formats of specific inputs are described in the [data readme](../data/README.md).

#### Outputs
The `pred` output is a dictionary which contains:
- `depth`: predicted depth map for the reference view
- `depth_uncertainty`: predicted uncertainty for the predicted depth map (optional)

The output type and shapes correspond to the input types and shapes, i.e.:
- numpy arrays with a prepended batch dimension  (e.g. `depth` has shape `N1HW` and type `np.ndarray`)
- numpy arrays without a batch dimension (e.g. `depth` has shape `1HW` and type `np.ndarray`)

The `aux` output is a dictionary which contains additional, model-specific outputs. These are only used for training 
or debugging and not further described here.

#### Resolution
Most models cannot handle input images at arbitrary resolutions. Models therefore internally upsize the images to
the next resolution that can be handled. 

The model output is often at a lower resolution as the input data.

---

## Internal implementation

Internally, all models have the following functions:
- a `input_adapter` function that converts input data into the models-specific format
- a `forward` function that runs a forward pass with the model (in non-pytorch models, this is the `__call__` function) 
- a `output_adapter` function that converts predictions from model-specific format to the `rmvd` format

The `run` function mentioned above to do inference, uses those three functions as follows:
```python
def run(images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
    no_batch_dim = (images[0].ndim == 3)
    if no_batch_dim:
        images, keyview_idx, poses, intrinsics, depth_range = \
            add_batch_dim(images, keyview_idx, poses, intrinsics, depth_range)

    sample = model.input_adapter(images=images, keyview_idx=keyview_idx, poses=poses,
                                 intrinsics=intrinsics, depth_range=depth_range)
    model_output = model(**sample)
    pred, aux = model.output_adapter(model_output)

    if no_batch_dim:
        pred, aux = remove_batch_dim(pred, aux)

    return pred, aux
```

In the following, we further describe the `input_adapter`, `forward`/`__call__` and `output_adapter` functions.

### The `input_adapter` function

The `input_adapter` function has the following interface: 
```python
def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
    # construct sample dict that contains all inputs in the model-specific format: sample = {..}
    return sample
```
The inputs to the `input_adapter` function are all `numpy` array with a batch dimension 
(e.g. images are `N3HW` and of type `np.ndarray`). The function then converts all inputs to the format that
is required by the model and returns this converted data as a dictionary where the keys are the parameter names
of the model's `forward` / `__call__` function. This allows to call `model(**sample)` where sample is the dictionary 
that is returned from the `input_adapter` function.

The conversion may for example include converting the inputs to `torch.Tensor`, moving them to the GPU if required, 
normalizing the images, etc.

### The `forward` function (for non-pytorch model this function is named `__call__`)
The `forward` function of each model expects data in the model-specific format and returns model-specific outputs.

Hence, in case all input data is already in the format required by the model, you can also do `model(**sample)`. 
This is used in the `rmvd` training code. Note that the `forward` function also expects input data to have a resolution
that is supported by the model.

### The `output_adapter` function
The `output_adapter` function has the following interface:
```python
def output_adapter(self, model_output):
    # construct pred and aux dicts from model_output
    # pred needs to have an item with key "depth" and value of type np.ndarray and shape N1HW
    return pred, aux
```
The output adapter converts model-specific outputs to the `pred` and `aux` dictionaries. The output types and shapes
need to be numpy arrays with a batch dimension (i.e. `depth` has shape `N1HW` and type `np.ndarray`). 

---

## Using custom models within the `rmvd` framework

If you want to use your own model within the framework, e.g. for evaluation, your model needs to have the
`input_adapter`, `forward`/`__call__` and `output_adapter` functions as described above.

Note: you don't have to add a `run` function your model. This function will be added automatically by calling
`rmvd.prepare_custom_model(model)`.

You can then use your custom model within the `rmvd` framework, for example to run inference, e.g.:
```python
import rmvd
model = CustomModel()
model = rmvd.prepare_custom_model(model)
dataset = rmvd.create_dataset("eth3d", "mvd", input_size=(384, 576))
sample = dataset[0]
pred, aux = model.run(**sample)
```
or to run evaluation, e.g.:
```python
import rmvd
model = CustomModel()
model = rmvd.prepare_custom_model(model)
eval = rmvd.create_evaluation(evaluation_type="mvd", out_dir="/tmp/eval_output", inputs=["intrinsics", "poses"])
dataset = rmvd.create_dataset("kitti", "mvd", input_size=(384, 1280))
results = eval(dataset=dataset, model=model)
```
