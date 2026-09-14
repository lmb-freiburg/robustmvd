# Data

The `rmvd` framework can load data from two sources:
- Commonly used research datasets (see [Common research datasets](#common-research-datasets))
- From a local folder: local folder that provides data in a structure that is compatible with the `rmvd` framework (see [Loading data from a local folder](#loading-data-from-a-local-folder))

Data can be loaded in different formats regarding the structure of the provided data. 
Currently, the following formats are supported:
- Multi-view depth (mvd): each sample contains multiple views and one view is assigned as keyview
Details on the different data formats are explained [below](#data-formats).

## Common research datasets

Currently, the `rmvd` framework supports the following datasets:
- KITTI
- ETH3D
- DTU
- ScanNet
- Tanks and Temples
- FlyingThings3D
- StaticThings3D
- BlendedMVS

Some datasets have different **splits**, e.g. `train` and `test`, or some custom splits, e.g. the `Eigen` split of KITTI. As described in the [beginning](#data) and in [Data formats](#data-formats), data can be loaded in different **formats**. Note that not all datasets and splits are available in all formats. 

**Setup**: All datasets need to be downloaded and some need to be preprocessed before they can be used. For some datasets, download scripts are provided; for others, the data has to be downloaded manually. The setup for each dataset is described below.
The root paths of the datasets need to be specified in the `paths.toml` file.

**Usage**: After the setup, datasets can be created with the `create_dataset` function by specifying the dataset name (including the split and format), e.g. `create_dataset("eth3d.robustmvd.mvd")`. Further usage instructions are explained [below](#the-create_dataset-function).

---

### The `paths.toml` file

The `paths.toml` file contains the root paths of the datasets. It can be located at:
- `rmvd/data/paths.toml` (prioretized)
- `~/rmvd_data_paths.toml` (useful when working on the `rmvd` framework and using `git`)

It is required to create such a `paths.toml` file at one of the above locations.
The file should contain the root paths of the datasets. The format is as follows:
```toml
[eth3d]
    root = '/tmp/eth3d/'

# ...
```

A template `path.tomls` file is provided at `rmvd/data/_paths.toml`. Further details
can be found in the respective dataset setup descriptions below.

---

The following describes the setup for each individual dataset.

### ETH3D
From the directory of this `README` file, execute the script `scripts/download_eth3d.sh` and specify the download 
target directory to download the dataset:
```bash
./scripts/download_eth3d.sh /path/to/eth3d
```
Then specify the download directory `/path/to/eth3d` in the `paths.toml` file.

Optionally, if you want to evaluate on ETH3D with undistorted images and depth maps (as proposed by
MVSAnywhere), execute the script `scripts/compute_eth3d_undistorted.py` from the directory of this `README`
file and specify the directory of the downloaded ETH3D dataset:
```bash
./scripts/compute_eth3d_undistorted.py /path/to/eth3d [--num_workers 6]
```
The script only needs `numpy`, `opencv` and `tqdm`, i.e. it runs in the normal `rmvd` environment.

---

### KITTI
Download the KITTI raw data from <https://www.cvlibs.net/datasets/kitti/raw_data.php> using
the "raw dataset download script (1 MB)" that is provided on the website. You need to register for this, 
therefore we can't provide a download script here. Move the "raw dataset download script" called
`raw_data_downloader.sh` to a directory `/path/to/kitti/raw_data` and execute it there.

Download the file "annotated depth maps data set (14 GB)" from 
<https://www.cvlibs.net/datasets/kitti/eval_depth_all.php> to a directory 
`/path/to/kitti/depth_completion_prediction/` and extract it there. The `data_depth_annotated.zip` file can be deleted.

Download the file "odometry ground truth poses (4MB)" from 
<https://www.cvlibs.net/datasets/kitti/eval_odometry.php> to a directory
`/path/to/kitti/odometry/` and extract it there. 

Then specify the KITTI directory `/path/to/kitti` in the `paths.toml` file.

---

### DTU

From the directory of this `README` file, execute the script `scripts/download_dtu.sh` and 
specify the download directory for the dataset:
```bash
./scripts/download_dtu.sh /path/to/dtu_raw
```

**Note**: the last download step of the script fetches the ground truth depth maps from Google Drive via `gdown`, 
which frequently fails, as Google Drive sometimes blocks the download.
In that case, either wait a while and execute the script again, or download the file manually in a browser from 
<https://drive.google.com/open?id=1LVy8tsWajG3uPTCYPSxDvVXFCdIYXaS-> and extract it in `/path/to/dtu_raw`, resulting 
in a directory `/path/to/dtu_raw/Depths`.

Then, from the directory of this `README` file, execute the script `scripts/convert_dtu.sh` to bring the dataset in the
structure that is required by the dataloader:
```bash
./scripts/convert_dtu.py /path/to/dtu_raw /path/to/dtu
```

Then specify the DTU directory (`/path/to/dtu`) in the `paths.toml` file. 
The directory `/path/to/dtu_raw` can be deleted.

---

### ScanNet
Download ScanNet by following the official instructions at <https://github.com/ScanNet/ScanNet>. 
You need to fill out a Terms of Use agreement for this, therefore we can't provide a download script here. 
You will receive a script named `download-scannet.py` to download the data. 
Download the full dataset to a directory `/path/to/scannet_orig` with this script as follows:
```bash
python download-scannet.py -o /path/to/scannet_orig/
```

Then create a new python2.7 virtual environment, e.g. via:
```bash
conda create -y -n scannetreader python=2.7
conda activate scannetreader
conda install -y numpy imageio=2.6.0 opencv tqdm==4.63
```

Then, from the directory of this `README` file, execute the script `scripts/convert_scannet.sh` to bring the dataset in 
the structure that is required by the dataloader:
```bash
./scripts/convert_scannet.py /path/to/scannet_orig /path/to/scannet
```

Then specify the ScanNet directory (`/path/to/scannet`) in the `paths.toml` file. 
The directory `/path/to/scannet_orig` can be deleted. The virtual environment `scannetreader` can be deleted.

---

### Tanks and Temples
From the directory of this `README` file, execute the script `scripts/download_tanks_and_temples.sh` and specify the 
download target directory to download the dataset:
```bash
./scripts/download_tanks_and_temples.sh /path/to/tanks_and_temples
```
Then specify the download directory (`/path/to/tanks_and_temples`) in the `paths.toml` file.

---

### FlyingThings3D
Download FlyingThings3D from 
[the dataset website](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html). You 
need the following files:
- RGB images (cleanpass) from the "full dataset"
- Disparity from the "full dataset"
- Camera data from the "full dataset"

Extract all downloaded files in the same directory, e.g. `/path/to/flyingthings3d`. The dataloader requires a different
storage structure than the original dataset. Execute the script `scripts/convert_flyingthings3d.py` to create the 
required storage structure in a directory called `/path/to/flyingthings3d_converted`:
```bash
./scripts/convert_flyingthings3d.py /path/to/flyingthings3d /path/to/flyingthings3d_converted
```

Note that the script partially creates links in the `/path/to/flyingthings3d_converted` directory, instead of
copying the actual files.

Then specify the converted directory `/path/to/flyingthings3d_converted` as
- `/path/to/flyingthings3d_converted/TRAIN` for the training set
- `/path/to/flyingthings3d_converted/TEST` for the test set

in the `paths.toml` file.

---

### StaticThings3D
From the directory of this `README` file, execute the script `scripts/download_staticthings3d.sh` and specify the 
download target directory to download the dataset:
```bash
./scripts/download_staticthings3d.sh /path/to/staticthings3d
```

The dataloader requires a different storage structure than the downloaded dataset. 
Execute the script `scripts/convert_staticthings3d.py` to create the 
required storage structure in a directory called `/path/to/staticthings3d_converted`:
```bash
./scripts/convert_staticthings3d.py /path/to/staticthings3d /path/to/staticthings3d_converted
```

Note that the script mostly creates links in the `/path/to/staticthings3d_converted` directory, instead of
copying the actual files.

Then specify the converted directory `/path/to/staticthings3d_converted` as
- `/path/to/staticthings3d_converted/TRAIN` for the training set (there is no test set!)

in the `paths.toml` file.

---

### BlendedMVS
Download the BlendedMVS low-res set (27.5GB) from https://github.com/YoYo000/BlendedMVS (we do not provide a download script,
as the data is hosted on OneDrive). Extract the files to a directory `/path/to/blendedmvs` (the directory should then contain 
the scene folders, e.g. `/path/to/blendedmvs/57f8d9bbe73f6760f10e916a`) and specify the 
directory (`/path/to/blendedmvs`) in the `paths.toml` file.

---

## Loading data from a local folder
The `rmvd` framework can load data from a local folder that contains data in a compatible structure.

As described in the [beginning](#data) and in [Data formats](#data-formats), data can be loaded in different **formats**. Note that not all local folder structures can be loaded in all formats. 

Currently, parsers for the following compatible structures with corresponding data formats are supported:
- rmvd sample folder structure; supported formats: `mvd`

**Usage**: Data from a local folder can be used as a dataset with the `create_dataset` function by specifying the path to the local folder, the local folder structure, and the format in which the data should be loaded.  Further usage instructions are explained [below](#the-create_dataset-function).

### `rmvd_sample` folder structure
This is a minimal folder structure that directly contains the data that is required for multi-view depth estimation. It is used for the `sample_data` folder in this repository. A dataset created from such a folder contains only a single sample.

---

## Data formats
Data can be loaded in different formats regarding the structure of the provided data.
The following describes the available data formats.

### Multi-view depth (mvd) data format
In the `mvd` data format, each sample is a dictionary with the following keys:
- `images`: a list of images. Each image is a numpy array of shape `(3, H, W)`, type `float32` and values from 0 to 255
- `poses`: a list of camera poses. Each camera pose is numpy array of shape `(4, 4)` and type `float32`. The reference
  coordinate system is the keyview coordinate system (for more information, see below)
- `intrinsics`: a list of camera intrinsics. Each camera intrinsic is numpy array of shape `(3, 3)` and type `float32`
- `keyview_idx`: integer that indicates the index of the keyview in the list of views, e.g. `images[keyview_idx]` is the
  keyview image
- `depth`: depth map for the keyview. This is a numpy array of shape `(1, H, W)` and type `float32`
- `invdepth`: inverse depth map for the keyview. This is a numpy array of shape `(1, H, W)` and type `float32`
- `depth_range`: minimum and maximum depth values for the view. This is a tuple of the form `(min_depth, max_depth)` where
both `min_depth` and `max_depth` are of type `np.float32`

### Intrinsics and camera poses
Intrinsics are numpy arrays of shape `(3, 3)` and given as follows:
```python
[[fx, 0, cx],
[0, fy, cy],
[0, 0, 1]]
```
The unit for the focal lengths `fx`, `fy` and the principal points `cx`, `cy` is pixels.

Camera poses are numpy arrays of shape `(4, 4)` and given as follows:
```python
[[r11, r12, r13, tx],
 [r21, r22, r23, ty],
 [r31, r32, r33, tz],
 [0, 0, 0, 1]]
```
Poses are given from the current view to a reference coordinate system. This means that the given transform transforms
the coordinate system of the current view into the reference coordinate system. In the code, this is indicated with
variable names of the form `view_to_ref_transform`. Equivalently, the transform transforms a point from reference
coordinates into current view coordinates: `p_cur = cur_to_ref_transform @ p_ref`. Note that the naming is a bit
unintuitive in this case. However, the advantage is an intuitive notation for chaining
transforms: `cur_to_key_transform = cur_to_ref_transform @ ref_to_key_transform`. 
The unit for the translation is meters.

The camera space is a right-handed coordinate system with the x-axis pointing to the right, the y-axis pointing down, and the z-axis pointing forwards. 

### Depth maps
Depths are provided with the key `depth` and are given in meters. Invalid depth values are set to 0.

Additionally, inverse depths (unit: 1/m) are provided with the key "invdepth". 
Invalid inverse depth values are set to 0. 

### Batched data format
Data can be loaded as batches of `N >= 1` samples. In this case, a batch dimension is added. A "batched" sample is
a dictionary the same keys as described above, but different shapes:
- `images`: a list of images. Each image is a numpy array of shape `(N, 3, H, W)`, type `float32` and values from 0 to 255
- `poses`: a list of camera poses. Each camera pose is numpy array of shape `(N, 4, 4)` and type `float32`
- `intrinsics`: a list of camera intrinsics. Each camera intrinsic is numpy array of shape `(N, 3, 3)` and type `float32`
- `keyview_idx`: numpy array of shape `(N,)` and type `int64`. Each element indicates the index of the keyview in each
  sample in the batch
- `depth`: depth map for the keyview. This is a numpy array of shape `(N, 1, H, W)` and type `float32`
- `invdepth`: inverse depth map for the keyview. This is a numpy array of shape `(N, 1, H, W)` and type `float32`
- `depth_range`: list of length 2 and where first element is a numpy array of shape `(N,)` and type `float32` that indicates
 the minimum depth values of each sample in the batch and the second element is a numpy array of shape `(N,)` and type `float32`
  that indicates the maximum depth values of each sample in the batch

### `torch` data format
Datasets can be created with the argument `to_torch=True`. In this case, all numpy arrays are converted to torch 
tensors and a batch dimension is prepended. A "batched" sample is a dictionary with the following items:
- `images`: a list of images. Each image is a `torch.Tensor` of shape [N, 3, H, W], type float32 and values from 0 to 255
- `poses`: a list of camera poses. Each camera pose is `torch.Tensor` of shape [N, 4, 4] and type float32
- `intrinsics`: a list of camera intrinsics. Each camera intrinsic is `torch.Tensor` of shape [N, 3, 3] and type float32
- `keyview_idx`: `torch.Tensor` of shape [N] and type int64. Each element indicates the index of the keyview in each
  sample in the batch
- `depth`: depth map for the keyview. This is a `torch.Tensor` of shape [N, 1, H, W] and type float32
- `invdepth`: inverse depth map for the keyview. This is a `torch.Tensor` of shape [N, 1, H, W] and type float32
- `depth_range`: list of length 2 and where first element is a `torch.Tensor` of shape [N] and type float32 where each
  element indicates the minimum depth of the respective sample in the batch and the second element is a `torch.Tensor`
  of shape [N] and type float32 where each element indicates the maximum depth of the respective sample in the batch

All datasets can be used within a torch dataloader, which automatically converts the data as with `to_torch=True`. For
details on using datasets within dataloaders, see below.

## Usage

### The `create_dataset` function
All datasets are created with the `create_dataset` function. 
The following describes the basic parameters of the `create_dataset` function.

#### `to_torch` parameter
A dataset can be created with the parameter `to_torch=True`, e.g. `create_dataset("eth3d.mvd", to_torch=True)`. In this
case, samples will be converted to torch data types, i.e. as torch tensors and not numpy arrays and with a prepended 
batch dimension.

#### `augmentations` parameter
A dataset can be created with the parameter `augmentations=[augmentation_1, augmentation_2, ..]`. The specified augmentations
will be applied to all loaded samples.

#### `input_size` parameter
A dataset can be created with the parameter `input_size=(height, width)`. The input images (not the ground truth
, e.g. depth maps) will then be rescaled to the specified resolution. 

#### `input_resize_backend` parameter
A dataset can be created with the parameter `input_resize_backend=backend_name`, which selects the backend that is used
to rescale the input images. This is only relevant if `input_size` is set.

The reason for this parameter is that different rescaling operations can produce slightly different results (e.g. handling of 
aliasing). When models were trained with a specific rescaling operation, it can be helpful to use the same 
rescaling operation at test time. It anyways makes sense to specify the rescaling operation in the dataset, because
then it can be done with multiple workers in parallel via the dataloader.

Note that in most cases, the differences between different rescaling operations are very small. The biggest differences
can be observed when the images are downscaled a lot (which is often the case for ETH3D samples).

### Creating a common research dataset
Common research datasets (KITTI etc.) can be created by passing the dataset name and data format (and optionally the split) to the `create_dataset` function:
```python
from rmvd import create_dataset
dataset = create_dataset(dataset_name_or_path=dataset_name, data_format=data_format)  # optional: split, e.g. split='robustmvd'; if not specified, the default split is used

# for example:
dataset = create_dataset(dataset_name_or_path="eth3d", data_format="mvd")  # will create the default eth3d.mvd split, which is 'robustmvd'
# other options for creating exactly the same dataset:
dataset = create_dataset(dataset_name_or_path="eth3d", data_format="mvd", split="robustmvd")  # explicitly specify the split
dataset = create_dataset(dataset_name_or_path="eth3d.mvd")  # specify the data_format and/or split in the dataset_name
dataset = create_dataset(dataset_name_or_path="eth3d.robustmvd.mvd")
dataset = create_dataset(dataset_name_or_path="eth3d.robustmvd", data_format="mvd")
dataset = create_dataset(dataset_name_or_path="eth3d.mvd", split="robustmvd")
```

It is required to indicate a dataset name and a data format. The split can be specified with the optional 
`split` parameter. If the split is not specified, the default split is used. 

Instead of using the `data_format` and `split` parameters of `create_dataset`, it is possible to specify
the data format and split within the `dataset_name` parameter. The format for this is 
`base_dataset_name.split.data_format`, for example `eth3d.robustmvd.mvd`.

For additional parameters that can be used with the `create_dataset` function, see above.

### Creating a dataset from a local folder
Datasets from local folders can be created by passing the folder path, folder parser name, and data format to the `create_dataset` function:
```python
from rmvd import create_dataset
dataset = create_dataset(dataset_name_or_path=folder_path, folder_parser_name=folder_parser_name, data_format=data_format)

# for example:
dataset = create_dataset(dataset_name_or_path="sample_data", folder_parser_name="rmvd_sample", data_format="mvd")
dataset = create_dataset(dataset_name_or_path="sample_data", folder_parser_name="rmvd_sample.mvd")  # specify the data_format in the folder_parser_name
```

It is required to indicate a dataset path, folder parser name, and a data format.

Instead of using the `data_format` parameter of `create_dataset`, it is possible to specify
the data format within the `folder_parser_name` parameter. The format for this is 
`folder_parser_name.data_format`, for example `rmvd_sample.mvd`.

For additional parameters that can be used with the `create_dataset` function, see above.

### Using a dataset
Datasets are of type `torch.utils.data.Dataset` and usage instructions can be found in the pytorch documentation.
The basic usage is:
```python
from rmvd import create_dataset
dataset = create_dataset("eth3d.mvd", input_size=(384, 576))
print(f"The dataset contains {len(dataset)} samples.")  # get number of samples in a dataset via len()
sample = dataset[0]  # get sample from dataset; the sample has no batch dimension
```

#### Torch dataloaders
It is possible to wrap a dataloader (`torch.utils.data.Dataloader`) around a dataset, as described in the pytorch 
documentation. The rmvd datasets contain a convenience method for this, which can be used as follows:
```python
from rmvd import create_dataset
dataset = create_dataset("eth3d.mvd", input_size=(384, 576))
dataloader = dataset.get_loader(batch_size=4, shuffle=False, num_workers=2)
```

#### Numpy dataloaders
The rmvd datasets contain a convenience method to create dataloaders that load batched data in `numpy` data types 
instead of `torch` data types:
```python
from rmvd import create_dataset
from rmvd.utils import numpy_collate
dataset = create_dataset("eth3d.mvd", input_size=(384, 576))
dataloader = dataset.get_loader(batch_size=4, shuffle=False, num_workers=2, collate_fn=numpy_collate)
```

---

## Dataset splits

### ETH3D
#### `robustmvd` split
This is the split introduced in "A Benchmark and a Baseline for Robust Multi-view Depth Estimation" by Schröppel et al. It is based
on the training split of the ETH3D High-res multi-view data, which consists of 13 sequences with in total 454 views.
For the `robustmvd` split, samples are defined based on 8 keyview from each sequence, resulting in a total of 104 
samples.

##### Source views:
Each  sample  contains  10  source  views,  using  the view  selection  provided  by
<https://github.com/FangjinhuaWang/PatchmatchNet>.

##### Depth range:
A depth range is not available. Instead, the minimum and maximum value of the ground truth depth maps are used as 
depth range.

#### `mvsanywhere` split
This is the split introduced in "MVSAnywhere: Zero-Shot Multi-View Stereo" by Izquierdo et al. It contains the same samples as the `robustmvd` split, but differs in using undistorted images and GT depth maps.

### KITTI
#### `robustmvd` split
This is the split introduced in "A Benchmark and a Baseline for Robust Multi-view Depth Estimation" by Schröppel et al. 
It is based on the commonly used Eigen test split, which contains 697 samples. It uses only samples of the Eigen
split where dense ground truth depth from Uhrig et al. is available (652 samples), and where ground truth poses
from the KITTI odometry benchmark are available (95 samples). It uses only samples where 10 views before
and 10 views after are available. This additional restriction leads to the final split with 93 samples.

##### Source views:
10 views before and 10 views after the keyview are uses as source views.

##### Depth range:
A depth range is not available. Instead, the minimum and maximum value of the ground truth depth maps are used as 
depth range.

#### `eigen_dense_depth_test` split
This is the split introduced in "Digging into Self-Supervised Monocular Depth Estimation" by Godard et al. in the 
supplementary section "D.1. Improved Ground Truth".
It is based on the commonly used Eigen test split, which contains 697 samples. It uses only samples of the Eigen
split where dense ground truth depth from Uhrig et al. is available. This results in a total number of 652 samples.

##### Source views:
None, i.e. this is a single-view dataset, where each sample only consists of a single view and no source views.

##### Depth range:
A depth range is not available. Instead, the minimum and maximum value of the ground truth depth maps are used as 
depth range.

#### `eigen_dense_depth_train` split
This split is based on the commonly used Eigen training split, which contains 22600 samples. It uses only samples of the Eigen
split where dense ground truth depth from Uhrig et al. is available. This results in a total number of 22285 samples.

##### Source views:
None, i.e. this is a single-view dataset, where each sample only consists of a single view and no source views.

##### Depth range:
A depth range is not available. Instead, the minimum and maximum value of the ground truth depth maps are used as 
depth range.

### DTU
#### `robustmvd` split
This is the split introduced in "A Benchmark and a Baseline for Robust Multi-view Depth Estimation" by Schröppel et al. 
It is based on the evaluation split used in MVSNet, which comprises 22 scans, where each scan has 49 frames. 
For the `robustmvd` split, 5 views of each scan are used as keyviews, resulting in a total of 110 samples.  

##### Source views:
Each sample contains 10 source views, using the view selection provided by 
<https://github.com/YoYo000/MVSNet>. 

##### Ground truth depth maps and depth ranges:
As ground truth depth, the depth maps from <https://github.com/YoYo000/MVSNet> are used. 
As ground truth depth range, the range 0.425m to 0.935m is used, which is common in many MVSNet-based works.

#### `robustmvd_paper` split
This split contains the same samples and source views as the `dtu.robustmvd.mvd` split and only differs in not using a fixed depth range of 0.425m to 0.935m.

Note:
- `dtu.robustmvd_paper.mvd` is the split that was actually used for the results in the RMVD paper (Table 2) 
- therefore this split is what `dtu.robustmvd.mvd` should be; however we accidentally released `dtu.robustmvd.mvd` *with* a fixed depth range of 0.425m to 0.935m and only found out much later
- as many follow-up works have evaluated on the RMVD benchmark in the meantime, we do not change the `dtu.robustmvd.mvd` split anymore, but instead release the original split as `dtu.robustmvd_paper.mvd`
- this affects only model evaluations where the `depth_range` is used as model input

##### Depth range:
No fixed depth range is used. Instead, the minimum and maximum value of the ground truth depth maps are used as 
depth range.

### ScanNet
#### `robustmvd` split
This is the split introduced in "A Benchmark and a Baseline for Robust Multi-view Depth Estimation" by Schröppel et al. 
It is based on the split from DeepV2D, which in turn extends the split by BA-Net and 
consists of 2000 samples taken from 90 of the 1513 totally available sequences. 
For the `robustmvd` split, every 10th sample of the original split is used, resulting in a total of 200 samples. 

##### Source views:
3 views before and 4 views after the keyview are uses as source views.

##### Depth range:
A depth range is not available. Instead, the minimum and maximum value of the ground truth depth maps are used as 
depth range.

##### Misc:
Images are resized to a resolution of 640x480px to match ground truth depth maps.

### Tanks and Temples
#### `robustmvd` split
This is the split introduced in "A Benchmark and a Baseline for Robust Multi-view Depth Estimation" by Schröppel et al. 
It is based on the scenes "Barn", "Courthouse", "Church", and "Ignatius" from the original training split.
Other scenes from the training split were not used, as they were not available for download at the time of writing. 
To speed up evaluation, only few images per scene are used as keyviews for the `robustmvd` split
(Barn: 18; Church: 25; Couthouse: 13; Ignatius: 13), resulting in a total of 69 samples. 

##### Camera poses and intrinsics
The construction of the test set starts with the provided image sets of the respective scenes. 
Camera poses and intrinsics are reconstructed by running COLMAP on the images. 
The reconstructed camera poses are aligned with the ground truth using the Tanks and Temples evaluation script 
<https://github.com/isl-org/TanksAndTemples/tree/master/python_toolbox/evaluation>. 

##### Ground truth depth maps and depth ranges:
To obtain ground truth depth maps for each image, the provided ground truth pointclouds for each scene are projected 
into the images and filtered manually for outliers. 
To determine the minimum depth range value, the minimum of the depth map and the visible 3d points estimated by COLMAP, 
is used.
To determine the maximum depth range value, the maximum of the depth map and the visible 3d points estimated by COLMAP, 
is used.

##### Source views:
10 source views are used, which are selected with the view selection script from
<https://github.com/FangjinhuaWang/PatchmatchNet/blob/main/colmap_input.py>.

### BlendedMVS
#### `robust_mvd` split
This is the training split used in "A Benchmark and a Baseline for Robust Multi-view Depth Estimation" by Schröppel et al. 
It is based on the official training split of BlendedMVS, which consists of 106 of the 113 available scenes
(see <https://github.com/YoYo000/BlendedMVS/blob/master/project_lists/BlendedMVS_training.txt>).
For the `robust_mvd` split, samples are defined to consist of a keyview and 4 source views. The official data
lists 10 source views for each keyview (with different scores for each source view). In this split, all combinations
of 4 source views out of this 10 source views are used.
Further, the official ground truth depth maps and depth ranges are used.

#### `train_mvsnet` split
This is the training split that was used in "BlendedMVS: A Large-scale Dataset for Generalized Multi-view Stereo Networks" by Yao et al.
to train MVSNet. It is based on the official training split of BlendedMVS, which consists of 106 of the 113 available scenes
(see <https://github.com/YoYo000/BlendedMVS/blob/master/project_lists/BlendedMVS_training.txt>). Samples are defined
to consist of a keyview and the 2 source views with the highest score. The official ground truth depth maps and depth ranges
are used.
