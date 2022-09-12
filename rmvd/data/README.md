# Datasets

Multiple existing datasets are supported. The datasets come in three different types:
- Multi-view depth (mvd): each sample contains multiple views and one view is assigned as keyview
- Multi-view stereo (mvs): each sample contains all available views of a scene
- Depth-from-video (v2d): each sample contains a video (i.e. sequence of views)

The following datasets are supported:
- KITTI
- ETH3D
- Sintel
- ScanNet
- Cityscapes

Some datasets have different splits, e.g. `train` and `test`, or some custom splits, e.g. the `Eigen` split of KITTI.

## Setup
All datasets need to be downloaded and some need to be preprocessed before they can be used. For some datasets, 
download scripts are provided; for others, the data has to be downloaded manually.
The root paths of the datasets need to be specified in the file `paths.toml`. 
The following describes the setup for each individual dataset.

### ETH3D
Execute the script `scripts/download_eth3d.sh` and specify the download target directory to download the dataset:
```bash
./scripts/download_eth3d.sh /path/to/download/dir
```
Then specify the download directory (`/path/to/download/dir`) in the `paths.toml` file.

### FlyingThings3D
Download FlyingThings3D from 
[the dataset website](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html). You 
need the following files:
- RGB images (cleanpass) from the "full dataset"
- Disparity from the "full dataset"
- Camera data from the "full dataset"

Extract all downloaded files in the same directory, e.g. `/path/to/FlyingThings3D`. The dataloader requires a different
storage structure than the original dataset. Execute the script `scripts/convert_flyingthings3d.py` to create the 
required storage structure in a directory called `/path/to/FlyingThings3D_converted`:
```bash
./scripts/convert_flyingthings3d.py /path/to/FlyingThings3D /path/to/FlyingThings3D_converted
```

Note that the script partially creates links in the `/path/to/FlyingThings3D_converted` directory, instead of
copying the actual files.

Then specify the converted directory (`/path/to/FlyingThings3D_converted`) in the `paths.toml` file.

### BlendedMVS
Download the BlendedMVS low-res set (27.5GB) from https://github.com/YoYo000/BlendedMVS. Extract the files to a
directory `/path/to/blendedmvs` (should then contain the scene folders, e.g. 
`/path/to/blendedmvs/57f8d9bbe73f6760f10e916a`) and specify the
directory (`/path/to/blendedmvs`) in the `paths.toml` file.

## Data format
Depending on the dataset type, the data is provided in a specific format. 
The following describes the format for each dataset type.

### Multi-view depth (mvd) data format
For mvd datasets each sample is a dictionary with the following keys:
- `images`: a list of images. Each image is a numpy array of shape (3, H, W), type float32 and values from 0 to 255
- `poses`: a list of camera poses. Each camera pose is numpy array of shape (4, 4) and type float32. The reference
  coordinate system is the keyview coordinate system (for more information, see below)
- `intrinsics`: a list of camera intrinsics. Each camera intrinsic is numpy array of shape (3, 3) and type float32
- `keyview_idx`: integer that indicates the index of the keyview in the list of views, e.g. `images[keyview_idx]` is the
  keyview image
- `depth`: depth map for the keyview. This is a numpy array of shape (1, H, W) and type float32
- `invdepth`: inverse depth map for the keyview. This is a numpy array of shape (1, H, W) and type float32
- `depth_range`: minimum and maximum depth values for the view. This is a tuple of the form (min_depth, max_depth)

### Intrinsics and camera poses
Intrinsics are numpy arrays of shape (3, 3) and given as follows:
```python
[[fx, 0, cx],
[0, fy, cy],
[0, 0, 1]]
```
The unit for the focal lengths fx, fy and the principal points cx, cy is pixels.

Camera poses are numpy arrays of shape (4, 4) and given as follows:
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

### Depth maps
Depths are provided with the key "depth" and are given in meters. Invalid depth values are set to 0.

Additionally, inverse depths (unit: 1/m) are provided with the key "invdepth". 
Invalid inverse depth values are set to 0. 

### Batched data format
Data can be loaded as batches of `N >= 1` samples. In this case, a batch dimension is added. A "batched" sample is
a dictionary the same keys as described above, but different shapes:
- `images`: a list of images. Each image is a numpy array of shape (N, 3, H, W), type float32 and values from 0 to 255
- `poses`: a list of camera poses. Each camera pose is numpy array of shape (N, 4, 4) and type float32
- `intrinsics`: a list of camera intrinsics. Each camera intrinsic is numpy array of shape (N, 3, 3) and type float32
- `keyview_idx`: numpy array of shape (N,) and type int64. Each element indicates the index of the keyview in each
  sample in the batch
- `depth`: depth map for the keyview. This is a numpy array of shape (N, 1, H, W) and type float32
- `invdepth`: inverse depth map for the keyview. This is a numpy array of shape (N, 1, H, W) and type float32
- `depth_range`: list of length 2 and where first element is a tuple with N floats that indicates the minimum depth
  values of each sample in the batch and the second element is a tuple with N floats that indicates the maximum depth
  values of each sample in the batch

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

### Creating a dataset
To create a dataset, use the `create_dataset` function:
```python
from rmvd import create_dataset
dataset = create_dataset(dataset_name=dataset_name, dataset_type=dataset_type)  # optional: split, e.g. split='train'

# for example:
dataset = create_dataset(dataset_name="eth3d", dataset_type="mvd")  # will create the default eth3d.mvd split, which is 'train'
# other options for creating exactly the same dataset:
dataset = create_dataset(dataset_name="eth3d", dataset_type="mvd", split="train")  # explicitly specify the split
dataset = create_dataset(dataset_name="eth3d.mvd")  # specify the dataset_type and/or split in the dataset_name param
dataset = create_dataset(dataset_name="eth3d.train.mvd")
dataset = create_dataset(dataset_name="eth3d.train", dataset_type="mvd")
dataset = create_dataset(dataset_name="eth3d.mvd", split="train")
```

It is required to indicate a dataset name and a dataset type. The split can be specified with the optional 
`split` parameter. If the split is not specified, the default split is used. 

Instead of using the `dataset_type` and `split` parameters of `create_dataset`, it is possible to specify
the dataset type and split within the `dataset_name` parameter. The format for this is 
`base_dataset_name.split.dataset_type`, for example `eth3d.train.mvd`.

#### `to_torch` parameter
A dataset can be created with the parameter `to_torch=True`, e.g. `create_dataset("eth3d.mvd", to_torch=True)`. In this
case, samples will be converted to torch format, i.e. as torch tensors and not numpy arrays and with a prepended 
batch dimension.

#### `aug_fcts` parameter
A dataset can be created with the parameter `aug_fcts=[your_aug_1, your_aug_2]`. The specified augmentation functions
will be applied to all loaded samples.

#### `input_size` parameter
A dataset can be created with the parameter `input_size=(height, width)`. The input images (not the ground truth
, e.g. depth maps) will then be rescaled to the specified resolution. 

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
The rmvd datasets contain a convenience method to create dataloaders that load batched data in `numpy` format instead
of `torch` format:
```python
from rmvd import create_dataset
from rmvd.utils import numpy_collate
dataset = create_dataset("eth3d.mvd", input_size=(384, 576))
dataloader = dataset.get_loader(batch_size=4, shuffle=False, num_workers=2, collate_fn=numpy_collate)
```
