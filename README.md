# Robust Multi-view Depth
[**Paper**](TODO) | [**Video**](TODO) | [**Project Page**](TODO)

**Robust** **M**ulti-**v**iew **D**epth (`robustmvd`) is a benchmark and framework for depth estimation 
from multiple input views with a focus on robust application independent of the target data. 
This repository contains evaluation code for the Robust Multi-view Depth Benchmark. 
Further, it contains implementations and weights of recent models together with inference scripts.

In the following, we first describe the setup, structure, and usage of the `rmvd` framework. 
Following this, we describe the usage of the Robust Multi-view Depth Benchmark. 

## The `rmvd` framework

### Setup

The code was tested with python 3.8 and PyTorch 1.9 on Ubuntu 20.04. 
For the setup, either the `requirements.txt` or the `setup.py` can be used.

To install the requirements, run:
```bash
pip install -r requirements.txt
```

To install the package using the `setup.py`, run:
```bash
python setup.py install
```
The package can then be imported via `import rmvd`.

To use the dataloaders from `rmvd`, datasets need to be downloaded and some need to be preprocessed before 
they can be used. For details, see [rmvd/data/README.md](rmvd/data/README.md).

### Structure

The `rmvd` framework contains dataloaders, models, evaluation/inference scripts for multi-view depth 
estimation. 

The setup and interface of the dataloaders is explained in [rmvd/data/README.md](rmvd/data/README.md).

The setup and interface of the models is explained in [rmvd/models/README.md](rmvd/models/README.md).

### Evaluation script
Evaluation is done with the script `eval.py`, for example:
```bash
python eval.py --model robust_mvd --dataset eth3d --setting mvd --input poses intrinsics --output /tmp/eval_output --input_width 1152 --input_height 768
```

The parameters `model`, `dataset` and `setting` are required. Note that not all models and datasets support all
evaluation settings. For an overview, see the [models](rmvd/models/README.md) and [data](rmvd/data/README.md) readme.

For further parameters, execute `python eval.py --help`.

### Programmatic evaluation

It is also possible to run the evaluation from python code, for example with:
```python
import rmvd
model = rmvd.create_model("robust_mvd", num_gpus=1)  # call with num_gpus=0 for CPU usage
eval = rmvd.create_evaluation(evaluation_type="mvd", out_dir="/tmp/eval_output", inputs=["intrinsics", "poses"])
dataset = rmvd.create_dataset("eth3d", "mvd", input_size=(384, 576))
results = eval(dataset=dataset, model=model)
```

Further details (e.g. additional function parameters, overview of available models, evaluations and datasets, ..)
are described in the READMEs mentioned above.

### Inference script
Evaluation is done with the script `inference.py`, for example:
```bash
python inference.py --model robust_mvd
```

### Programmatic inference
`rmvd` models can be used programmatically, e.g.:
```python
import rmvd
model = rmvd.create_model("robust_mvd")
dataset = rmvd.create_dataset("eth3d", "mvd", input_size=(384, 576))
sample = dataset[0]
pred, aux = model.run(**sample)
```

### Conventions and design decisions
Within this package, we use the following conventions:
- all data is in float32 format
- all data on an image grid uses CHW format (e.g. images are 3HW, depth maps 1HW); batches use NCHW format (e.g. images
  are N3HW, depth maps N1HW)
- models output predictions potentially at a downscaled resolution
- resolutions are indicated as (height, width) everywhere
- if the depth range of a scene is unknown, we consider a default depth range of (0.1m, 100m)
- all evaluations use numpy arrays as input and outputs
- all evaluations use a batch size of 1 for consistent runtime measurements
- GT depth values / inverse depth values of <=0 indicate invalid values
- predicted depth / inverse depth values of ==0 indiacte invalid values

## Robust Multi-view Depth Benchmark

The Robust Multi-view Depth Benchmark aims to evaluate robust multi-view depth estimation on arbitrary real-world data.
As proxy for this, it uses test sets based on multiple diverse, existing datasets and evaluates in a zero-shot fashion.

It supports multiple different input modalities:
- images
- intrinsics
- ground truth poses
- ground truth depth range (minimum and maximum values of the ground truth depth map)

It optionally supports alignment of predicted and ground truth depth maps to account for scale-ambiguity of some models.

Depth and uncertainty estimation performance is measured with the following metrics:
- Absolute relative error (rel)
- Inliers with a threshold of 1.03
- Sparsification Error Curves
- Area Under Sparsification Error (AUSE)

The following describes how to evaluate on the benchmark.

### Evaluation of models within the `rmvd` framework

### Evaluation of custom models

## TODOs
- [ ] add all models that were evaluated in the publication to the `rmvd` framework
- [ ] add more datasets: KITTI, ScanNet, DTU, Tanks and Temples
- [ ] add project page including an overview of the benchmark and a leaderboard
- [ ] add code to train the models

## Changelog
### Sept 13 2022 First version

## Citation
This is the official repository for the publication:
> **A Benchmark and a Baseline for Robust Multi-view Depth Estimation**
>
> [Philipp Schröppel*](https://lmb.informatik.uni-freiburg.de/people/schroepp), [Jan Bechtold*](https://lmb.informatik.uni-freiburg.de/people/bechtolj), [Artemij Amiranashvili](https://lmb.informatik.uni-freiburg.de/people/amiranas) and [Thomas Brox](https://lmb.informatik.uni-freiburg.de/people/brox)
> 
> **3DV 2022**

If you find our work useful, please consider citing:
```bibtex
@inproceedings{schroeppel2022robust,
  author     = {Philipp Schr\"oppel and Jan Bechtold and Artemij Amiranashvili and Thomas Brox},
  booktitle  = {Proceedings of the International Conference on {3D} Vision ({3DV})},
  title      = {A Benchmark and a Baseline for Robust Multi-view Depth Estimation},
  year       = {2022}
}
```
