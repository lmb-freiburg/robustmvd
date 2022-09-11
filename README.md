# Robust Multi-view Depth

Task: given multiple images of a scene, estimate the depth map for the keyimage. 

**Rob**ust **D**epth (**robd**) is a framework and benchmark for this task with a focus on robust application 
independent of the target data. It contains implementations and weights of recent models, training/evaluation/inference 
scripts, and benchmark results.

It supports multiple different input modalities:
- images
- intrinsics
- ground truth poses
- ground truth depth range (minimum and maximum values of the ground truth depth map)

robd is mainly focused on models that are implemented in pytorch, but provides means for evaluating tensorflow models. 

## Setup

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
The package can then be imported via `import robd`.

To use the dataloaders from robd, datasets need to be downloaded and some need to be preprocessed before 
they can be used. For details, see [robd/data/README.md](robd/data/README.md).

## Structure

The framework contains dataloaders, models, training/evaluation/inference scripts for multi-view depth estimation.
Further, it contains a set of tools for visualizing and reporting results.

The setup and interface of the dataloaders is explained in [robd/data/README.md](robd/data/README.md).

The setup and interface of the models is explained in [robd/models/README.md](robd/models/README.md).

The usage of the training, evaluation and inference scripts is explained below.

The usage of the tools is explained in [robd/tools/README.md](robd/tools/README.md).

## Evaluation script
Evaluation is done with the script `eval.py`, for example:
```bash
python eval.py --model robust_mvd --dataset eth3d --setting mvd --input poses intrinsics --output /tmp/eval_output --input_width 1152 --input_height 768
```

The parameters `model`, `dataset` and `setting` are required. Note that not all models and datasets support all
evaluation settings. For an overview, see the [models](robd/models/README.md) and [data](robd/data/README.md) readme.

For further parameters, execute `python eval.py --help`.

## Programmatic evaluation

It is also possible to run the evaluation from python code, for example with:
```python
import robd
model = robd.create_model("robust_mvd")  # call with num_gpus=0 for CPU usage
eval = robd.create_evaluation(evaluation_type="mvd", out_dir="/tmp/eval_output", inputs=["intrinsics", "poses"])
dataset = robd.create_dataset("eth3d", "mvd", input_size=(384, 576))
results = eval(dataset=dataset, model=model)
```

For further details, see the documentation. 

## Inference script
TODO

## Programmatic inference
`robd` models can be used programmatically, e.g.:
```python
import robd
model = robd.create_model("robust_mvd")
dataset = robd.create_dataset("eth3d", "mvd", to_torch=True, input_size=(384, 576))
sample = dataset[0]
pred, aux = model(**sample)  # TODO: add visualization call
```

## Training script
TODO

## Compatibility with tensorflow models
TODO

## Results

Results are provided for all models that are included in this framework. The results are obtained with the script
`eval.sh` and can be found in the `results` directory. 

TODO.

## Conventions and design decisions
Within this package, we use the following conventions:
- all data is in float32 format
- all data on an image grid uses CHW format (e.g. images are 3HW, depth maps 1HW); batches use NCHW format (e.g. images
  are N3HW, depth maps N1HW)
- models output predictions potentially at a downscaled resolution
- resolutions are indicated as (height, width) everywhere
- if the depth range of a scene is unknown, we consider a default depth range of (0.1m, 100m)
- all evaluations use numpy arrays as input and outputs to stay framework agnostic (TODO)
- all evaluations use batch size 1 in order to avoid confusing runtime measurements
- Training supports only pytorch models
- Evaluation uses numpy tensors, for cross-framework evaluation. All models need to have input and output adapter
  functions to convert to the required framework data format
- GT Depth values / Inverse depth values of <=0 indicate invalid values
- Predicted Depth / Inverse depth values of ==0 indiacte invalid values