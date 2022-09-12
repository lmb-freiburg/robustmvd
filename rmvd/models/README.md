# Models

rmvd contains implementations of depth estimation models. The usage of the models is described in the following.

## Usage

### Initialization

To initialize a model, use the `create_model` function:
```python
from rmvd import create_model
model = create_model(model_name, pretrained=True, weights=None, train=False, num_gpus=1)  # optional: model-specific parameters
```

#### Model variants
Some models support multiple variants/configurations. They are represented by different model names. 
For all list of all models including their configurations, use the `list_models` function:
```python
from rmvd import list_models
print(list_models())
```

#### Weights

If `pretrained` is set to True, the default pretrained weights for the model will be used. The default weights
are automatically downloaded at first use. 
Alternatively, custom weights can be loaded by providing the path to the weights with the `weights` parameter.

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

#### Internal implementation

Internally, the `run` function works as follows:
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

##### The `input_adapter` function

The `input_adapter` function has the following interface: 
```python
def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
    # construct sample dict that contains all inputs in the model-specific format: sample = {..}
    return sample
```
The inputs to the `input_adapter` function are all `numpy` array with a batch dimension 
(e.g. images are `N3HW` and of type `np.ndarray`). The function then converts all inputs to the format that
is required by the model and returns this converted data as a dictionary where the keys are the parameter names
of the model's `__call__` function. This allows to call `model(**sample)` where sample is the dictionary that is 
return from the `input_adapter` function.

The conversion may for example include converting the inputs to `torch.Tensor`, moving them to the GPU if required, 
normalizing the images, etc.

##### The `__call__` function
The `__call__` function of each model expects data in the model-specific format and returns model-specific outputs.

Hence, in case all input data is already in the format required by the model, you can also do `model(**sample)`. 
This is used in the provided code for training. 

##### The `output_adapter` function
The `output_adapter` function has the following interface:
```python
def output_adapter(self, model_output):
    # construct pred and aux dicts from model_output
    # pred needs to have an item with key "depth" and value of type np.ndarray and shape N1HW
    return pred, aux
```
The output adapter converts model-specific outputs to the `pred` and `aux` dictionaries. The output types and shapes
need to be numpy arrays with a batch dimension (i.e. `depth` has shape `N1HW` and type `np.ndarray`). 

#### Resolution
Most models cannot handle input images at arbitrary resolutions. Models therefore internally downsize the images to
the next resolution that can be handled. 

The model output is often at a lower resolution as the input data.

## Using a custom model within the `robustmvd` framework

If you want to use your own model within the framework, e.g. for evaluation, your model needs to have:
- a `input_adapter` function
- a `__call__` function (in `torch` basically equivalent to the `forward` function) 
- a `output_adapter` function

Note: you don't have to add a `run` function your model. If required, this function will be added automatically.

## Wrapped Models

The framework contains two types of model implementations:
1. Native model implementations: models that are implemented in pytorch within the framework and that 
adhere to the interfaces of the framework
2. Model wrappers: wrappers around existing implementations

The usage of wrapped models is a bit more involved, as it is required to download the original code 
and set up an appropriate environment. For each wrapped model, a short setup guide is therefore provided in the 
following.
