import importlib
import collections
import re

import numpy as np
import torch
from torch._six import string_classes


def get_function(name):  # from https://github.com/aschampion/diluvian/blob/master/diluvian/util.py
    mod_name, func_name = name.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    return func


def get_class(name):
    return get_function(name)


def module_exists(name):
    spec = importlib.util.find_spec(name)
    return (spec is not None)


def function_exists(name):
    mod_name, fct_name = name.rsplit('.', 1)
    spec = importlib.util.find_spec(mod_name)
    if spec is None:
        return False
    else:
        mod = importlib.import_module(mod_name)
        return hasattr(mod, fct_name)


def class_exists(name):
    return function_exists(name)


def get_full_class_name(obj):
    module = obj.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return obj.__class__.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + obj.__class__.__name__


def transform_from_rot_trans(R, t):
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1])).astype(np.float32)


def invert_transform(T):
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    R_inv = R.T
    t_inv = np.dot(-R.T, t)
    return transform_from_rot_trans(R_inv, t_inv)


def to_cuda(data, device=None):
    if isinstance(data, dict):
        return {k: to_cuda(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_cuda(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_cuda(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.cuda(device=device)
    else:
        return data


def to_torch(data):  # TODO: this only works for non-batched input data; rename to torch_collate
    return torch.utils.data._utils.collate.default_collate([data])


def to_numpy(data):
    pass # TODO


def numpy_collate(batch):
    # adapted from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    err_msg_format = "numpy_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {}"

    if batch is None:
        return None

    elem = batch[0]
    elem_type = type(elem)

    if elem is None:
        assert all(elem is None for elem in batch)
        return None

    elif isinstance(elem, torch.Tensor):
        return numpy_collate([b.cpu().detach().numpy() for b in batch])

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return np.stack(batch, 0)
        elif elem.shape == ():  # scalars
            return batch  # do nothing

    elif isinstance(elem, float):
        return np.array(batch)

    elif isinstance(elem, int):
        return np.array(batch)

    elif isinstance(elem, string_classes):
        return batch

    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: numpy_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: numpy_collate([d[key] for d in batch]) for key in elem}

    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(numpy_collate(samples) for samples in zip(*batch)))

    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [numpy_collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([numpy_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [numpy_collate(samples) for samples in transposed]

    raise TypeError(err_msg_format.format(elem_type))


def get_torch_model_device(model):
    # make sure that all parameters are on the same device:
    it = iter(model.parameters())
    is_cuda = next(it).is_cuda
    device = next(it).device
    if not all((elem.device == device) for elem in it):
        raise RuntimeError('All model parameters need to be on the same device')
    return device


def check_torch_model_cuda(model):
    # make sure that model parameters are all on the GPU or all on the CPU:
    it = iter(model.parameters())
    is_cuda = next(it).is_cuda
    if not all((elem.is_cuda == is_cuda) for elem in it):
        raise RuntimeError('All model parameters need to be on the same device')
    return is_cuda
