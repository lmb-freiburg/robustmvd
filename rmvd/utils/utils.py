import importlib
import collections
import os.path as osp
import re
import random

import numpy as np
import torch
import pytoml


string_classes = (str, bytes)


def set_random_seed(seed=None, use_gpu=True):
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_depth_range(depth=None, invdepth=None, default_min_depth=0.1, default_max_depth=100.):
    if depth is not None:
        mask = depth > 0
        if mask.any():
            min_depth = np.min(depth[mask])
            max_depth = np.max(depth[mask])
            depth_range = (min_depth, max_depth)
            return depth_range
        
    if invdepth is not None:
        mask = invdepth > 0
        if mask.any():
            min_invdepth = np.min(invdepth[mask])
            max_invdepth = np.max(invdepth[mask])
            depth_range = (1./max_invdepth, 1./min_invdepth)
            return depth_range
        
    return (np.float32(default_min_depth), np.float32(default_max_depth))


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


def trans_from_transform(T):  # can handle np arrays and torch tensors
    if T.ndim == 2:
        t = T[0:3, 3]
    elif T.ndim == 3:
        t = T[:, 0:3, 3]
    else:
        raise ValueError("Transform must have 2 or 3 dimensions, but has {} dimensions.".format(T.ndim))

    return t


def rot_from_transform(T):  # can handle np arrays and torch tensors
    if T.ndim == 2:
        R = T[0:3, 0:3]
    elif T.ndim == 3:
        R = T[:, 0:3, 0:3]
    else:
        raise ValueError("Transform must have 2 or 3 dimensions, but has {} dimensions.".format(T.ndim))

    return R


def _transform_from_rot_trans_np(R, t):
    if R.ndim == 2:
        assert t.ndim == 1, "If rotation has no batch dimension, translation must have 1 dimension, but has {} dimensions.".format(t.ndim)
        t = t[..., np.newaxis]
        last_row = np.array([0, 0, 0, 1], dtype=R.dtype)
        T = np.vstack((np.hstack([R, t]), last_row))  # [4, 4]

    elif R.ndim == 3:
        N = R.shape[0]
        assert t.ndim == 2, "If rotation has a batch dimension, translation must have 2 dimensions, but has {} dimensions.".format(t.ndim)
        assert t.shape[0] == N, "Batch dimension of rotation and translation must match, but have {} and {}.".format(R.shape[0], t.shape[0])
        t = t[..., np.newaxis]  # [N, 3, 1]
        T = np.concatenate((R, t), -1)  # [N, 3, 4]
        last_row = np.array([0, 0, 0, 1], dtype=T.dtype)
        last_row = np.tile(last_row[np.newaxis, np.newaxis, :], (N, 1, 1))  # [N, 1, 4]
        T = np.concatenate((T, last_row), 1)  # [N, 4, 4]

    else:
        raise ValueError("Rotation must have 2 or 3 dimensions, but has {} dimensions.".format(R.ndim))

    return T


def _transform_from_rot_trans_torch(R, t):
    if R.ndim == 2:
        assert t.ndim == 1, "If rotation has no batch dimension, translation must have 1 dimension, but has {} dimensions.".format(t.ndim)
        t = t.unsqueeze(-1)  # [3, 1]
        T = torch.vstack((torch.hstack([R, t]), torch.tensor([0, 0, 0, 1], dtype=R.dtype, device=R.device)))  # [4, 4]

    elif R.ndim == 3:
        N = R.shape[0]
        assert t.ndim == 2, "If rotation has a batch dimension, translation must have 2 dimensions, but has {} dimensions.".format(t.ndim)
        assert t.shape[0] == N, "Batch dimension of rotation and translation must match, but have {} and {}.".format(R.shape[0], t.shape[0])
        t = t.unsqueeze(-1)  # [N, 3, 1]
        T = torch.cat((R, t), -1)  # [N, 3, 4]
        last_row = torch.tensor([0, 0, 0, 1], dtype=T.dtype, device=T.device)
        last_row = last_row.unsqueeze(0).unsqueeze(0).expand(N, 1, 4)  # [N, 1, 4]
        T = torch.cat((T, last_row), 1)  # [N, 4, 4]

    else:
        raise ValueError("Rotation must have 2 or 3 dimensions, but has {} dimensions.".format(R.ndim))

    return T


def transform_from_rot_trans(R, t):  # can handle np arrays and torch tensors
    if isinstance(R, np.ndarray):
        return _transform_from_rot_trans_np(R, t)
    elif isinstance(R, torch.Tensor):
        return _transform_from_rot_trans_torch(R, t)
    else:
        raise ValueError("Rotation/Translation must be numpy arrays or torch tensors.")


def _invert_transform_np(T):
    R = rot_from_transform(T)  # [3, 3] or [N, 3, 3]
    t = trans_from_transform(T)  # [3] or [N, 3]
    t = t[..., np.newaxis]  # [3, 1] or [N, 3, 1]
    R_inv = np.swapaxes(R, -1, -2)  # [3, 3] or [N, 3, 3]
    t_inv = np.matmul(-R_inv, t)[..., 0]  # [3] or [N, 3]
    return transform_from_rot_trans(R_inv, t_inv)


def _invert_transform_torch(T):
    R = rot_from_transform(T)  # [3, 3] or [N, 3, 3]
    t = trans_from_transform(T)  # [3] or [N, 3]
    t = t.unsqueeze(-1)  # [3, 1] or [N, 3, 1]
    R_inv = torch.transpose(R, -1, -2)  # [3, 3] or [N, 3, 3]
    t_inv = torch.matmul(-R_inv, t).squeeze(-1)  # [3] or [N, 3]
    return transform_from_rot_trans(R_inv, t_inv)


def invert_transform(T):  # can handle np arrays and torch tensors
    if isinstance(T, np.ndarray):
        return _invert_transform_np(T)
    elif isinstance(T, torch.Tensor):
        return _invert_transform_torch(T)
    else:
        raise ValueError("Transform must be a numpy array or a torch tensor.")


def rotx(t):                                                                    
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], 
                     [0, c, -s],
                     [0, s, c]])                                                                                                                                                              


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


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


def torch_collate(batch):
    if batch is None:
        return None

    return torch.utils.data._utils.collate.default_collate(batch)


def to_torch(data, device=None):
    # adapted from torch.utils.data._utils.collate.default_convert
    np_str_obj_array_pattern = re.compile(r'[SaUO]')

    if data is None:
        return None

    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data, device=device)
    elif isinstance(data, collections.abc.Mapping):
        try:
            return elem_type({key: to_torch(data[key], device=device) for key in data})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`...
            return {key: to_torch(data[key], device=device) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(to_torch(d, device=device) for d in data))
    elif isinstance(data, tuple):
        return [to_torch(d, device=device) for d in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        try:
            return elem_type([to_torch(d, device=device) for d in data])
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [to_torch(d, device=device) for d in data]
    else:
        return data


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
            return np.array(batch)

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


def to_numpy(data):
    if data is None:
        return None

    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        return data
    elif isinstance(data, collections.abc.Mapping):
        try:
            return elem_type({key: to_numpy(data[key]) for key in data})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: to_numpy(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(to_numpy(d) for d in data))
    elif isinstance(data, tuple):
        return [to_numpy(d) for d in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        try:
            return elem_type([to_numpy(d) for d in data])
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [to_numpy(d) for d in data]
    else:
        return data


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


def count_torch_model_parameters(model, requires_grad_only=False):
    params = model.parameters()  # shared/tied parameters are deduplicated by parameters()
    return sum(p.numel() for p in params if p.requires_grad or not requires_grad_only)


def select_by_index(l, idx):
    """Select an element from a list by an index. Supports data batches with different indices.

    Args:
        l (list): List with potentially batched data items.
        idx: idx can be an integer in case of non-batched data or in case samples in the batch have the same index.
            Alternatively, idx can be an iterable that contains indices for each sample in the batch separately.
    """
    if isinstance(idx, int):
        ret = l[idx]
    else:
        indices = idx
        ret = []
        for batch_idx, idx in enumerate(indices):
            ret.append(l[idx][batch_idx])

        if isinstance(ret[0], np.ndarray):
            ret = np.stack(ret, 0)
        else:
            ret = torch.stack(ret, 0)

    return ret


def exclude_index(l, exclude_idx):
    """Selects all element from a list, excluding a specific index. Supports data batches with different indices.

    Args:
        l (list): List with potentially batched data items.
        idx: idx can be an integer in case of non-batched data or in case samples in the batch have the same index.
            Alternatively, idx can be an iterable that contains indices for each sample in the batch separately.
    """
    if isinstance(exclude_idx, int):
        ret = [ele for idx, ele in enumerate(l) if idx != exclude_idx]
    else:
        exclude_indices = exclude_idx
        ret = []
        for batch_idx, exclude_idx in enumerate(exclude_indices):
            ret.append([ele[batch_idx] for idx, ele in enumerate(l) if idx != exclude_idx])

        if len(l) == 1 and len(ret) > 0 and all(len(ele) == 0 for ele in ret):
            return []

        if len(ret) > 0 and all([len(ele) > 0 for ele in ret]):
            transposed = list(zip(*ret))
            if isinstance(transposed[0][0], np.ndarray):
                ret = [np.stack(ele, 0) for ele in transposed]
            else:
                ret = [torch.stack(ele, 0) for ele in transposed]

    return ret


def insert_at_index(l, idx, ele):
    """Insert an element into a list at an index. Supports data batches with different indices.

    Args:
        l (list): List with potentially batched data items. In case idx is batched, each element in l must be batched with the same first dimension as idx.
        idx: idx can be an integer in case of non-batched data or in case all samples in the batch should be inserted at the same index.
            Alternatively, idx can be an iterable that contains indices for each sample in the batch separately.
        ele: The element to insert. In case idx is batched, ele must be batched with the same first dimension as idx.
    Returns:
        list: New list with length len(l) + 1.
    """
    def _to_int(x):
        # robustly convert numpy/torch scalars to Python int
        if isinstance(x, int):
            return x
        if hasattr(x, "item"):
            try:
                return int(x.item())
            except Exception:
                pass
        return int(x)

    n = len(l)

    if isinstance(idx, int):
        pos = idx
        if pos < 0:
            pos = 0
        if pos > n:
            pos = n
        ret = l[:pos] + [ele] + l[pos:]

    else:
        indices = [_to_int(i) for i in idx]
        indices = [0 if i < 0 else (n if i > n else i) for i in indices]

        ret = []
        for out_pos in range(n + 1):
            pieces = []
            for batch_idx, ins_pos in enumerate(indices):
                if ins_pos == out_pos:
                    pieces.append(ele[batch_idx])
                else:
                    src = out_pos if out_pos < ins_pos else out_pos - 1
                    pieces.append(l[src][batch_idx])

            if isinstance(pieces[0], np.ndarray):
                stacked = np.stack(pieces, 0)
            else:
                stacked = torch.stack(pieces, 0)

            ret.append(stacked)

    return ret


def batched_index(l, eles):
    """Gets the indices of a batch of elements in a list.

    This is basically the equivalent of l.index(ele) in a batched version.

    Args:
        l (list): List with batched data items.
        eles: Iterable that contains elemenets for each sample in the batch.
    """
    ret = []
    for batch_idx, ele in enumerate(eles):
        batch_l = [x[batch_idx] for x in l]
        ret.append(batch_l.index(ele))

    if isinstance(ret[0], np.ndarray):
        ret = np.stack(ret, 0)
    else:
        ret = torch.stack(ret, 0)

    return ret


def get_paths_file(local_paths_file, home_paths_file, readme):
    """Get the paths.toml file, either from the local (prioritized) or the home location."""
    if osp.exists(local_paths_file):
        paths_file = local_paths_file
    elif osp.exists(home_paths_file):
        paths_file = home_paths_file
    else:
        raise FileNotFoundError(f"No paths.toml file found at {local_paths_file} or {home_paths_file}. Please create "
                                f"a paths.toml file as specified in the {readme} file.")

    return paths_file


def get_paths(paths_file):
    with open(paths_file, 'r') as paths_file:
        return pytoml.load(paths_file)


def get_path(paths_file, *keys):
    current = get_paths(paths_file)

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]

    if not isinstance(current, (str, list)):
        return None

    if isinstance(current, list):
        return [osp.expanduser(p) for p in current]
    return osp.expanduser(current)


def normalize(x, dim=None, eps=1e-9):
    norm = torch.linalg.norm(x, dim=dim, keepdim=True)
    return x / (norm + eps)
