"""Model Registry.

Based on the model registry from the timm package ( https://github.com/rwightman/pytorch-image-models ).
"""


_model_entrypoints = {}


def register_model(model_entrypoint):
    """Register a model."""
    model_name = model_entrypoint.__name__
    _model_entrypoints[model_name] = model_entrypoint
    return model_entrypoint


def list_models():  # TODO: add filter, e.g. to list all model variants
    """List all available models."""
    models = _model_entrypoints.keys()
    return list(sorted(models))


def has_model(name):
    """Check if model is registered."""
    return name in _model_entrypoints


def get_model(name):
    """Get model entrypoint by name."""
    return _model_entrypoints[name]







# def register_model(model_entrypoint):
#     """Register a model."""
#     # use entrpypoint name as model name:
#     model_name = model_entrypoint.__name__
#     _model_entrypoints[model_name] = model_entrypoint
#
#     # check if the model has a defaults_cfg in the module of the model:
#     mod = sys.modules[model_entrypoint.__module__]
#     if hasattr(mod, 'defaults_cfg') and model_name in mod.defaults_cfg:
#         _model_cfgs[model_name] = mod.default_cfgs[model_name]
#
#     return model_entrypoint