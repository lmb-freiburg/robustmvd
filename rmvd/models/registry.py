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
