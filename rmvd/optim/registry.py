"""optimizer Registry.

Based on the model registry from the timm package ( https://github.com/rwightman/pytorch-image-models ).
"""


_optimizer_entrypoints = {}


def register_optimizer(optimizer_entrypoint):
    """Register an optimizer."""
    optimizer_name = optimizer_entrypoint.__name__
    _optimizer_entrypoints[optimizer_name] = optimizer_entrypoint
    return optimizer_entrypoint


def list_optimizers():
    """List all available optimizers."""
    optimizers = _optimizer_entrypoints.keys()
    optimizers = list(sorted(optimizers))
    return optimizers


def has_optimizer(name):
    """Check if optimizer is registered."""
    return name in _optimizer_entrypoints


def get_optimizer(name):
    """Get optimizer entrypoint by name."""
    assert has_optimizer(name), f'The requested optimizer "{name}" does not exist. Available optimizers are: {" ".join(list_optimizers())}'
    return _optimizer_entrypoints[name]


def list_schedulers():
    return []  # TODO
