from .registry import get_optimizer


def create_optimizer(name, **kwargs):
    """Creates an optimizer.

    Args:
        name (str): The name of the optimizer to create.

    Keyword Args:
        **kwargs: Additional arguments to pass to the optimizer.
    """
    optimizer_entrypoint = get_optimizer(name=name)
    optimizer = optimizer_entrypoint(**kwargs)
    return optimizer


def create_scheduler():
    return None  # TODO
