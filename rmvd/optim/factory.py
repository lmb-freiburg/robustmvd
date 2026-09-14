from .registry import _get_optimizer, _get_scheduler, has_optimizer, has_scheduler, list_optimizers, list_schedulers


def create_optimizer(name, model, **kwargs):
    """Creates an optimizer.

    Args:
        name (str): The name of the optimizer to create.
        model (torch.nn.Module): The model to optimize.

    Keyword Args:
        **kwargs: Additional arguments to pass to the optimizer.
    """
    assert has_optimizer(name), f'The requested optimizer "{name}" does not exist. Available optimizers are: {" ".join(list_optimizers())}'
    optimizer_entrypoint = _get_optimizer(name=name)
    optimizer = optimizer_entrypoint(model=model, **kwargs)
    optimizer.name = name
    return optimizer


def create_scheduler(name, optimizer, **kwargs):
    """Creates a scheduler.

    Args:
        name (str): The name of the scheduler to create.
        optimizer (torch.optim.Optimizer): The optimizer to schedule.

    Keyword Args:
        **kwargs: Additional arguments to pass to the scheduler.
    """
    assert has_scheduler(name), f'The requested scheduler "{name}" does not exist. Available schedulers are: {" ".join(list_schedulers())}'
    scheduler_entrypoint = _get_scheduler(name=name)
    scheduler = scheduler_entrypoint(optimizer=optimizer, **kwargs)
    scheduler.name = name
    return scheduler
