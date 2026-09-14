from .registry import _get_loss, has_loss, list_losses


def create_loss(name, **kwargs):
    """Creates a loss.

    Args:
        name (str): The name of the loss to create.

    Keyword Args:
        **kwargs: Additional arguments to pass to the loss.
    """
    assert has_loss(name), f'The requested loss "{name}" does not exist. Available losses are: {" ".join(list_losses())}'
    loss_entrypoint = _get_loss(name=name)
    loss = loss_entrypoint(**kwargs)
    return loss
