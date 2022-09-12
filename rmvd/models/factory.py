from .registry import get_model
from .helpers import add_run_function


def create_model(name, pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    """Creates a model.

    Args:
        name (str): The name of the model to create.
        pretrained (bool): Whether to load the default pretrained weights for the model.
        weights (str): Path to custom weights to be loaded. Overrides `pretrained`.
        train (bool): Whether to put the model in train mode.
        num_gpus (int): Number of GPUs to be used from the model.

    Keyword Args:
        **kwargs: Additional arguments to pass to the model.
    """
    model_entrypoint = get_model(name=name)
    model = model_entrypoint(pretrained=pretrained, weights=weights, train=train, num_gpus=num_gpus, **kwargs)
    add_run_function(model)
    return model
