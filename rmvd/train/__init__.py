from .training import Training


def create_training(training_type, **kwargs):
    assert training_type in list_trainings(), \
        f'The requested training type "{training_type}" does not exist. ' \
        f'Available training types are: {" ".join(list_trainings())}'

    if training_type == 'mvd':
        from .multi_view_depth_training import MultiViewDepthTraining
        return MultiViewDepthTraining(**kwargs)
    else:
        raise NotImplementedError(f'Training type "{training_type}" is not implemented.')


def list_trainings():
    return ['mvd']
