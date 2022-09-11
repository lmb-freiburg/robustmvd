from .training import Training


def create_training(**kwargs):
    return Training(**kwargs)


def list_trainings():
    return ['mvd']
