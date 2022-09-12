from .multi_view_depth_evaluation import MultiViewDepthEvaluation


def create_evaluation(evaluation_type, **kwargs):
    if evaluation_type == 'mvd':
        return MultiViewDepthEvaluation(**kwargs)


def list_evaluations():
    return ['mvd']
