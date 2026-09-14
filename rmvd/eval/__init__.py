def create_evaluation(evaluation_type, **kwargs):
    assert evaluation_type in list_evaluations(), \
        f'The requested evaluation type "{evaluation_type}" does not exist. ' \
        f'Available evaluation types are: {" ".join(list_evaluations())}'

    if evaluation_type == 'mvd':
        from .multi_view_depth_evaluation import MultiViewDepthEvaluation
        return MultiViewDepthEvaluation(**kwargs)
    elif evaluation_type == 'robustmvd':
        from .robust_mvd_benchmark import RobustMultiViewDepthBenchmark
        return RobustMultiViewDepthBenchmark(**kwargs)
    else:
        raise NotImplementedError(f'Evaluation type "{evaluation_type}" is not implemented.')


def list_evaluations():
    return ['mvd', 'robustmvd']
