def create_inference(inference_type, **kwargs):
    assert inference_type in list_inference_types(), \
        f'The requested inference type "{inference_type}" does not exist. ' \
        f'Available inference types are: {" ".join(list_inference_types())}'

    if inference_type == 'mvd':
        from .multi_view_depth_inference import MultiViewDepthInference
        return MultiViewDepthInference(**kwargs)
    else:
        raise NotImplementedError(f'Inference type "{inference_type}" is not implemented.')


def list_inference_types():
    return ['mvd']
