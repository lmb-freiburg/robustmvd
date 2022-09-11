import robd.data

def test_build_dataset_names():
    assert robd.data.registry._build_dataset_name('eth3d.train.mvd') == 'eth3d.train.mvd'
    assert robd.data.registry._build_dataset_name('eth3d.train.mvd', dataset_type='mvd') == 'eth3d.train.mvd'
    assert robd.data.registry._build_dataset_name('eth3d.train.mvd', split='train') == 'eth3d.train.mvd'
    assert robd.data.registry._build_dataset_name('eth3d.train.mvd', dataset_type='mvd', split='train') == 'eth3d.train.mvd'
    assert robd.data.registry._build_dataset_name('eth3d.train', dataset_type='mvd') == 'eth3d.train.mvd'
    assert robd.data.registry._build_dataset_name('eth3d.mvd', split='train') == 'eth3d.train.mvd'
    assert robd.data.registry._build_dataset_name('eth3d.mvd') == 'eth3d.train.mvd'
    assert robd.data.registry._build_dataset_name('eth3d', dataset_type='mvd', split='train') == 'eth3d.train.mvd'
    assert robd.data.registry._build_dataset_name('eth3d.train') == 'eth3d.train'
    assert robd.data.registry._build_dataset_name('eth3d') == 'eth3d'


def test_split_dataset_names():
    assert robd.data.registry._split_dataset_name("eth3d.train.mvd", None, None) == ("eth3d", "mvd", "train")
    assert robd.data.registry._split_dataset_name("eth3d.train", "mvd", None) == ("eth3d", "mvd", "train")
    assert robd.data.registry._split_dataset_name("eth3d", "mvd", "train") == ("eth3d", "mvd", "train")
    assert robd.data.registry._split_dataset_name("eth3d.train", "mvd", "train") == ("eth3d", "mvd", "train")
    assert robd.data.registry._split_dataset_name("eth3d.mvd", "mvd", "train") == ("eth3d", "mvd", "train")
    assert robd.data.registry._split_dataset_name("eth3d.mvd", None, "train") == ("eth3d", "mvd", "train")
    assert robd.data.registry._split_dataset_name("eth3d.mvd", None, None) == ("eth3d", "mvd", "train")
    assert robd.data.registry._split_dataset_name("eth3d", "mvd", None) == ("eth3d", "mvd", "train")
