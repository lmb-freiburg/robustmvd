from .factory import create_dataset, create_compound_dataset
from .registry import register_dataset, list_base_datasets, list_dataset_types, list_splits, list_datasets, has_dataset

# import all datasets; this triggers the registration of the datasets in the registry
from .eth3d import ETH3DTrainRobustMVD
from .kitti import KITTITest
from .dtu import DTUTest
# from .flyingthings3d import FlyingThings3DSeq4Train
# from .blendedmvs import BlendedMVSSeq4Train
