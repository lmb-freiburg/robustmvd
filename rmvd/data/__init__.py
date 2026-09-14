from .factory import create_dataset, create_compound_dataset, create_augmentation, create_batch_augmentation, \
    create_dataset_from_registry, create_dataset_from_cfg, create_dataset_from_folder, has_dataset_cfg
from .registry import register_dataset, list_data_formats, list_splits, list_datasets, has_dataset
from .registry import register_folder_parser, list_folder_parsers, has_folder_parser
from .registry import register_augmentation, list_augmentations, has_augmentation, register_batch_augmentation, list_batch_augmentations, has_batch_augmentation
from .transforms import list_resize_backends, DEFAULT_RESIZE_BACKEND

# import all datasets; this triggers the registration of the datasets in the registry
from .eth3d import ETH3DTrainRobustMVD, ETH3DTrainMVSAnywhere
from .kitti import KITTIRobustMVD, KITTIEigenDenseDepthTest, KITTIEigenDenseDepthTrain
from .dtu import DTURobustMVD, DTURobustMVDPaper
from .scannet import ScanNetRobustMVD
from .tanks_and_temples import TanksAndTemplesTrainRobustMVD
from .flyingthings3d import FlyingThings3DSeq4Train
from .staticthings3d import StaticThings3DSeq4Train
from .blendedmvs import BlendedMVSSeq4TrainSmall, BlendedMVSMVSNetTrain
from .rmvd_sample_parser import RmvdSampleMVDParser

# import all augmentations; this triggers the registration of the augmentations in the registry
from .augmentations import robust_mvd_augmentations_staticthings3d, robust_mvd_augmentations_blendedmvs, supervised_monodepth2_augmentations
from .batch_augmentations import robust_mvd_batch_augmentations
