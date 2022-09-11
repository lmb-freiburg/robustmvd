import argparse
import os
import os.path as osp
import math

import torch
import numpy as np
from PIL import Image

from robust_mvd_model.robust_mvd_model import RobustMVDModel
import utils.vis as vis


def transform_from_rot_trans(R, t):
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1])).astype(np.float32)


def invert_transform(T):
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    R_inv = R.T
    t_inv = np.dot(-R.T, t)
    return transform_from_rot_trans(R_inv, t_inv)


def preprocess_view(image, intrinsics, to_ref_transform):
    w_orig, h_orig = image.size
    h_in = int(math.ceil(h_orig / 64.0) * 64.0)
    w_in = int(math.ceil(w_orig / 64.0) * 64.0)

    image = image.resize((w_in, h_in), Image.BILINEAR)
    image = np.array(image)
    image = ((image / 255.0) - 0.4).astype(np.float32)
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, 0)  # 1, 3, H, W
    image = torch.from_numpy(image).float().cuda()

    scale_arr = np.array([[1 / w_orig] * 3, [1 / h_orig] * 3, [1.] * 3])
    intrinsics *= scale_arr
    intrinsics = np.expand_dims(np.expand_dims(intrinsics, axis=0), axis=0).astype(np.float32)
    intrinsics = torch.from_numpy(intrinsics).float().cuda()

    to_ref_transform = np.expand_dims(np.expand_dims(to_ref_transform, axis=0), axis=0).astype(np.float32)
    to_ref_transform = torch.from_numpy(to_ref_transform).float().cuda()

    return image, intrinsics, to_ref_transform


def load_data(path):
    key_path = osp.join(path, "key")
    src_path = osp.join(path, "source")
    src_paths = sorted([osp.join(src_path, x) for x in os.listdir(src_path)])

    image_key = Image.open(osp.join(key_path, 'image.png'))
    intrinsics_key = np.load(osp.join(key_path, 'K.npy'))
    key_to_ref_transform = np.load(osp.join(key_path, 'to_ref_transform.npy'))
    ref_to_key_transform = invert_transform(key_to_ref_transform)
    w_orig, h_orig = image_key.size
    image_key, intrinsics_key, _ = preprocess_view(image_key, intrinsics_key, key_to_ref_transform)

    images_source = []
    source_to_key_transforms = []
    intrinsics_sources = []
    for src_path in src_paths:
        image_source = Image.open(osp.join(src_path, 'image.png'))
        intrinsics_source = np.load(osp.join(src_path, 'K.npy'))
        source_to_ref_transform = np.load(osp.join(src_path, 'to_ref_transform.npy'))
        source_to_key_transform = np.dot(source_to_ref_transform, ref_to_key_transform)
        image_source, intrinsics_source, source_to_key_transform = preprocess_view(image_source, intrinsics_source, source_to_key_transform)

        images_source.append(image_source)
        source_to_key_transforms.append(source_to_key_transform)
        intrinsics_sources.append(intrinsics_source)

    return image_key, intrinsics_key, images_source, source_to_key_transforms, intrinsics_sources, h_orig, w_orig


def write_pred(pred_invdepth, pred_uncertainty, pred_depth):
    os.makedirs(args.output, exist_ok=True)
    np.save(osp.join(args.output, "pred_invdepth.npy"), pred_invdepth)
    np.save(osp.join(args.output, "pred_uncertainty.npy"), pred_uncertainty)
    np.save(osp.join(args.output, "pred_depth.npy"), pred_depth)

    vis.np2d(pred_invdepth).save(osp.join(args.output, "pred_invdepth.png"))
    vis.np2d(pred_depth).save(osp.join(args.output, "pred_depth.png"))
    vis.np2d(pred_uncertainty, image_range_text_off=True).save(osp.join(args.output, "pred_uncertainty.png"))


@torch.no_grad()
def run(args):
    print("Processing data from {} with weights from {}.".format(args.input, args.weights))

    model = RobustMVDModel()
    model.load_state_dict(torch.load(args.weights))
    model.cuda()
    model.eval()

    image_key, intrinsics_key, images_source, source_to_key_transforms, intrinsics_sources, h_orig, w_orig = load_data(args.input)
    pred = model(image_key, intrinsics_key, images_source, source_to_key_transforms, intrinsics_sources)

    pred_invdepth = pred['pred_invdepth'].cpu().numpy()
    pred_uncertainty = pred['pred_invdepth_uncertainty'].cpu().numpy()

    pred_invdepth = np.array(Image.fromarray(pred_invdepth.squeeze()).resize((w_orig, h_orig), Image.NEAREST)).astype(np.float32)
    pred_uncertainty = np.array(Image.fromarray(pred_uncertainty.squeeze()).resize((w_orig, h_orig), Image.NEAREST)).astype(np.float32)
    pred_depth = 1 / (pred_invdepth + 1e-9)

    write_pred(pred_invdepth, pred_uncertainty, pred_depth)

    print("Done. Output written to {}.".format(args.output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default="sample_data", help="Path to folder with input data.")
    parser.add_argument('--output', default="sample_data/out", help="Path to folder for output data.")
    parser.add_argument('--weights', default="weights/robustmvd.pt", help="Weights for the Robust MVD model.")
    args = parser.parse_args()

    run(args)
