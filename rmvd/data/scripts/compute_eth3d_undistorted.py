#!/usr/bin/env python3
import argparse
import os
import os.path as osp
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm

SCENES = ['courtyard', 'delivery_area', 'electro', 'facade', 'kicker', 'meadow', 'office', 'pipes',
          'playground', 'relief', 'relief_2', 'terrace', 'terrains']

IMAGE_DIR = osp.join('images', 'dslr_images')
IMAGE_OUT_DIR = osp.join('images', 'dslr_images_undistorted_by_rmvd')
IMAGE_REF_DIR = osp.join('images', 'dslr_images_undistorted')
DEPTH_DIR = osp.join('ground_truth_depth', 'dslr_images')
DEPTH_OUT_DIR = osp.join('ground_truth_depth', 'dslr_images_undistorted_by_rmvd')
CALIB_DIR = 'dslr_calibration_jpg'
CALIB_REF_DIR = 'dslr_calibration_undistorted'

ROW_CHUNK = 256  # rows per chunk when building the per-camera maps, to bound peak memory


def read_cameras_txt(path):
    """Read a COLMAP cameras.txt. Returns {camera_id: (model, width, height, params)}."""
    cameras = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            tokens = line.split()
            camera_id = int(tokens[0])
            model = tokens[1]
            width, height = int(tokens[2]), int(tokens[3])
            params = np.array([float(t) for t in tokens[4:]], dtype=np.float64)
            cameras[camera_id] = (model, width, height, params)
    return cameras


def read_image_camera_ids(path):
    """Read a COLMAP images.txt. Returns {basename: camera_id}, ignoring the poses and POINTS2D."""
    images = {}
    with open(path, 'r') as f:
        lines = f.readlines()

    header_done = False
    expect_points = False
    for line in lines:
        if line.startswith('#'):
            continue
        if expect_points:  # the POINTS2D line that follows every image line, possibly empty
            expect_points = False
            continue
        if not line.strip():
            continue
        header_done = True
        tokens = line.split()
        images[osp.basename(tokens[9])] = int(tokens[8])
        expect_points = True

    assert header_done, f"No image entries found in {path}."
    return images


def thin_prism_distort(u, v, dist):
    """Apply the thin prism / Brown distortion to normalized points. dist is k1,k2,p1,p2,k3,k4,sx1,sy1."""
    k1, k2, p1, p2, k3, k4, sx1, sy1 = dist
    u2, v2, uv = u * u, v * v, u * v
    r2 = u2 + v2
    radial = 1. + r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)))
    du = 2. * p1 * uv + p2 * (r2 + 2. * u2) + sx1 * r2
    dv = 2. * p2 * uv + p1 * (r2 + 2. * v2) + sy1 * r2
    return u * radial + du, v * radial + dv


def fisheye_project(x, y, dist):
    """Project normalized points to distorted normalized points (ETH3D BenchmarkCamera::Distort)."""
    r = np.sqrt(x * x + y * y)
    scale = np.where(r > 1e-6, np.arctan(r) / np.where(r > 1e-6, r, 1.), 1.)
    return thin_prism_distort(x * scale, y * scale, dist)


def fisheye_unproject(dx, dy, dist, max_iters=100, rel_step=1e-6, max_step_norm=1e-10):
    """Invert fisheye_project exactly the way COLMAP's IterativeUndistortion does."""
    eps = np.finfo(np.float64).eps
    x0 = np.array(dx, dtype=np.float64)
    y0 = np.array(dy, dtype=np.float64)
    x, y = x0.copy(), y0.copy()

    def delta(u, v):  # COLMAP's Distortion() returns the offset, not the distorted point
        du, dv = thin_prism_distort(u, v, dist)
        return du - u, dv - v

    for _ in range(max_iters):
        step_x = np.maximum(eps, np.abs(rel_step * x))
        step_y = np.maximum(eps, np.abs(rel_step * y))
        dx_c, dy_c = delta(x, y)
        dx_xf, dy_xf = delta(x + step_x, y)
        dx_xb, dy_xb = delta(x - step_x, y)
        dx_yf, dy_yf = delta(x, y + step_y)
        dx_yb, dy_yb = delta(x, y - step_y)
        j00 = 1. + (dx_xf - dx_xb) / (2. * step_x)
        j10 = (dy_xf - dy_xb) / (2. * step_x)
        j01 = (dx_yf - dx_yb) / (2. * step_y)
        j11 = 1. + (dy_yf - dy_yb) / (2. * step_y)
        rx, ry = x + dx_c - x0, y + dy_c - y0
        det = j00 * j11 - j01 * j10
        sx = (j11 * rx - j01 * ry) / det
        sy = (j00 * ry - j10 * rx) / det
        x, y = x - sx, y - sy
        if np.max(sx * sx + sy * sy) < max_step_norm:
            break

    pu, pv = thin_prism_distort(x, y, dist)
    converged = (pu - x0) ** 2 + (pv - y0) ** 2 < 1e-8

    theta = np.sqrt(x * x + y * y)
    in_front = theta < np.pi / 2.  # beyond 90 degrees the equidistant fisheye cannot be undistorted
    theta_cos_theta = theta * np.cos(theta)
    safe = theta_cos_theta > eps  # COLMAP leaves the point unscaled outside this range
    scale = np.where(safe, np.sin(theta) / np.where(safe, theta_cos_theta, 1.), 1.)
    return x * scale, y * scale, converged & in_front


def undistort_camera(width, height, params, blank_pixels=0., min_scale=0.2, max_scale=2.):
    """Reimplementation of COLMAP's UndistortCamera for THIN_PRISM_FISHEYE input cameras.

    Returns the PINHOLE target camera as (width, height, [fx, fy, cx, cy]).
    """
    fx, fy, cx, cy = params[:4]
    dist = params[4:]

    def to_undistorted_image(px, py):
        x, y, _ = fisheye_unproject((px - cx) / fx, (py - cy) / fy, dist)
        return x * fx + cx, y * fy + cy

    ys = np.arange(height, dtype=np.float64) + 0.5
    xs = np.arange(width, dtype=np.float64) + 0.5
    left_x, _ = to_undistorted_image(np.full_like(ys, 0.5), ys)
    right_x, _ = to_undistorted_image(np.full_like(ys, width - 0.5), ys)
    _, top_y = to_undistorted_image(xs, np.full_like(xs, 0.5))
    _, bottom_y = to_undistorted_image(xs, np.full_like(xs, height - 0.5))

    # scale such that the undistorted image contains all pixels of the distorted image:
    min_scale_x = min(cx / (cx - left_x.min()), (width - 0.5 - cx) / (right_x.max() - cx))
    min_scale_y = min(cy / (cy - top_y.min()), (height - 0.5 - cy) / (bottom_y.max() - cy))
    # scale such that there are no blank pixels in the undistorted image:
    max_scale_x = max(cx / (cx - left_x.max()), (width - 0.5 - cx) / (right_x.min() - cx))
    max_scale_y = max(cy / (cy - top_y.max()), (height - 0.5 - cy) / (bottom_y.min() - cy))

    scale_x = 1. / (min_scale_x * blank_pixels + max_scale_x * (1. - blank_pixels))
    scale_y = 1. / (min_scale_y * blank_pixels + max_scale_y * (1. - blank_pixels))
    scale_x = float(np.clip(scale_x, min_scale, max_scale))
    scale_y = float(np.clip(scale_y, min_scale, max_scale))

    out_width = int(max(1., scale_x * width))  # truncation, as in COLMAP
    out_height = int(max(1., scale_y * height))
    out_cx = cx * out_width / width
    out_cy = cy * out_height / height

    return out_width, out_height, np.array([fx, fy, out_cx, out_cy], dtype=np.float64)


def compute_backward_map(params, out_width, out_height, out_params):
    """Target pixel -> source pixel map for cv2.remap (OpenCV pixel-center indices, float32)."""
    fx, fy, cx, cy = params[:4]
    dist = params[4:]
    out_fx, out_fy, out_cx, out_cy = out_params

    map_x = np.empty((out_height, out_width), dtype=np.float32)
    map_y = np.empty((out_height, out_width), dtype=np.float32)
    xs = (np.arange(out_width, dtype=np.float64) + 0.5 - out_cx) / out_fx

    for y0 in range(0, out_height, ROW_CHUNK):
        y1 = min(y0 + ROW_CHUNK, out_height)
        ys = (np.arange(y0, y1, dtype=np.float64) + 0.5 - out_cy) / out_fy
        grid_x, grid_y = np.meshgrid(xs, ys)
        dx, dy = fisheye_project(grid_x, grid_y, dist)
        map_x[y0:y1] = dx * fx + cx - 0.5  # COLMAP corner convention -> OpenCV pixel centers
        map_y[y0:y1] = dy * fy + cy - 0.5

    return map_x, map_y


def compute_forward_map(width, height, params, out_params):
    """Source pixel -> target pixel index map, for forward scattering of the sparse depth points.

    Returns (target_x, target_y, valid), each of shape (height, width). target_* are int32 pixel
    indices into the target image; valid marks source pixels that could be undistorted.
    """
    fx, fy, cx, cy = params[:4]
    dist = params[4:]
    out_fx, out_fy, out_cx, out_cy = out_params

    target_x = np.empty((height, width), dtype=np.int32)
    target_y = np.empty((height, width), dtype=np.int32)
    valid = np.empty((height, width), dtype=bool)
    xs = (np.arange(width, dtype=np.float64) + 0.5 - cx) / fx

    for y0 in range(0, height, ROW_CHUNK):
        y1 = min(y0 + ROW_CHUNK, height)
        ys = (np.arange(y0, y1, dtype=np.float64) + 0.5 - cy) / fy
        grid_x, grid_y = np.meshgrid(xs, ys)
        x, y, ok = fisheye_unproject(grid_x, grid_y, dist)
        # corner-convention target coordinate -> index of the containing pixel:
        target_x[y0:y1] = np.floor(x * out_fx + out_cx).astype(np.int32)
        target_y[y0:y1] = np.floor(y * out_fy + out_cy).astype(np.int32)
        valid[y0:y1] = ok

    return target_x, target_y, valid


def check_maps_consistent(params, out_params, forward_map, num_samples=100000, tolerance=1.):
    """Sanity check: round-tripping source pixels through both maps must return the source pixels."""
    fx, fy, cx, cy = params[:4]
    dist = params[4:]
    out_fx, out_fy, out_cx, out_cy = out_params
    target_x, target_y, valid = forward_map
    height, width = valid.shape

    rng = np.random.default_rng(0)
    ys = rng.integers(0, height, num_samples)
    xs = rng.integers(0, width, num_samples)
    ok = valid[ys, xs]
    ys, xs = ys[ok], xs[ok]

    # project the centers of the target pixels back to the source image:
    nx = (target_x[ys, xs] + 0.5 - out_cx) / out_fx
    ny = (target_y[ys, xs] + 0.5 - out_cy) / out_fy
    dx, dy = fisheye_project(nx, ny, dist)
    err_x = np.abs(dx * fx + cx - 0.5 - xs)
    err_y = np.abs(dy * fy + cy - 0.5 - ys)

    max_err = float(max(err_x.max(), err_y.max())) if len(xs) > 0 else 0.
    assert max_err < tolerance, f"Forward/backward map round trip is off by {max_err:.3f} px."
    return max_err


def warp_image(image, backward_map):
    """Undistort an image by backward mapping with bilinear interpolation, as COLMAP does."""
    map_x, map_y = backward_map
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def warp_depth(depth, forward_map, out_width, out_height):
    """Undistort a sparse depth map by forward scattering. Returns (depth, num_in, num_out, num_oob)."""
    target_x, target_y, valid = forward_map
    keep = valid & np.isfinite(depth) & (depth > 0.)
    num_in = int(keep.sum())

    tx, ty, values = target_x[keep], target_y[keep], depth[keep]
    inside = (tx >= 0) & (tx < out_width) & (ty >= 0) & (ty < out_height)
    num_oob = int(num_in - inside.sum())
    tx, ty, values = tx[inside], ty[inside], values[inside]

    out = np.full((out_height, out_width), np.inf, dtype=np.float32)
    np.minimum.at(out, (ty, tx), values)  # nearest surface wins where points collide
    num_out = int(np.isfinite(out).sum())

    return out, num_in, num_out, num_oob


def compare_to_reference(image, reference_path):
    """Compare an undistorted image against the officially provided one. Returns (mean, max)."""
    assert osp.isfile(reference_path), (
        f"{reference_path} is missing, so the undistortion cannot be verified against it. "
        f"Re-run scripts/download_eth3d.sh to fetch it.")
    reference = cv2.imread(reference_path)
    assert reference is not None, f"Could not read {reference_path}."
    assert reference.shape == image.shape, (
        f"{reference_path} has shape {reference.shape}, but we produced {image.shape}.")
    diff = np.abs(image.astype(np.int32) - reference.astype(np.int32))
    return float(diff.mean()), float(diff.max())


def undistort_scene(root, scene, write_images=False, progress=True):
    """Undistort all images and depth maps of one scene. Returns a dict of summary statistics."""
    scene_path = osp.join(root, scene)
    cameras = read_cameras_txt(osp.join(scene_path, CALIB_DIR, 'cameras.txt'))
    images = read_image_camera_ids(osp.join(scene_path, CALIB_DIR, 'images.txt'))

    out_cameras = {}
    for camera_id, (model, width, height, params) in cameras.items():
        assert model == 'THIN_PRISM_FISHEYE', \
            f"{scene} camera {camera_id}: expected THIN_PRISM_FISHEYE, got {model}."
        out_width, out_height, out_params = undistort_camera(width, height, params)
        out_cameras[camera_id] = ('PINHOLE', out_width, out_height, out_params)

    verify_cameras(scene, out_cameras, osp.join(scene_path, CALIB_REF_DIR, 'cameras.txt'))

    if write_images:
        os.makedirs(osp.join(scene_path, IMAGE_OUT_DIR), exist_ok=True)
    os.makedirs(osp.join(scene_path, DEPTH_OUT_DIR), exist_ok=True)

    stats = {'scene': scene, 'num_images': 0, 'image_diffs': [], 'roundtrip_err': 0.,
             'depth_in': 0, 'depth_out': 0, 'depth_oob': 0}

    names_per_camera = {}
    for name, camera_id in images.items():
        names_per_camera.setdefault(camera_id, []).append(name)

    pbar = tqdm(total=len(images), desc=f"Undistorting {scene}") if progress else None
    # process camera by camera, so that only one camera's maps are held in memory at a time:
    for camera_id in sorted(names_per_camera):
        _, width, height, params = cameras[camera_id]
        _, out_width, out_height, out_params = out_cameras[camera_id]

        backward_map = compute_backward_map(params, out_width, out_height, out_params)
        forward_map = compute_forward_map(width, height, params, out_params)
        stats['roundtrip_err'] = max(
            stats['roundtrip_err'], check_maps_consistent(params, out_params, forward_map))

        for name in sorted(names_per_camera[camera_id]):
            image = cv2.imread(osp.join(scene_path, IMAGE_DIR, name))
            assert image is not None, f"Could not read {osp.join(scene_path, IMAGE_DIR, name)}."
            undistorted_image = warp_image(image, backward_map)
            stats['image_diffs'].append(compare_to_reference(
                undistorted_image, osp.join(scene_path, IMAGE_REF_DIR, name)))
            if write_images:
                out_path = osp.join(scene_path, IMAGE_OUT_DIR, osp.splitext(name)[0] + '.png')
                cv2.imwrite(out_path, undistorted_image, [cv2.IMWRITE_PNG_COMPRESSION, 3])

            depth = np.fromfile(osp.join(scene_path, DEPTH_DIR, name),
                                dtype=np.float32).reshape(height, width)
            undistorted_depth, num_in, num_out, num_oob = warp_depth(
                depth, forward_map, out_width, out_height)
            stats['depth_in'] += num_in
            stats['depth_out'] += num_out
            stats['depth_oob'] += num_oob
            undistorted_depth.tofile(osp.join(scene_path, DEPTH_OUT_DIR, name))

            stats['num_images'] += 1
            if pbar is not None:
                pbar.update(1)

        del backward_map, forward_map

    if pbar is not None:
        pbar.close()

    return stats


def verify_cameras(scene, out_cameras, reference_path):
    """Assert that the computed target cameras match the officially provided undistorted ones."""
    assert osp.isfile(reference_path), (
        f"{scene}: {reference_path} is missing. It is the record of the undistorted cameras that "
        f"the outputs of this script belong to; re-run scripts/download_eth3d.sh to fetch it.")

    reference = read_cameras_txt(reference_path)
    for camera_id, (_, width, height, params) in out_cameras.items():
        assert camera_id in reference, f"{scene}: camera {camera_id} missing in {reference_path}."
        ref_model, ref_width, ref_height, ref_params = reference[camera_id]
        assert ref_model == 'PINHOLE', f"{scene}: camera {camera_id} reference is {ref_model}."
        assert (width, height) == (ref_width, ref_height), (
            f"{scene}: camera {camera_id} size {width}x{height} != "
            f"official {ref_width}x{ref_height}.")
        assert np.allclose(params, ref_params, atol=0.01), (
            f"{scene}: camera {camera_id} intrinsics {params} != official {ref_params}.")


def undistort_scene_star(kwargs):
    return undistort_scene(**kwargs)


def print_summary(all_stats):
    print("")
    print(f"{'scene':<16}{'imgs':>6}{'img mean diff':>15}{'img max diff':>14}"
          f"{'rt err [px]':>13}{'gt pts in':>14}{'gt pts out':>14}{'oob':>12}")
    total = {'num_images': 0, 'depth_in': 0, 'depth_out': 0, 'depth_oob': 0}
    all_diffs = []
    for stats in all_stats:
        diffs = stats['image_diffs']
        all_diffs += diffs
        mean_diff = f"{np.mean([d[0] for d in diffs]):.3f}"
        max_diff = f"{max(d[1] for d in diffs):.0f}"
        print(f"{stats['scene']:<16}{stats['num_images']:>6}{mean_diff:>15}{max_diff:>14}"
              f"{stats['roundtrip_err']:>13.4f}{stats['depth_in']:>14}{stats['depth_out']:>14}"
              f"{stats['depth_oob']:>12}")
        for key in total:
            total[key] += stats[key]

    mean_diff = f"{np.mean([d[0] for d in all_diffs]):.3f}"
    max_diff = f"{max(d[1] for d in all_diffs):.0f}"
    print(f"{'TOTAL':<16}{total['num_images']:>6}{mean_diff:>15}{max_diff:>14}{'':>13}"
          f"{total['depth_in']:>14}{total['depth_out']:>14}{total['depth_oob']:>12}")
    kept = 100. * total['depth_out'] / total['depth_in']
    print(f"\nGround truth depth points kept: {kept:.3f}% "
          f"({total['depth_in'] - total['depth_out']} lost, of which "
          f"{total['depth_oob']} out of bounds and the rest to collisions).")
    print("Image differences are against the officially provided undistorted images, which are "
          "JPEGs;\nvalues around 1/255 mean and around 16 max are the expected JPEG noise floor.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Undistort the ETH3D DSLR images and ground truth depth maps.")
    parser.add_argument('root', type=str, help="Path to the downloaded ETH3D dataset.")
    parser.add_argument('--scenes', type=str, nargs='+', default=SCENES,
                        help=f"Scenes to process. Default: {' '.join(SCENES)}")
    parser.add_argument('--write_images', action='store_true', default=False,
                        help="Also write the undistorted images. Off by default, because ETH3D "
                             "provides equivalent ones in images/dslr_images_undistorted.")
    parser.add_argument('--num_workers', type=int, default=1,
                        help="Number of scenes to process in parallel. Needs ~2GB per worker.")
    args = parser.parse_args()

    jobs = [{'root': args.root, 'scene': scene, 'write_images': args.write_images,
             'progress': args.num_workers == 1}
            for scene in args.scenes]

    print(f"Undistorting {len(jobs)} ETH3D scene(s) in {args.root}.")
    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            all_stats = list(tqdm(pool.imap(undistort_scene_star, jobs), total=len(jobs),
                                  desc="Processed scenes"))
    else:
        all_stats = [undistort_scene_star(job) for job in jobs]

    print_summary(all_stats)
    print("\nDone")
