import os
import sys
import shutil
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.data_utils import save_image, load_image
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.easy_utils import read_camera, write_camera

# 把项目根目录加入 sys.path 以便导入 longstream 包
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from longstream.utils.gt_pose import anchor_w2c_sequence, save_gt_pose_npy
from scripts.preprocess_vkitti2 import parse_vkitti2_extrinsic_txt


# Define some global variables for vkitti2
vkitti2_img_pattern0 = 'frames/rgb/Camera_0/rgb_{frame:05d}.jpg'
vkitti2_img_pattern1 = 'frames/rgb/Camera_1/rgb_{frame:05d}.jpg'
vkitti2_dpt_pattern0 = 'frames/depth/Camera_0/depth_{frame:05d}.png'
vkitti2_dpt_pattern1 = 'frames/depth/Camera_1/depth_{frame:05d}.png'

# Define some global variables for EasyVolcap
easyvolcap_img_dir = 'images'
easyvolcap_dpt_dir = 'depths'
easyvolcap_cam_dir = 'cameras'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/original/vkitti2")
    parser.add_argument("--easyvolcap_root", default="data/vkitti2")
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument('--scenes', nargs='+', default=[])
    args = parser.parse_args()

    # Global variables
    vkitti2_root = args.data_root
    easyvolcap_root = args.easyvolcap_root

    # Define per-scene preprocessing function
    def process_scene(scene):
        # Input and output paths
        src_root = join(vkitti2_root, scene)
        tar_root = join(easyvolcap_root, scene)

        # Load the raw camera parameters
        ixts = np.loadtxt(join(src_root, 'intrinsic.txt'), skiprows=1)
        exts = np.loadtxt(join(src_root, 'extrinsic.txt'), skiprows=1)

        # Check if there is any missing camera parameters or images or depth maps
        if not (
            len(ixts) // 2 == len(exts) // 2
            == len(os.listdir(join(src_root, dirname(vkitti2_img_pattern0))))
            == len(os.listdir(join(src_root, dirname(vkitti2_img_pattern1))))
            == len(os.listdir(join(src_root, dirname(vkitti2_dpt_pattern0))))
            == len(os.listdir(join(src_root, dirname(vkitti2_dpt_pattern1))))
        ):
            log(red(f"Missing camera parameters or images or depth maps in {cyan(scene)}"))
            return

        # Output camera parameters
        camera0 = dotdict()
        camera1 = dotdict()

        def process_view(sidx, tidx):
            # Load camera parameters
            K0, K1 = np.eye(3), np.eye(3)
            K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2] = ixts[tidx * 2 + 0][2:]
            K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2] = ixts[tidx * 2 + 1][2:]
            w2c0 = exts[tidx * 2 + 0][2:].reshape(4, 4)
            w2c1 = exts[tidx * 2 + 1][2:].reshape(4, 4)

            # Load depth maps
            dpt0 = cv2.imread(
                join(src_root, vkitti2_dpt_pattern0.format(frame=sidx)),
                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
            ) / 100.  # https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/
            dpt1 = cv2.imread(
                join(src_root, vkitti2_dpt_pattern1.format(frame=sidx)),
                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
            ) / 100.  # https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/

            # Camera parameters
            camera0[f'{tidx:05d}'] = dotdict()
            camera1[f'{tidx:05d}'] = dotdict()
            cam0 = camera0[f'{tidx:05d}']
            cam1 = camera1[f'{tidx:05d}']
            cam0.H, cam0.W = dpt0.shape[0], dpt0.shape[1]
            cam1.H, cam1.W = dpt1.shape[0], dpt1.shape[1]
            cam0.K, cam0.R, cam0.T = K0, w2c0[:3, :3], w2c0[:3, 3:]
            cam1.K, cam1.R, cam1.T = K1, w2c1[:3, :3], w2c1[:3, 3:]
            cam0.D = np.zeros((1, 5), dtype=np.float32)
            cam1.D = np.zeros((1, 5), dtype=np.float32)

            # Link/copy RGB image (cross-platform)
            img0_path = join(tar_root, easyvolcap_img_dir, '00', f'{tidx:05d}.jpg')
            if not exists(img0_path):
                os.makedirs(dirname(img0_path), exist_ok=True)
                _link_or_copy(abspath(join(src_root, vkitti2_img_pattern0.format(frame=sidx))), img0_path)
            img1_path = join(tar_root, easyvolcap_img_dir, '01', f'{tidx:05d}.jpg')
            if not exists(img1_path):
                os.makedirs(dirname(img1_path), exist_ok=True)
                _link_or_copy(abspath(join(src_root, vkitti2_img_pattern1.format(frame=sidx))), img1_path)

            # Write the depth map
            dpt0_path = join(tar_root, easyvolcap_dpt_dir, '00', f'{tidx:05d}.exr')
            if not exists(dpt0_path):
                os.makedirs(dirname(dpt0_path), exist_ok=True)
                save_image(dpt0_path, dpt0.astype(np.float32))
            dpt1_path = join(tar_root, easyvolcap_dpt_dir, '01', f'{tidx:05d}.exr')
            if not exists(dpt1_path):
                os.makedirs(dirname(dpt1_path), exist_ok=True)
                save_image(dpt1_path, dpt1.astype(np.float32))


        # Find all views
        inds = [
            int(f[-9:-4]) for f in sorted(
                os.listdir(join(src_root, dirname(vkitti2_img_pattern0)))
            ) if f.endswith('.jpg')
        ]

        # Process all views parallelly
        parallel_execution(
            inds,
            list(range(len(inds))),
            action=process_view,
            sequential=False,
            num_workers=args.num_workers,
            print_progress=True,
        )

        # Write the camera data (cameras/<cam>/extri.yml + intri.yml = 主真值)
        write_camera(camera0, join(tar_root, easyvolcap_cam_dir, '00'))
        write_camera(camera1, join(tar_root, easyvolcap_cam_dir, '01'))

        # Export gt_poses.npy cache for each camera
        export_vkitti2_gt_pose_cache(src_root, tar_root, inds)

        # Logging
        log(green(f"Processed scene {cyan(scene)}, total {len(inds):04d} views"))


    # Process all scenes
    if len(args.scenes) > 0:
        scenes = args.scenes
    else:
        # Get all scenes
        scenes = [
            f for f in sorted(os.listdir(vkitti2_root))
                if os.path.isdir(join(vkitti2_root, f))
        ]

    # Each scene of vkitti2 has some sub-scenes
    subscenes = []
    for scene in scenes:
        subscenes += [
            join(scene, f) for f in sorted(os.listdir(join(vkitti2_root, scene)))
                if os.path.isdir(join(vkitti2_root, scene, f))
        ]
    log(yellow(f"Found {len(subscenes)} sub-scenes in total"))

    # Process each scene
    for subscene in subscenes:
        process_scene(subscene)


def _link_or_copy(src: str, dst: str) -> None:
    """跨平台 symlink / copy helper：优先 symlink，不支持时退回 shutil.copy2。"""
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def export_vkitti2_gt_pose_cache(src_root: str, tar_root: str, inds: list) -> None:
    """在 tar_root 生成 gt_poses_00.npy / gt_poses_01.npy（锚定到第 0 帧的 w2c）。"""
    extrinsic_path = join(src_root, 'extrinsic.txt')
    if not exists(extrinsic_path):
        return
    for cam_id, cam_name in [(0, '00'), (1, '01')]:
        raw_poses = parse_vkitti2_extrinsic_txt(extrinsic_path, camera_id=cam_id)
        if not raw_poses:
            continue
        sorted_keys = sorted(raw_poses.keys())
        w2c_seq = np.stack([raw_poses[k] for k in sorted_keys], axis=0)
        anchored = anchor_w2c_sequence(w2c_seq)
        save_gt_pose_npy(join(tar_root, f'gt_poses_{cam_name}.npy'), anchored)
        # 同时生成不带后缀的 gt_poses.npy（兼容旧流程，默认取 Camera 0）
        if cam_id == 0:
            save_gt_pose_npy(join(tar_root, 'gt_poses.npy'), anchored)


if __name__ == "__main__":
    main()
