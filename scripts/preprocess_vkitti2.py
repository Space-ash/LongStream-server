#!/usr/bin/env python3
"""
将 vKITTI2 的 extrinsic.txt 解析为以第一帧为原点的相对绝对位姿，
并保存为 gt_poses.npy（形状 [N, 4, 4]）。

vKITTI2 extrinsic.txt 列顺序（首行为表头，跳过）：
    frame  cameraID  r11 r12 r13 t1  r21 r22 r23 t2  r31 r32 r33 t3
矩阵含义：world-to-camera 外参（相机外参矩阵 [R|t]）。

用法：
    python scripts/preprocess_vkitti2.py \
        --extrinsic /path/to/vkitti2/Scene01/15-deg-left/extrinsic.txt \
        --output    /path/to/prepared_inputs/.../gt_poses.npy \
        [--camera_id 0]
"""

import argparse
import os
from typing import Dict

import numpy as np


def process_vkitti2_poses(txt_path: str, camera_id: int = 0) -> Dict[int, np.ndarray]:
    """
    解析 vKITTI2 extrinsic.txt，返回 {frame_index: 4x4 相对位姿} 字典。

    第一帧位姿为单位矩阵（Identity），后续帧位姿均以第一帧为坐标原点：
        T_relative[i] = inv(T_world_curr[0]) @ T_world_curr[i]

    Args:
        txt_path:  extrinsic.txt 文件路径。
        camera_id: 仅处理该 ID 对应的相机（0=左相机，1=右相机）。

    Returns:
        {frame_idx: ndarray(4,4, float32)}
    """
    raw_data = np.loadtxt(txt_path, skiprows=1)
    if raw_data.ndim == 1:
        raw_data = raw_data[np.newaxis, :]

    T_world_to_start: np.ndarray | None = None
    poses: Dict[int, np.ndarray] = {}

    for row in raw_data:
        f_idx = int(row[0])
        cam_id = int(row[1])
        if cam_id != camera_id:
            continue

        # 从行中第 2~13 列组装 4×4 外参矩阵（world-to-camera）
        v = row[2:14]
        T_world_curr = np.array(
            [
                [v[0], v[1], v[2],  v[3]],
                [v[4], v[5], v[6],  v[7]],
                [v[8], v[9], v[10], v[11]],
                [0.0,  0.0,  0.0,   1.0],
            ],
            dtype=np.float64,
        )

        if T_world_to_start is None:
            # 第一帧：记录其逆矩阵，用于后续相对化处理
            T_world_to_start = np.linalg.inv(T_world_curr)

        # 核心逻辑：T_relative = inv(T[0]) @ T[i]
        # 第一帧时结果为 Identity；后续帧表达相对于第一帧的位姿
        T_relative = T_world_to_start @ T_world_curr
        poses[f_idx] = T_relative.astype(np.float32)

    return poses


def poses_dict_to_array(poses: Dict[int, np.ndarray]) -> np.ndarray:
    """将 {frame_idx -> 4x4} 字典按帧号升序转换为 [N, 4, 4] 连续数组。"""
    sorted_keys = sorted(poses.keys())
    return np.stack([poses[k] for k in sorted_keys], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="预处理 vKITTI2 外参文件，生成以第一帧为原点的相对绝对位姿 .npy"
    )
    parser.add_argument(
        "--extrinsic", required=True, help="vKITTI2 extrinsic.txt 路径"
    )
    parser.add_argument(
        "--output", required=True, help="输出 gt_poses.npy 的保存路径"
    )
    parser.add_argument(
        "--camera_id",
        type=int,
        default=0,
        help="要处理的相机 ID（0=左相机，1=右相机），默认 0",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.extrinsic):
        raise FileNotFoundError(f"外参文件不存在：{args.extrinsic}")

    poses = process_vkitti2_poses(args.extrinsic, camera_id=args.camera_id)
    if not poses:
        raise RuntimeError(
            f"在 {args.extrinsic} 中未找到 camera_id={args.camera_id} 的位姿数据"
        )

    poses_array = poses_dict_to_array(poses)  # [N, 4, 4]

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    np.save(args.output, poses_array)
    print(
        f"[preprocess_vkitti2] 已保存 {len(poses_array)} 帧位姿至 {args.output}"
    )
    print(f"  数组形状: {poses_array.shape}，dtype: {poses_array.dtype}")
    print(f"  第 0 帧（应为单位矩阵）:\n{poses_array[0]}")


if __name__ == "__main__":
    main()
