"""
帧质量过滤模块：在推理前剔除模糊帧和冗余帧。

函数 `is_high_quality` 可在数据加载阶段逐帧调用，
从方案说明「优化1：输入预筛选」中提取。
"""

from typing import Optional

import cv2
import numpy as np


def is_high_quality(
    img: np.ndarray,
    prev_img: Optional[np.ndarray],
    blur_threshold: float = 100.0,
    motion_threshold: float = 0.02,
) -> bool:
    """
    判断当前帧是否满足质量要求。

    检测逻辑：
      1. 模糊检测：计算灰度图的拉普拉斯方差；低于 blur_threshold 则视为模糊帧。
      2. 冗余帧检测：与前一帧的均值绝对差（归一化到 0~1）；
                     低于 motion_threshold 则视为静止/重复帧。

    Args:
        img:              当前帧，形状 (H, W, C)，BGR 或 RGB，uint8。
        prev_img:         前一帧（相同形状），首帧时传入 None。
        blur_threshold:   拉普拉斯方差阈值；越大对清晰度要求越高。
        motion_threshold: 帧间差阈值（0-1）；越大对运动量要求越高。

    Returns:
        True 表示帧质量合格，可送入模型；False 表示应跳过该帧。
    """
    # ---- 1. 模糊检测 ----
    if img.ndim == 3:
        # 统一按 BGR/RGB 均适用的方式转灰度（cv2 要求 BGR，但对模糊检测影响极小）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.astype(np.uint8, copy=False)

    blur_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_val < blur_threshold:
        return False

    # ---- 2. 冗余帧检测 ----
    if prev_img is not None:
        diff = float(
            np.mean(
                np.abs(
                    img.astype(np.float32) - prev_img.astype(np.float32)
                )
            )
        ) / 255.0
        if diff < motion_threshold:
            return False

    return True
