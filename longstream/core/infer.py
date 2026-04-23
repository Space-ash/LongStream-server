import argparse
import os
import yaml
import cv2
import numpy as np
import torch
from PIL import Image

from longstream.core.model import LongStreamModel
from longstream.data.dataloader import LongStreamDataLoader
from longstream.streaming.keyframe_selector import KeyframeSelector
from longstream.streaming.refresh import run_batch_refresh, run_streaming_refresh
from longstream.utils.vendor.models.components.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)
from longstream.utils.camera import compose_abs_from_rel
from longstream.utils.depth import colorize_depth, unproject_depth_to_points
from longstream.utils.sky_mask import compute_sky_mask
from longstream.io.save_points import save_pointcloud
from longstream.io.save_poses_txt import save_w2c_txt, save_intri_txt, save_rel_pose_txt
from longstream.io.save_images import save_image_sequence, save_video
from longstream.io.frame_index_map import save_frame_index_map


def _to_uint8_rgb(images):
    imgs = images.detach().cpu().numpy()
    imgs = np.clip(imgs, 0.0, 1.0)
    imgs = (imgs * 255.0).astype(np.uint8)
    return imgs


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _apply_sky_mask(depth, mask):
    if mask is None:
        return depth
    m = (mask > 0).astype(np.float32)
    return depth * m


def _to_homogeneous_w2c(extri_np):
    """
    将预测位姿统一转换为齐次 4x4 w2c 矩阵序列。

    `pose_encoding_to_extri_intri` 当前返回的是 [S, 3, 4] 的 [R|t]，
    但后续 GT 校正、矩阵求逆、点云变换都假设使用 [S, 4, 4]。
    这里统一补最后一行 [0, 0, 0, 1]，避免下游混用 3x4 / 4x4。
    """
    extri_np = np.asarray(extri_np)
    if extri_np.ndim != 3:
        raise ValueError(
            f"Expected extrinsics with shape [S,3,4] or [S,4,4], got {extri_np.shape}"
        )
    if extri_np.shape[-2:] == (4, 4):
        return extri_np.astype(np.float32, copy=False)
    if extri_np.shape[-2:] != (3, 4):
        raise ValueError(
            f"Expected extrinsics with shape [S,3,4] or [S,4,4], got {extri_np.shape}"
        )

    S = extri_np.shape[0]
    homogeneous = np.zeros((S, 4, 4), dtype=extri_np.dtype)
    homogeneous[:, :3, :] = extri_np
    homogeneous[:, 3, 3] = 1.0
    return homogeneous.astype(np.float32, copy=False)


def _camera_points_to_world(points, extri):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(extri[:3, :3], dtype=np.float64)
    t = np.asarray(extri[:3, 3], dtype=np.float64)
    world = (R.T @ (pts.T - t[:, None])).T
    return world.astype(np.float32, copy=False)


def _mask_points_and_colors(points, colors, mask):
    pts = points.reshape(-1, 3)
    cols = None if colors is None else colors.reshape(-1, 3)
    if mask is None:
        return pts, cols
    valid = mask.reshape(-1) > 0
    pts = pts[valid]
    if cols is not None:
        cols = cols[valid]
    return pts, cols


def _combine_masks(mask_a, mask_b):
    """将两个掩码按位 AND 合并。None 表示全通（不限制）。"""
    if mask_a is None and mask_b is None:
        return None
    if mask_a is None:
        return (mask_b > 0).astype(np.uint8)
    if mask_b is None:
        return (mask_a > 0).astype(np.uint8)
    return ((mask_a > 0) & (mask_b > 0)).astype(np.uint8)


def _resize_long_edge(arr, long_edge_size, interpolation):
    h, w = arr.shape[:2]
    scale = float(long_edge_size) / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(arr, (new_w, new_h), interpolation=interpolation)


def _prepare_mask_for_model(
    mask, size, crop, patch_size, target_shape, square_ok=False
):
    if mask is None:
        return None
    long_edge = (
        round(size * max(mask.shape[1] / mask.shape[0], mask.shape[0] / mask.shape[1]))
        if size == 224
        else size
    )
    mask = _resize_long_edge(mask, long_edge, cv2.INTER_NEAREST)

    h, w = mask.shape[:2]
    cx, cy = w // 2, h // 2
    if size == 224:
        half = min(cx, cy)
        target_w = 2 * half
        target_h = 2 * half
        if crop:
            mask = mask[cy - half : cy + half, cx - half : cx + half]
        else:
            mask = cv2.resize(
                mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
            )
    else:
        halfw = ((2 * cx) // patch_size) * (patch_size // 2)
        halfh = ((2 * cy) // patch_size) * (patch_size // 2)
        if not square_ok and w == h:
            halfh = int(3 * halfw / 4)
        target_w = 2 * halfw
        target_h = 2 * halfh
        if crop:
            mask = mask[cy - halfh : cy + halfh, cx - halfw : cx + halfw]
        else:
            mask = cv2.resize(
                mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
            )

    if mask.shape[:2] != tuple(target_shape):
        mask = cv2.resize(
            mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return mask


def _save_full_pointcloud(path, point_chunks, color_chunks, max_points=None, seed=0):
    if not point_chunks:
        return
    points = np.concatenate(point_chunks, axis=0)
    colors = None
    if color_chunks and len(color_chunks) == len(point_chunks):
        colors = np.concatenate(color_chunks, axis=0)
    if max_points is not None and len(points) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(points), size=max_points, replace=False)
        points = points[keep]
        if colors is not None:
            colors = colors[keep]
    np.save(os.path.splitext(path)[0] + ".npy", points.astype(np.float32, copy=False))
    save_pointcloud(path, points, colors=colors, max_points=None, seed=seed)


def _decode_predicted_extri_intri(outputs, keyframe_indices, H, W):
    """统一解码预测位姿，返回 (extri_np, intri_np, rel_pose_enc_or_None)。"""
    rel_pose_enc = None
    if "rel_pose_enc" in outputs:
        rel_pose_enc = outputs["rel_pose_enc"][0]
        abs_pose_enc = compose_abs_from_rel(rel_pose_enc, keyframe_indices[0])
        extri, intri = pose_encoding_to_extri_intri(
            abs_pose_enc[None], image_size_hw=(H, W)
        )
    elif "pose_enc" in outputs:
        extri, intri = pose_encoding_to_extri_intri(
            outputs["pose_enc"][0][None], image_size_hw=(H, W)
        )
    else:
        return None, None, None
    extri_np = _to_homogeneous_w2c(extri[0].detach().cpu().numpy())
    intri_np = intri[0].detach().cpu().numpy()
    return extri_np, intri_np, rel_pose_enc


def _apply_depth_corrections(depth_np, scale_corrections):
    """逐帧应用尺度校正到深度图（就地修改并返回）。"""
    S = depth_np.shape[0]
    for i in range(S):
        depth_np[i] = depth_np[i] * float(scale_corrections[i])
    return depth_np


# ============================================================
# GPS 松耦合尺度融合工具函数
# ============================================================

def _w2c_to_camera_centers(extri_np):
    """
    从 [S, 4, 4] w2c 矩阵序列提取相机中心坐标 [S, 3]（世界坐标系）。

    数学推导：w2c 满足 p_cam = R @ p_world + t，
    故相机中心 = -R^T @ t。
    使用 np.float64 保证累加精度。
    """
    if extri_np.ndim != 3 or extri_np.shape[-2:] != (4, 4):
        raise ValueError(
            f"_w2c_to_camera_centers: 期望 [S,4,4]，实际 {extri_np.shape}"
        )
    extri = np.asarray(extri_np, dtype=np.float64)
    R = extri[:, :3, :3]   # [S, 3, 3]
    t = extri[:, :3, 3]    # [S, 3]
    # center_i = -(R_i^T @ t_i)，向量化：-einsum('nji,nj->ni', R, t)
    centers = -np.einsum('nji,nj->ni', R, t)  # [S, 3]
    return centers  # float64


def _camera_centers_to_w2c(template_extri_np, centers_xyz):
    """
    将修正后的相机中心坐标回写为 [S, 4, 4] w2c 矩阵，严格保留原始旋转。

    由 p_cam = R @ p_world + t 及 center = -R^T @ t，
    反推 t = -R @ center。
    """
    if template_extri_np.ndim != 3 or template_extri_np.shape[-2:] != (4, 4):
        raise ValueError(
            f"_camera_centers_to_w2c: 期望 [S,4,4]，实际 {template_extri_np.shape}"
        )
    extri = np.asarray(template_extri_np, dtype=np.float64)
    centers = np.asarray(centers_xyz, dtype=np.float64)
    result = extri.copy()
    R = extri[:, :3, :3]   # [S, 3, 3]
    # new_t_i = -R_i @ center_i
    new_t = -np.einsum('nij,nj->ni', R, centers)  # [S, 3]
    result[:, :3, 3] = new_t
    # 确保最后一行仍为 [0, 0, 0, 1]
    result[:, 3, :] = 0.0
    result[:, 3, 3] = 1.0
    return result.astype(np.float32, copy=False)


def _compute_gps_loose_scale_corrections(
    extri_np,
    gps_xyz,
    trigger_distance_m=5.0,
    min_pred_distance=1e-4,
    max_frames_threshold=50,
    sequence_name=None,
    image_paths=None,
):
    """
    基于段内位移动态触发的松耦合 GPS 尺度校正（线性插值版本）。

    算法逻辑：
    - vo_disp = ||pred_centers[i] - pred_centers[seg_start_idx]||（段起点到当前帧直线位移）
    - 触发条件：当前帧有有效 GPS，且 vo_disp >= trigger_distance_m，且 vo_disp >= min_pred_distance
    - 静止/慢速门控：(i - seg_start_idx) > max_frames_threshold 且 vo_disp < trigger_distance_m 时，
      按 scale_prev 落地当前段并重置段起点，防止长期静止累积异常分母。
    - d_true <= 0 时按 scale_prev 落地并重置，不产生零尺度。
    - 有效触发时对段 [seg_start_idx, i] 做 scale_prev -> scale_new 线性插值，
      并同步更新 corrected_centers。
    - 尾段 flush 沿用 last_applied_scale；全程未触发则退化为不做额外校正。
    - 旋转分量完整保留，仅修正平移尺度。

    参数：
        extri_np             : [S, 4, 4] float32，预测 w2c 矩阵
        gps_xyz              : [S, 3] float32，相机中心（世界坐标，由 GT 位姿提取）
        trigger_distance_m   : 触发尺度计算的最小段内直线位移（单位与场景一致）
        min_pred_distance    : 预测位移下限，低于此值时不触发（防止除零）
        max_frames_threshold : 静止门控：段内帧数超此值且未达位移阈值时强制落地

    返回：
        scale_corrections : [S] float32，逐帧尺度因子（用于深度/点云）
        corrected_extri   : [S, 4, 4] float32，尺度重锚后的 w2c 矩阵
    """
    if extri_np.ndim != 3 or extri_np.shape[-2:] != (4, 4):
        raise ValueError(
            f"_compute_gps_loose_scale_corrections: 期望 [S,4,4]，实际 {extri_np.shape}"
        )
    S = extri_np.shape[0]
    gps = np.asarray(gps_xyz, dtype=np.float64)    # [N_gps, 3]
    n_gps = len(gps)

    # 提取预测相机中心（float64 保证精度）
    pred_centers = _w2c_to_camera_centers(extri_np)  # [S, 3], float64
    corrected_centers = pred_centers.copy()

    scale_corrections = np.ones(S, dtype=np.float32)

    seg_start_idx = 0
    anchor_pred = pred_centers[0].copy()       # 段起点的预测中心
    anchor_corr = corrected_centers[0].copy()  # 段起点的校正中心（初始与预测相同）
    scale_prev = np.float64(1.0)               # 上一段有效尺度，初始为 1.0
    last_applied_scale = np.float64(1.0)       # 最近一次有效尺度，用于尾段 flush

    def _flush_segment(seg_start, seg_end_inclusive, sc):
        """将段 [seg_start, seg_end_inclusive] 按常量尺度 sc 落地。"""
        for j in range(seg_start, seg_end_inclusive + 1):
            delta = pred_centers[j] - anchor_pred
            corrected_centers[j] = anchor_corr + delta * sc
            scale_corrections[j] = float(sc)

    for i in range(1, S):
        # 段内位移：从段起点到当前帧的直线距离
        vo_disp = np.float64(np.linalg.norm(pred_centers[i] - pred_centers[seg_start_idx]))

        # ------------------------------------------------------------------
        # 静止/慢速门控：段内帧数超阈值且未达位移触发条件，按 scale_prev 落地并重置
        # ------------------------------------------------------------------
        if (i - seg_start_idx) > max_frames_threshold and vo_disp < trigger_distance_m:
            _flush_segment(seg_start_idx, i, scale_prev)
            seg_start_idx = i
            anchor_pred = pred_centers[i].copy()
            anchor_corr = corrected_centers[i].copy()
            continue

        # 当前帧无有效 GPS 时跳过触发判断
        if i >= n_gps:
            continue

        # 触发条件：段内位移足够大
        if vo_disp < trigger_distance_m or vo_disp < min_pred_distance:
            continue

        # 计算该段的真实直线距离
        d_true = float(np.linalg.norm(gps[i] - gps[seg_start_idx]))

        if d_true <= 0.0:
            # GPS 位移为零或数据异常，按 scale_prev 落地并重置，不产生零尺度
            _flush_segment(seg_start_idx, i, scale_prev)
            seg_start_idx = i
            anchor_pred = pred_centers[i].copy()
            anchor_corr = corrected_centers[i].copy()
            continue

        scale_new = np.float64(d_true) / vo_disp
        trigger_image = "N/A"
        if image_paths is not None and 0 <= i < len(image_paths):
            trigger_image = os.path.basename(str(image_paths[i]))
        print(
            "[longstream][gps_correction_trigger]"
            f" seq={sequence_name if sequence_name is not None else 'unknown'}"
            f" frame_idx={i}"
            f" image={trigger_image}"
            f" window_start_idx={seg_start_idx}"
            f" vo_disp={float(vo_disp):.6f}"
            f" d_true={float(d_true):.6f}"
            f" scale_new={float(scale_new):.6f}",
            flush=True,
        )

        # 线性插值：对段 [seg_start_idx, i] 每帧 j 做 scale_prev -> scale_new 平滑过渡，
        # 并同步更新 corrected_centers（否则 abs_pose_corrected.txt 仍存在段间阶跃）
        seg_len = max(i - seg_start_idx, 1)
        for j in range(seg_start_idx, i + 1):
            alpha = (j - seg_start_idx) / seg_len
            scale_j = scale_prev + alpha * (scale_new - scale_prev)
            delta = pred_centers[j] - anchor_pred
            corrected_centers[j] = anchor_corr + delta * scale_j
            scale_corrections[j] = float(scale_j)

        # 更新段状态，以当前触发帧为新段起点
        scale_prev = scale_new
        last_applied_scale = scale_new
        seg_start_idx = i
        anchor_pred = pred_centers[i].copy()
        anchor_corr = corrected_centers[i].copy()

    # ------------------------------------------------------------------
    # 尾段 flush：用 last_applied_scale 覆盖剩余未触发帧。
    # 若全程从未触发过，last_applied_scale == 1.0，行为退化为不做额外校正。
    # ------------------------------------------------------------------
    if seg_start_idx < S - 1:
        for j in range(seg_start_idx + 1, S):
            delta = pred_centers[j] - anchor_pred
            corrected_centers[j] = anchor_corr + delta * last_applied_scale
            scale_corrections[j] = float(last_applied_scale)

    # 将校正后的相机中心回写为 w2c 矩阵（保留旋转，仅更新平移）
    corrected_extri = _camera_centers_to_w2c(extri_np, corrected_centers)
    return scale_corrections, corrected_extri


def run_inference_cfg(cfg: dict):
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device_type = torch.device(device).type
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    infer_cfg = cfg.get("inference", {})
    output_cfg = cfg.get("output", {})

    # --- optimizations 配置 ---
    opt_cfg = cfg.get("optimizations", {})
    filter_cfg = opt_cfg.get("filter", {})
    corr_cfg = opt_cfg.get("correction", {})
    frame_filter_enabled = bool(
        filter_cfg.get("frame_filter_enabled", filter_cfg.get("enabled", False))
    )
    confidence_filter_enabled = bool(
        filter_cfg.get(
            "confidence_filter_enabled",
            filter_cfg.get("enabled", False),
        )
    )
    correction_enabled = bool(corr_cfg.get("enabled", False))
    gps_trigger_distance_m = float(corr_cfg.get("gps_trigger_distance_m", 5.0))
    min_pred_distance = float(corr_cfg.get("min_pred_distance", 1e-4))
    max_frames_threshold = int(corr_cfg.get("max_frames_threshold", 50))
    confidence_threshold = float(filter_cfg.get("confidence_threshold", 0.5))

    print(f"[longstream] device={device}", flush=True)
    model = LongStreamModel(model_cfg).to(device)
    model.eval()
    print("[longstream] model ready", flush=True)

    # 将帧质量过滤配置合并入 data_cfg传给 LongStreamDataLoader
    data_cfg_with_filter = dict(data_cfg)
    if filter_cfg:
        data_cfg_with_filter["filter"] = filter_cfg
    loader = LongStreamDataLoader(data_cfg_with_filter)

    keyframe_stride = int(infer_cfg.get("keyframe_stride", 8))
    keyframe_mode = infer_cfg.get("keyframe_mode", "fixed")
    refresh = int(
        infer_cfg.get("refresh", int(infer_cfg.get("keyframes_per_batch", 3)) + 1)
    )
    if refresh < 2:
        raise ValueError(
            "refresh must be >= 2 because it counts both keyframe endpoints"
        )
    mode = infer_cfg.get("mode", "streaming_refresh")
    if mode == "streaming":
        mode = "streaming_refresh"
    streaming_mode = infer_cfg.get("streaming_mode", "causal")
    window_size = int(infer_cfg.get("window_size", 5))

    selector = KeyframeSelector(
        min_interval=keyframe_stride,
        max_interval=keyframe_stride,
        force_first=True,
        mode="random" if keyframe_mode == "random" else "fixed",
    )

    out_root = output_cfg.get("root", "outputs")
    _ensure_dir(out_root)
    save_videos = bool(output_cfg.get("save_videos", True))
    save_points = bool(output_cfg.get("save_points", True))
    save_frame_points = bool(output_cfg.get("save_frame_points", True))
    save_depth = bool(output_cfg.get("save_depth", True))
    save_images = bool(output_cfg.get("save_images", True))
    mask_sky = bool(output_cfg.get("mask_sky", True))
    max_full_pointcloud_points = output_cfg.get("max_full_pointcloud_points", None)
    if max_full_pointcloud_points is not None:
        max_full_pointcloud_points = int(max_full_pointcloud_points)
    max_frame_pointcloud_points = output_cfg.get("max_frame_pointcloud_points", None)
    if max_frame_pointcloud_points is not None:
        max_frame_pointcloud_points = int(max_frame_pointcloud_points)
    skyseg_path = output_cfg.get(
        "skyseg_path",
        os.path.join(os.path.dirname(__file__), "..", "..", "skyseg.onnx"),
    )

    with torch.no_grad():
        for seq in loader:
            images = seq.images
            B, S, C, H, W = images.shape
            print(
                f"[longstream] sequence {seq.name}: inference start ({S} frames)",
                flush=True,
            )

            is_keyframe, keyframe_indices = selector.select_keyframes(
                S, B, images.device
            )

            rel_pose_cfg = infer_cfg.get("rel_pose_head_cfg", {"num_iterations": 4})

            if mode == "batch_refresh":
                outputs = run_batch_refresh(
                    model,
                    images,
                    is_keyframe,
                    keyframe_indices,
                    streaming_mode,
                    keyframe_stride,
                    refresh,
                    rel_pose_cfg,
                )
            elif mode == "streaming_refresh":
                outputs = run_streaming_refresh(
                    model,
                    images,
                    is_keyframe,
                    keyframe_indices,
                    streaming_mode,
                    window_size,
                    refresh,
                    rel_pose_cfg,
                )
            else:
                raise ValueError(f"Unsupported inference mode: {mode}")
            print(f"[longstream] sequence {seq.name}: inference done", flush=True)
            if device_type == "cuda":
                torch.cuda.empty_cache()

            seq_dir = os.path.join(out_root, seq.name)
            _ensure_dir(seq_dir)

            frame_ids = list(range(S))
            rgb = _to_uint8_rgb(images[0].permute(0, 2, 3, 1))

            extri_np = None  # 在 if/elif 块外初始化，便于后续校正逻辑判断
            intri_np = None

            # ============================================================
            # 统一解码预测位姿（只做一次）
            # ============================================================
            extri_np, intri_np, rel_pose_enc = _decode_predicted_extri_intri(
                outputs, keyframe_indices, H, W
            )

            if extri_np is not None:
                pose_dir = os.path.join(seq_dir, "poses")
                _ensure_dir(pose_dir)
                save_w2c_txt(
                    os.path.join(pose_dir, "abs_pose.txt"), extri_np, frame_ids
                )
                save_intri_txt(os.path.join(pose_dir, "intri.txt"), intri_np, frame_ids)
                if rel_pose_enc is not None:
                    save_rel_pose_txt(
                        os.path.join(pose_dir, "rel_pose.txt"), rel_pose_enc, frame_ids
                    )

            # ============================================================
            # GPS 松耦合尺度校正：仅修正尺度，保留纯视觉旋转
            # scale_corrections 同时用于深度和点云
            # ============================================================
            scale_corrections = np.ones(S, dtype=np.float32)
            if correction_enabled and extri_np is not None and getattr(seq, 'gps_xyz', None) is not None:
                scale_corrections, extri_np = _compute_gps_loose_scale_corrections(
                    extri_np=extri_np,
                    gps_xyz=seq.gps_xyz,
                    trigger_distance_m=gps_trigger_distance_m,
                    min_pred_distance=min_pred_distance,
                    max_frames_threshold=max_frames_threshold,
                    sequence_name=seq.name,
                    image_paths=seq.image_paths,
                )
                triggered = int(np.sum(scale_corrections != 1.0))
                print(
                    f"[longstream][gps_correction] seq={seq.name}"
                    f" triggered_frames={triggered}/{S}"
                    f" scale_range=[{scale_corrections.min():.4f}, {scale_corrections.max():.4f}]",
                    flush=True,
                )
                # 保存尺度重锚后的位姿
                pose_dir = os.path.join(seq_dir, "poses")
                _ensure_dir(pose_dir)
                save_w2c_txt(
                    os.path.join(pose_dir, "abs_pose_corrected.txt"), extri_np, frame_ids
                )

            # ============================================================
            # 保存 frame_index_map.json（筛帧映射）
            # ============================================================
            save_fmap = bool(filter_cfg.get("save_frame_index_map", True))
            if save_fmap and seq.original_frame_indices is not None:
                save_frame_index_map(
                    os.path.join(seq_dir, "frame_index_map.json"),
                    seq.original_frame_indices,
                    image_paths=seq.image_paths,
                )

            if save_images:
                print(f"[longstream] sequence {seq.name}: saving rgb", flush=True)
                rgb_dir = os.path.join(seq_dir, "images", "rgb")
                save_image_sequence(rgb_dir, list(rgb))
                if save_videos:
                    save_video(
                        os.path.join(seq_dir, "images", "rgb.mp4"),
                        os.path.join(rgb_dir, "frame_*.png"),
                    )

            sky_masks = None
            if mask_sky:
                raw_sky_masks = compute_sky_mask(
                    seq.image_paths, skyseg_path, os.path.join(seq_dir, "sky_masks")
                )
                if raw_sky_masks is not None:
                    sky_masks = [
                        _prepare_mask_for_model(
                            mask,
                            size=int(data_cfg.get("size", 518)),
                            crop=bool(data_cfg.get("crop", False)),
                            patch_size=int(data_cfg.get("patch_size", 14)),
                            target_shape=(H, W),
                        )
                        for mask in raw_sky_masks
                    ]

            if save_depth and outputs.get("depth") is not None:
                print(f"[longstream] sequence {seq.name}: saving depth", flush=True)
                depth = outputs["depth"][0, :, :, :, 0].detach().cpu().numpy()
                # 应用尺度补偿系数（来自 GT 位姿校正）
                if correction_enabled and np.any(scale_corrections != 1.0):
                    depth = _apply_depth_corrections(depth, scale_corrections)
                depth_dir = os.path.join(seq_dir, "depth", "dpt")
                _ensure_dir(depth_dir)
                color_dir = os.path.join(seq_dir, "depth", "dpt_plasma")
                _ensure_dir(color_dir)

                # 提前提取 depth_conf（容错：若不存在则置 None）
                _raw_depth_conf = outputs.get("depth_conf")
                depth_conf_np = None
                if _raw_depth_conf is not None:
                    try:
                        dc_arr = _raw_depth_conf[0].detach().cpu().numpy()  # [S, H, W, ?]
                        if dc_arr.ndim == 4:
                            dc_arr = dc_arr[..., 0]  # [S, H, W]
                        depth_conf_np = dc_arr
                    except Exception:
                        depth_conf_np = None

                color_frames = []
                for i in range(S):
                    d = depth[i]
                    sky_m = sky_masks[i] if (sky_masks is not None and sky_masks[i] is not None) else None
                    conf_m = None
                    if confidence_filter_enabled and depth_conf_np is not None:
                        conf_m = (depth_conf_np[i] > confidence_threshold).astype(np.uint8)
                    combined = _combine_masks(sky_m, conf_m)
                    if combined is not None:
                        d = _apply_sky_mask(d, combined)
                    np.save(os.path.join(depth_dir, f"frame_{i:06d}.npy"), d)
                    colored = colorize_depth(d, cmap="plasma")
                    Image.fromarray(colored).save(
                        os.path.join(color_dir, f"frame_{i:06d}.png")
                    )
                    color_frames.append(colored)
                if save_videos:
                    save_video(
                        os.path.join(seq_dir, "depth", "dpt_plasma.mp4"),
                        os.path.join(color_dir, "frame_*.png"),
                    )

            # ============================================================
            # 点云导出：统一使用校正后的 extri_np/intri_np
            # ============================================================
            if save_points:
                print(
                    f"[longstream] sequence {seq.name}: saving point clouds", flush=True
                )
                # --- point_head 分支 ---
                if outputs.get("world_points") is not None and extri_np is not None:
                    pts_dir = os.path.join(seq_dir, "points", "point_head")
                    _ensure_dir(pts_dir)
                    pts = outputs["world_points"][0].detach().cpu().numpy()
                    # 提取 world_points_conf（容错）
                    _raw_wpc = outputs.get("world_points_conf")
                    wpc_np = None
                    if _raw_wpc is not None:
                        try:
                            wpc_arr = _raw_wpc[0].detach().cpu().numpy()  # [S, H, W, ?]
                            if wpc_arr.ndim == 4:
                                wpc_arr = wpc_arr[..., 0]
                            wpc_np = wpc_arr
                        except Exception:
                            wpc_np = None
                    full_pts = []
                    full_cols = []
                    for i in range(S):
                        pts_cam = pts[i]
                        # 尺度校正（与 dpt_unproj 保持一致）
                        if correction_enabled and scale_corrections[i] != 1.0:
                            pts_cam = pts_cam * float(scale_corrections[i])
                        pts_world = _camera_points_to_world(pts_cam, extri_np[i])
                        pts_world = pts_world.reshape(pts[i].shape)
                        sky_m = sky_masks[i] if (sky_masks is not None and sky_masks[i] is not None) else None
                        conf_m = None
                        if confidence_filter_enabled and wpc_np is not None:
                            conf_m = (wpc_np[i] > confidence_threshold).astype(np.uint8)
                        valid_mask = _combine_masks(sky_m, conf_m)
                        pts_i, cols_i = _mask_points_and_colors(
                            pts_world,
                            rgb[i],
                            valid_mask,
                        )
                        if save_frame_points:
                            save_pointcloud(
                                os.path.join(pts_dir, f"frame_{i:06d}.ply"),
                                pts_i,
                                colors=cols_i,
                                max_points=max_frame_pointcloud_points,
                                seed=i,
                            )
                        if len(pts_i):
                            full_pts.append(pts_i)
                            full_cols.append(cols_i)
                    _save_full_pointcloud(
                        os.path.join(seq_dir, "points", "point_head_full.ply"),
                        full_pts,
                        full_cols,
                        max_points=max_full_pointcloud_points,
                        seed=0,
                    )

                # --- dpt_unproj 分支 ---
                if (
                    outputs.get("depth") is not None
                    and extri_np is not None
                    and intri_np is not None
                ):
                    depth_for_pts = outputs["depth"][0, :, :, :, 0]
                    # 对深度做尺度校正（与保存的深度一致）
                    if correction_enabled and np.any(scale_corrections != 1.0):
                        depth_for_pts = depth_for_pts.clone()
                        for i in range(S):
                            depth_for_pts[i] = depth_for_pts[i] * float(scale_corrections[i])

                    dpt_pts_dir = os.path.join(seq_dir, "points", "dpt_unproj")
                    _ensure_dir(dpt_pts_dir)
                    full_pts = []
                    full_cols = []
                    intri_torch = torch.from_numpy(intri_np).to(depth_for_pts.device)

                    # 提取 depth_conf（容错，dpt_unproj 分支独立提取）
                    _raw_dconf = outputs.get("depth_conf")
                    dconf_np = None
                    if _raw_dconf is not None:
                        try:
                            dconf_arr = _raw_dconf[0].detach().cpu().numpy()  # [S, H, W, ?]
                            if dconf_arr.ndim == 4:
                                dconf_arr = dconf_arr[..., 0]
                            dconf_np = dconf_arr
                        except Exception:
                            dconf_np = None

                    for i in range(S):
                        d = depth_for_pts[i]
                        pts_cam = unproject_depth_to_points(
                            d[None], intri_torch[i : i + 1]
                        )[0]
                        R_np = extri_np[i, :3, :3]
                        t_np = extri_np[i, :3, 3]
                        pts_cam_np = pts_cam.cpu().numpy().reshape(-1, 3)
                        pts_world = (
                            R_np.T @ (pts_cam_np.T - t_np[:, None])
                        ).T.astype(np.float32)
                        sky_m = sky_masks[i] if (sky_masks is not None and sky_masks[i] is not None) else None
                        conf_m = None
                        if confidence_filter_enabled and dconf_np is not None:
                            conf_m = (dconf_np[i] > confidence_threshold).astype(np.uint8)
                        valid_mask = _combine_masks(sky_m, conf_m)
                        pts_i, cols_i = _mask_points_and_colors(
                            pts_world,
                            rgb[i],
                            valid_mask,
                        )
                        if save_frame_points:
                            save_pointcloud(
                                os.path.join(dpt_pts_dir, f"frame_{i:06d}.ply"),
                                pts_i,
                                colors=cols_i,
                                max_points=max_frame_pointcloud_points,
                                seed=i,
                            )
                        if len(pts_i):
                            full_pts.append(pts_i)
                            full_cols.append(cols_i)
                    _save_full_pointcloud(
                        os.path.join(seq_dir, "points", "dpt_unproj_full.ply"),
                        full_pts,
                        full_cols,
                        max_points=max_full_pointcloud_points,
                        seed=1,
                    )
            del outputs
            if device_type == "cuda":
                torch.cuda.empty_cache()


def run_inference(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    run_inference_cfg(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_inference(args.config)


if __name__ == "__main__":
    main()
