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


def _compute_segment_corrections(extri_np, gt_poses_arr, S, interval, align_mode):
    """
    段级校正：在每个锚帧计算完整校正变换 C = GT @ inv(pred)，
    并将其传播到段内所有帧，系统性修正长期拼接漂移。

    段尺度因子由段内 pred/GT 帧间相对平移范数的中位数比值决定（而非
    单个锚帧的绝对平移），这样对世界原点不敏感且对离群帧更稳健。

    align_mode="full":
        对段内每帧传播完整校正 corrected[i] = C_anchor @ pred[i]，
        同时修正旋转和平移漂移。
    align_mode="scale_only":
        仅将段尺度比例传播到段内每帧的平移分量。

    Returns:
        (scale_corrections, corrected_extri_np)
        scale_corrections: 逐帧尺度因子，用于深度和点云。
    """
    if extri_np.ndim != 3 or extri_np.shape[-2:] != (4, 4):
        raise ValueError(
            f"Expected predicted extrinsics with shape [S,4,4], got {extri_np.shape}"
        )
    if gt_poses_arr.ndim != 3 or gt_poses_arr.shape[-2:] != (4, 4):
        raise ValueError(
            f"Expected GT poses with shape [S,4,4], got {gt_poses_arr.shape}"
        )

    corrected = extri_np.copy()
    scale_corrections = np.ones(S, dtype=np.float32)

    n_gt = len(gt_poses_arr)

    # 收集有 GT 对应的锚帧
    anchors = []
    for seg_start in range(0, S, interval):
        if seg_start >= n_gt:
            break
        anchors.append(seg_start)

    if not anchors:
        return scale_corrections, corrected

    for seg_idx, a in enumerate(anchors):
        seg_end = anchors[seg_idx + 1] if seg_idx + 1 < len(anchors) else S

        # ---- 段尺度：用段内帧间相对运动中位数比值 ----
        seg_scale = _estimate_segment_scale(extri_np, gt_poses_arr, a, seg_end)

        if align_mode == "full":
            # 完整校正变换: C @ pred[a] ≈ gt[a]
            C_a = gt_poses_arr[a] @ np.linalg.inv(extri_np[a])
            for i in range(a, seg_end):
                corrected[i] = C_a @ extri_np[i]
        else:
            # 仅缩放平移分量
            for i in range(a, seg_end):
                corrected[i, :3, 3] = extri_np[i, :3, 3] * seg_scale

        scale_corrections[a:seg_end] = seg_scale

    return scale_corrections, corrected


def _estimate_segment_scale(extri_np, gt_poses_arr, seg_start, seg_end):
    """
    估算段 [seg_start, seg_end) 的 GT/pred 尺度比。

    方法：收集段内所有连续帧对 (i, i+1)（要求 GT 可用），计算帧间平移
    范数比 ||Δt_gt|| / ||Δt_pred||，取中位数作为段尺度。

    当帧间运动不足时退回到锚帧绝对平移范数比。
    """
    n_gt = len(gt_poses_arr)
    ratios = []
    upper = min(seg_end, n_gt)
    for i in range(seg_start, upper - 1):
        j = i + 1
        if j >= n_gt:
            break
        dt_pred = np.linalg.norm(extri_np[j, :3, 3] - extri_np[i, :3, 3])
        dt_gt = np.linalg.norm(gt_poses_arr[j, :3, 3] - gt_poses_arr[i, :3, 3])
        if dt_pred > 1e-6:
            ratios.append(dt_gt / dt_pred)

    if ratios:
        return float(np.median(ratios))

    # 退回锚帧绝对平移范数比
    t_pred_norm = float(np.linalg.norm(extri_np[seg_start, :3, 3]))
    t_gt_norm = float(np.linalg.norm(gt_poses_arr[seg_start, :3, 3]))
    if t_pred_norm > 1e-6 or t_gt_norm > 1e-6:
        return t_gt_norm / (t_pred_norm + 1e-8)
    return 1.0


def _apply_depth_corrections(depth_np, scale_corrections):
    """逐帧应用尺度校正到深度图（就地修改并返回）。"""
    S = depth_np.shape[0]
    for i in range(S):
        depth_np[i] = depth_np[i] * float(scale_corrections[i])
    return depth_np


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
    correction_enabled = bool(corr_cfg.get("enabled", False))
    correction_interval = max(1, int(corr_cfg.get("interval", 10)))
    align_mode = str(corr_cfg.get("align_mode", "full"))

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
            # GT 位姿校正：段级尺度 + 锚帧硬重置
            # scale_corrections 同时用于深度和点云
            # ============================================================
            scale_corrections = np.ones(S, dtype=np.float32)
            if correction_enabled and extri_np is not None and seq.gt_poses is not None:
                scale_corrections, extri_np = _compute_segment_corrections(
                    extri_np, seq.gt_poses, S, correction_interval, align_mode
                )
                for seg_start in range(0, S, correction_interval):
                    if seg_start >= len(seq.gt_poses):
                        break
                    print(
                        f"[longstream][correction] seq={seq.name}"
                        f" frame={seg_start} scale={scale_corrections[seg_start]:.4f}",
                        flush=True,
                    )
                # 保存校正后位姿
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

                color_frames = []
                for i in range(S):
                    d = depth[i]
                    if sky_masks is not None and sky_masks[i] is not None:
                        d = _apply_sky_mask(d, sky_masks[i])
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
                    full_pts = []
                    full_cols = []
                    for i in range(S):
                        pts_cam = pts[i]
                        # 尺度校正（与 dpt_unproj 保持一致）
                        if correction_enabled and scale_corrections[i] != 1.0:
                            pts_cam = pts_cam * float(scale_corrections[i])
                        pts_world = _camera_points_to_world(pts_cam, extri_np[i])
                        pts_world = pts_world.reshape(pts[i].shape)
                        pts_i, cols_i = _mask_points_and_colors(
                            pts_world,
                            rgb[i],
                            None if sky_masks is None else sky_masks[i],
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
                        pts_i, cols_i = _mask_points_and_colors(
                            pts_world,
                            rgb[i],
                            None if sky_masks is None else sky_masks[i],
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
