import argparse
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from longstream.preprocess import (
    default_preprocess_config_path,
    load_preprocess_config,
    prepare_images_to_generalizable,
    prepare_video_to_generalizable,
)
from longstream.preprocess.config import deep_update


DEFAULT_DYNAMIC_CLASS_IDS = (0, 1, 2, 3, 5, 7)


def _natural_sort_key(path: str):
    stem = Path(path).stem
    parts = []
    token = ""
    is_digit = None
    for ch in stem:
        curr_digit = ch.isdigit()
        if is_digit is None or curr_digit == is_digit:
            token += ch
        else:
            parts.append((0, int(token)) if is_digit else (1, token.lower()))
            token = ch
        is_digit = curr_digit
    if token:
        parts.append((0, int(token)) if is_digit else (1, token.lower()))
    return parts, Path(path).name.lower()


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(path)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image


def _write_image(path: Path, image: np.ndarray):
    ext = path.suffix.lower()
    if ext == ".png":
        ok, buf = cv2.imencode(ext, image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    elif ext in {".jpg", ".jpeg"}:
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        raise ValueError(f"unsupported image extension: {path.suffix}")
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    _ensure_dir(path.parent)
    buf.tofile(str(path))


def _resize_mask(mask: np.ndarray, image_shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = image_shape_hw
    if mask.shape[:2] == (h, w):
        return mask.astype(np.uint8, copy=False)
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def _copy_scene_metadata(src_scene_root: Path, dst_scene_root: Path):
    for subdir in ["cameras", "depths", "masks", "vis_depths"]:
        src = src_scene_root / subdir
        dst = dst_scene_root / subdir
        if src.exists() and src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)


def _copy_prepared_scene(
    src_root: Path, dst_root: Path, scene_name: str, overwrite: bool = True
):
    src_scene_root = src_root / scene_name
    dst_scene_root = dst_root / scene_name
    if not src_scene_root.exists():
        raise FileNotFoundError(src_scene_root)
    if dst_scene_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"target scene already exists: {dst_scene_root}. "
                "Enable overwrite to replace it."
            )
        shutil.rmtree(dst_scene_root)
    shutil.copytree(src_scene_root, dst_scene_root)
    _ensure_dir(dst_root)
    with open(dst_root / "data_roots.txt", "w", encoding="utf-8") as f:
        f.write(f"{scene_name}\n")


@dataclass
class InstanceSegmentation:
    class_id: int
    class_name: str
    confidence: float
    mask: np.ndarray
    area: int
    bbox_xyxy: Tuple[float, float, float, float]


@dataclass
class FrameSegmentation:
    instances: List[InstanceSegmentation]
    semantic_mask: np.ndarray


@dataclass
class MotionDecision:
    instance_index: int
    class_id: int
    class_name: str
    confidence: float
    area: int
    bbox_xyxy: Tuple[float, float, float, float]
    sampled_points: int
    median_error: float
    mean_error: float
    dynamic_ratio: float
    median_flow_magnitude: float
    is_dynamic: bool
    reason: str
    raw_is_dynamic: Optional[bool] = None
    track_id: Optional[int] = None
    temporal_vote_dynamic_count: int = 0
    temporal_vote_total_count: int = 0
    temporal_dynamic_fraction: float = 0.0
    temporal_window_size: int = 1
    temporal_refined: bool = False


@dataclass
class GeometryDebug:
    model: str
    valid: bool
    matched_points: int
    inlier_points: int
    inlier_ratio: float


@dataclass
class FramePairResult:
    original_image: np.ndarray
    instances: List[InstanceSegmentation]
    masked_image: np.ndarray
    dynamic_mask: np.ndarray
    semantic_mask: np.ndarray
    static_mask: np.ndarray
    overlay_image: np.ndarray
    comparison_image: np.ndarray
    object_decisions: List[MotionDecision] = field(default_factory=list)
    geometry_model: str = "homography"
    geometry_matrix: Optional[np.ndarray] = None
    geometry_debug: Optional[GeometryDebug] = None


class SemanticFlowDynamicMasker:
    def __init__(
        self,
        model_path: str = "./model/yolo11n-seg.pt",
        dynamic_class_ids: Sequence[int] = DEFAULT_DYNAMIC_CLASS_IDS,
        geometry_model: str = "fundamental",
        conf: float = 0.25,
        iou: float = 0.5,
        max_corners: int = 800,
        lk_win_size: Tuple[int, int] = (21, 21),
        farneback_levels: int = 3,
        farneback_winsize: int = 21,
        farneback_backward_winsize: Optional[int] = None,
        pixel_error_threshold: float = 3.0,
        dynamic_ratio_threshold: float = 0.35,
        min_motion_magnitude: float = 0.75,
        min_mask_area: int = 80,
        max_mask_points: int = 2000,
        homography_ransac_threshold: float = 3.0,
        fundamental_ransac_threshold: float = 1.5,
        dilation_kernel_size: int = 5,
        dilation_iterations: int = 2,
        temporal_consistency_enabled: bool = True,
        temporal_window_size: int = 7,
        temporal_vote_ratio_threshold: float = 0.6,
        temporal_min_track_length: int = 3,
        temporal_match_iou_threshold: float = 0.2,
        temporal_max_center_distance_ratio: float = 0.75,
        fallback_keep_semantic_if_geometry_fails: bool = True,
        random_seed: int = 0,
    ):
        self.yolo_model = YOLO(model_path)
        self.dynamic_class_ids = tuple(int(x) for x in dynamic_class_ids)
        self.geometry_model = str(geometry_model).lower()
        if self.geometry_model not in {"fundamental", "homography"}:
            raise ValueError(
                f"unsupported geometry_model: {self.geometry_model}. "
                "Expected 'fundamental' or 'homography'."
            )
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_corners = int(max_corners)
        self.lk_win_size = tuple(int(x) for x in lk_win_size)
        self.farneback_levels = int(farneback_levels)
        self.farneback_winsize = int(farneback_winsize)
        self.farneback_backward_winsize = int(
            farneback_backward_winsize
            if farneback_backward_winsize is not None
            else farneback_winsize
        )
        self.pixel_error_threshold = float(pixel_error_threshold)
        self.dynamic_ratio_threshold = float(dynamic_ratio_threshold)
        self.min_motion_magnitude = float(min_motion_magnitude)
        self.min_mask_area = int(min_mask_area)
        self.max_mask_points = int(max_mask_points)
        self.homography_ransac_threshold = float(homography_ransac_threshold)
        self.fundamental_ransac_threshold = float(fundamental_ransac_threshold)
        self.dilation_kernel_size = int(dilation_kernel_size)
        self.dilation_iterations = int(dilation_iterations)
        self.temporal_consistency_enabled = bool(temporal_consistency_enabled)
        self.temporal_window_size = max(1, int(temporal_window_size))
        self.temporal_vote_ratio_threshold = float(temporal_vote_ratio_threshold)
        self.temporal_min_track_length = max(1, int(temporal_min_track_length))
        self.temporal_match_iou_threshold = float(temporal_match_iou_threshold)
        self.temporal_max_center_distance_ratio = float(
            temporal_max_center_distance_ratio
        )
        self.fallback_keep_semantic_if_geometry_fails = bool(
            fallback_keep_semantic_if_geometry_fails
        )
        self.rng = np.random.default_rng(random_seed)
        self.class_names = self.yolo_model.names

    @classmethod
    def from_preprocess_config(cls, preprocess_cfg: Dict):
        dynamic_cfg = dict(preprocess_cfg.get("dynamic_filter", {}))
        model_path = dynamic_cfg.pop("model_path", "./model/yolo11n-seg.pt")
        dynamic_class_ids = dynamic_cfg.pop(
            "dynamic_class_ids", DEFAULT_DYNAMIC_CLASS_IDS
        )
        if (
            "fallback_keep_semantic_if_geometry_fails" not in dynamic_cfg
            and "fallback_keep_semantic_if_homography_fails" in dynamic_cfg
        ):
            dynamic_cfg["fallback_keep_semantic_if_geometry_fails"] = dynamic_cfg.pop(
                "fallback_keep_semantic_if_homography_fails"
            )
        dynamic_cfg.pop("enabled", None)
        dynamic_cfg.pop("save_preview_video", None)
        dynamic_cfg.pop("preview_fps", None)
        return cls(
            model_path=model_path,
            dynamic_class_ids=dynamic_class_ids,
            **dynamic_cfg,
        )

    def _segment_frames(self, frames: Sequence[np.ndarray]) -> List[FrameSegmentation]:
        results = self.yolo_model(
            list(frames),
            classes=list(self.dynamic_class_ids),
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        segmentations: List[FrameSegmentation] = []
        for frame, result in zip(frames, results):
            h, w = frame.shape[:2]
            instances: List[InstanceSegmentation] = []
            semantic_mask = np.zeros((h, w), dtype=np.uint8)

            boxes = result.boxes
            masks = result.masks
            if boxes is not None and masks is not None and len(boxes) > 0:
                mask_data = masks.data.detach().cpu().numpy()
                cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
                confs = boxes.conf.detach().cpu().numpy().astype(float)
                xyxy = boxes.xyxy.detach().cpu().numpy().astype(float)

                for idx in range(len(cls_ids)):
                    mask = (mask_data[idx] > 0.5).astype(np.uint8)
                    mask = _resize_mask(mask, (h, w))
                    area = int(mask.sum())
                    if area <= 0:
                        continue
                    class_id = int(cls_ids[idx])
                    class_name = str(self.class_names.get(class_id, class_id))
                    instances.append(
                        InstanceSegmentation(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=float(confs[idx]),
                            mask=mask,
                            area=area,
                            bbox_xyxy=tuple(float(v) for v in xyxy[idx]),
                        )
                    )
                    semantic_mask = np.maximum(semantic_mask, mask)

            segmentations.append(
                FrameSegmentation(instances=instances, semantic_mask=semantic_mask * 255)
            )
        return segmentations

    def _estimate_background_geometry(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_semantic_mask: np.ndarray,
        curr_semantic_mask: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], GeometryDebug]:
        prev_bg = cv2.bitwise_not(prev_semantic_mask)
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=0.01,
            minDistance=8,
            mask=prev_bg,
        )

        min_points = 8 if self.geometry_model == "fundamental" else 4
        if prev_pts is None or len(prev_pts) < min_points:
            return None, GeometryDebug(self.geometry_model, False, 0, 0, 0.0)

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_pts,
            None,
            winSize=self.lk_win_size,
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        if curr_pts is None or status is None:
            return None, GeometryDebug(self.geometry_model, False, 0, 0, 0.0)

        valid = status.reshape(-1) == 1
        good_prev = prev_pts.reshape(-1, 2)[valid]
        good_curr = curr_pts.reshape(-1, 2)[valid]

        if len(good_prev) < min_points:
            return None, GeometryDebug(
                self.geometry_model, False, int(len(good_prev)), 0, 0.0
            )

        h, w = curr_gray.shape[:2]
        keep = (
            (good_curr[:, 0] >= 0)
            & (good_curr[:, 0] < w)
            & (good_curr[:, 1] >= 0)
            & (good_curr[:, 1] < h)
        )
        if curr_semantic_mask is not None:
            xi = np.clip(good_curr[:, 0].astype(np.int32), 0, w - 1)
            yi = np.clip(good_curr[:, 1].astype(np.int32), 0, h - 1)
            keep &= curr_semantic_mask[yi, xi] == 0

        good_prev = good_prev[keep]
        good_curr = good_curr[keep]
        if len(good_prev) < min_points:
            return None, GeometryDebug(
                self.geometry_model, False, int(len(good_prev)), 0, 0.0
            )

        if self.geometry_model == "fundamental":
            matrix, inliers = cv2.findFundamentalMat(
                good_prev,
                good_curr,
                cv2.FM_RANSAC,
                self.fundamental_ransac_threshold,
                0.99,
            )
            if matrix is not None and matrix.shape != (3, 3):
                matrix = matrix[:3, :3]
        else:
            matrix, inliers = cv2.findHomography(
                good_prev,
                good_curr,
                cv2.RANSAC,
                self.homography_ransac_threshold,
            )

        if matrix is None or inliers is None:
            return None, GeometryDebug(
                self.geometry_model, False, int(len(good_prev)), 0, 0.0
            )

        inlier_count = int(inliers.ravel().sum())
        matched_points = int(len(good_prev))
        inlier_ratio = float(inlier_count / max(matched_points, 1))
        return matrix, GeometryDebug(
            self.geometry_model, True, matched_points, inlier_count, inlier_ratio
        )

    def _compute_dense_flow(
        self, src_gray: np.ndarray, dst_gray: np.ndarray, winsize: int
    ) -> np.ndarray:
        return cv2.calcOpticalFlowFarneback(
            src_gray,
            dst_gray,
            None,
            pyr_scale=0.5,
            levels=self.farneback_levels,
            winsize=winsize,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

    def _compute_dense_flows(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        forward_flow = self._compute_dense_flow(
            prev_gray,
            curr_gray,
            self.farneback_winsize,
        )
        backward_flow = self._compute_dense_flow(
            curr_gray,
            prev_gray,
            self.farneback_backward_winsize,
        )
        return forward_flow, backward_flow

    def _sample_mask_points(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        if len(x_coords) > self.max_mask_points:
            indices = self.rng.choice(len(x_coords), size=self.max_mask_points, replace=False)
            y_coords = y_coords[indices]
            x_coords = x_coords[indices]
        return y_coords.astype(np.int32), x_coords.astype(np.int32)

    def _classify_instance_motion(
        self,
        instance_index: int,
        instance: InstanceSegmentation,
        backward_flow: np.ndarray,
        geometry_matrix: Optional[np.ndarray],
    ) -> MotionDecision:
        if instance.area < self.min_mask_area:
            return MotionDecision(
                instance_index=instance_index,
                class_id=instance.class_id,
                class_name=instance.class_name,
                confidence=instance.confidence,
                area=instance.area,
                bbox_xyxy=instance.bbox_xyxy,
                sampled_points=0,
                median_error=0.0,
                mean_error=0.0,
                dynamic_ratio=0.0,
                median_flow_magnitude=0.0,
                is_dynamic=False,
                reason="mask_too_small",
                raw_is_dynamic=False,
            )

        if geometry_matrix is None:
            return MotionDecision(
                instance_index=instance_index,
                class_id=instance.class_id,
                class_name=instance.class_name,
                confidence=instance.confidence,
                area=instance.area,
                bbox_xyxy=instance.bbox_xyxy,
                sampled_points=0,
                median_error=0.0,
                mean_error=0.0,
                dynamic_ratio=0.0,
                median_flow_magnitude=0.0,
                is_dynamic=not self.fallback_keep_semantic_if_geometry_fails,
                reason=f"{self.geometry_model}_unavailable",
                raw_is_dynamic=not self.fallback_keep_semantic_if_geometry_fails,
            )

        y_coords, x_coords = self._sample_mask_points(instance.mask)
        if len(x_coords) < 8:
            return MotionDecision(
                instance_index=instance_index,
                class_id=instance.class_id,
                class_name=instance.class_name,
                confidence=instance.confidence,
                area=instance.area,
                bbox_xyxy=instance.bbox_xyxy,
                sampled_points=int(len(x_coords)),
                median_error=0.0,
                mean_error=0.0,
                dynamic_ratio=0.0,
                median_flow_magnitude=0.0,
                is_dynamic=False,
                reason="not_enough_points",
                raw_is_dynamic=False,
            )

        current_pts = np.stack(
            [x_coords.astype(np.float32), y_coords.astype(np.float32)], axis=1
        )
        backward_disp = backward_flow[y_coords, x_coords]
        prev_pts = current_pts + backward_disp
        flow_magnitude = np.linalg.norm(backward_disp, axis=1)
        h, w = instance.mask.shape[:2]
        valid = (
            np.isfinite(prev_pts[:, 0])
            & np.isfinite(prev_pts[:, 1])
            & (prev_pts[:, 0] >= 0)
            & (prev_pts[:, 0] < w)
            & (prev_pts[:, 1] >= 0)
            & (prev_pts[:, 1] < h)
        )
        if not np.any(valid):
            return MotionDecision(
                instance_index=instance_index,
                class_id=instance.class_id,
                class_name=instance.class_name,
                confidence=instance.confidence,
                area=instance.area,
                bbox_xyxy=instance.bbox_xyxy,
                sampled_points=0,
                median_error=0.0,
                mean_error=0.0,
                dynamic_ratio=0.0,
                median_flow_magnitude=0.0,
                is_dynamic=False,
                reason="no_valid_correspondence_points",
                raw_is_dynamic=False,
            )
        current_pts = current_pts[valid]
        prev_pts = prev_pts[valid]
        flow_magnitude = flow_magnitude[valid]

        if self.geometry_model == "fundamental":
            prev_h = cv2.convertPointsToHomogeneous(prev_pts).reshape(-1, 3)
            lines = (geometry_matrix @ prev_h.T).T
            denom = np.linalg.norm(lines[:, :2], axis=1)
            valid = denom > 1e-6
            if not np.any(valid):
                return MotionDecision(
                    instance_index=instance_index,
                    class_id=instance.class_id,
                    class_name=instance.class_name,
                    confidence=instance.confidence,
                    area=instance.area,
                    bbox_xyxy=instance.bbox_xyxy,
                    sampled_points=0,
                    median_error=0.0,
                    mean_error=0.0,
                    dynamic_ratio=0.0,
                    median_flow_magnitude=0.0,
                    is_dynamic=False,
                    reason="degenerate_epipolar_lines",
                    raw_is_dynamic=False,
                )
            lines = lines[valid]
            current_pts = current_pts[valid]
            flow_magnitude = flow_magnitude[valid]
            errors = np.abs(
                lines[:, 0] * current_pts[:, 0]
                + lines[:, 1] * current_pts[:, 1]
                + lines[:, 2]
            ) / np.linalg.norm(lines[:, :2], axis=1)
        else:
            src_pts = prev_pts.reshape(-1, 1, 2)
            pred_pts = cv2.perspectiveTransform(src_pts, geometry_matrix).reshape(-1, 2)
            errors = np.linalg.norm(current_pts - pred_pts, axis=1)

        median_error = float(np.median(errors))
        mean_error = float(np.mean(errors))
        dynamic_ratio = float(np.mean(errors > self.pixel_error_threshold))
        median_flow_magnitude = float(np.median(flow_magnitude))

        error_dynamic = (
            median_error > self.pixel_error_threshold
            or dynamic_ratio > self.dynamic_ratio_threshold
        )
        motion_significant = median_flow_magnitude > self.min_motion_magnitude
        is_dynamic = bool(error_dynamic and motion_significant)
        reason = "dynamic" if is_dynamic else "background_consistent"

        return MotionDecision(
            instance_index=instance_index,
            class_id=instance.class_id,
            class_name=instance.class_name,
            confidence=instance.confidence,
            area=instance.area,
            bbox_xyxy=instance.bbox_xyxy,
            sampled_points=int(len(x_coords)),
            median_error=median_error,
            mean_error=mean_error,
            dynamic_ratio=dynamic_ratio,
            median_flow_magnitude=median_flow_magnitude,
            is_dynamic=is_dynamic,
            reason=reason,
            raw_is_dynamic=is_dynamic,
        )

    def _dilate_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.dilation_kernel_size <= 1 or self.dilation_iterations <= 0:
            return mask
        kernel = np.ones(
            (self.dilation_kernel_size, self.dilation_kernel_size), dtype=np.uint8
        )
        return cv2.dilate(mask, kernel, iterations=self.dilation_iterations)

    def _bbox_iou(
        self,
        bbox_a: Tuple[float, float, float, float],
        bbox_b: Tuple[float, float, float, float],
    ) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union_area = area_a + area_b - inter_area
        if union_area <= 0.0:
            return 0.0
        return float(inter_area / union_area)

    def _bbox_center_and_diag(
        self, bbox: Tuple[float, float, float, float]
    ) -> Tuple[np.ndarray, float]:
        x1, y1, x2, y2 = bbox
        center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
        diag = float(np.hypot(max(0.0, x2 - x1), max(0.0, y2 - y1)))
        return center, max(diag, 1.0)

    def _render_result_visuals(self, result: FramePairResult) -> None:
        dynamic_mask = np.zeros(result.original_image.shape[:2], dtype=np.uint8)
        static_mask = np.zeros(result.original_image.shape[:2], dtype=np.uint8)
        decision_by_index = {
            decision.instance_index: decision for decision in result.object_decisions
        }

        for idx, instance in enumerate(result.instances):
            decision = decision_by_index.get(idx)
            if decision is None:
                continue
            if decision.is_dynamic:
                dynamic_mask = np.maximum(dynamic_mask, instance.mask.astype(np.uint8) * 255)
            else:
                static_mask = np.maximum(static_mask, instance.mask.astype(np.uint8) * 255)

        dynamic_mask = self._dilate_mask(dynamic_mask)
        masked_image = result.original_image.copy()
        masked_image[dynamic_mask > 0] = 0
        overlay_image = self._build_overlay(result.original_image, dynamic_mask, static_mask)
        comparison_image = np.hstack([result.original_image, overlay_image, masked_image])

        result.dynamic_mask = dynamic_mask
        result.static_mask = static_mask
        result.masked_image = masked_image
        result.overlay_image = overlay_image
        result.comparison_image = comparison_image

    def _match_to_previous_track(
        self,
        prev_result: FramePairResult,
        curr_decision: MotionDecision,
        used_prev_indices: set,
    ) -> Optional[int]:
        curr_center, curr_diag = self._bbox_center_and_diag(curr_decision.bbox_xyxy)
        best_idx = None
        best_score = None

        for prev_decision in prev_result.object_decisions:
            prev_idx = prev_decision.instance_index
            if prev_idx in used_prev_indices:
                continue
            if prev_decision.class_id != curr_decision.class_id:
                continue

            iou = self._bbox_iou(prev_decision.bbox_xyxy, curr_decision.bbox_xyxy)
            prev_center, prev_diag = self._bbox_center_and_diag(prev_decision.bbox_xyxy)
            norm_dist = float(
                np.linalg.norm(curr_center - prev_center) / max(curr_diag, prev_diag, 1.0)
            )
            if (
                iou < self.temporal_match_iou_threshold
                and norm_dist > self.temporal_max_center_distance_ratio
            ):
                continue

            score = iou - 0.1 * norm_dist
            if best_score is None or score > best_score:
                best_score = score
                best_idx = prev_idx

        return best_idx

    def _assign_temporal_tracks(
        self, results: List[FramePairResult]
    ) -> Dict[int, List[Tuple[int, int]]]:
        tracks: Dict[int, List[Tuple[int, int]]] = {}
        next_track_id = 0

        for frame_idx, result in enumerate(results):
            if frame_idx == 0:
                for decision in result.object_decisions:
                    decision.track_id = next_track_id
                    tracks[next_track_id] = [(frame_idx, decision.instance_index)]
                    next_track_id += 1
                continue

            prev_result = results[frame_idx - 1]
            used_prev_indices = set()
            prev_track_ids = {
                decision.instance_index: decision.track_id
                for decision in prev_result.object_decisions
            }

            for decision in result.object_decisions:
                prev_idx = self._match_to_previous_track(
                    prev_result,
                    decision,
                    used_prev_indices,
                )
                if prev_idx is not None:
                    used_prev_indices.add(prev_idx)
                    track_id = prev_track_ids.get(prev_idx)
                    if track_id is not None:
                        decision.track_id = track_id
                        tracks.setdefault(track_id, []).append(
                            (frame_idx, decision.instance_index)
                        )
                        continue

                decision.track_id = next_track_id
                tracks[next_track_id] = [(frame_idx, decision.instance_index)]
                next_track_id += 1

        return tracks

    def _apply_temporal_consistency(self, results: List[FramePairResult]) -> None:
        for result in results:
            for decision in result.object_decisions:
                if decision.raw_is_dynamic is None:
                    decision.raw_is_dynamic = decision.is_dynamic

        if (
            not self.temporal_consistency_enabled
            or self.temporal_window_size <= 1
            or not results
        ):
            for result in results:
                self._render_result_visuals(result)
            return

        tracks = self._assign_temporal_tracks(results)
        radius = self.temporal_window_size // 2
        decision_lookup = {
            (frame_idx, decision.instance_index): decision
            for frame_idx, result in enumerate(results)
            for decision in result.object_decisions
        }

        for track in tracks.values():
            if not track:
                continue
            track = sorted(track, key=lambda item: item[0])
            for pos, key in enumerate(track):
                decision = decision_lookup[key]
                start = max(0, pos - radius)
                end = min(len(track), pos + radius + 1)
                window_keys = track[start:end]
                vote_total = len(window_keys)
                vote_dynamic = sum(
                    1
                    for item in window_keys
                    if bool(decision_lookup[item].raw_is_dynamic)
                )
                dynamic_fraction = float(vote_dynamic / max(vote_total, 1))
                refined_is_dynamic = bool(decision.raw_is_dynamic)
                if vote_total >= self.temporal_min_track_length:
                    refined_is_dynamic = (
                        dynamic_fraction >= self.temporal_vote_ratio_threshold
                    )
                decision.temporal_vote_dynamic_count = vote_dynamic
                decision.temporal_vote_total_count = vote_total
                decision.temporal_dynamic_fraction = dynamic_fraction
                decision.temporal_window_size = self.temporal_window_size
                decision.temporal_refined = refined_is_dynamic != bool(
                    decision.raw_is_dynamic
                )
                if decision.temporal_refined:
                    decision.reason = (
                        f"temporal_vote_refined_to_"
                        f"{'dynamic' if refined_is_dynamic else 'static'}"
                    )
                decision.is_dynamic = refined_is_dynamic

        for result in results:
            self._render_result_visuals(result)

    def _build_overlay(
        self,
        image: np.ndarray,
        dynamic_mask: np.ndarray,
        static_mask: np.ndarray,
    ) -> np.ndarray:
        overlay = image.copy()
        dynamic_region = dynamic_mask > 0
        static_region = static_mask > 0
        overlay[static_region] = (
            overlay[static_region].astype(np.float32) * 0.5
            + np.array([0, 255, 0], dtype=np.float32) * 0.5
        ).astype(np.uint8)
        overlay[dynamic_region] = (
            overlay[dynamic_region].astype(np.float32) * 0.3
            + np.array([0, 0, 255], dtype=np.float32) * 0.7
        ).astype(np.uint8)
        return overlay

    def process_frame_pair(
        self, prev_img: np.ndarray, curr_img: np.ndarray
    ) -> FramePairResult:
        prev_seg, curr_seg = self._segment_frames([prev_img, curr_img])
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

        geometry_matrix, geometry_debug = self._estimate_background_geometry(
            prev_gray,
            curr_gray,
            prev_seg.semantic_mask,
            curr_seg.semantic_mask,
        )
        _forward_flow, backward_flow = self._compute_dense_flows(prev_gray, curr_gray)

        object_decisions: List[MotionDecision] = []

        for idx, instance in enumerate(curr_seg.instances):
            decision = self._classify_instance_motion(
                idx, instance, backward_flow, geometry_matrix
            )
            object_decisions.append(decision)
        result = FramePairResult(
            original_image=curr_img,
            instances=curr_seg.instances,
            masked_image=np.zeros_like(curr_img),
            dynamic_mask=np.zeros(curr_img.shape[:2], dtype=np.uint8),
            semantic_mask=curr_seg.semantic_mask,
            static_mask=np.zeros(curr_img.shape[:2], dtype=np.uint8),
            overlay_image=np.zeros_like(curr_img),
            comparison_image=np.zeros((curr_img.shape[0], curr_img.shape[1] * 3, 3), dtype=np.uint8),
            object_decisions=object_decisions,
            geometry_model=self.geometry_model,
            geometry_matrix=geometry_matrix,
            geometry_debug=geometry_debug,
        )
        self._render_result_visuals(result)
        return result

    def _list_image_paths(self, image_dir: Path) -> List[Path]:
        paths = [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        return sorted(paths, key=lambda p: _natural_sort_key(str(p)))

    def process_prepared_scene(
        self,
        prepared_root: str,
        masked_root: str,
        scene_name: str,
        camera_id: str = "00",
        save_preview_video: bool = True,
        preview_fps: float = 10.0,
    ) -> Dict:
        prepared_root_path = Path(prepared_root).expanduser().resolve()
        masked_root_path = Path(masked_root).expanduser().resolve()
        src_scene_root = prepared_root_path / scene_name
        src_image_dir = src_scene_root / "images" / camera_id
        if not src_image_dir.is_dir():
            raise FileNotFoundError(src_image_dir)

        dst_scene_root = masked_root_path / scene_name
        dst_image_dir = dst_scene_root / "images" / camera_id
        debug_root = dst_scene_root / "dynamic_filter_debug" / camera_id
        for path in [
            dst_image_dir,
            debug_root / "dynamic_masks",
            debug_root / "semantic_masks",
            debug_root / "overlays",
            debug_root / "comparisons",
            debug_root / "reports",
        ]:
            _ensure_dir(path)

        _copy_scene_metadata(src_scene_root, dst_scene_root)
        with open(masked_root_path / "data_roots.txt", "w", encoding="utf-8") as f:
            f.write(f"{scene_name}\n")

        image_paths = self._list_image_paths(src_image_dir)
        if not image_paths:
            raise RuntimeError(f"no images found in: {src_image_dir}")

        first_image = _read_image(image_paths[0])
        _write_image(dst_image_dir / image_paths[0].name, first_image)

        preview_writer = None
        preview_path = debug_root / "dynamic_filter_preview.mp4"
        summary_records = []

        def _write_preview(frame: np.ndarray):
            nonlocal preview_writer
            if not save_preview_video:
                return
            if preview_writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                preview_writer = cv2.VideoWriter(
                    str(preview_path), fourcc, float(preview_fps), (w, h)
                )
            preview_writer.write(frame)

        first_compare = np.hstack([first_image, first_image, first_image])
        _write_image(debug_root / "comparisons" / image_paths[0].name, first_compare)
        _write_preview(first_compare)
        summary_records.append(
            {
                "frame_name": image_paths[0].name,
                "is_reference_frame": True,
                "object_decisions": [],
            }
        )

        prev_image = first_image
        frame_results: List[Tuple[Path, FramePairResult]] = []
        for image_path in image_paths[1:]:
            curr_image = _read_image(image_path)
            result = self.process_frame_pair(prev_image, curr_image)
            frame_results.append((image_path, result))
            prev_image = curr_image

        self._apply_temporal_consistency([result for _, result in frame_results])

        for image_path, result in frame_results:
            _write_image(dst_image_dir / image_path.name, result.masked_image)
            _write_image(
                debug_root / "dynamic_masks" / image_path.with_suffix(".png").name,
                result.dynamic_mask,
            )
            _write_image(
                debug_root / "semantic_masks" / image_path.with_suffix(".png").name,
                result.semantic_mask,
            )
            _write_image(debug_root / "overlays" / image_path.name, result.overlay_image)
            _write_image(
                debug_root / "comparisons" / image_path.name, result.comparison_image
            )
            _write_preview(result.comparison_image)

            report = {
                "frame_name": image_path.name,
                "is_reference_frame": False,
                "geometry_model": result.geometry_model,
                "geometry_debug": asdict(result.geometry_debug)
                if result.geometry_debug is not None
                else None,
                "homography_debug": asdict(result.geometry_debug)
                if result.geometry_model == "homography"
                and result.geometry_debug is not None
                else None,
                "object_decisions": [asdict(item) for item in result.object_decisions],
            }
            with open(
                debug_root / "reports" / f"{image_path.stem}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(report, f, indent=2, ensure_ascii=True)
            summary_records.append(report)

        if preview_writer is not None:
            preview_writer.release()

        summary = {
            "prepared_root": str(prepared_root_path),
            "masked_root": str(masked_root_path),
            "scene_name": scene_name,
            "camera_id": camera_id,
            "num_frames": len(image_paths),
            "preview_video": str(preview_path) if save_preview_video else None,
            "geometry_model": self.geometry_model,
            "temporal_consistency_enabled": self.temporal_consistency_enabled,
            "temporal_window_size": self.temporal_window_size,
            "temporal_vote_ratio_threshold": self.temporal_vote_ratio_threshold,
            "dynamic_classes": list(self.dynamic_class_ids),
            "reports": summary_records,
        }
        with open(
            debug_root / "summary.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)
        return summary

    def prepare_and_process_video(
        self,
        video_path: str,
        prepared_root: str,
        masked_root: str,
        scene_name: str = "runtime_scene",
        camera_id: str = "00",
        target_fps: float = 5.0,
        image_ext: str = "png",
        overwrite: bool = True,
    ) -> Dict:
        prepare_video_to_generalizable(
            video_path=video_path,
            prepared_root=prepared_root,
            scene_name=scene_name,
            camera_id=camera_id,
            target_fps=target_fps,
            image_ext=image_ext,
            overwrite=overwrite,
        )
        return self.process_prepared_scene(
            prepared_root=prepared_root,
            masked_root=masked_root,
            scene_name=scene_name,
            camera_id=camera_id,
        )

    def prepare_and_process_images(
        self,
        source_path: str,
        prepared_root: str,
        masked_root: str,
        scene_name: str = "runtime_scene",
        camera_id: str = "00",
        image_ext: str = "png",
        overwrite: bool = True,
        recursive: bool = False,
    ) -> Dict:
        prepare_images_to_generalizable(
            source_path=source_path,
            prepared_root=prepared_root,
            scene_name=scene_name,
            camera_id=camera_id,
            image_ext=image_ext,
            overwrite=overwrite,
            recursive=recursive,
        )
        return self.process_prepared_scene(
            prepared_root=prepared_root,
            masked_root=masked_root,
            scene_name=scene_name,
            camera_id=camera_id,
        )


DynamicMasker = SemanticFlowDynamicMasker


def run_preprocess_pipeline(preprocess_cfg: Dict) -> Dict:
    input_cfg = dict(preprocess_cfg.get("input", {}))
    io_cfg = dict(preprocess_cfg.get("io", {}))
    video_cfg = dict(preprocess_cfg.get("video", {}))
    dynamic_cfg = dict(preprocess_cfg.get("dynamic_filter", {}))

    input_mode = str(input_cfg.get("mode", "video")).lower()
    source_path = input_cfg.get("source_path", None)
    recursive_images = bool(input_cfg.get("recursive_images", False))

    scene_name = str(io_cfg.get("scene_name", "runtime_scene"))
    camera_id = str(io_cfg.get("camera_id", "00"))
    image_ext = str(io_cfg.get("image_ext", "png")).lower()
    overwrite = bool(io_cfg.get("overwrite", True))
    normalized_root = str(io_cfg.get("normalized_root", "prepared_inputs/preprocess/raw"))
    final_root = str(io_cfg.get("final_root", normalized_root))

    if input_mode in {"video", "images"} and not source_path:
        raise ValueError("input.source_path must be set for video/images preprocessing")

    if input_mode == "video":
        prepare_video_to_generalizable(
            video_path=source_path,
            prepared_root=normalized_root,
            scene_name=scene_name,
            camera_id=camera_id,
            target_fps=video_cfg.get("target_fps", 0.0),
            image_ext=image_ext,
            overwrite=overwrite,
        )
    elif input_mode == "images":
        prepare_images_to_generalizable(
            source_path=source_path,
            prepared_root=normalized_root,
            scene_name=scene_name,
            camera_id=camera_id,
            image_ext=image_ext,
            overwrite=overwrite,
            recursive=recursive_images,
        )
    elif input_mode != "prepared":
        raise ValueError(f"unsupported input.mode: {input_mode}")

    if bool(dynamic_cfg.get("enabled", False)):
        masker = SemanticFlowDynamicMasker.from_preprocess_config(preprocess_cfg)
        summary = masker.process_prepared_scene(
            prepared_root=normalized_root,
            masked_root=final_root,
            scene_name=scene_name,
            camera_id=camera_id,
            save_preview_video=bool(dynamic_cfg.get("save_preview_video", True)),
            preview_fps=float(dynamic_cfg.get("preview_fps", 10.0)),
        )
        summary["dynamic_filter_enabled"] = True
        summary["final_root"] = summary["masked_root"]
        return summary

    normalized_root_path = Path(normalized_root).expanduser().resolve()
    final_root_path = Path(final_root).expanduser().resolve()
    if normalized_root_path != final_root_path:
        _copy_prepared_scene(
            normalized_root_path,
            final_root_path,
            scene_name=scene_name,
            overwrite=overwrite,
        )

    summary = {
        "input_mode": input_mode,
        "dynamic_filter_enabled": False,
        "scene_name": scene_name,
        "camera_id": camera_id,
        "normalized_root": str(normalized_root_path),
        "final_root": str(final_root_path),
        "source_path": source_path,
    }
    summary_path = final_root_path / scene_name / "preprocess_summary.json"
    _ensure_dir(summary_path.parent)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_preprocess_config_path())
    parser.add_argument("--input-mode", choices=["video", "images", "prepared"], default=None)
    parser.add_argument("--source-path", default=None)
    parser.add_argument("--normalized-root", default=None)
    parser.add_argument("--final-root", default=None)
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument("--image-ext", choices=["png", "jpg", "jpeg"], default=None)
    parser.add_argument("--recursive-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable-dynamic-filter", action="store_true")
    args = parser.parse_args()

    preprocess_cfg = load_preprocess_config(args.config)
    overrides: Dict = {"input": {}, "io": {}, "video": {}, "dynamic_filter": {}}
    if args.input_mode is not None:
        overrides["input"]["mode"] = args.input_mode
    if args.source_path is not None:
        overrides["input"]["source_path"] = args.source_path
    if args.recursive_images:
        overrides["input"]["recursive_images"] = True
    if args.normalized_root is not None:
        overrides["io"]["normalized_root"] = args.normalized_root
    if args.final_root is not None:
        overrides["io"]["final_root"] = args.final_root
    if args.scene_name is not None:
        overrides["io"]["scene_name"] = args.scene_name
    if args.camera_id is not None:
        overrides["io"]["camera_id"] = args.camera_id
    if args.image_ext is not None:
        overrides["io"]["image_ext"] = args.image_ext
    if args.overwrite:
        overrides["io"]["overwrite"] = True
    if args.target_fps is not None:
        overrides["video"]["target_fps"] = args.target_fps
    if args.model_path is not None:
        overrides["dynamic_filter"]["model_path"] = args.model_path
    if args.disable_dynamic_filter:
        overrides["dynamic_filter"]["enabled"] = False

    preprocess_cfg = deep_update(preprocess_cfg, overrides)
    summary = run_preprocess_pipeline(preprocess_cfg)

    final_root = summary.get("final_root", summary.get("masked_root"))
    print(
        "[dynamic-masker] done "
        f"scene={summary['scene_name']} final_root={final_root} "
        f"dynamic_filter_enabled={summary['dynamic_filter_enabled']}"
    )


if __name__ == "__main__":
    main()
