import torch
from typing import Dict, Any, List

from longstream.streaming.stream_session import StreamSession

_SEQUENCE_OUTPUT_KEYS = {
    "pose_enc",
    "rel_pose_enc",
    "world_points",
    "world_points_conf",
    "depth",
    "depth_conf",
}
_SCALAR_OUTPUT_KEYS = {
    "predicted_scale_factor",
    "global_scale",
}


def _refresh_intervals(refresh: int) -> int:
    refresh = int(refresh)
    if refresh < 2:
        raise ValueError("refresh must be >= 2")
    return refresh - 1


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def _move_scalar_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _append_batch_output(
    stitched_tensors: Dict[str, List[torch.Tensor]],
    stitched_scalars: Dict[str, Any],
    output: Dict[str, Any],
    actual_frames: int,
    slice_start: int,
) -> None:
    for key in _SEQUENCE_OUTPUT_KEYS:
        value = output.get(key)
        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim < 2 or value.shape[1] != actual_frames:
            continue
        stitched_tensors.setdefault(key, []).append(
            value[:, slice_start:].detach().cpu()
        )

    for key in _SCALAR_OUTPUT_KEYS:
        if key in output:
            stitched_scalars[key] = _move_scalar_to_cpu(output[key])


def _finalize_stitched_batches(
    stitched_tensors: Dict[str, List[torch.Tensor]],
    stitched_scalars: Dict[str, Any],
) -> Dict[str, Any]:
    stitched_output: Dict[str, Any] = {}
    for key, chunks in stitched_tensors.items():
        if not chunks:
            continue
        stitched_output[key] = (
            chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=1)
        )
    stitched_output.update(stitched_scalars)
    return stitched_output


def run_batch_refresh(
    model,
    images,
    is_keyframe,
    keyframe_indices,
    mode: str,
    keyframe_stride: int,
    refresh: int,
    rel_pose_cfg,
    loop_closure_cfg: dict = None,
    local_ba_cfg: dict = None,
    gps_xyz=None,
):
    B, S = images.shape[:2]
    device = _model_device(model)

    lc_active = (loop_closure_cfg or {}).get("enabled", False)
    ba_active = (local_ba_cfg or {}).get("enabled", False)
    if lc_active or ba_active:
        import warnings
        warnings.warn(
            "[refresh] loop_closure / local_ba are only supported in "
            "streaming_refresh mode. Ignoring for batch_refresh.",
            stacklevel=2,
        )

    refresh_intervals = _refresh_intervals(refresh)
    frames_per_batch = refresh_intervals * keyframe_stride + 1
    step_frames = refresh_intervals * keyframe_stride

    stitched_tensors: Dict[str, List[torch.Tensor]] = {}
    stitched_scalars: Dict[str, Any] = {}
    num_batches = (S + step_frames - 1) // step_frames
    for batch_idx in range(num_batches):
        start_frame = batch_idx * step_frames
        end_frame = min(start_frame + frames_per_batch, S)
        batch_images = images[:, start_frame:end_frame].to(device, non_blocking=True)
        batch_is_keyframe = (
            is_keyframe[:, start_frame:end_frame].clone()
            if is_keyframe is not None
            else None
        )
        batch_keyframe_indices = (
            keyframe_indices[:, start_frame:end_frame].clone()
            if keyframe_indices is not None
            else None
        )

        if batch_idx > 0 and batch_is_keyframe is not None:
            batch_is_keyframe[:, 0] = True
            if batch_keyframe_indices is not None:
                batch_keyframe_indices[:, 0] = start_frame

        if batch_keyframe_indices is not None:
            batch_keyframe_indices = batch_keyframe_indices - start_frame
            batch_keyframe_indices = torch.clamp(
                batch_keyframe_indices, 0, end_frame - start_frame - 1
            )

        batch_rel_pose_inputs = None
        if rel_pose_cfg is not None and batch_is_keyframe is not None:
            batch_is_keyframe = batch_is_keyframe.to(device, non_blocking=True)
            if batch_keyframe_indices is not None:
                batch_keyframe_indices = batch_keyframe_indices.to(
                    device, non_blocking=True
                )
            batch_rel_pose_inputs = {
                "is_keyframe": batch_is_keyframe,
                "keyframe_indices": batch_keyframe_indices,
                "num_iterations": rel_pose_cfg.get("num_iterations", 4),
            }
        elif batch_is_keyframe is not None:
            batch_is_keyframe = batch_is_keyframe.to(device, non_blocking=True)

        batch_output = model(
            images=batch_images,
            mode=mode,
            rel_pose_inputs=batch_rel_pose_inputs,
            is_keyframe=batch_is_keyframe,
        )

        _append_batch_output(
            stitched_tensors,
            stitched_scalars,
            batch_output,
            actual_frames=end_frame - start_frame,
            slice_start=0 if batch_idx == 0 else 1,
        )
        del batch_output
        del batch_images
        del batch_is_keyframe
        del batch_keyframe_indices

    return _finalize_stitched_batches(stitched_tensors, stitched_scalars)


def run_streaming_refresh(
    model,
    images,
    is_keyframe,
    keyframe_indices,
    mode: str,
    window_size: int,
    refresh: int,
    rel_pose_cfg,
    loop_closure_cfg: dict = None,
    local_ba_cfg: dict = None,
    gps_xyz=None,
):
    B, S = images.shape[:2]
    device = _model_device(model)
    refresh_intervals = _refresh_intervals(refresh)
    session = StreamSession(
        model,
        mode=mode,
        window_size=window_size,
        loop_closure_cfg=loop_closure_cfg,
        local_ba_cfg=local_ba_cfg,
        gps_xyz=gps_xyz,
    )
    # Pass GPS to loop manager (set again now that session is constructed)
    if gps_xyz is not None and session.loop_manager is not None:
        import numpy as np
        gps_arr = np.asarray(gps_xyz, dtype=np.float32) if not hasattr(gps_xyz, "numpy") else gps_xyz.numpy().astype(np.float32)
        session.loop_manager.set_gps_xyz(gps_arr)

    keyframe_count = 0
    segment_start = 0
    for s in range(S):
        frame_images = images[:, s : s + 1].to(device, non_blocking=True)
        is_keyframe_s = (
            is_keyframe[:, s : s + 1].to(device, non_blocking=True)
            if is_keyframe is not None
            else None
        )
        if keyframe_indices is not None:
            keyframe_indices_s = keyframe_indices[:, s : s + 1].clone() - segment_start
            keyframe_indices_s = torch.clamp(keyframe_indices_s, min=0)
            keyframe_indices_s = keyframe_indices_s.to(device, non_blocking=True)
        else:
            keyframe_indices_s = None

        # Global keyframe index for this frame (before segment adjustment)
        global_kf_idx_s = int(keyframe_indices[0, s].item()) if keyframe_indices is not None else s

        session.forward_stream(
            frame_images,
            is_keyframe=is_keyframe_s,
            keyframe_indices=keyframe_indices_s,
            record=True,
            global_kf_idx=global_kf_idx_s,
        )
        if is_keyframe_s is None or not bool(is_keyframe_s.item()) or s <= 0:
            del frame_images
            if is_keyframe_s is not None:
                del is_keyframe_s
            if keyframe_indices_s is not None:
                del keyframe_indices_s
            continue
        keyframe_count += 1
        if keyframe_count % refresh_intervals == 0:
            session.clear_cache_only()
            segment_start = s
            if keyframe_indices_s is not None:
                keyframe_indices_self = torch.zeros_like(keyframe_indices_s)
            else:
                keyframe_indices_self = None
            session.forward_stream(
                frame_images,
                is_keyframe=is_keyframe_s,
                keyframe_indices=keyframe_indices_self,
                record=False,
                global_kf_idx=global_kf_idx_s,
            )
        del frame_images
        if is_keyframe_s is not None:
            del is_keyframe_s
        if keyframe_indices_s is not None:
            del keyframe_indices_s

    # Merge sequence predictions
    outputs = session.get_all_predictions()

    # Shut down async workers first so all pending BA/PGO jobs finish before
    # we collect their results.  get_loop_closure_outputs() is non-blocking
    # and reads the shared result buffers, so it must come AFTER shutdown().
    session.shutdown()

    # Merge loop closure / PGO / BA results
    lc_outputs = session.get_loop_closure_outputs()
    if lc_outputs.get("optimized_w2c") is not None:
        outputs["optimized_w2c"] = lc_outputs["optimized_w2c"]
    if lc_outputs.get("loop_edges"):
        outputs["loop_edges"] = lc_outputs["loop_edges"]
    if lc_outputs.get("local_ba_windows"):
        outputs["local_ba_windows"] = lc_outputs["local_ba_windows"]

    return outputs
