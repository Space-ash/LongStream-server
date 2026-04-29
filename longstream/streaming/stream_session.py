import torch


class StreamSession:
    def __init__(
        self,
        model,
        mode: str,
        window_size: int = 5,
        keep_first_frame_anchor: bool = True,
        loop_closure_cfg: dict = None,
        local_ba_cfg: dict = None,
        gps_xyz=None,
        async_loop_closure: bool = True,
    ):
        self.model = model
        self.core_model = getattr(model, "longstream", model)
        self.mode = mode
        self.window_size = window_size
        self.keep_first_frame_anchor = keep_first_frame_anchor

        if self.mode not in ["causal", "window"]:
            raise ValueError(f"Unsupported attention mode: {self.mode}")

        self.aggregator_kv_cache_depth = self.core_model.aggregator.depth
        self.use_camera_head = self.core_model.camera_head is not None
        if self.use_camera_head:
            self.camera_head_kv_cache_depth = self.core_model.camera_head.trunk_depth
            self.camera_head_iterations = 4
        else:
            self.camera_head_kv_cache_depth = 0
            self.camera_head_iterations = 0

        self.use_rel_pose_head = (
            hasattr(self.core_model, "rel_pose_head")
            and self.core_model.rel_pose_head is not None
        )
        if self.use_rel_pose_head:
            self.rel_pose_head_trunk_depth = self.core_model.rel_pose_head.trunk_depth
            self.rel_pose_head_iterations = 4

        # --- Loop closure / BA integration ---
        lc_cfg = loop_closure_cfg or {}
        ba_cfg = local_ba_cfg or {}
        self.loop_enabled = bool(lc_cfg.get("enabled", False))
        self.local_ba_enabled = bool(ba_cfg.get("enabled", False))
        self._return_feature_cache = self.loop_enabled or self.local_ba_enabled

        self.loop_manager = None
        if self._return_feature_cache:
            from longstream.streaming.loop_closure import LoopClosureManager

            device_str = str(next(self.core_model.parameters()).device)
            self.loop_manager = LoopClosureManager(
                rel_pose_head=(
                    self.core_model.rel_pose_head if self.use_rel_pose_head else None
                ),
                lc_cfg=lc_cfg,
                ba_cfg=ba_cfg,
                device=device_str,
                patch_size=int(getattr(self.core_model, "patch_size", 14)),
                embed_dim=int(getattr(self.core_model, "embed_dim", 1024)),
            )
            if gps_xyz is not None:
                import numpy as np
                self.loop_manager.set_gps_xyz(
                    gps_xyz if isinstance(gps_xyz, type(None))
                    else (gps_xyz if hasattr(gps_xyz, "shape") else None)
                )

        # Global frame counter for loop manager (incremented per record=True call)
        self._global_frame_id = 0

        self.clear()

    def _clear_predictions(self):
        self.sequence_predictions = {}
        self.scalar_predictions = {}

    def _update_predictions(self, predictions):
        sequence_keys = [
            "pose_enc",
            "rel_pose_enc",
            "world_points",
            "world_points_conf",
            "depth",
            "depth_conf",
        ]
        scalar_keys = ["predicted_scale_factor", "global_scale"]

        for k in sequence_keys:
            if k in predictions:
                self.sequence_predictions.setdefault(k, []).append(
                    predictions[k].detach().cpu()
                )

        for k in scalar_keys:
            if k in predictions:
                value = predictions[k]
                self.scalar_predictions[k] = (
                    value.detach().cpu() if isinstance(value, torch.Tensor) else value
                )

    def _clear_cache(self):
        self.aggregator_kv_cache_list = [
            [None, None] for _ in range(self.aggregator_kv_cache_depth)
        ]
        if self.use_camera_head:
            self.camera_head_kv_cache_list = [
                [[None, None] for _ in range(self.camera_head_kv_cache_depth)]
                for _ in range(self.camera_head_iterations)
            ]
        else:
            self.camera_head_kv_cache_list = None
        if self.use_rel_pose_head:
            self.rel_pose_kv_cache_list = [
                [[None, None] for _ in range(self.rel_pose_head_trunk_depth)]
                for _ in range(self.rel_pose_head_iterations)
            ]
        else:
            self.rel_pose_kv_cache_list = None

    def _update_cache(
        self, aggregator_kv_cache_list, camera_head_kv_cache_list, frame_hw
    ):
        if self.mode == "causal":
            self.aggregator_kv_cache_list = aggregator_kv_cache_list
            if self.use_camera_head:
                self.camera_head_kv_cache_list = camera_head_kv_cache_list
            return

        if self.mode == "window":
            h, w = frame_hw
            P = (
                h
                * w
                // self.core_model.aggregator.patch_size
                // self.core_model.aggregator.patch_size
                + self.core_model.aggregator.patch_start_idx
            )

            for k in range(2):
                for i in range(self.aggregator_kv_cache_depth):
                    cache_size = aggregator_kv_cache_list[i][k].size(2)
                    if self.keep_first_frame_anchor:
                        if cache_size <= P:
                            self.aggregator_kv_cache_list[i][
                                k
                            ] = aggregator_kv_cache_list[i][k].contiguous()
                        elif cache_size <= self.window_size * P:
                            self.aggregator_kv_cache_list[i][
                                k
                            ] = aggregator_kv_cache_list[i][k].contiguous()
                        else:
                            anchor = aggregator_kv_cache_list[i][k][:, :, :P]
                            recent_start = cache_size - (self.window_size - 1) * P
                            recent = aggregator_kv_cache_list[i][k][:, :, recent_start:]
                            self.aggregator_kv_cache_list[i][k] = torch.cat(
                                [anchor, recent], dim=2
                            ).contiguous()
                    else:
                        start_idx = max(0, cache_size - self.window_size * P)
                        self.aggregator_kv_cache_list[i][k] = aggregator_kv_cache_list[
                            i
                        ][k][:, :, start_idx:].contiguous()

            if camera_head_kv_cache_list is not None:
                for k in range(2):
                    for i in range(self.camera_head_iterations):
                        for j in range(self.camera_head_kv_cache_depth):
                            cache_size = camera_head_kv_cache_list[i][j][k].size(2)
                            if self.keep_first_frame_anchor:
                                if cache_size <= 1:
                                    self.camera_head_kv_cache_list[i][j][
                                        k
                                    ] = camera_head_kv_cache_list[i][j][k].contiguous()
                                elif cache_size <= self.window_size:
                                    self.camera_head_kv_cache_list[i][j][
                                        k
                                    ] = camera_head_kv_cache_list[i][j][k].contiguous()
                                else:
                                    anchor = camera_head_kv_cache_list[i][j][k][
                                        :, :, :1
                                    ]
                                    recent_start = cache_size - (self.window_size - 1)
                                    recent = camera_head_kv_cache_list[i][j][k][
                                        :, :, recent_start:
                                    ]
                                    self.camera_head_kv_cache_list[i][j][k] = torch.cat(
                                        [anchor, recent], dim=2
                                    ).contiguous()
                            else:
                                start_idx = max(0, cache_size - self.window_size)
                                self.camera_head_kv_cache_list[i][j][
                                    k
                                ] = camera_head_kv_cache_list[i][j][k][
                                    :, :, start_idx:
                                ].contiguous()
            return

        raise ValueError(f"Unsupported attention mode: {self.mode}")

    def _get_cache(self):
        return self.aggregator_kv_cache_list, self.camera_head_kv_cache_list

    def get_all_predictions(self):
        predictions = {}
        for key, chunks in self.sequence_predictions.items():
            if not chunks:
                continue
            predictions[key] = (
                chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=1)
            )
        predictions.update(self.scalar_predictions)
        return predictions

    def get_last_prediction(self):
        last_predictions = {}
        keys_to_extract = [
            "pose_enc",
            "rel_pose_enc",
            "world_points",
            "world_points_conf",
            "depth",
            "depth_conf",
            "predicted_scale_factor",
        ]
        for k in keys_to_extract:
            if k in self.sequence_predictions and self.sequence_predictions[k]:
                last_predictions[k] = self.sequence_predictions[k][-1][:, -1:]
            elif k in self.scalar_predictions:
                last_predictions[k] = self.scalar_predictions[k]
        return last_predictions

    def clear(self):
        self._clear_predictions()
        self._clear_cache()
        self._global_frame_id = 0
        if self.loop_manager is not None:
            self.loop_manager.clear()
        if self.use_rel_pose_head:
            if hasattr(self.core_model.rel_pose_head, "_keyframe_tokens_cache"):
                self.core_model.rel_pose_head._keyframe_tokens_cache = {}
            if hasattr(self.core_model.rel_pose_head, "_current_frame_id"):
                self.core_model.rel_pose_head._current_frame_id = 0
            if hasattr(self.core_model.rel_pose_head, "_frame_info"):
                self.core_model.rel_pose_head._frame_info = []

    def clear_cache_only(self):
        self._clear_cache()
        if self.use_rel_pose_head:
            if hasattr(self.core_model.rel_pose_head, "_keyframe_tokens_cache"):
                self.core_model.rel_pose_head._keyframe_tokens_cache = {}
            if hasattr(self.core_model.rel_pose_head, "_current_frame_id"):
                self.core_model.rel_pose_head._current_frame_id = 0
            if hasattr(self.core_model.rel_pose_head, "_frame_info"):
                self.core_model.rel_pose_head._frame_info = []

    def get_loop_closure_outputs(self) -> dict:
        """Return loop closure / PGO / BA results (non-blocking, may be partial)."""
        if self.loop_manager is None:
            return {}
        return {
            "optimized_w2c": self.loop_manager.get_optimized_w2c(),
            "loop_edges": self.loop_manager.get_loop_edges(),
            "local_ba_windows": self.loop_manager.get_local_ba_windows(),
        }

    def shutdown(self) -> None:
        """Gracefully shut down async workers (call after sequence ends)."""
        if self.loop_manager is not None:
            self.loop_manager.shutdown()

    def forward_stream(
        self, images, is_keyframe=None, keyframe_indices=None, record: bool = True,
        global_kf_idx: int = 0,
    ):
        aggregator_kv_cache_list, camera_head_kv_cache_list = self._get_cache()

        rel_pose_inputs = None
        if (
            self.use_rel_pose_head
            and is_keyframe is not None
            and keyframe_indices is not None
        ):
            rel_pose_inputs = {
                "is_keyframe": is_keyframe,
                "keyframe_indices": keyframe_indices,
                "kv_cache_list": self.rel_pose_kv_cache_list,
            }

        outputs = self.model(
            images=images,
            mode=self.mode,
            aggregator_kv_cache_list=aggregator_kv_cache_list,
            camera_head_kv_cache_list=camera_head_kv_cache_list,
            rel_pose_inputs=rel_pose_inputs,
            is_keyframe=is_keyframe,
            return_feature_cache=self._return_feature_cache,
        )

        if record:
            self._update_predictions(outputs)

        camera_head_kv_cache_list = outputs.get("camera_head_kv_cache_list", None)
        depth_hw = (
            outputs["depth"].shape[2:4] if "depth" in outputs else images.shape[-2:]
        )
        self._update_cache(
            outputs["aggregator_kv_cache_list"], camera_head_kv_cache_list, depth_hw
        )

        if self.use_rel_pose_head and "rel_pose_kv_cache_list" in outputs:
            rel_pose_kv_cache = outputs["rel_pose_kv_cache_list"]
            if self.mode == "causal":
                self.rel_pose_kv_cache_list = rel_pose_kv_cache
            elif self.mode == "window":
                for k in range(2):
                    for i in range(self.rel_pose_head_iterations):
                        for j in range(self.rel_pose_head_trunk_depth):
                            if rel_pose_kv_cache[i][j][k] is None:
                                continue
                            cache_len = rel_pose_kv_cache[i][j][k].size(2)
                            if self.keep_first_frame_anchor:
                                if cache_len <= 1:
                                    self.rel_pose_kv_cache_list[i][j][
                                        k
                                    ] = rel_pose_kv_cache[i][j][k].contiguous()
                                elif cache_len <= self.window_size:
                                    self.rel_pose_kv_cache_list[i][j][
                                        k
                                    ] = rel_pose_kv_cache[i][j][k].contiguous()
                                else:
                                    anchor = rel_pose_kv_cache[i][j][k][:, :, :1]
                                    recent_start = cache_len - (self.window_size - 1)
                                    recent = rel_pose_kv_cache[i][j][k][
                                        :, :, recent_start:
                                    ]
                                    self.rel_pose_kv_cache_list[i][j][k] = torch.cat(
                                        [anchor, recent], dim=2
                                    ).contiguous()
                            else:
                                start_idx = max(0, cache_len - self.window_size)
                                self.rel_pose_kv_cache_list[i][j][
                                    k
                                ] = rel_pose_kv_cache[i][j][k][
                                    :, :, start_idx:
                                ].contiguous()

        # ---------------------------------------------------------------
        # Loop closure: feed per-frame features to manager (non-blocking)
        # ---------------------------------------------------------------
        if record and self._return_feature_cache and self.loop_manager is not None:
            ft = outputs.get("feature_tokens")
            psi = outputs.get("patch_start_idx_cache")
            if ft is not None and psi is not None:
                # Extract scalar pose_enc for current frame [D]
                pe_key = "rel_pose_enc" if "rel_pose_enc" in outputs else "pose_enc"
                pe_raw = outputs.get(pe_key)
                if pe_raw is not None:
                    pose_enc_s = pe_raw[0, -1].detach().cpu().float()  # [D]
                else:
                    pose_enc_s = torch.zeros(9)

                # Depth for current frame [H, W]
                depth_out = outputs.get("depth")
                if depth_out is not None:
                    depth_s = depth_out[0, -1, :, :, 0].detach().cpu().float()
                else:
                    depth_s = torch.zeros(depth_hw[0], depth_hw[1])

                # Decode intrinsics from FoV fields of current frame's pose encoding.
                # pose_enc "absT_quaR_FoV" layout: [T(3), quat(4), fov_h, fov_w]
                import numpy as _np
                import logging as _logging
                _lc_logger = _logging.getLogger(__name__)
                intri_np = None
                if pose_enc_s.shape[0] >= 9:
                    try:
                        H_d, W_d = int(depth_hw[0]), int(depth_hw[1])
                        fov_h = float(pose_enc_s[7].item())
                        fov_w = float(pose_enc_s[8].item())
                        if (
                            _np.isfinite(fov_h) and abs(fov_h) > 1e-6
                            and _np.isfinite(fov_w) and abs(fov_w) > 1e-6
                        ):
                            import math as _math
                            fy = (H_d / 2.0) / _math.tan(fov_h / 2.0)
                            fx = (W_d / 2.0) / _math.tan(fov_w / 2.0)
                            intri_np = _np.array(
                                [[fx, 0.0, W_d / 2.0],
                                 [0.0, fy, H_d / 2.0],
                                 [0.0, 0.0, 1.0]],
                                dtype=_np.float32,
                            )
                        else:
                            _lc_logger.warning(
                                "[StreamSession] frame %d: invalid FoV in pose_enc "
                                "(fov_h=%.4f fov_w=%.4f); BA/fusion skipped.",
                                self._global_frame_id, fov_h, fov_w,
                            )
                    except Exception as _exc:
                        _lc_logger.warning(
                            "[StreamSession] frame %d: intri decode failed (%s); "
                            "BA/fusion skipped.", self._global_frame_id, _exc,
                        )
                else:
                    _lc_logger.warning(
                        "[StreamSession] frame %d: pose_enc too short (len=%d), "
                        "no FoV; BA/fusion skipped.",
                        self._global_frame_id, pose_enc_s.shape[0],
                    )

                # Feature tokens: take last frame slice [1, 1, P_total, 2*embed_dim]
                ft_s = ft[:, -1:, :, :].cpu()  # [1, 1, P_total, 2*embed_dim]

                self.loop_manager.on_frame(
                    frame_id=self._global_frame_id,
                    feature_tokens=ft_s,
                    patch_start_idx=int(psi),
                    depth=depth_s,
                    pose_enc=pose_enc_s,
                    global_kf_idx=global_kf_idx,
                    image_hw=tuple(depth_hw),
                    intri=intri_np,
                )
                # Release large tensors immediately — must not persist in outputs
                del ft_s, ft, psi
                outputs.pop("feature_tokens", None)
                outputs.pop("patch_start_idx_cache", None)

            self._global_frame_id += 1

        return outputs
