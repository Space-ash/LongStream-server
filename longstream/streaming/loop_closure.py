"""
longstream/streaming/loop_closure.py

Loop Closure Detection, Pose Graph Optimization (PGO),
Feature-Metric Local Bundle Adjustment (BA), and Open3D Point Cloud Fusion.

Pipeline (all async relative to forward_stream):
  LoopClosureManager.on_frame()
    -> FeatureFrameCache  (ring buffer, CPU-only)
    -> FaissLoopIndex     (descriptor similarity search)
    -> [async] FeatureMetricLocalBA + Open3DFusion
    -> [async] PoseGraphOptimizer (g2o-python or scipy fallback)

Key constraints (see blueprint §6):
  - All cached tensors must be .detach().cpu()  (float16 for patch features)
  - BA and PGO run in background ThreadPoolExecutor
  - forward_stream() never blocks on BA / PGO completion
  - Gradients must NEVER flow back to LongStream backbone parameters
"""

from __future__ import annotations

import dataclasses
import logging
import random
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports (lazy, so normal inference is unaffected)
# ---------------------------------------------------------------------------

def _try_import_faiss():
    try:
        import faiss  # noqa: F401
        return faiss
    except ImportError:
        logger.warning(
            "[loop_closure] faiss-cpu not available. "
            "Loop closure descriptor search will be disabled. "
            "Install with: pip install faiss-cpu"
        )
        return None


def _try_import_g2o():
    try:
        import g2o  # noqa: F401
        return g2o
    except ImportError:
        logger.info(
            "[loop_closure] g2o-python not available. "
            "PGO will fall back to scipy.optimize."
        )
        return None


def _try_import_open3d():
    try:
        import open3d as o3d  # noqa: F401
        return o3d
    except ImportError:
        logger.warning(
            "[loop_closure] open3d not available. "
            "Point cloud fusion will be skipped. "
            "Install with: pip install open3d"
        )
        return None


# ---------------------------------------------------------------------------
# SE(3) utilities — delegate to the stable vendor implementation
# ---------------------------------------------------------------------------

from longstream.utils.vendor.models.components.utils.se3 import se3_exp as _vendor_se3_exp  # noqa: E402


def _apply_pose_delta(w2c_init: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Apply SE(3) perturbation delta to w2c (left multiplication).

    delta convention (matches vendor se3_exp):
      delta[..., :3] = translation (v)
      delta[..., 3:] = rotation angle-axis (omega)

    Uses the vendor se3_exp with Taylor-series near-zero handling so that
    gradients remain well-conditioned when delta is initialised to zero.
    """
    return _vendor_se3_exp(delta) @ w2c_init


def _np_to_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32, copy=False)).to(device)


# ---------------------------------------------------------------------------
# Descriptor utilities
# ---------------------------------------------------------------------------

_PROJ_MAT_CACHE: Dict[Tuple[int, int, int], np.ndarray] = {}


def _get_random_projection(in_dim: int, out_dim: int, seed: int = 42) -> np.ndarray:
    """Johnson-Lindenstrauss random orthogonal projection (cached by key)."""
    key = (in_dim, out_dim, seed)
    if key not in _PROJ_MAT_CACHE:
        rng = np.random.default_rng(seed)
        M = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
        # Orthogonalise via QR for better conditioning
        Q, _ = np.linalg.qr(M)
        _PROJ_MAT_CACHE[key] = Q[:, :out_dim].astype(np.float32)
    return _PROJ_MAT_CACHE[key]


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def _compute_descriptor(
    patch_tokens: torch.Tensor,  # [N_patch, C] CPU
    proj_mat: np.ndarray,        # [C, faiss_dim]
) -> np.ndarray:
    """Mean-pool patch tokens, project to faiss_dim, L2-normalise."""
    mean_feat = patch_tokens.float().mean(dim=0).numpy()  # [C]
    desc = mean_feat @ proj_mat                            # [faiss_dim]
    desc = desc / max(np.linalg.norm(desc), 1e-12)
    return desc.astype(np.float32)


def _reduce_patch_features(
    patch_tokens: torch.Tensor,  # [N_patch, C] CPU
    proj_mat: np.ndarray,        # [C, feature_dim]
) -> torch.Tensor:
    """Project patch tokens to feature_dim; return float16 CPU tensor."""
    p = patch_tokens.float().numpy() @ proj_mat           # [N_patch, feature_dim]
    return torch.from_numpy(p).half()


# ---------------------------------------------------------------------------
# Frame buffer entry
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FrameEntry:
    frame_id: int
    descriptor: np.ndarray           # [faiss_dim] float32, L2-normalised
    pose_token: torch.Tensor         # [embed_dim*2] cpu float16
    patch_tokens: torch.Tensor       # [N_patch, feature_dim] cpu float16
    depth: torch.Tensor              # [H, W] cpu float32
    pose_enc: torch.Tensor           # [D] cpu float32, rel_pose_enc or pose_enc
    global_kf_idx: int               # global keyframe index for this frame
    intri: Optional[np.ndarray]      # [3,3] float32; None when intrinsics unavailable
    image_hw: Tuple[int, int]        # (H, W) of the feature map (model output size)
    p_total: int                     # total tokens per frame (for RelPoseHead fake call)
    w2c_init: Optional[np.ndarray]   # [4,4] float32, decoded on demand
    rgb: Optional[np.ndarray]        # [H, W, 3] uint8, optional for point cloud colour


# ---------------------------------------------------------------------------
# Loop edge record
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class LoopEdge:
    """Detected loop between hist_id and curr_id frames."""
    hist_id: int
    curr_id: int
    score: float
    T_ij_meas: Optional[np.ndarray]  # [4,4] float32; None → fall back to trajectory in PGO


# ---------------------------------------------------------------------------
# Feature Frame Cache (ring buffer)
# ---------------------------------------------------------------------------

class FeatureFrameCache:
    """Ring buffer backed by a dict for O(1) lookup by frame_id."""

    def __init__(self, max_size: int = 3000):
        self.max_size = max_size
        self._data: Dict[int, FrameEntry] = {}
        self._insertion_order: List[int] = []

    def add(self, entry: FrameEntry) -> None:
        if entry.frame_id in self._data:
            self._data[entry.frame_id] = entry
            return
        if len(self._data) >= self.max_size:
            oldest = self._insertion_order.pop(0)
            self._data.pop(oldest, None)
        self._data[entry.frame_id] = entry
        self._insertion_order.append(entry.frame_id)

    def get(self, frame_id: int) -> Optional[FrameEntry]:
        return self._data.get(frame_id)

    def get_window(self, center_id: int, radius: int) -> List[FrameEntry]:
        result = []
        for fid in range(center_id - radius, center_id + radius + 1):
            e = self._data.get(fid)
            if e is not None:
                result.append(e)
        return result

    def latest_ids(self, n: int) -> List[int]:
        return self._insertion_order[-n:]

    def __len__(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# FAISS Loop Index
# ---------------------------------------------------------------------------

class FaissLoopIndex:
    """
    Incrementally-built FAISS index for per-frame descriptor similarity search.
    Falls back to brute-force numpy cosine search if faiss is unavailable.
    """

    def __init__(self, dim: int, index_type: str = "flat"):
        self._dim = dim
        self._faiss = _try_import_faiss()
        self._frame_ids: List[int] = []
        self._descriptors: List[np.ndarray] = []  # kept for numpy fallback

        if self._faiss is not None:
            if index_type == "hnsw":
                self._index = self._faiss.IndexHNSWFlat(dim, 32)
            else:
                self._index = self._faiss.IndexFlatIP(dim)
        else:
            self._index = None

    def add(self, descriptor: np.ndarray, frame_id: int) -> None:
        self._frame_ids.append(frame_id)
        self._descriptors.append(descriptor.copy())
        if self._index is not None:
            self._index.add(descriptor.reshape(1, -1).astype(np.float32))

    def search(
        self,
        descriptor: np.ndarray,
        topk: int,
        min_frame_gap: int,
        current_frame_id: int,
    ) -> List[Tuple[int, float]]:
        """Return [(frame_id, score)] sorted by score descending, skipping recent frames."""
        if len(self._frame_ids) == 0:
            return []

        eligible_mask = np.array(
            [fid for fid in self._frame_ids if (current_frame_id - fid) >= min_frame_gap]
        )
        if len(eligible_mask) == 0:
            return []

        if self._index is not None:
            k = min(topk + min_frame_gap, self._index.ntotal)
            scores, indices = self._index.search(
                descriptor.reshape(1, -1).astype(np.float32), k
            )
            candidates = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._frame_ids):
                    continue
                fid = self._frame_ids[idx]
                if (current_frame_id - fid) >= min_frame_gap:
                    candidates.append((fid, float(score)))
            candidates.sort(key=lambda x: -x[1])
            return candidates[:topk]
        else:
            # Numpy brute-force fallback
            descs = np.stack([
                self._descriptors[i]
                for i, fid in enumerate(self._frame_ids)
                if (current_frame_id - fid) >= min_frame_gap
            ])  # [N_eligible, D]
            sims = descs @ descriptor  # [N_eligible]
            top_k = min(topk, len(sims))
            top_idx = np.argpartition(sims, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(-sims[top_idx])]

            eligible_fids = [
                fid for fid in self._frame_ids
                if (current_frame_id - fid) >= min_frame_gap
            ]
            return [(eligible_fids[i], float(sims[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Pose Graph Optimizer
# ---------------------------------------------------------------------------

class PoseGraphOptimizer:
    """
    PGO with g2o-python as primary backend and scipy fallback.

    Edge format: (i, j, T_ij_measured) where T_ij = T_w2c[j] @ inv(T_w2c[i]).
    """

    def optimize(
        self,
        w2c_init: np.ndarray,                               # [S, 4, 4]
        all_edges: List[Tuple[int, int, np.ndarray]],       # (i, j, T_ij [4,4])
        gps_xyz: Optional[np.ndarray] = None,               # [S, 3] camera centres
        gps_weight: float = 0.1,
    ) -> np.ndarray:
        """Returns optimised w2c [S, 4, 4] float32."""
        g2o = _try_import_g2o()
        if g2o is not None:
            try:
                return self._optimize_g2o(g2o, w2c_init, all_edges, gps_xyz, gps_weight)
            except Exception as exc:
                logger.warning(
                    f"[PGO] g2o failed ({exc}), falling back to scipy."
                )
        return self._optimize_scipy(w2c_init, all_edges, gps_xyz, gps_weight)

    # ------------------------------------------------------------------
    # g2o backend
    # ------------------------------------------------------------------

    def _optimize_g2o(
        self,
        g2o,
        w2c_init: np.ndarray,
        all_edges: List[Tuple[int, int, np.ndarray]],
        gps_xyz: Optional[np.ndarray] = None,
        gps_weight: float = 0.1,
    ) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer = g2o.SparseOptimizer()
        optimizer.set_algorithm(algorithm)
        optimizer.set_verbose(False)

        S = len(w2c_init)
        for i in range(S):
            v = g2o.VertexSE3Expmap()
            v.set_id(i)
            R = w2c_init[i, :3, :3]
            t = w2c_init[i, :3, 3]
            q = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
            # g2o SE3Quat constructor: (R, t) or quaternion representation
            try:
                v.set_estimate(g2o.SE3Quat(R, t))
            except Exception:
                # Fallback: some distributions use different constructor
                v.set_estimate(g2o.SE3Quat(
                    np.array([q[3], q[0], q[1], q[2]]), t
                ))
            v.set_fixed(i == 0)
            optimizer.add_vertex(v)

        info = np.eye(6) * 500.0
        for (i, j, T_ij) in all_edges:
            R_ij = T_ij[:3, :3]
            t_ij = T_ij[:3, 3]
            e = g2o.EdgeSE3Expmap()
            e.set_vertex(0, optimizer.vertex(i))
            e.set_vertex(1, optimizer.vertex(j))
            try:
                e.set_measurement(g2o.SE3Quat(R_ij, t_ij))
            except Exception:
                q_ij = Rotation.from_matrix(R_ij).as_quat()
                e.set_measurement(g2o.SE3Quat(
                    np.array([q_ij[3], q_ij[0], q_ij[1], q_ij[2]]), t_ij
                ))
            e.set_information(info)
            optimizer.add_edge(e)

        # ------------------------------------------------------------------
        # GPS unary priors (translation-only anchor vertices + binary edges)
        # SE3Expmap tangent-space layout: [ω_x, ω_y, ω_z, t_x, t_y, t_z]
        # → translation occupies indices 3–6, rotation indices 0–3.
        # We create a fixed "GPS anchor" vertex for each constrained frame
        # with the GPS-derived w2c translation, then add a binary edge that
        # carries near-zero rotation weight and user-specified translation weight.
        # ------------------------------------------------------------------
        if gps_xyz is not None:
            gps_trans_weight = gps_weight * 500.0  # scale to match binary edge info
            info_gps = np.zeros((6, 6), dtype=np.float64)
            info_gps[3, 3] = info_gps[4, 4] = info_gps[5, 5] = gps_trans_weight
            info_gps[0, 0] = info_gps[1, 1] = info_gps[2, 2] = 1e-6  # soft rotation

            id_identity_meas = g2o.SE3Quat()

            for i in range(min(S, len(gps_xyz))):
                c_world = gps_xyz[i]
                if not np.isfinite(c_world).all():
                    continue
                # Camera centre in world → w2c translation: t = -R @ c_world
                R_i = w2c_init[i, :3, :3].astype(np.float64)
                t_gps = -(R_i @ c_world.astype(np.float64))

                v_gps = g2o.VertexSE3Expmap()
                v_gps.set_id(S + i)
                q_i = Rotation.from_matrix(R_i).as_quat()
                try:
                    v_gps.set_estimate(g2o.SE3Quat(R_i, t_gps))
                except Exception:
                    v_gps.set_estimate(g2o.SE3Quat(
                        np.array([q_i[3], q_i[0], q_i[1], q_i[2]]), t_gps
                    ))
                v_gps.set_fixed(True)
                optimizer.add_vertex(v_gps)

                e_gps = g2o.EdgeSE3Expmap()
                e_gps.set_vertex(0, optimizer.vertex(S + i))  # fixed GPS anchor
                e_gps.set_vertex(1, optimizer.vertex(i))       # frame to optimize
                try:
                    e_gps.set_measurement(id_identity_meas)
                except Exception:
                    e_gps.set_measurement(g2o.SE3Quat(np.eye(3), np.zeros(3)))
                e_gps.set_information(info_gps)
                optimizer.add_edge(e_gps)

        optimizer.initialize_optimization()
        optimizer.optimize(20)

        result = np.zeros_like(w2c_init)
        for i in range(S):
            T = optimizer.vertex(i).estimate().matrix()
            result[i] = T.astype(np.float32)
        return result

    # ------------------------------------------------------------------
    # scipy fallback backend
    # ------------------------------------------------------------------

    def _optimize_scipy(
        self,
        w2c_init: np.ndarray,
        all_edges: List[Tuple[int, int, np.ndarray]],
        gps_xyz: Optional[np.ndarray],
        gps_weight: float,
    ) -> np.ndarray:
        from scipy.optimize import minimize
        from scipy.spatial.transform import Rotation

        S = len(w2c_init)

        def _w2c_to_params(T: np.ndarray) -> np.ndarray:
            r = Rotation.from_matrix(T[:3, :3]).as_rotvec()
            return np.concatenate([r, T[:3, 3]])

        def _params_to_w2c_list(x: np.ndarray) -> List[np.ndarray]:
            Ts = [w2c_init[0].copy()]
            for i in range(1, S):
                params = x[(i - 1) * 6: i * 6]
                R = Rotation.from_rotvec(params[:3]).as_matrix()
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = params[3:]
                Ts.append(T)
            return Ts

        x0 = np.zeros((S - 1) * 6, dtype=np.float64)
        for i in range(1, S):
            x0[(i - 1) * 6: i * 6] = _w2c_to_params(w2c_init[i])

        def _cost(x: np.ndarray) -> float:
            Ts = _params_to_w2c_list(x)
            total = 0.0
            for (i, j, T_ij_meas) in all_edges:
                T_ij_pred = Ts[j] @ np.linalg.inv(Ts[i])
                rot_res = T_ij_pred[:3, :3].T @ T_ij_meas[:3, :3] - np.eye(3)
                total += np.sum(rot_res ** 2)
                total += np.sum((T_ij_pred[:3, 3] - T_ij_meas[:3, 3]) ** 2)
            if gps_xyz is not None:
                for i, T in enumerate(Ts):
                    R, t = T[:3, :3], T[:3, 3]
                    centre = -R.T @ t
                    if i < len(gps_xyz) and np.isfinite(gps_xyz[i]).all():
                        total += gps_weight * np.sum((centre - gps_xyz[i]) ** 2)
            return float(total)

        result = minimize(_cost, x0, method="L-BFGS-B", options={"maxiter": 100})
        opt_Ts = _params_to_w2c_list(result.x)
        out = np.zeros((S, 4, 4), dtype=np.float32)
        for i, T in enumerate(opt_Ts):
            out[i] = T.astype(np.float32)
        return out


# ---------------------------------------------------------------------------
# Feature-Metric Local Bundle Adjustment
# ---------------------------------------------------------------------------

class FeatureMetricLocalBA:
    """
    Feature-metric local BA over a small window of frames.

    Optimises per-frame pose_delta (SE3, 6-DoF) and depth_delta (low-res additive).
    Backbone parameters are NEVER in the optimizer — gradients flow only through
    the projection grid back to pose_delta and depth_delta.
    """

    def __init__(self, cfg: dict):
        self.steps = int(cfg.get("steps", 30))
        self.lr_pose = float(cfg.get("lr_pose", 1e-3))
        self.lr_depth = float(cfg.get("lr_depth", 5e-4))
        self.max_pairs = int(cfg.get("max_pairs", 64))
        self.lambda_depth = float(cfg.get("lambda_depth", 0.01))
        self.lambda_pose = float(cfg.get("lambda_pose", 1e-3))
        self.patch_size: int = 14  # updated by LoopClosureManager

    def optimize(
        self,
        window_frames: List[FrameEntry],
        device: str,
    ) -> Tuple[List[np.ndarray], List[torch.Tensor]]:
        """
        Returns:
            opt_w2c_list: list of [4,4] np.float32
            opt_depth_list: list of [H,W] cpu float32 tensors
        """
        if len(window_frames) < 2:
            return (
                [f.w2c_init for f in window_frames],
                [f.depth for f in window_frames],
            )

        M = len(window_frames)
        # --- Move features / depths / intrinsics to device (detached) ---
        patch_feats = [
            f.patch_tokens.to(device).float().detach()  # [N_patch, feature_dim]
            for f in window_frames
        ]
        depths_init = [
            f.depth.to(device).float().detach()          # [H, W]
            for f in window_frames
        ]
        intris = [
            _np_to_tensor(f.intri, device)               # [3, 3]
            for f in window_frames
        ]
        w2c_init_list = [
            _np_to_tensor(
                f.w2c_init if f.w2c_init is not None else np.eye(4, dtype=np.float32),
                device,
            )
            for f in window_frames
        ]

        # Optimisation variables
        pose_delta = nn.Parameter(torch.zeros(M, 6, device=device))
        depth_delta = nn.Parameter(torch.zeros(
            M, 1,
            max(depths_init[0].shape[0] // 4, 1),
            max(depths_init[0].shape[1] // 4, 1),
            device=device,
        ))
        optimizer = torch.optim.Adam(
            [
                {"params": [pose_delta], "lr": self.lr_pose},
                {"params": [depth_delta], "lr": self.lr_depth},
            ]
        )

        # Build patch grid info
        H, W = depths_init[0].shape
        N_patch = patch_feats[0].shape[0]
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        # Recalculate to match actual N_patch
        if h_patches * w_patches != N_patch:
            h_patches = int(N_patch ** 0.5)
            w_patches = N_patch // max(h_patches, 1)

        # Build pairs: all combinations within window, up to max_pairs
        pairs = [
            (i, j) for i in range(M) for j in range(M)
            if i != j and abs(i - j) <= 4
        ]
        if len(pairs) > self.max_pairs:
            pairs = random.sample(pairs, self.max_pairs)

        with torch.enable_grad():
            for _step in range(self.steps):
                optimizer.zero_grad()

                # Current w2c for each frame
                w2c_cur = [
                    _apply_pose_delta(w2c_init_list[i], pose_delta[i])
                    for i in range(M)
                ]

                # Current depths (low-res delta upsampled + init)
                depths_cur = [
                    F.interpolate(
                        depth_delta[i].unsqueeze(0), size=(H, W), mode="bilinear",
                        align_corners=False,
                    ).squeeze() + depths_init[i]
                    for i in range(M)
                ]

                total_loss = torch.zeros(1, device=device)
                for src_idx, tgt_idx in pairs:
                    loss = _feature_projection_loss(
                        patch_feats[src_idx],
                        depths_cur[src_idx],
                        intris[src_idx],
                        w2c_cur[src_idx],
                        patch_feats[tgt_idx],
                        intris[tgt_idx],
                        w2c_cur[tgt_idx],
                        h_patches, w_patches, self.patch_size,
                    )
                    total_loss = total_loss + loss

                total_loss = (
                    total_loss
                    + self.lambda_depth * depth_delta.abs().mean()
                    + self.lambda_pose * pose_delta.pow(2).sum()
                )
                total_loss.backward()
                optimizer.step()

        with torch.no_grad():
            opt_w2c = [
                _apply_pose_delta(w2c_init_list[i], pose_delta[i])
                .detach().cpu().numpy()
                for i in range(M)
            ]
            opt_depths = [
                (
                    F.interpolate(
                        depth_delta[i].unsqueeze(0).detach(),
                        size=(H, W), mode="bilinear", align_corners=False,
                    ).squeeze() + depths_init[i]
                ).cpu()
                for i in range(M)
            ]

        del pose_delta, depth_delta, optimizer, patch_feats, depths_init, w2c_cur
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        return opt_w2c, opt_depths


def _feature_projection_loss(
    src_feats: torch.Tensor,    # [N_patch, C]
    src_depth: torch.Tensor,    # [H, W]
    src_intri: torch.Tensor,    # [3, 3]
    src_w2c: torch.Tensor,      # [4, 4]
    tgt_feats: torch.Tensor,    # [N_patch, C]
    tgt_intri: torch.Tensor,    # [3, 3]
    tgt_w2c: torch.Tensor,      # [4, 4]
    h_patches: int,
    w_patches: int,
    patch_size: int,
) -> torch.Tensor:
    """
    Compute feature-projection photometric loss between two frames.
    Projects src patch centres through src_depth into tgt camera, samples tgt_feats,
    returns mean Charbonnier loss on 1 - cosine_similarity.
    """
    device = src_feats.device
    eps_charbonnier = 1e-3

    # Patch centre pixel coordinates (float)
    py = torch.arange(h_patches, device=device, dtype=torch.float32)
    px = torch.arange(w_patches, device=device, dtype=torch.float32)
    py, px = torch.meshgrid(py, px, indexing="ij")   # [Hp, Wp]
    u = (px.reshape(-1) + 0.5) * patch_size          # [N_patch]
    v = (py.reshape(-1) + 0.5) * patch_size          # [N_patch]

    N = u.shape[0]
    if N != src_feats.shape[0]:
        # Mismatch: safely truncate / zero-pad
        N = min(N, src_feats.shape[0])
        u, v = u[:N], v[:N]

    # Sample depth at patch centres (bilinear, from full-res depth map)
    H, W = src_depth.shape
    u_norm_s = 2.0 * u / max(W - 1, 1) - 1.0
    v_norm_s = 2.0 * v / max(H - 1, 1) - 1.0
    grid_s = torch.stack([u_norm_s, v_norm_s], dim=-1).reshape(1, N, 1, 2)
    z = F.grid_sample(
        src_depth.reshape(1, 1, H, W), grid_s,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    ).reshape(N).clamp(min=1e-3)  # [N]

    # Unproject src patch centres to src camera space
    fx_s, fy_s = src_intri[0, 0], src_intri[1, 1]
    cx_s, cy_s = src_intri[0, 2], src_intri[1, 2]
    x_cam = (u - cx_s) * z / fx_s
    y_cam = (v - cy_s) * z / fy_s
    pts_src_cam = torch.stack([x_cam, y_cam, z], dim=-1)  # [N, 3]

    # Transform to world: pts_w = R_s^T @ (pts_cam - t_s)
    R_s = src_w2c[:3, :3]
    t_s = src_w2c[:3, 3]
    pts_world = (R_s.t() @ (pts_src_cam.t() - t_s.unsqueeze(1))).t()  # [N, 3]

    # Project to tgt camera space
    R_t = tgt_w2c[:3, :3]
    t_t = tgt_w2c[:3, 3]
    pts_tgt_cam = (R_t @ pts_world.t() + t_t.unsqueeze(1)).t()  # [N, 3]

    z_tgt = pts_tgt_cam[:, 2].clamp(min=1e-3)
    fx_t, fy_t = tgt_intri[0, 0], tgt_intri[1, 1]
    cx_t, cy_t = tgt_intri[0, 2], tgt_intri[1, 2]
    u_tgt = fx_t * pts_tgt_cam[:, 0] / z_tgt + cx_t  # [N]
    v_tgt = fy_t * pts_tgt_cam[:, 1] / z_tgt + cy_t  # [N]

    # Convert to patch-grid normalised coordinates
    u_norm = 2.0 * (u_tgt / patch_size) / max(w_patches - 1, 1) - 1.0
    v_norm = 2.0 * (v_tgt / patch_size) / max(h_patches - 1, 1) - 1.0

    # Grid-sample tgt patch features: tgt_feats [N, C] -> [1, C, Hp, Wp]
    C = tgt_feats.shape[1]
    tgt_grid = tgt_feats[:N].reshape(1, h_patches, w_patches, C).permute(0, 3, 1, 2)
    sample_grid = torch.stack([u_norm, v_norm], dim=-1).reshape(1, N, 1, 2)
    sampled = F.grid_sample(
        tgt_grid, sample_grid, mode="bilinear",
        padding_mode="zeros", align_corners=True,
    ).squeeze(0).squeeze(-1).t()  # [N, C]

    # Validity mask
    valid = (u_norm.abs() <= 1.0) & (v_norm.abs() <= 1.0) & (z_tgt > 1e-3)
    if valid.sum() < 4:
        return torch.tensor(0.0, device=device)

    # Charbonnier loss on (1 - cosine_similarity)
    s_f = F.normalize(src_feats[:N][valid], dim=-1)
    t_f = F.normalize(sampled[valid], dim=-1)
    cos_sim = (s_f * t_f).sum(dim=-1)
    residual = 1.0 - cos_sim
    return ((residual.pow(2) + eps_charbonnier ** 2).sqrt()).mean()


# ---------------------------------------------------------------------------
# Open3D Point Cloud Fusion
# ---------------------------------------------------------------------------

class Open3DFusion:
    """Statistical outlier removal + voxel downsampling via Open3D."""

    def __init__(self, voxel_size: float = 0.03, nb_neighbors: int = 20, std_ratio: float = 2.0):
        self.voxel_size = voxel_size
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

    def fuse_window(
        self,
        window_frames: List[FrameEntry],
        opt_w2c_list: List[np.ndarray],
        opt_depth_list: List[torch.Tensor],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fuse optimised-depth window into a cleaned point cloud.

        Returns:
            points:  [N, 3] float32 world coordinates
            colors:  [N, 3] uint8 or None
        """
        o3d = _try_import_open3d()
        if o3d is None:
            return np.empty((0, 3), dtype=np.float32), None

        all_pts: List[np.ndarray] = []
        all_cols: List[np.ndarray] = []

        for frame, w2c, depth_t in zip(window_frames, opt_w2c_list, opt_depth_list):
            depth_np = depth_t.cpu().float().numpy()    # [H, W]
            H, W = depth_np.shape
            intri = frame.intri
            fx, fy = float(intri[0, 0]), float(intri[1, 1])
            cx, cy = float(intri[0, 2]), float(intri[1, 2])

            ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            z = depth_np.reshape(-1)
            valid = (np.isfinite(z)) & (z > 0)
            z = z[valid]
            x = ((xs.reshape(-1)[valid] - cx) * z / max(fx, 1e-12)).astype(np.float32)
            y = ((ys.reshape(-1)[valid] - cy) * z / max(fy, 1e-12)).astype(np.float32)

            pts_cam = np.stack([x, y, z], axis=1)  # [Nv, 3]
            R, t = w2c[:3, :3], w2c[:3, 3]
            pts_world = (R.T @ (pts_cam.T - t[:, None])).T.astype(np.float32)
            all_pts.append(pts_world)

            if frame.rgb is not None:
                rgb_flat = frame.rgb.reshape(-1, 3)[valid]
                all_cols.append(rgb_flat)

        if not all_pts:
            return np.empty((0, 3), dtype=np.float32), None

        pts = np.concatenate(all_pts, axis=0)
        cols = np.concatenate(all_cols, axis=0) if all_cols and len(all_cols) == len(all_pts) else None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        if cols is not None:
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)

        # Voxel downsample
        pcd = pcd.voxel_down_sample(self.voxel_size)
        # Statistical outlier removal
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio
        )

        out_pts = np.asarray(pcd.points, dtype=np.float32)
        out_cols = (
            (np.asarray(pcd.colors) * 255).astype(np.uint8)
            if len(pcd.colors) > 0 else None
        )
        return out_pts, out_cols


# ---------------------------------------------------------------------------
# Loop Closure Manager
# ---------------------------------------------------------------------------

class LoopClosureManager:
    """
    Orchestrates per-frame loop closure detection, PGO, local BA and
    point cloud fusion. All heavy jobs run in a background ThreadPoolExecutor
    so forward_stream() is never blocked.

    Usage::

        manager = LoopClosureManager(rel_pose_head, lc_cfg, ba_cfg, device, patch_size, embed_dim)
        # inside forward_stream():
        manager.on_frame(frame_id, feature_tokens, patch_start_idx,
                         depth, pose_enc, global_kf_idx, image_hw, intri_hw)
        # retrieve async results:
        opt_w2c = manager.get_optimized_w2c()
    """

    def __init__(
        self,
        rel_pose_head,           # LongStream.rel_pose_head (read-only inference)
        lc_cfg: dict,
        ba_cfg: dict,
        device: str,
        patch_size: int = 14,
        embed_dim: int = 1024,
    ):
        self._rel_pose_head = rel_pose_head
        self._device = device
        self._patch_size = patch_size
        self._embed_dim = embed_dim

        # Config
        self._faiss_dim = int(lc_cfg.get("faiss_dim", 128))
        self._min_frame_gap = int(lc_cfg.get("min_frame_gap", 80))
        self._topk = int(lc_cfg.get("topk", 5))
        self._sim_threshold = float(lc_cfg.get("similarity_threshold", 0.72))
        self._pgo_enabled = bool(lc_cfg.get("pgo_enabled", True))
        self._use_gps_prior = bool(lc_cfg.get("use_gps_prior", True))
        # Whether to include GPS priors in the g2o backend (same flag, both backends)
        self._use_gps_prior_g2o = self._use_gps_prior
        # Whether to estimate the loop T_rel in a background thread (default: True).
        # Set async_rel_pose: false in config for synchronous/debug mode.
        self._async_rel_pose = bool(lc_cfg.get("async_rel_pose", True))

        self._ba_enabled = bool(ba_cfg.get("enabled", False))
        ba_max_cached = int(ba_cfg.get("max_cached_frames", 3000))
        self._window_radius = int(ba_cfg.get("window_radius", 4))
        self._feature_dim = int(ba_cfg.get("feature_dim", 128))

        # Sub-components
        self._cache = FeatureFrameCache(max_size=ba_max_cached)
        self._faiss_index = FaissLoopIndex(self._faiss_dim)
        self._pgo = PoseGraphOptimizer()
        self._ba = FeatureMetricLocalBA(ba_cfg)
        self._ba.patch_size = patch_size
        self._fusion = Open3DFusion(
            voxel_size=float(ba_cfg.get("voxel_size", 0.03)),
            nb_neighbors=int(ba_cfg.get("nb_neighbors", 20)),
            std_ratio=float(ba_cfg.get("std_ratio", 2.0)),
        )

        # Projection matrices (fixed random, CPU numpy)
        token_dim = 2 * embed_dim  # aggregated tokens are 2*embed_dim
        self._desc_proj = _get_random_projection(token_dim, self._faiss_dim)
        self._feat_proj = _get_random_projection(token_dim, self._feature_dim)

        # Accumulated pose data for w2c reconstruction
        self._acc_pose_enc: List[torch.Tensor] = []   # each [D] cpu float32
        self._acc_kf_idx: List[int] = []              # global keyframe indices

        # Output buffers
        self._loop_edges: List[LoopEdge] = []
        self._optimized_w2c: Optional[np.ndarray] = None  # [S, 4, 4]
        self._local_ba_windows: List[dict] = []
        self._opt_w2c_lock = threading.Lock()
        self._gps_xyz: Optional[np.ndarray] = None

        # Async executor (1 worker for sequential BA/PGO)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending_futures: List[Future] = []
        self._pgo_dirty = False

    def set_gps_xyz(self, gps_xyz: Optional[np.ndarray]) -> None:
        """Set GPS camera-centre array [S,3] for PGO prior."""
        self._gps_xyz = gps_xyz

    # ------------------------------------------------------------------
    # Main entry point called by StreamSession per frame
    # ------------------------------------------------------------------

    def on_frame(
        self,
        frame_id: int,
        feature_tokens: torch.Tensor,      # [1, 1, P_total, 2*embed_dim] CPU
        patch_start_idx: int,
        depth: torch.Tensor,               # [H, W] CPU float32
        pose_enc: torch.Tensor,            # [D] CPU float32
        global_kf_idx: int,
        image_hw: Tuple[int, int],         # (H, W) model output
        intri: Optional[np.ndarray] = None,  # [3,3] float32 (can be None)
        rgb: Optional[np.ndarray] = None,  # [H, W, 3] uint8
    ) -> None:
        # Accumulate pose data
        self._acc_pose_enc.append(pose_enc.cpu().float())
        self._acc_kf_idx.append(global_kf_idx)

        # Extract tokens — all strictly CPU & detached
        full_tokens = feature_tokens[0, 0]  # [P_total, 2*embed_dim]
        p_total = int(full_tokens.shape[0])
        pose_token = full_tokens[0].half().detach().cpu()
        patch_tokens_raw = full_tokens[patch_start_idx:].detach().cpu()  # [N_patch, 2*embed_dim]
        patch_tokens_reduced = _reduce_patch_features(patch_tokens_raw, self._feat_proj)

        descriptor = _compute_descriptor(patch_tokens_raw, self._desc_proj)

        if intri is None:
            logger.warning(
                f"[LoopClosure] frame {frame_id}: intrinsics unavailable; "
                "BA and point cloud fusion will be skipped for this frame."
            )

        entry = FrameEntry(
            frame_id=frame_id,
            descriptor=descriptor,
            pose_token=pose_token,
            patch_tokens=patch_tokens_reduced,
            depth=depth.cpu().float().detach(),
            pose_enc=pose_enc.cpu().float(),
            global_kf_idx=global_kf_idx,
            intri=intri,  # May be None; downstream BA/fusion must check
            image_hw=image_hw,
            p_total=p_total,
            w2c_init=None,   # decoded lazily
            rgb=rgb,
        )
        self._cache.add(entry)
        self._faiss_index.add(descriptor, frame_id)

        # Poll completed async jobs before submitting new ones
        self._poll_futures()

        # Loop candidate search
        if frame_id >= self._min_frame_gap:
            candidates = self._faiss_index.search(
                descriptor, self._topk, self._min_frame_gap, frame_id
            )
            good = [
                (fid, score)
                for fid, score in candidates
                if score >= self._sim_threshold
            ]
            if good:
                best_hist_id, best_score = good[0]

                if self._async_rel_pose:
                    # --- Async path (default) ---
                    # Snapshot the two pose tokens so the background task is self-contained.
                    hist_entry_snap = self._cache.get(best_hist_id)
                    if hist_entry_snap is not None:
                        hist_pose_snap = hist_entry_snap.pose_token.float().clone().cpu()
                        curr_pose_snap = full_tokens[0].float().detach().clone().cpu()
                        self._submit_loop_detection_job(
                            hist_id=best_hist_id,
                            best_score=best_score,
                            curr_frame_id=frame_id,
                            hist_pose_token=hist_pose_snap,
                            curr_pose_token=curr_pose_snap,
                            p_total=p_total,
                            image_hw=image_hw,
                        )
                    else:
                        # hist entry evicted from cache; append edge with no T_rel
                        with self._opt_w2c_lock:
                            self._loop_edges.append(LoopEdge(
                                hist_id=best_hist_id, curr_id=frame_id,
                                score=best_score, T_ij_meas=None,
                            ))
                        self._pgo_dirty = True
                    # BA doesn't depend on T_rel; submit immediately
                    if self._ba_enabled:
                        self._submit_ba_job(best_hist_id, frame_id)
                else:
                    # --- Sync path (debug mode: async_rel_pose: false) ---
                    T_ij_meas = self._estimate_loop_T_rel(
                        hist_id=best_hist_id,
                        curr_pose_token=full_tokens[0].float(),
                        p_total=p_total,
                        image_hw=image_hw,
                    )
                    with self._opt_w2c_lock:
                        self._loop_edges.append(LoopEdge(
                            hist_id=best_hist_id,
                            curr_id=frame_id,
                            score=best_score,
                            T_ij_meas=T_ij_meas,
                        ))
                    logger.info(
                        f"[LoopClosure] frame {frame_id} → {best_hist_id} "
                        f"(score={best_score:.3f}, "
                        f"T_meas={'ok' if T_ij_meas is not None else 'fallback'})"
                    )
                    self._pgo_dirty = True
                    if self._ba_enabled:
                        self._submit_ba_job(best_hist_id, frame_id)

        # Trigger async PGO if new loop edges accumulated
        if self._pgo_dirty and self._pgo_enabled:
            self._pgo_dirty = False
            self._submit_pgo_job(frame_id + 1)

    # ------------------------------------------------------------------
    # Async job submission
    # ------------------------------------------------------------------

    def _submit_loop_detection_job(
        self,
        hist_id: int,
        best_score: float,
        curr_frame_id: int,
        hist_pose_token: torch.Tensor,   # [embed_dim_x2] CPU float32, pre-snapshotted
        curr_pose_token: torch.Tensor,   # [embed_dim_x2] CPU float32, pre-snapshotted
        p_total: int,
        image_hw: Tuple[int, int],
    ) -> None:
        """
        Submit async task: estimate T_rel with RelPoseHead, then append LoopEdge
        and set _pgo_dirty so the next on_frame() triggers PGO.

        Because this runs in the same single-worker executor, the new task will
        be processed after any already-queued BA/PGO jobs without risk of deadlock.
        """
        num_frames_snap = curr_frame_id + 1
        pgo_enabled = self._pgo_enabled

        def _loop_detection_task():
            T_ij_meas = self._estimate_loop_T_rel_from_tokens(
                hist_pose_token=hist_pose_token,
                curr_pose_token=curr_pose_token,
                p_total=p_total,
                image_hw=image_hw,
            )
            with self._opt_w2c_lock:
                self._loop_edges.append(LoopEdge(
                    hist_id=hist_id,
                    curr_id=curr_frame_id,
                    score=best_score,
                    T_ij_meas=T_ij_meas,
                ))
            logger.info(
                f"[LoopClosure] frame {curr_frame_id} → {hist_id} "
                f"(score={best_score:.3f}, "
                f"T_meas={'ok' if T_ij_meas is not None else 'fallback'})"
            )
            # Run PGO synchronously inside this worker task.
            # Calling _run_pgo_now() here is safe: it does NOT submit to the
            # executor again, so there is no risk of RuntimeError if shutdown
            # has already started (or is about to start) on the calling side.
            if pgo_enabled:
                self._run_pgo_now(num_frames_snap)

        future = self._executor.submit(_loop_detection_task)
        self._pending_futures.append(future)

    def _estimate_loop_T_rel(
        self,
        hist_id: int,
        curr_pose_token: torch.Tensor,  # [2*embed_dim] float32, any device
        p_total: int,
        image_hw: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Thin wrapper: look up hist pose token from cache, then delegate."""
        if self._rel_pose_head is None:
            return None
        hist_entry = self._cache.get(hist_id)
        if hist_entry is None:
            return None
        return self._estimate_loop_T_rel_from_tokens(
            hist_pose_token=hist_entry.pose_token.float().cpu(),
            curr_pose_token=curr_pose_token.float().cpu(),
            p_total=p_total,
            image_hw=image_hw,
        )

    def _estimate_loop_T_rel_from_tokens(
        self,
        hist_pose_token: torch.Tensor,  # [2*embed_dim] CPU float32
        curr_pose_token: torch.Tensor,  # [2*embed_dim] CPU float32
        p_total: int,
        image_hw: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Estimate relative pose T_ij (curr ← hist) using RelPoseHead.

        Builds a fake 2-frame token tensor where only the CLS/pose token
        (index 0) is populated — RelPoseHead only reads tokens[:,:,0,:] so
        the patch slots are irrelevant and left as zero.

        Returns [4,4] float32 T_rel, or None on failure (caller falls back
        to trajectory-derived measurement).
        """
        if self._rel_pose_head is None:
            return None

        device = self._device
        embed_dim_x2 = hist_pose_token.shape[0]  # 2 * embed_dim

        try:
            with torch.no_grad():
                # Build fake_tokens [1, 2, P_total, 2*embed_dim].
                # RelPoseHead only uses tokens[:, :, 0, :] so the rest stays zero.
                fake_tokens = torch.zeros(
                    1, 2, p_total, embed_dim_x2,
                    dtype=torch.float32, device=device,
                )
                fake_tokens[0, 0, 0, :] = hist_pose_token.to(device)
                fake_tokens[0, 1, 0, :] = curr_pose_token.to(device)

                is_kf = torch.tensor([[True, False]], device=device)
                kf_idx = torch.tensor([[0, 0]], device=device)

                out = self._rel_pose_head(
                    aggregated_tokens_list=[fake_tokens],
                    keyframe_indices=kf_idx,
                    is_keyframe=is_kf,
                    num_iterations=4,
                    mode="full",
                    kv_cache_list=None,
                )

            rel_pose_enc = out.get("pose_enc")  # [1, 2, 9]
            if rel_pose_enc is None:
                return None

            # compose_abs_from_rel: frame 1's abs pose = T_rel(1) @ T_abs(keyframe 0)
            # → extri[0,1] is the w2c of curr in hist's coordinate system
            from longstream.utils.camera import compose_abs_from_rel
            from longstream.utils.vendor.models.components.utils.pose_enc import (
                pose_encoding_to_extri_intri,
            )

            abs_enc = compose_abs_from_rel(rel_pose_enc[0], kf_idx[0])  # [2, 9]
            extri, _ = pose_encoding_to_extri_intri(
                abs_enc.unsqueeze(0),  # [1, 2, 9]
                image_size_hw=image_hw,
                build_intrinsics=False,
            )  # extri [1, 2, 3, 4]

            extri_np = extri[0].detach().cpu().numpy()  # [2, 3, 4]
            w2c_hist = np.eye(4, dtype=np.float32)
            w2c_hist[:3, :] = extri_np[0]
            w2c_curr = np.eye(4, dtype=np.float32)
            w2c_curr[:3, :] = extri_np[1]

            # T_ij = T_w2c[curr] @ inv(T_w2c[hist])  ≡  T_rel(curr ← hist)
            T_ij = (w2c_curr @ np.linalg.inv(w2c_hist)).astype(np.float32)
            return T_ij

        except Exception as exc:
            logger.warning(f"[RelPoseHead] loop T_rel estimation failed: {exc}")
            return None

    def _submit_ba_job(self, hist_id: int, curr_id: int) -> None:
        """Decode w2c for both windows and submit BA to thread pool."""
        win_a = self._cache.get_window(hist_id, self._window_radius)
        win_b = self._cache.get_window(curr_id, self._window_radius)

        # Deduplicate by frame_id
        seen = set()
        window_frames = []
        for fe in win_a + win_b:
            if fe.frame_id not in seen:
                seen.add(fe.frame_id)
                window_frames.append(fe)
        window_frames.sort(key=lambda e: e.frame_id)

        # Filter out frames without valid intrinsics (required for projection loss)
        window_frames = [
            fe for fe in window_frames
            if fe.intri is not None and np.isfinite(fe.intri).all()
        ]
        if len(window_frames) < 2:
            logger.debug("[LocalBA] window skipped: insufficient frames with valid intrinsics")
            return

        # Decode w2c for the window (cheap, CPU-only)
        n_acc = len(self._acc_pose_enc)
        for fe in window_frames:
            if fe.w2c_init is None and fe.frame_id < n_acc:
                fe.w2c_init = self._decode_w2c_for_frame(fe.frame_id, fe.image_hw)

        # Copy window_frames to avoid mutation during async run
        window_snapshot = list(window_frames)
        device = self._device
        ba = self._ba

        def _ba_task():
            try:
                opt_w2c, opt_depths = ba.optimize(window_snapshot, device)
                with self._opt_w2c_lock:
                    self._local_ba_windows.append({
                        "frame_ids": [f.frame_id for f in window_snapshot],
                        "opt_w2c": opt_w2c,
                        "opt_depths": [d.detach().cpu() for d in opt_depths],
                    })
                    # Update w2c_init in cache
                    for fe, w2c in zip(window_snapshot, opt_w2c):
                        cached = self._cache.get(fe.frame_id)
                        if cached is not None:
                            cached.w2c_init = w2c
                logger.info(
                    f"[LocalBA] window ({window_snapshot[0].frame_id}"
                    f"…{window_snapshot[-1].frame_id}) done"
                )
            except Exception as exc:
                logger.warning(f"[LocalBA] failed: {exc}", exc_info=True)

        future = self._executor.submit(_ba_task)
        self._pending_futures.append(future)

    def _run_pgo_now(self, num_frames: int) -> None:
        """Execute PGO synchronously on the calling thread.

        Safe to call from inside an executor worker task because it never
        calls ``self._executor.submit()``; there is therefore no risk of a
        ``RuntimeError: cannot schedule new futures after shutdown`` when
        the sequence ends and ``shutdown()`` has already been requested.
        """
        # Collect w2c for all frames; decode missing ones
        w2c_list = []
        for fid in range(num_frames):
            entry = self._cache.get(fid)
            if entry is not None and entry.w2c_init is not None:
                w2c_list.append(entry.w2c_init.copy())
            elif fid < len(self._acc_pose_enc):
                w2c = self._decode_w2c_for_frame(fid, None)
                w2c_list.append(w2c if w2c is not None else np.eye(4, dtype=np.float32))
            else:
                break

        if len(w2c_list) < 2:
            return

        w2c_arr = np.stack(w2c_list, axis=0)  # [S, 4, 4]
        S = len(w2c_arr)

        # Snapshot _loop_edges under lock to avoid concurrent-modification issues
        with self._opt_w2c_lock:
            loop_edges_snap = list(self._loop_edges)
        edges: List[Tuple[int, int, np.ndarray]] = []
        for i in range(S - 1):
            T_ij = w2c_arr[i + 1] @ np.linalg.inv(w2c_arr[i])
            edges.append((i, i + 1, T_ij.astype(np.float32)))

        for edge in loop_edges_snap:
            hist_id, curr_id = edge.hist_id, edge.curr_id
            if hist_id < S and curr_id < S:
                if edge.T_ij_meas is not None:
                    # Use RelPoseHead-derived measurement (independent of drift)
                    T_ij = edge.T_ij_meas.astype(np.float32)
                else:
                    # Fallback: derive from current trajectory (inherits drift)
                    logger.debug(
                        f"[PGO] loop ({hist_id}\u2192{curr_id}): "
                        "no RelPoseHead measurement, using trajectory fallback"
                    )
                    T_ij = (w2c_arr[curr_id] @ np.linalg.inv(w2c_arr[hist_id])).astype(np.float32)
                edges.append((hist_id, curr_id, T_ij))

        gps = self._gps_xyz[:S] if (self._gps_xyz is not None and self._use_gps_prior) else None
        try:
            opt_w2c = self._pgo.optimize(w2c_arr, edges, gps)
            with self._opt_w2c_lock:
                self._optimized_w2c = opt_w2c
            logger.info(f"[PGO] optimised {S} frames.")
        except Exception as exc:
            logger.warning(f"[PGO] failed: {exc}", exc_info=True)

    def _submit_pgo_job(self, num_frames: int) -> None:
        """Submit PGO as a background job via the executor.

        Use this from the main thread (inside ``on_frame``) where
        ``shutdown()`` has not been called yet.  Do NOT call this from
        inside a worker task — use ``_run_pgo_now()`` directly instead.
        """
        future = self._executor.submit(lambda: self._run_pgo_now(num_frames))
        self._pending_futures.append(future)

    def _poll_futures(self) -> None:
        self._pending_futures = [f for f in self._pending_futures if not f.done()]

    # ------------------------------------------------------------------
    # w2c decoding helper
    # ------------------------------------------------------------------

    def _decode_w2c_for_frame(
        self,
        frame_id: int,
        image_hw: Optional[Tuple[int, int]],
    ) -> Optional[np.ndarray]:
        """
        Decode absolute w2c [4,4] for frame_id from accumulated pose_enc.
        Tries to use the cached image_hw; falls back to a 518x518 placeholder.
        """
        try:
            from longstream.utils.camera import compose_abs_from_rel
            from longstream.utils.vendor.models.components.utils.pose_enc import (
                pose_encoding_to_extri_intri,
            )

            n = min(frame_id + 1, len(self._acc_pose_enc))
            if n == 0:
                return None

            rel_pose_stacked = torch.stack(self._acc_pose_enc[:n], dim=0)  # [n, D]
            kf_indices = torch.tensor(
                self._acc_kf_idx[:n], dtype=torch.long
            )  # [n]

            abs_pose_enc = compose_abs_from_rel(
                rel_pose_stacked, kf_indices
            )  # [n, D]

            hw = image_hw if image_hw is not None else (518, 518)
            extri, _ = pose_encoding_to_extri_intri(
                abs_pose_enc[frame_id: frame_id + 1].unsqueeze(0),
                image_size_hw=hw,
            )
            w2c_34 = extri[0, 0].detach().cpu().numpy()  # [3, 4]
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :] = w2c_34
            return w2c
        except Exception as exc:
            logger.debug(f"[LoopClosure] _decode_w2c_for_frame({frame_id}): {exc}")
            return None

    # ------------------------------------------------------------------
    # Result accessors (non-blocking)
    # ------------------------------------------------------------------

    def get_optimized_w2c(self) -> Optional[np.ndarray]:
        """Return latest PGO result [S,4,4] or None if not yet available."""
        with self._opt_w2c_lock:
            return self._optimized_w2c

    def get_loop_edges(self) -> List[Tuple[int, int, float]]:
        """Return loop edges as (hist_id, curr_id, score) tuples (backward-compatible)."""
        with self._opt_w2c_lock:
            return [(e.hist_id, e.curr_id, e.score) for e in self._loop_edges]

    def get_local_ba_windows(self) -> List[dict]:
        with self._opt_w2c_lock:
            return list(self._local_ba_windows)

    def shutdown(self) -> None:
        """Graceful shutdown: wait for all pending jobs (up to 60 s)."""
        self._executor.shutdown(wait=True, cancel_futures=False)

    def clear(self) -> None:
        """Reset all state (called between sequences)."""
        self._cache = FeatureFrameCache(max_size=self._cache.max_size)
        self._faiss_index = FaissLoopIndex(self._faiss_dim)
        self._acc_pose_enc.clear()
        self._acc_kf_idx.clear()
        self._loop_edges.clear()
        self._local_ba_windows.clear()
        self._pgo_dirty = False
        with self._opt_w2c_lock:
            self._optimized_w2c = None
