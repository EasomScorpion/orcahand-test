"""bone_matcher — MediaPipe 21 landmarks → sim 17 qpos。

设计哲学：
    - **启发式初值**为主：用 MediaPipe curl_norm + 局部 X 偏移估算每个关节。
    - **可选 IK 微调**：用 MuJoCo mj_jacBody + Levenberg-Marquardt 让 sim 5 指尖逼近
      MediaPipe 给出的（缩放后）指尖坐标。**默认开启**但有 fallback：当 IK 让
      残差变大就退回到启发式初值。
    - **不依赖 calibration_data.json**：因为里面的 qpos 是 CurlSolver 算的，不能
      当 ground truth。

关节语义（v1 right hand，参考右 MJCF）：
    - 17 actuator = wrist + thumb(mcp/abd/pip/dip) + 4×(abd/mcp/pip)
    - thumb_pip/dip 不影响 thumb_dp body（链上没下游 body）→ IK 无法调整这俩关节
    - 4 指的 pip 也基本不影响 ip body（axis 是 0,-1,0 且 ip 是 leaf）→ IK 只用
      wrist + 5×(abd/mcp) + thumb_abd 这 11 自由度真正"动" 5 指尖
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from orca_sim.retarget.sim_skeleton import SimSkeleton
from orca_sim.retarget.curl_solver import CurlSolver


WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


@dataclass(frozen=True)
class MediaPipeHand:
    """21 landmarks (米, 右手系) 的强类型视图。

    **坐标系约定**：
        MediaPipe Hands 输出的 ``world_landmarks`` 是「手背朝上」的右手系——
        ``+Y`` 指向手背方向，``+Z`` 指向远离手腕方向（指尖朝前是 ``-Z``）。
        MuJoCo 默认是 ``+Y`` 向上，``+Z`` 指向世界背后。要让 MediaPipe 跟 sim
        对齐，需要：

            sim_x = mp_x    (x 方向不变)
            sim_y = -mp_y   (y 翻转：MediaPipe 手背→sim 向下)
            sim_z = -mp_z   (z 翻转：MediaPipe 指尖向前→sim 向后)

        设 ``flip_yz=False`` 可禁用此翻转——在你已经做了坐标系转换时。
    """

    wrist: np.ndarray
    thumb: np.ndarray
    index: np.ndarray
    middle: np.ndarray
    ring: np.ndarray
    pinky: np.ndarray

    @classmethod
    def from_landmarks(cls, lm: np.ndarray, flip_yz: bool = True) -> "MediaPipeHand":
        if lm.shape != (21, 3):
            raise ValueError(f"expected (21, 3), got {lm.shape}")
        arr = lm.copy()
        if flip_yz:
            arr[:, 1] *= -1.0
            arr[:, 2] *= -1.0
        return cls(
            wrist=arr[0].copy(),
            thumb=arr[1:5].copy(),
            index=arr[5:9].copy(),
            middle=arr[9:13].copy(),
            ring=arr[13:17].copy(),
            pinky=arr[17:21].copy(),
        )


@dataclass
class BoneMatcherConfig:
    """BoneMatcher 配置。"""

    max_iterations: int = 12
    lm_damping: float = 1e-3
    warm_start: bool = True
    enforce_limits: bool = True
    # IK 失败阈值：如果 IK 让 tip 总残差 > init 总残差 × 1.5，回退到启发式初值
    fallback_residual_factor: float = 1.5
    # 启发式初值选项
    use_heuristic_init: bool = True


class BoneMatcher:
    """把 MediaPipe 21 landmarks 映射到 sim 17 关节角。

    主流程：
        1. 启发式初值（**复用 CurlSolver**——它对张直/握拳已经能区分手势）
        2. IK 微调：让 sim 5 指尖方向逼近 MediaPipe 给出的方向（不假设距离相等）
        3. 限位

    用法::

        skel = SimSkeleton.from_model(env.unwrapped.model)
        matcher = BoneMatcher(skel, env.unwrapped.model, env.unwrapped.data)
        qpos = matcher.solve(landmarks_world, prev_qpos=prev_qpos)
    """

    def __init__(
        self,
        skel: SimSkeleton,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: BoneMatcherConfig | None = None,
    ):
        self.skel = skel
        self.model = model
        self.data = data
        self.config = config or BoneMatcherConfig()
        self._last_qpos: np.ndarray | None = None
        # 复用 CurlSolver 当启发式初值算法——它已经过验证且能区分手势
        self._curl = CurlSolver(model, data)

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def solve(
        self,
        landmarks_world: np.ndarray,
        prev_qpos: np.ndarray | None = None,
        data: mujoco.MjData | None = None,
    ) -> np.ndarray:
        """MediaPipe 21 landmarks → sim 17 qpos。

        Parameters
        ----------
        landmarks_world
            shape ``(21, 3)``，米，MediaPipe 右手世界系（已 flip X/Z）。
        prev_qpos
            显式提供的上一帧 qpos（17，）。用作热启动；仅当调用方**主动**传才用，
            不会自动从 ``_last_qpos`` 拿（避免测试 / 离线批量运算时被错误热启动污染）。
        data
            与 :class:`SimSkeleton` 对应的 MjData，**会被就地修改**。
        """
        cfg = self.config

        # 启发式 init 是 sim 端真实的目标——绝不用 prev_qpos 替代。
        # prev_qpos 只在启发式 init 与历史一致时用作 blend（避免 IK 卡死在错误目标）。
        if cfg.use_heuristic_init:
            qpos_init = self._heuristic_init(landmarks_world, data.model)
        else:
            qpos_init = data.model.qpos0[:17].copy()
        if cfg.warm_start and prev_qpos is not None:
            # 启发式 init 已经有"目标 qpos"——prev_qpos 只微调（blend 0.3），
            # 避免完全用 prev_qpos 当初值导致 IK 卡在「上一帧的 qpos」。
            qpos_init = 0.7 * qpos_init + 0.3 * prev_qpos

        # MediaPipeHand 包装（用于 IK 残差计算）
        hand = MediaPipeHand.from_landmarks(landmarks_world)

        # IK 微调（直接优化 5 指尖目标）
        qpos_ik = self._ik_refine(qpos_init, hand, data.model, data)

        # 比较启发式残差 vs IK 残差
        ctrlrange = data.model.actuator_ctrlrange
        qpos = qpos_ik
        if self._is_init_better(qpos_init, qpos_ik, hand, data.model, data):
            qpos = qpos_init

        # 限位
        if cfg.enforce_limits:
            qpos = np.clip(qpos, ctrlrange[:, 0], ctrlrange[:, 1]).astype(np.float64)
        self._last_qpos = qpos.copy()
        return qpos

    def reset(self) -> None:
        """清空热启动历史。"""
        self._last_qpos = None

    # ------------------------------------------------------------------
    # 启发式初值
    # ------------------------------------------------------------------
    def _heuristic_init(self, lm: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
        """启发式初值：直接复用 :class:`CurlSolver`。

        之前尝试自己写 curl_norm 估算 + 局部 X 投影，但 thumb_abd 与 wrist
        估算不准，导致 4 根手指（open vs peace）几乎不可分。CurlSolver 是
        已经过验证的现成算法，直接调它。
        """
        return self._curl.solve(lm)

    @staticmethod
    def _local_x(wrist: np.ndarray, middle_mcp: np.ndarray) -> np.ndarray:
        """手掌局部 X 方向（垂直于 palm_axis）。"""
        palm = middle_mcp - wrist
        n = np.linalg.norm(palm)
        if n < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        palm = palm / n
        up = np.array([0.0, 1.0, 0.0])
        local_x = np.cross(up, palm)
        ln = np.linalg.norm(local_x)
        if ln < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        return local_x / ln

    # ------------------------------------------------------------------
    # IK 微调
    # ------------------------------------------------------------------
    def _ik_refine(
        self,
        qpos: np.ndarray,
        hand: MediaPipeHand,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> np.ndarray:
        """跑 IK 微调当前 qpos：让 sim 5 指尖方向逼近 MediaPipe 5 指尖方向。

        距离归一化：把 MediaPipe 5 指尖相对 wrist 整体归一化到 unit vector，
        避免「人手大 sim 手小」的距离不匹配。
        """
        cfg = self.config

        # 5 指尖相对 wrist —— 归一化为方向向量
        target_dirs = {}
        for k, tip in zip(
            ("thumb_dp", "index_ip", "middle_ip", "ring_ip", "pinky_ip"),
            [hand.thumb[3], hand.index[3], hand.middle[3], hand.ring[3], hand.pinky[3]],
        ):
            v = tip - hand.wrist
            vn = np.linalg.norm(v)
            target_dirs[k] = v / max(vn, 1e-6)

        # media 5 指尖相对 wrist 的「距离」，作为 sim 距离权重
        target_dists = {}
        for k, tip in zip(
            ("thumb_dp", "index_ip", "middle_ip", "ring_ip", "pinky_ip"),
            [hand.thumb[3], hand.index[3], hand.middle[3], hand.ring[3], hand.pinky[3]],
        ):
            target_dists[k] = float(np.linalg.norm(tip - hand.wrist))

        body_ids = {
            k: self.skel.get_bone(v).body_id
            for k, v in zip(
                ("thumb_dp", "index_ip", "middle_ip", "ring_ip", "pinky_ip"),
                self.skel.fingertip_body_names,
            )
        }

        # sim wrist（固定，用初始 qpos 算一次）
        data.qpos[:17] = qpos
        mujoco.mj_forward(model, data)
        palm_id = self.skel.bones[self.skel.palm_body_name].body_id
        sim_wrist = data.xpos[palm_id, :3].copy()

        n_targets = 5 * 3
        nv = 17
        J = np.zeros((n_targets, nv))
        jacp = np.zeros((3, model.nv))

        # 初始 resid
        init_sim_tips = {
            k: data.xpos[body_ids[k], :3] - sim_wrist for k in body_ids
        }
        init_resid = self._total_tip_resid(init_sim_tips, target_dirs, target_dists)
        last_resid = init_resid

        best_qpos = qpos.copy()
        best_resid = init_resid

        for it in range(cfg.max_iterations):
            sim_tips = {
                k: data.xpos[body_ids[k], :3] - sim_wrist for k in body_ids
            }
            # 残差 = (sim_dir - tgt_dir) × tgt_dist
            err = np.zeros(n_targets)
            for i, k in enumerate(body_ids):
                d = target_dists[k]
                err[3 * i:3 * i + 3] = (sim_tips[k] - target_dirs[k] * d).flatten()
            resid = float(np.linalg.norm(err)) / max(np.sqrt(sum(target_dists.values()) ** 2), 1e-3)
            if resid < 5e-3 or resid > last_resid * cfg.fallback_residual_factor:
                break

            for i, k in enumerate(body_ids):
                mujoco.mj_jacBody(model, data, jacp, None, body_ids[k])
                J[3 * i:3 * i + 3, :] = jacp[:, :17]

            H = J.T @ J + cfg.lm_damping * max(np.trace(J.T @ J) / 17, 1e-9) * np.eye(nv)
            try:
                dq = np.linalg.solve(H, J.T @ err)
            except np.linalg.LinAlgError:
                break
            dq = np.clip(dq, -0.2, 0.2)
            qpos_new = np.clip(
                qpos + dq,
                model.actuator_ctrlrange[:, 0],
                model.actuator_ctrlrange[:, 1],
            )
            data.qpos[:17] = qpos_new
            mujoco.mj_forward(model, data)
            sim_tips_new = {
                k: data.xpos[body_ids[k], :3] - sim_wrist for k in body_ids
            }
            new_resid = self._total_tip_resid(sim_tips_new, target_dirs, target_dists)
            last_resid = resid
            if new_resid < best_resid:
                best_resid = new_resid
                best_qpos = qpos_new.copy()
                qpos = qpos_new
            else:
                # 不接受变差的步
                data.qpos[:17] = qpos
                break
        return best_qpos

    @staticmethod
    def _total_tip_resid(
        sim_tips: dict[str, np.ndarray],
        target_dirs: dict[str, np.ndarray],
        target_dists: dict[str, np.ndarray],
    ) -> float:
        """5 指尖方向距离的 L2。"""
        total = 0.0
        for k in sim_tips:
            d = target_dists[k]
            resid = sim_tips[k] - target_dirs[k] * d
            total += float(np.dot(resid, resid))
        return total ** 0.5

    def _is_init_better(
        self,
        qpos_init: np.ndarray,
        qpos_ik: np.ndarray,
        hand: MediaPipeHand,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> bool:
        """如果启发式初值比 IK 输出更接近目标方向，返回 True。"""
        target_dirs = {}
        target_dists = {}
        for k, tip in zip(
            ("thumb_dp", "index_ip", "middle_ip", "ring_ip", "pinky_ip"),
            [hand.thumb[3], hand.index[3], hand.middle[3], hand.ring[3], hand.pinky[3]],
        ):
            v = tip - hand.wrist
            target_dirs[k] = v / max(np.linalg.norm(v), 1e-6)
            target_dists[k] = float(np.linalg.norm(tip - hand.wrist))

        body_ids = {
            k: self.skel.get_bone(v).body_id
            for k, v in zip(
                ("thumb_dp", "index_ip", "middle_ip", "ring_ip", "pinky_ip"),
                self.skel.fingertip_body_names,
            )
        }

        palm_id = self.skel.bones[self.skel.palm_body_name].body_id
        scores = []
        for qpos in (qpos_init, qpos_ik):
            data.qpos[:17] = qpos
            mujoco.mj_forward(model, data)
            sim_wrist = data.xpos[palm_id, :3].copy()
            sim_tips = {k: data.xpos[body_ids[k], :3] - sim_wrist for k in body_ids}
            r = self._total_tip_resid(sim_tips, target_dirs, target_dists)
            scores.append(r)
        return scores[0] < scores[1]


# ----------------------------------------------------------------------
# 私有常量
# ----------------------------------------------------------------------
_MCP_GAIN = 1.0

_FINGER_SPECS = {
    "index": dict(MCP=INDEX_MCP, PIP=INDEX_PIP, DIP=INDEX_DIP, TIP=INDEX_TIP,
                  mcp_aidx=6, pip_aidx=7, abd_aidx=5),
    "middle": dict(MCP=MIDDLE_MCP, PIP=MIDDLE_PIP, DIP=MIDDLE_DIP, TIP=MIDDLE_TIP,
                   mcp_aidx=9, pip_aidx=10, abd_aidx=8),
    "ring": dict(MCP=RING_MCP, PIP=RING_PIP, DIP=RING_DIP, TIP=RING_TIP,
                 mcp_aidx=12, pip_aidx=13, abd_aidx=11),
    "pinky": dict(MCP=PINKY_MCP, PIP=PINKY_PIP, DIP=PINKY_DIP, TIP=PINKY_TIP,
                  mcp_aidx=15, pip_aidx=16, abd_aidx=14),
}

# sim abd 旋转轴方向：CurlSolver 用统一 -1 是错的。每根手指单独调
_ABD_AXIS_SIGN = {
    "index": -1.0,
    "middle": -1.0,
    "ring": +1.0,
    "pinky": +1.0,
}
_ABD_X_SCALE_REL = 1.0

_THUMB_PIP_GAIN = 1.88
_THUMB_MCP_RATIO = 0.5
_THUMB_DIP_RATIO = 0.5


__all__ = ["MediaPipeHand", "BoneMatcher", "BoneMatcherConfig"]
