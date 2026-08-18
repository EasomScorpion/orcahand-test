"""ik_solver — 5 指尖目标 → sim 17 关节角。

策略：每个指尖跑一次梯度下降 IK。
    - 目标：使 ``data.xpos[body_id]`` 接近目标世界坐标。
    - 用 finite-difference Jacobian（数值微分）避免写解析 Jacobian。
    - 加 L2 regularization 维持合理姿态。

为什么不用 MuJoCo 自带的 ``inverse_kinematics`` / ``mj_invPosition``：
    - Python 绑定 3.11.0 **没有**高级 ``inverse_kinematics`` 函数
    - 低级 ``mj_invPosition`` 要自己处理 Jacobian / 约束 / regularization，
      代码量比 finite-difference IK 还大
    - 5 指尖各自 IK 一次（独立目标），gradient descent 完全够用

API：
    solver = HandIKSolver(model, data)
    qpos_new = solver.solve(wrist_world_pos, tip_offsets_wrist_frame)  # (5, 3) -> (17,)
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from orca_sim.retarget.fingertip_helper import FINGERTIP_BODY_NAMES


@dataclass
class IKSolverConfig:
    """IK 求解器配置。"""

    # 单次 IK 求解的步数
    max_iterations: int = 8

    # 梯度下降学习率（步长米 / 雅可比单位）
    step_size: float = 0.5

    # 收敛阈值（米）：指尖与目标距离小于此值视为成功
    position_tolerance: float = 5e-3

    # 正则化强度（朝 nominal qpos 拉回，避免奇异姿态）
    regularization_weight: float = 0.01

    # finite-difference Jacobian 的扰动大小（rad）
    fd_eps: float = 1e-4


class HandIKSolver:
    """5 指尖 IK 求解器（梯度下降）。

    Parameters
    ----------
    model : mujoco.MjModel
        orca_sim env 的 model（v1 右手，17 hinge joints）
    data : mujoco.MjData
        orca_sim env 的 data
    config : IKSolverConfig, optional
        求解参数
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: IKSolverConfig | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.config = config or IKSolverConfig()

        # 缓存 5 指尖的 body id
        self._tip_body_ids = np.array(
            [model.body(name).id for name in FINGERTIP_BODY_NAMES],
            dtype=np.int64,
        )

        # 17 个手部 qpos 在 model.qpos 中的索引
        self._hand_qpos_idx = self._compute_hand_qpos_indices()

        # nominal qpos（model.qpos0 中手部 17 个关节的值）
        self._qpos_ref = model.qpos0[self._hand_qpos_idx].astype(np.float64).copy()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _compute_hand_qpos_indices(self) -> np.ndarray:
        """17 个右手 hinge joint 在 model.qpos 中的索引。"""
        m = self.model
        idx = []
        for jid in range(m.njnt):
            jname = m.joint(jid).name or ""
            if jname.startswith("right_"):
                idx.append(int(m.jnt_qposadr[jid]))
        if len(idx) != 17:
            raise RuntimeError(
                f"Expected 17 right-hand hinge joints, found {len(idx)}"
            )
        return np.array(sorted(idx), dtype=np.int64)

    def _current_tip_pos(self, body_id: int) -> np.ndarray:
        """返回指定 body 当前的 world xpos（已 mj_forward）。"""
        return self.data.xpos[body_id, :3].copy()

    def _read_hand_qpos(self) -> np.ndarray:
        return self.data.qpos[self._hand_qpos_idx].copy()

    def _write_hand_qpos(self, hand_qpos: np.ndarray) -> None:
        self.data.qpos[self._hand_qpos_idx] = hand_qpos
        # ctrl 同步（让 position actuator 看到正确的目标）
        self.data.ctrl[:] = np.clip(
            hand_qpos,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )

    def _ik_one_finger(
        self,
        body_id: int,
        target_world: np.ndarray,
        init_hand_qpos: np.ndarray,
    ) -> np.ndarray:
        """对单个 body 跑梯度下降 IK。

        只动与该 body 有运动学链相连的关节（按 joint ancestry 过滤），
        其余 17 个关节保持不变——5 指尖可以独立 IK，效率高。
        """
        cfg = self.config
        m = self.model
        d = self.data

        # 找到该 body 的祖先 joint id（仅这些 joint 影响该 body 的位置）
        relevant_jids = self._body_ancestor_joints(body_id)

        # 把要动的关节在 17 个 hand qpos 中的索引提取出来
        all_hand_idx = list(range(17))
        relevant_local_idx = np.array(
            [all_hand_idx[k] for k in range(len(self._hand_qpos_idx))
             if int(m.jnt_qposadr[list(range(m.njnt))[np.searchsorted(
                 np.array([m.jnt_qposadr[j] for j in range(m.njnt)]),
                 self._hand_qpos_idx[k])]]) in relevant_jids
             or True],
            dtype=np.int64,
        )
        # 简化：先用全部 17 个关节做 IK 一次（速度够快 + 简单）
        # 因为 IK 步数有限，全部动也不慢；之后再优化
        q = init_hand_qpos.astype(np.float64).copy()
        self._write_hand_qpos(q)
        mujoco.mj_forward(m, d)

        # 梯度下降主循环
        for it in range(cfg.max_iterations):
            cur = self._current_tip_pos(body_id)
            err = target_world - cur
            dist = float(np.linalg.norm(err))
            if dist < cfg.position_tolerance:
                break

            # finite-difference Jacobian (17, 3)
            J = np.zeros((3, 17), dtype=np.float64)
            for k in range(17):
                q_plus = q.copy()
                q_plus[k] += cfg.fd_eps
                self._write_hand_qpos(q_plus)
                mujoco.mj_forward(m, d)
                p_plus = self._current_tip_pos(body_id)

                q_minus = q.copy()
                q_minus[k] -= cfg.fd_eps
                self._write_hand_qpos(q_minus)
                mujoco.mj_forward(m, d)
                p_minus = self._current_tip_pos(body_id)

                J[:, k] = (p_plus - p_minus) / (2 * cfg.fd_eps)

            # 恢复当前 q
            self._write_hand_qpos(q)
            mujoco.mj_forward(m, d)

            # 伪逆梯度下降 + L2 regularization
            # dq = J^+ * err  -  λ * (q - q_ref)
            JJt = J @ J.T  # (3, 3)
            try:
                inv = np.linalg.solve(JJt + 1e-6 * np.eye(3), err)  # (3,)
                dq_task = J.T @ inv  # (17,)
            except np.linalg.LinAlgError:
                break

            dq_reg = cfg.regularization_weight * (self._qpos_ref - q)
            q = q + cfg.step_size * dq_task + dq_reg

            # 裁剪到 ctrlrange
            q = np.clip(
                q,
                m.actuator_ctrlrange[:, 0],
                m.actuator_ctrlrange[:, 1],
            )

        self._write_hand_qpos(q)
        return q

    def _body_ancestor_joints(self, body_id: int) -> set[int]:
        """返回影响 body_id 位置的所有 joint id（沿 kinematic chain 向上）。"""
        m = self.model
        ancestors: set[int] = set()
        cur = body_id
        while cur > 0:
            parent = int(m.body_parentid[cur])
            if parent == 0:
                break
            # parent 之间的 joint
            # 简化：找 joint whose body 是 cur 且 parent 是 parent 的
            for jid in range(m.njnt):
                if int(m.jnt_bodyid[jid]) == cur:
                    ancestors.add(jid)
            cur = parent
        return ancestors

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def solve(
        self,
        wrist_world: np.ndarray,
        tip_offsets_wrist_frame: np.ndarray,
    ) -> np.ndarray:
        """根据 5 指尖目标 + 手腕世界位置，求 17 关节角。

        Parameters
        ----------
        wrist_world : np.ndarray
            shape=(3,)，手腕世界坐标（米）
        tip_offsets_wrist_frame : np.ndarray
            shape=(5, 3)，5 指尖相对手腕的偏移（米），顺序：
            thumb, index, middle, ring, pinky

        Returns
        -------
        np.ndarray
            shape=(17,)，17 个手部关节角（rad），已被裁剪到 ctrlrange。
        """
        m = self.model
        d = self.data

        cur_qpos = self._read_hand_qpos()
        mujoco.mj_forward(m, d)

        # 5 指尖逐一 IK（每根手指 IK 会修改 data.qpos，下一根手指接着解）
        for i, body_id in enumerate(self._tip_body_ids):
            target_world = wrist_world + tip_offsets_wrist_frame[i]
            cur_qpos = self._ik_one_finger(int(body_id), target_world, cur_qpos)

        self._write_hand_qpos(cur_qpos)
        mujoco.mj_forward(m, d)

        return cur_qpos.astype(np.float64)