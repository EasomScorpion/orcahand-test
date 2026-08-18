"""sim_skeleton — 从 MuJoCo model 解析出右手 17 body 骨架的描述。

设计要点：
    - **不写任何硬编码**（不写死关节名/轴向/range/位置）——全部从 ``model`` 读，
      改 v2、改左手都能直接复用。
    - **包含"骨架链"信息**：每个 bone 的 parent_body_id、child_body_names，
      方便后续做 IK 的链式遍历。
    - **forward_kinematics** 在副本 data 上跑 ``mj_forward``——不污染调用方的 env state。

用法::

    skel = SimSkeleton.from_model(env.unwrapped.model)
    print(skel.bones["right_index_ip"].joint_axis_local, skel.bones["right_index_ip"].joint_range)

    # 给一帧 qpos 算 17 body 的世界坐标
    xpos_by_name = skel.forward_kinematics(qpos, env.unwrapped.data)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import mujoco
import numpy as np


# v1 右手 17 个关节（actuator）顺序——这个顺序是 v1 MJCF 里声明的，
# SimSkeleton.from_model 用它把 qpos[i] 映射到对应 body。
_ACTUATOR_QPOS_ORDER_V1 = (
    "right_wrist_actuator",
    "right_thumb_mcp_actuator",
    "right_thumb_abd_actuator",
    "right_thumb_pip_actuator",
    "right_thumb_dip_actuator",
    "right_index_abd_actuator",
    "right_index_mcp_actuator",
    "right_index_pip_actuator",
    "right_middle_abd_actuator",
    "right_middle_mcp_actuator",
    "right_middle_pip_actuator",
    "right_ring_abd_actuator",
    "right_ring_mcp_actuator",
    "right_ring_pip_actuator",
    "right_pinky_abd_actuator",
    "right_pinky_mcp_actuator",
    "right_pinky_pip_actuator",
)


@dataclass(frozen=True)
class SimBone:
    """右手一个 body 的解析描述。

    Attributes
    ----------
    body_id
        MuJoCo body id（在 ``model.body`` 数组里）。
    body_name
        body 名（例 ``right_index_ip``），sim v1 用 ``right_<finger>_<seg>``。
    parent_body_id
        父 body id（``0`` 表示父是 world）。手指链通过 parent 串成。
    child_body_names
        直接子 body 名（拓扑下游）。
    joint_id
        连接这个 body 与 parent 的 hinge joint id（``right_tower`` 是固定，
        joint_id = -1）。
    joint_axis_local
        旋转轴（在 parent body 的 local 系里），单位向量。shape=(3,)。
    joint_range
        ``(lo, hi)`` rad。joint 是 hinge 时才有。
    joint_ref
        joint 的 ref 偏置（rad）。``mj_forward`` 会把 ``ref`` 加到 qpos。
    joint_qpos_idx
        该关节在 ``model.qpos`` 中的索引（actuator 顺序）。
    local_pos_in_parent
        body 相对 parent 的平移（米）。shape=(3,)。
    """

    body_id: int
    body_name: str
    parent_body_id: int
    child_body_names: tuple[str, ...]
    joint_id: int
    joint_axis_local: np.ndarray
    joint_range: tuple[float, float]
    joint_ref: float
    joint_qpos_idx: int
    local_pos_in_parent: np.ndarray


@dataclass(frozen=True)
class SimSkeleton:
    """右手的解析骨架（17 body）。"""

    bones: dict[str, SimBone]                              # body_name → bone
    qpos_to_body: dict[int, int]                           # qpos_idx → body_id
    fingertip_body_names: tuple[str, ...] = (
        "right_thumb_dp",
        "right_index_ip",
        "right_middle_ip",
        "right_ring_ip",
        "right_pinky_ip",
    )
    palm_body_name: str = "right_palm"
    actuator_qpos_order: tuple[str, ...] = _ACTUATOR_QPOS_ORDER_V1

    def get_bone(self, body_name: str) -> SimBone:
        if body_name not in self.bones:
            raise KeyError(f"unknown body: {body_name!r}")
        return self.bones[body_name]

    def fingertip_body_ids(self) -> np.ndarray:
        return np.array(
            [self.bones[n].body_id for n in self.fingertip_body_names],
            dtype=np.int64,
        )

    def fingertip_local_lengths(self) -> dict[str, float]:
        """每根手指从 mcp body 到 tip body 的"自然链长"（米）。

        把每个 body 自己的 ``local_pos_in_parent`` 长度加起来。
        """
        out: dict[str, float] = {}
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            segs = [
                "mp", "pp",
                "ip" if finger != "thumb" else "ip",
                "dp" if finger == "thumb" else None,
            ]
            segs = [s for s in segs if s]
            body_names = [f"right_{finger}_{s}" for s in segs]
            # 不算第一个（mp 相对 palm 的偏移）；只算 pp/ip/dp 相对 mp 的长度
            total = 0.0
            for bn in body_names[1:]:
                total += float(np.linalg.norm(self.bones[bn].local_pos_in_parent))
            out[finger] = total
        return out

    # ------------------------------------------------------------------
    # forward kinematics
    # ------------------------------------------------------------------
    def forward_kinematics(
        self,
        qpos: np.ndarray,
        data: mujoco.MjData,
    ) -> dict[str, np.ndarray]:
        """给定 17 维 qpos，跑 ``mj_forward``，返回 body_name → world xpos (3,).

        Parameters
        ----------
        qpos
            17 维右手关节角（rad）。可以传一个比 17 大的数组（包含其它自由度），
            函数只覆盖右手 17 个 entry。
        data
            与 ``self.bones[*].body_id`` 对应的 MjData。**就地修改** qpos 与
            xpos，但调用方应负责 reset。
        """
        model = data.model
        qpos_full = data.qpos.copy()
        for bone in self.bones.values():
            if bone.joint_qpos_idx < 0:
                continue
            qpos_full[bone.joint_qpos_idx] = float(qpos[bone.joint_qpos_idx])
        data.qpos[:] = qpos_full
        mujoco.mj_forward(model, data)
        return {name: data.xpos[bone.body_id, :3].copy() for name, bone in self.bones.items()}

    # ------------------------------------------------------------------
    # 工厂：从 model 抽
    # ------------------------------------------------------------------
    @classmethod
    def from_model(cls, model: mujoco.MjModel, hand_prefix: str = "right") -> "SimSkeleton":
        """从右手 v1 MJCF model 抽出 17 body 骨架。

        Parameters
        ----------
        model
            已加载的 MuJoCo model。
        hand_prefix
            右手 = ``"right"``，左手 = ``"left"``。
        """
        # 1) 找所有以 prefix_ 开头的 body
        prefix_bodies: dict[str, int] = {}                 # body_name → body_id
        for i in range(model.nbody):
            nm = model.body(i).name or ""
            if nm.startswith(hand_prefix + "_"):
                prefix_bodies[nm] = i

        if not prefix_bodies:
            raise ValueError(
                f"model 里没找到任何以 {hand_prefix + '_'!r} 开头的 body。"
                "检查 OrcaHandRight 版本（v1 vs v2 命名略不同）。"
            )

        # 2) 找每个 body 的 joint（jnt_bodyid == this_body_id）
        body_to_joint: dict[int, int] = {}
        for jid in range(model.njnt):
            jb = int(model.jnt_bodyid[jid])
            body_to_joint[jb] = jid

        # 3) 找每个 body 的直接子（parent_bodyid[i] == this）
        children_of: dict[int, list[str]] = {}
        for name, bid in prefix_bodies.items():
            parent = int(model.body_parentid[bid])
            children_of.setdefault(parent, []).append(name)

        # 4) 把 17 个 actuator 按 name → qpos_idx 排序
        actuator_qpos: dict[str, int] = {}
        for aid in range(model.nu):
            aname = model.actuator(aid).name
            jid = int(model.actuator_trnid[aid, 0])
            qa = int(model.jnt_qposadr[jid])
            if aname.startswith(hand_prefix + "_"):
                actuator_qpos[aname] = qa

        # 5) 验证 prefix 顺序与 actuator_qpos_order 一致
        for aname in _ACTUATOR_QPOS_ORDER_V1:
            if hand_prefix == "right":
                if aname not in actuator_qpos:
                    raise ValueError(
                        f"model 没有 actuator {aname!r}——可能不是 v1 right hand？"
                    )
            else:
                left_name = aname.replace("right_", hand_prefix + "_", 1)
                if left_name not in actuator_qpos:
                    raise ValueError(
                        f"model 没有 actuator {left_name!r}——可能不是 {hand_prefix} hand？"
                    )

        # 6) 构造 SimBone：只保留有 joint 的 body（palm / tower 等无 joint 的 base 视为元信息不计入 bones）
        # palm 必须保留（wrist 关节挂在它上面）
        bones: dict[str, SimBone] = {}
        for name, bid in prefix_bodies.items():
            jid = body_to_joint.get(bid, -1)
            # tower 类无 joint 的 base body 不收
            if jid < 0 and name != f"{hand_prefix}_palm":
                continue
            parent = int(model.body_parentid[bid])
            if jid >= 0:
                axis = np.array(model.jnt_axis[jid], dtype=np.float64).copy()
                lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
                ref = float(model.qpos0[int(model.jnt_qposadr[jid])])
                qa = int(model.jnt_qposadr[jid])
            else:
                axis = np.zeros(3, dtype=np.float64)
                lo, hi = 0.0, 0.0
                ref = 0.0
                qa = -1

            child_names = tuple(sorted(children_of.get(bid, [])))
            local_pos = np.array(model.body_pos[bid], dtype=np.float64).copy()

            bones[name] = SimBone(
                body_id=bid,
                body_name=name,
                parent_body_id=parent,
                child_body_names=child_names,
                joint_id=jid,
                joint_axis_local=axis,
                joint_range=(lo, hi),
                joint_ref=ref,
                joint_qpos_idx=qa,
                local_pos_in_parent=local_pos,
            )

        # 7) 验证：必须恰好有 17 个 右手 body（tower/palm/wrist 视为 base）
        expected_count = 17
        if len(bones) != expected_count:
            raise ValueError(
                f"期望 {expected_count} 个 {hand_prefix}_ body，实际 {len(bones)}。"
                f"已抽到: {sorted(bones.keys())}"
            )

        # 8) palm body 名（root，含 wrist 关节）
        palm_candidates = [
            n for n in bones if n.endswith("_palm") and n.startswith(hand_prefix)
        ]
        if len(palm_candidates) != 1:
            raise ValueError(
                f"找不到唯一的 palm body: {palm_candidates}"
            )
        palm_name = palm_candidates[0]

        # 9) 5 指尖 body 名（按 thumb→pinky 顺序）
        tip_names = tuple(
            f"{hand_prefix}_{finger}_dp" if finger == "thumb"
            else f"{hand_prefix}_{finger}_ip"
            for finger in ("thumb", "index", "middle", "ring", "pinky")
        )
        for tn in tip_names:
            if tn not in bones:
                raise ValueError(f"找不到指尖 body {tn!r}")

        return cls(
            bones=bones,
            qpos_to_body={b.joint_qpos_idx: b.body_id for b in bones.values() if b.joint_qpos_idx >= 0},
            fingertip_body_names=tip_names,
            palm_body_name=palm_name,
        )


__all__ = ["SimBone", "SimSkeleton"]
