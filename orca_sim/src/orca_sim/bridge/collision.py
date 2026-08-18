"""sim 自碰撞守卫：从 ``env.data.contact`` 过滤出同只手内部的自碰撞。

orca_sim v1 MJCF 已对相邻手指段做了 ``<exclude>``，所以这里读到的接触
都是「真正的」自碰撞，而不是相邻段的假阳性。
用户构造 env 时需显式传 ``skin=False``，让骨头 geom 参与碰撞（与真机一致）。

设计：
    - 阈值 ``penetration_threshold``：只把 ``dist < threshold`` 的接触视为真碰撞；
      MuJoCo 接触求解有微小松弛，纯粹贴合（dist≈0）不算。
    - **按 geom 所属 body 的名字过滤**：v1 右手的 geom 名都是空字符串（``<geom>`` 节点
      没起名），但所属 body（``right_tower`` / ``right_palm`` / ``right_thumb_mp`` …）
      都有名字。cube / floor / 左手 / 世界的 body 不以 ``{hand}_`` 开头。
    - **必须两侧 body 都属于目标 hand** 才算自碰撞；这同时排除了
      right↔cube、right↔floor、左手↔右手、左手↔cube 等非自碰撞接触对。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContactPair:
    """一个自碰撞接触对。"""

    body1: str  # 所属 body 名（比 geom 名更有意义，因为 v1 的 geom 多为无名）
    body2: str
    dist: float  # 负值 = 穿透深度


class CollisionGuard:
    """sim 自碰撞守卫（按 geom 所属 body 过滤）。

    Parameters
    ----------
    env : gymnasium.Env
        orca_sim 的 env（任意 ``OrcaHandXxx`` 子类）；可访问 ``env.unwrapped.model`` / ``data``。
    hand : str
        目标手（``"right"`` / ``"left"``）。orca_sim v1 MJCF 中右手 body 以 ``right_`` 开头，
        左手 ``left_`` 开头。默认 ``"right"``。
    penetration_threshold : float
        穿透深度阈值（负数）。``dist < threshold`` 才视为真碰撞。默认 ``-1e-4``。
    """

    def __init__(
        self,
        env: Any,
        *,
        hand: str = "right",
        penetration_threshold: float = -1e-4,
    ) -> None:
        if hand not in ("right", "left"):
            raise ValueError(f"hand must be 'right' or 'left', got {hand!r}")
        self.env = env
        self.hand = hand
        self.threshold = float(penetration_threshold)

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def _body_name(self, model: Any, geom_id: int) -> str:
        """把 geom_id 解析为所属 body 的名字（无名 body 用 ``geom_{id}`` 占位）。"""
        body_id = int(model.geom_bodyid[geom_id])
        name = model.body(body_id).name
        return name if name else f"geom_{geom_id}"

    def self_contacts(self) -> list[ContactPair]:
        """返回所有「真自碰撞」接触对。

        判定条件：两侧 geom 所属 body 都以 ``{hand}_`` 开头。
        """
        out: list[ContactPair] = []
        model = self.env.unwrapped.model
        data = self.env.unwrapped.data
        prefix = f"{self.hand}_"
        ncon = int(data.ncon)
        for i in range(ncon):
            c = data.contact[i]
            dist = float(c.dist)
            if dist >= self.threshold:
                continue
            b1 = self._body_name(model, int(c.geom1))
            b2 = self._body_name(model, int(c.geom2))
            if not (b1.startswith(prefix) and b2.startswith(prefix)):
                continue
            out.append(ContactPair(body1=b1, body2=b2, dist=dist))
        return out

    def is_self_colliding(self) -> bool:
        """是否有任何自碰撞。"""
        return len(self.self_contacts()) > 0