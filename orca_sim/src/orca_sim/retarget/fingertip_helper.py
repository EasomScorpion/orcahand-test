"""fingertip_helper — sim 端 5 指尖世界坐标读取。

v1 右手 MJCF 没有给指尖打 ``<site>`` 标签，但每个手指的 leaf body 有名字：
    - right_thumb_dp    (拇指指尖)
    - right_index_ip    (食指指尖)
    - right_middle_ip   (中指指尖)
    - right_ring_ip     (无名指指尖)
    - right_pinky_ip    (小拇指指尖)

用 ``data.xpos[body_id, :3]`` 直接拿 5 指尖的世界坐标。
"""

from __future__ import annotations

import numpy as np

# 5 指尖 leaf body 名字（顺序固定：thumb → index → middle → ring → pinky）
FINGERTIP_BODY_NAMES: tuple[str, ...] = (
    "right_thumb_dp",
    "right_index_ip",
    "right_middle_ip",
    "right_ring_ip",
    "right_pinky_ip",
)


def fingertip_body_ids(model) -> np.ndarray:
    """返回 5 个指尖 body 在 model.body 数组中的 id（int64，shape=(5,)）。

    顺序对应 :data:`FINGERTIP_BODY_NAMES`。
    """
    return np.array(
        [model.body(name).id for name in FINGERTIP_BODY_NAMES],
        dtype=np.int64,
    )


def fingertip_positions(model, data) -> np.ndarray:
    """返回当前 ``data`` 下 5 个指尖的世界坐标，shape=(5, 3)，单位米。

    Parameters
    ----------
    model : mujoco.MjModel
        orca_sim env 的 ``model``
    data : mujoco.MjData
        orca_sim env 的 ``data``

    Returns
    -------
    np.ndarray
        ``shape=(5, 3)``，5 指尖的 (x, y, z) 世界坐标。
    """
    ids = fingertip_body_ids(model)
    return data.xpos[ids, :3].copy()


def wrist_position(model, data) -> np.ndarray:
    """手腕（palm）世界坐标，shape=(3,)，单位米。"""
    palm_id = model.body("right_palm").id
    return data.xpos[palm_id, :3].copy()