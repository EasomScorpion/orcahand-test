"""17 舵机 ↔ orca_sim v1 右手 actuator 的映射加载与查询。

数据源：
    :file:`orca_sim/src/orca_sim/bridge/data/servo_joint_mapping.json`
    （由项目维护；目前 v1 右手 + skin=False + 17 STS3215）

设计：
    - JSON 中 ``servo_id_to_actuator`` 的 key 是字符串形式的 servo_id（"1".."17"）。
    - 每个 entry 含 ``chinese`` / ``sim_actuator`` / ``joint_role``。
    - bridge 不硬编码 sim index，启动时通过 ``mujoco`` 的 ``model.actuator(name).id`` 解析。
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any, Mapping as MappingType

# JSON 资源路径（与本文件同目录）
_DATA_PATH = pathlib.Path(__file__).parent / "data" / "servo_joint_mapping.json"


@dataclass(frozen=True)
class ServoEntry:
    """单个 servo_id 的映射条目。"""

    servo_id: int               # 1..17
    chinese: str                # 中文关节名（仅展示）
    sim_actuator: str           # orca_sim MJCF 中的 actuator 名
    joint_role: str             # pip / mcp / abd / wrist
    flip_direction: bool = True # 是否翻转 rad→raw 映射的方向（详见 deploy.py）


# dict 类型别名，方便静态检查与类型标注
Mapping = MappingType[str, ServoEntry]


def load_joint_mapping(path: pathlib.Path | None = None) -> Mapping:
    """加载 17 舵机映射表。

    Parameters
    ----------
    path : 可选，自定义 JSON 路径；默认使用 :file:`bridge/data/servo_joint_mapping.json`。

    Returns
    -------
    Mapping[str, ServoEntry]
        以 ``str(servo_id)`` 为键的字典，例如 ``{"1": ServoEntry(servo_id=1, chinese="食指指尖", sim_actuator="right_index_pip", joint_role="pip"), ...}``。

    Raises
    ------
    FileNotFoundError
        JSON 文件不存在。
    ValueError
        JSON 内容不合法（缺字段、servo_id 越界、重复）。
    """
    p = pathlib.Path(path) if path else _DATA_PATH
    if not p.exists():
        raise FileNotFoundError(f"servo joint mapping JSON not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    table = raw.get("servo_id_to_actuator")
    if not isinstance(table, dict):
        raise ValueError(f"missing 'servo_id_to_actuator' in {p}")

    out: dict[str, ServoEntry] = {}
    for sid_str, info in table.items():
        if not isinstance(info, dict):
            raise ValueError(f"bad entry for servo_id={sid_str}")
        for required in ("chinese", "sim_actuator", "joint_role"):
            if required not in info:
                raise ValueError(
                    f"servo_id={sid_str} missing field {required!r}"
                )
        try:
            sid = int(sid_str)
        except (TypeError, ValueError):
            raise ValueError(f"bad servo_id key: {sid_str!r}")
        if sid < 1 or sid > 17:
            raise ValueError(f"servo_id {sid} out of range 1..17")
        if sid_str in out:
            raise ValueError(f"duplicate servo_id {sid_str}")
        out[sid_str] = ServoEntry(
            servo_id=sid,
            chinese=str(info["chinese"]),
            sim_actuator=str(info["sim_actuator"]),
            joint_role=str(info["joint_role"]),
            flip_direction=bool(info.get("flip_direction", True)),
        )

    # 校验：1..17 必须全部存在
    missing = [str(i) for i in range(1, 18) if str(i) not in out]
    if missing:
        raise ValueError(f"missing servo_ids in mapping: {missing}")

    return out


# ----------------------------------------------------------------------
# 运行时查询辅助
# ----------------------------------------------------------------------
def resolve_sim_indices(
    mapping: Mapping, model: Any
) -> dict[int, int]:
    """把映射表解析为 ``{servo_id: sim_actuator_index}``。

    MuJoCo 的 actuator name 在 MJCF 中带 ``_actuator`` 后缀（如
    ``right_index_pip_actuator``），而本模块的 mapping 用的是 joint 名
    （``right_index_pip``）。这里先尝试 joint 名，再尝试 ``<name>_actuator``。

    Parameters
    ----------
    mapping : Mapping
        :func:`load_joint_mapping` 返回的对象。
    model : mujoco.MjModel
        orca_sim 的 ``env.unwrapped.model``。

    Returns
    -------
    dict[int, int]
        ``{servo_id (1..17): sim_actuator_index (0..16)}``。
    """
    out: dict[int, int] = {}
    for sid_str, entry in mapping.items():
        # 先尝试 joint 名，再尝试 <name>_actuator 形式
        candidates = (entry.sim_actuator, f"{entry.sim_actuator}_actuator")
        sim_idx = None
        for cand in candidates:
            try:
                sim_idx = int(model.actuator(cand).id)
                break
            except KeyError:
                continue
        if sim_idx is None:
            raise KeyError(
                f"servo_id={sid_str} sim_actuator={entry.sim_actuator!r} "
                f"not found in model (tried {candidates})"
            )
        out[entry.servo_id] = sim_idx
    return out


def sim_actuator_for_servo(mapping: Mapping, servo_id: int) -> str:
    """反向查表：servo_id → sim actuator 名。"""
    sid_str = str(servo_id)
    if sid_str not in mapping:
        raise KeyError(f"unknown servo_id {servo_id}; valid: 1..17")
    return mapping[sid_str].sim_actuator