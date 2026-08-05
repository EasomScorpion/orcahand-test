"""sim ctrlrange 与 xdat 范围的对齐检查（advisory）。

orca_sim v1 MJCF 已按 ORCA 真实手逆向建模，理论上 sim ctrlrange 与 xdat 转弧度后的
范围应该粗对齐。本模块在启动时计算两者量程比 + 零点差，给出 WARN 但不阻塞。

启动用法：
    deployer.alignment_report()   # 返回 list[(servo_id, sim_actuator, range_ratio, zero_diff_rad)]
"""

from __future__ import annotations

import math
from typing import Any

# xdat 转弧度的常数（与 xdat_tool / FTServo_Python 文档一致）
_XDAT_DEG_PER_UNIT = 360.0 / 4095.0


def _xdat_rad_range(info: dict[str, float]) -> tuple[float, float]:
    """xdat [min_angle, max_angle] raw → [low, high] rad（不补偿 ofs）。"""
    low_deg = info["raw_low"] * _XDAT_DEG_PER_UNIT
    high_deg = info["raw_high"] * _XDAT_DEG_PER_UNIT
    return low_deg * math.pi / 180.0, high_deg * math.pi / 180.0


def check_alignment(
    mapping: Any, info_by_servo: dict[int, dict[str, float]]
) -> list[tuple[int, str, float, float]]:
    """对每只舵机计算「sim ctrlrange 量程 / xdat 量程」与「零点差」。

    Parameters
    ----------
    mapping : Mapping
        :func:`load_joint_mapping` 返回的对象。
    info_by_servo : dict[int, dict]
        ``SimToRealDeployer._info`` 缓存；key 是 servo_id，value 含
        ``sim_low / sim_high / raw_low / raw_high``。

    Returns
    -------
    list[tuple[int, str, float, float]]
        ``[(servo_id, sim_actuator_name, range_ratio, zero_diff_rad), ...]``。
        ``range_ratio`` ≈ 1 表示量程对得齐；``zero_diff_rad`` ≈ 0 表示零点对得齐。
    """
    rows: list[tuple[int, str, float, float]] = []
    for entry in mapping.values():
        sid = entry.servo_id
        info = info_by_servo[sid]
        sim_range = info["sim_high"] - info["sim_low"]
        xdat_low_rad, xdat_high_rad = _xdat_rad_range(info)
        xdat_range = xdat_high_rad - xdat_low_rad
        range_ratio = (sim_range / xdat_range) if xdat_range > 1e-9 else float("nan")
        zero_diff = info["sim_low"] - xdat_low_rad
        rows.append((sid, entry.sim_actuator, range_ratio, zero_diff))
    return rows


def format_alignment_report(
    rows: list[tuple[int, str, float, float]],
    *,
    ratio_warn_threshold: float = 0.20,
    zero_warn_threshold_rad: float = 0.10,
) -> str:
    """把 ``check_alignment`` 结果格式化为可读的多行字符串，带 WARN 标记。"""
    lines = ["servo_id | sim_actuator            | range_ratio | zero_diff_rad"]
    lines.append("-" * 70)
    for sid, name, ratio, zd in rows:
        warn = ""
        if abs(ratio - 1.0) > ratio_warn_threshold:
            warn += " [RANGE-WARN]"
        if abs(zd) > zero_warn_threshold_rad:
            warn += " [ZERO-WARN]"
        lines.append(
            f"{sid:>7} | {name:<23} | {ratio:>10.3f}  | {zd:>+10.4f}{warn}"
        )
    return "\n".join(lines)