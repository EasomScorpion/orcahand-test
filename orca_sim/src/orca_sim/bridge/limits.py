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
    """xdat [min_angle, max_angle] raw → [low, high] rad（不补偿 ofs）。

    注意：``info['raw_low']`` / ``info['raw_high']`` 在方向翻转（默认）下是
    颠倒的（``raw_low`` 实际存的是 ``xdat.max_angle``）；这里用
    ``min/max`` 始终保证输出「小值→大值」的弧度区间。
    """
    raw_lo = float(info["raw_low"])
    raw_hi = float(info["raw_high"])
    raw_min, raw_max = (raw_lo, raw_hi) if raw_lo <= raw_hi else (raw_hi, raw_lo)
    low_deg = raw_min * _XDAT_DEG_PER_UNIT
    high_deg = raw_max * _XDAT_DEG_PER_UNIT
    return low_deg * math.pi / 180.0, high_deg * math.pi / 180.0


def check_alignment(
    mapping: Any, info_by_servo: dict[int, dict[str, float]]
) -> list[tuple[int, str, float, float]]:
    """对每只舵机计算「sim ctrlrange 量程 / xdat 量程」与「中点差」。

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
        ``[(servo_id, sim_actuator_name, range_ratio, mid_diff_rad), ...]``。
        ``range_ratio`` ≈ 1 表示量程对得齐；``mid_diff_rad`` 是「sim 中点
        对应的 raw 减去 xdat 中点 raw」再换回弧度的差，≈ 0 表示中点对得齐。

    .. note::
        旧版本这里返回 ``zero_diff_rad = sim_low - xdat_low_rad``，但
        sim_low 与 xdat_low 在不同坐标系下相减没物理意义；现在改为
        mid_diff_rad（量纲一致），更直观。
    """
    rows: list[tuple[int, str, float, float]] = []
    for entry in mapping.values():
        sid = entry.servo_id
        info = info_by_servo[sid]
        sim_range = info["sim_high"] - info["sim_low"]
        xdat_low_rad, xdat_high_rad = _xdat_rad_range(info)
        xdat_range = xdat_high_rad - xdat_low_rad
        range_ratio = (sim_range / xdat_range) if xdat_range > 1e-9 else float("nan")
        sim_mid = (info["sim_low"] + info["sim_high"]) / 2.0
        xdat_mid_rad = (xdat_low_rad + xdat_high_rad) / 2.0
        mid_diff = sim_mid - xdat_mid_rad
        rows.append((sid, entry.sim_actuator, range_ratio, mid_diff))
    return rows


def detailed_alignment(
    mapping: Any, info_by_servo: dict[int, dict[str, float]]
) -> list[dict[str, Any]]:
    """每个 servo 的详细诊断（sim range / raw range / mid / 不对称 / 死区告警）。

    返回 list[dict]，每个 dict 包含：
        servo_id, sim_actuator, chinese, flip_direction,
        sim_low, sim_high, sim_range, sim_mid,
        raw_low, raw_high, raw_range, raw_mid,
        xdat_min, xdat_max, xdat_mid_rad,
        range_ratio, mid_diff_rad,
        asymmetry_ratio,           # 0 = 完全对称中点；越大越偏
        flags                      # str，空格分隔的告警 [WARN_RATIO, WARN_ZERO,
                                   # WARN_DEADZONE, WARN_ASYMMETRY, WARN_MID_MISMATCH]
    """
    out: list[dict[str, Any]] = []
    for entry in mapping.values():
        sid = entry.servo_id
        info = info_by_servo[sid]
        sim_low = float(info["sim_low"])
        sim_high = float(info["sim_high"])
        raw_low = float(info["raw_low"])
        raw_high = float(info["raw_high"])
        sim_range = sim_high - sim_low
        raw_range = abs(raw_high - raw_low)
        sim_mid = (sim_low + sim_high) / 2.0
        raw_mid = (raw_low + raw_high) / 2.0
        xdat_low_rad, xdat_high_rad = _xdat_rad_range(info)
        xdat_mid_rad = (xdat_low_rad + xdat_high_rad) / 2.0
        xdat_range = xdat_high_rad - xdat_low_rad
        range_ratio = (sim_range / xdat_range) if xdat_range > 1e-9 else float("nan")
        mid_diff_rad = sim_mid - xdat_mid_rad

        # 死区检查：raw_range < 100（< 8.8°）时舵机几乎转不动
        dead_zone = raw_range < 100
        # 不对称度：raw_mid 偏离 2048 的程度
        asymmetry_ratio = abs(raw_mid - 2048.0) / max(raw_range / 2.0, 1.0)

        # 中点匹配：sim_mid 应该映射到 raw_mid（容差 ±5 raw）
        raw_at_sim_mid = (
            raw_low + (sim_mid - sim_low) / max(sim_range, 1e-9) * (raw_high - raw_low)
        )
        mid_diff = abs(raw_at_sim_mid - raw_mid)

        flags: list[str] = []
        if abs(range_ratio - 1.0) > 0.20:
            flags.append("WARN_RATIO")
        if abs(mid_diff_rad) > 0.10:
            flags.append("WARN_ZERO")
        if dead_zone:
            flags.append("WARN_DEADZONE")
        if asymmetry_ratio > 0.30:
            flags.append("WARN_ASYMMETRY")
        if mid_diff > 5.0:
            flags.append("WARN_MID_MISMATCH")

        out.append({
            "servo_id": sid,
            "sim_actuator": entry.sim_actuator,
            "chinese": entry.chinese,
            "flip_direction": entry.flip_direction,
            "sim_low": sim_low,
            "sim_high": sim_high,
            "sim_range": sim_range,
            "sim_mid": sim_mid,
            "raw_low": raw_low,
            "raw_high": raw_high,
            "raw_range": raw_range,
            "raw_mid": raw_mid,
            "xdat_min": int(min(raw_low, raw_high)),
            "xdat_max": int(max(raw_low, raw_high)),
            "xdat_mid_rad": xdat_mid_rad,
            "range_ratio": range_ratio,
            "mid_diff_rad": mid_diff_rad,
            "asymmetry_ratio": asymmetry_ratio,
            "mid_diff_raw": mid_diff,
            "flags": " ".join(flags),
        })
    return out


def format_detailed_alignment(rows: list[dict[str, Any]]) -> str:
    """把 :func:`detailed_alignment` 的结果格式化成可读的多行字符串。"""
    lines = [
        "servo | sim_actuator            | flip | sim_range   | raw_range    | "
        "ratio  | mid_diff | asym   | flags"
    ]
    lines.append("-" * 110)
    for r in rows:
        sid = r["servo_id"]
        lines.append(
            f"{sid:>5} | {r['sim_actuator']:<23} | "
            f"{'Y' if r['flip_direction'] else 'N':<4} | "
            f"{r['sim_range']:>+.4f} | "
            f"{int(r['xdat_min']):>4}-{int(r['xdat_max']):<4} ({int(r['raw_range']):>4}) | "
            f"{r['range_ratio']:>5.2f} | "
            f"{r['mid_diff_rad']:>+8.4f} | "
            f"{r['asymmetry_ratio']:>5.2f} | "
            f"{r['flags']}"
        )
    return "\n".join(lines)


def format_alignment_report(
    rows: list[tuple[int, str, float, float]],
    *,
    ratio_warn_threshold: float = 0.20,
    mid_warn_threshold_rad: float = 0.10,
) -> str:
    """把 ``check_alignment`` 结果格式化为可读的多行字符串，带 WARN 标记。"""
    lines = ["servo_id | sim_actuator            | range_ratio | mid_diff_rad"]
    lines.append("-" * 70)
    for sid, name, ratio, md in rows:
        warn = ""
        if abs(ratio - 1.0) > ratio_warn_threshold:
            warn += " [RANGE-WARN]"
        if abs(md) > mid_warn_threshold_rad:
            warn += " [MID-WARN]"
        lines.append(
            f"{sid:>7} | {name:<23} | {ratio:>10.3f}  | {md:>+10.4f}{warn}"
        )
    return "\n".join(lines)