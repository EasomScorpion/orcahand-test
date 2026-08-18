#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sim ↔ 真机 手工校准工具。

目的
----
orca_sim 的 MJCF 是按真机逆向建模的，但 v1 模型 + skin=False 与你的打印件
未必严格对齐。当 sim 滑条在边缘位置滑动时，真机看起来"没反应"——可能原因：

1. **方向反了**（用 ``servo_joint_mapping.json`` 的 ``flip_direction`` 修）
2. **死区**：xdat min/max 范围太窄（< 100 raw，< 8.8°），舵机几乎转不动
3. **中点偏**：xdat 限位不对称中点，导致 sim `rad=0` ≠ 真机 2048
4. **量程不匹配**：sim ctrlrange 与 xdat 转弧度后的范围相差 > 20%

本工具自动把所有这些潜在问题 print 成一张表，并让你**在每只舵机上
采样几个点**（sim 拖到某值 → 真机转到某 raw），自动算出线性 fit 并建议
正确的 ``flip_direction`` 和 zero offset。

用法
----
1. 静态分析（不需要真机）::

       python scripts/calibrate_sim_to_real.py inspect --xdat-dir ../FTServo_Python/参数

   输出每个 servo 的：sim range / xdat range / 中点 / 不对称 / 告警。

2. 交互采样（要 GUI + 真机）::

       # 终端 1：起 GUI
       python FTServo_Python/test/servo_console.py --remote-tcp-port 8765

       # 终端 2：弹 viewer，按提示输入每个采样点的真实 raw
       python scripts/calibrate_sim_to_real.py sample \
           --xdat-dir ../FTServo_Python/参数 \
           --output calibration.csv

   脚本驱动 sim 到 N 个采样点（默认 5 个：sim_low, 25%, 50%, 75%, sim_high），
   对每个 servo 输出「sim rad → 你应该让真机转到 raw = ?」并接收你报告的
   真机 raw；最后把 CSV + 推荐配置写盘。

3. 自动拟合 + 给出 ``flip_direction`` 建议::

       python scripts/calibrate_sim_to_real.py fit --csv calibration.csv

   读 CSV，对每只舵机跑线性 fit：
   - 斜率 > 0 → ``flip_direction=false``（sim 与真机方向一致）
   - 斜率 < 0 → ``flip_direction=true``（翻转）
   - 截距偏离 raw=2048 太大 → 报警「可能需要加 zero offset」
   输出 JSON 建议（可手动合并进 servo_joint_mapping.json）。
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

# 让脚本能 import orca_sim
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================================
# 子命令 1: inspect
# ============================================================================
def cmd_inspect(args: argparse.Namespace) -> int:
    """静态检查 sim ctrlrange ↔ xdat 范围对齐情况。"""
    from orca_sim import OrcaHandRight
    from orca_sim.bridge import build_local_deployer, detailed_alignment, format_detailed_alignment

    env = OrcaHandRight(version="v1", skin=False)
    env.reset(seed=0)
    try:
        deployer = build_local_deployer(
            env=env, xdat_dir=args.xdat_dir, collision_check=False,
        )
        rows = detailed_alignment(deployer.mapping, deployer._info)
        print(format_detailed_alignment(rows))

        # 总结告警
        warned = [r for r in rows if r["flags"]]
        print(f"\n{'='*60}")
        if warned:
            print(f"{len(warned)}/17 servo 有告警：")
            for r in warned:
                print(f"  servo {r['servo_id']:>2} ({r['sim_actuator']}): {r['flags']}")
        else:
            print("全部 17 servo 都在合理范围。")
        return 0
    finally:
        env.close()


# ============================================================================
# 子命令 2: sample（交互）
# ============================================================================
DEFAULT_SAMPLE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass
class SamplePoint:
    """一个采样点：sim 拖到 rad 时期望 raw = ?（待人工报告）。"""
    servo_id: int
    sim_actuator: str
    fraction: float       # 0=sim_low, 1=sim_high
    sim_rad: float
    target_raw: int       # bridge 给的目标 raw
    actual_raw: int | None  # 人工报告的真机实际 raw


def _drive_sim_to(env: Any, qpos_array: np.ndarray) -> None:
    """强制 sim qpos 到指定值（不调用 env.step；只 mj_forward 让 ctrl 跟 qpos 一致）。"""
    import mujoco
    nq = int(env.unwrapped.model.nq)
    q = np.array(env.unwrapped.data.qpos, copy=True)
    q[:nq] = qpos_array[:nq]
    env.unwrapped.data.qpos[:] = q
    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)


def cmd_sample(args: argparse.Namespace) -> int:
    """交互式采样：驱动 sim → 输出期望 raw → 接收人工报告。"""
    from orca_sim import OrcaHandRight
    from orca_sim.bridge import build_local_deployer

    env = OrcaHandRight(version="v1", skin=False)
    env.reset(seed=0)
    try:
        deployer = build_local_deployer(
            env=env, xdat_dir=args.xdat_dir, collision_check=False,
        )

        nq = int(env.unwrapped.model.nq)
        # 每个 servo 的 qpos index
        fractions = DEFAULT_SAMPLE_FRACTIONS

        # 第一遍：构造所有 SamplePoint（target_raw 由 bridge 算）
        all_points: list[SamplePoint] = []
        # 对每个采样点：把每个 servo 都设到对应 fraction 位置
        for frac in fractions:
            for entry in deployer.mapping.values():
                sid = entry.servo_id
                info = deployer._info[sid]
                sim_idx = info["sim_idx"]
                sim_rad = info["sim_low"] + frac * (info["sim_high"] - info["sim_low"])
                # 构造 qpos
                qpos = np.array(env.unwrapped.data.qpos, copy=True)
                if sim_idx < nq:
                    qpos[sim_idx] = sim_rad
                # 让所有其它 servo 保持在当前值（避免自碰撞）
                # 计算 raw
                target_raw = deployer._rad_to_raw(info, sim_rad)
                all_points.append(SamplePoint(
                    servo_id=sid,
                    sim_actuator=entry.sim_actuator,
                    fraction=frac,
                    sim_rad=float(sim_rad),
                    target_raw=int(target_raw),
                    actual_raw=None,
                ))

        # 打印人类可读的表头
        print(f"Will drive sim to {len(fractions)} fractions × 17 servos = {len(all_points)} sample points")
        print(f"Output CSV → {args.output}\n")
        print("=" * 80)
        print(f"{'sid':>3} | {'fraction':>8} | {'sim_rad':>+9} | {'target_raw':>9} | {'actual_raw':>10}")
        print("-" * 80)

        actual_rows: list[dict[str, Any]] = []
        for pt in all_points:
            # 把 sim 拖到 pt.sim_rad（仅这一只 servo 动，其它保持）
            qpos = np.array(env.unwrapped.data.qpos, copy=True)
            info = deployer._info[pt.servo_id]
            sim_idx = info["sim_idx"]
            if sim_idx < nq:
                qpos[sim_idx] = pt.sim_rad
            _drive_sim_to(env, qpos)

            # 提示用户 + 接收输入
            print(f"{pt.servo_id:>3} | {pt.fraction:>8.2f} | {pt.sim_rad:>+9.4f} | {pt.target_raw:>9} | ", end="", flush=True)
            line = sys.stdin.readline().strip()
            if not line:
                actual_raw = None
            else:
                try:
                    actual_raw = int(line)
                except ValueError:
                    actual_raw = None
            pt.actual_raw = actual_raw
            actual_rows.append({
                "servo_id": pt.servo_id,
                "sim_actuator": pt.sim_actuator,
                "fraction": pt.fraction,
                "sim_rad": pt.sim_rad,
                "target_raw": pt.target_raw,
                "actual_raw": actual_raw if actual_raw is not None else "",
            })
            print()

        # 写 CSV
        with args.output.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["servo_id", "sim_actuator", "fraction", "sim_rad", "target_raw", "actual_raw"],
            )
            w.writeheader()
            w.writerows(actual_rows)
        print(f"\nSaved {len(actual_rows)} rows → {args.output}")
        print(f"下一步：python scripts/calibrate_sim_to_real.py fit --csv {args.output}")
        return 0
    finally:
        env.close()


# ============================================================================
# 子命令 3: fit（自动分析 CSV）
# ============================================================================
@dataclass
class ServoFitResult:
    servo_id: int
    sim_actuator: str
    n_points: int                # 有效点数
    slope: float                 # raw / rad；> 0 表示 sim 与真机方向一致
    intercept: float             # rad=0 时的 raw
    r_squared: float             # 拟合度，1.0 = 完美线性
    suggested_flip: bool         # 推荐的 flip_direction 值
    zero_offset_raw: float       # 推荐补偿的零偏移 (rad → raw 偏移)
    warnings: list[str]


def _linear_fit(sim_rads: list[float], actual_raws: list[int]) -> tuple[float, float, float]:
    """最小二乘线性 fit：raw = slope * rad + intercept。返回 (slope, intercept, R²)。"""
    x = np.asarray(sim_rads, dtype=np.float64)
    y = np.asarray(actual_raws, dtype=np.float64)
    # y = a*x + b
    a, b = np.polyfit(x, y, 1)
    y_pred = a * x + b
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 1.0
    return float(a), float(b), r2


def _analyze_servo(sid: int, sim_actuator: str, points: list[dict]) -> ServoFitResult:
    """分析一个 servo 的采样点，给出 fit + 建议。"""
    valid = [(float(p["sim_rad"]), int(p["actual_raw"])) for p in points if p["actual_raw"] != ""]
    warnings: list[str] = []
    if len(valid) < 2:
        return ServoFitResult(
            servo_id=sid, sim_actuator=sim_actuator,
            n_points=len(valid),
            slope=0.0, intercept=0.0, r_squared=0.0,
            suggested_flip=True, zero_offset_raw=0.0,
            warnings=["数据点不足（< 2），无法拟合"],
        )

    sim_rads, actual_raws = zip(*valid)
    slope, intercept, r2 = _linear_fit(list(sim_rads), list(actual_raws))

    # 方向：slope > 0 → sim 与真机方向一致 → flip_direction = false
    suggested_flip = slope < 0
    # 截距偏离 raw=2048（中点）的差
    zero_offset_raw = intercept - 2048.0
    if abs(zero_offset_raw) > 50:
        warnings.append(f"zero offset 偏离 {zero_offset_raw:+.0f} raw")
    if r2 < 0.95:
        warnings.append(f"R²={r2:.3f} < 0.95，非线性或采样点异常")
    if abs(slope) < 100:
        warnings.append(f"|slope|={abs(slope):.0f} 太小，舵机可能几乎不动")
    if abs(slope) > 10000:
        warnings.append(f"|slope|={abs(slope):.0f} 太大，可能填错单位")

    return ServoFitResult(
        servo_id=sid, sim_actuator=sim_actuator,
        n_points=len(valid),
        slope=slope, intercept=intercept, r_squared=r2,
        suggested_flip=suggested_flip, zero_offset_raw=zero_offset_raw,
        warnings=warnings,
    )


def cmd_fit(args: argparse.Namespace) -> int:
    """读 CSV，对每只舵机跑线性 fit，输出 JSON 建议。"""
    if not args.csv.exists():
        print(f"CSV 不存在：{args.csv}", file=sys.stderr)
        return 1

    # 读 CSV
    rows: list[dict[str, str]] = []
    with args.csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 按 servo 分组
    by_servo: dict[int, list[dict]] = {}
    for r in rows:
        sid = int(r["servo_id"])
        by_servo.setdefault(sid, []).append(r)

    # 对每只舵机 fit
    results: list[ServoFitResult] = []
    for sid in sorted(by_servo.keys()):
        points = by_servo[sid]
        sim_actuator = points[0]["sim_actuator"] if points else "?"
        results.append(_analyze_servo(sid, sim_actuator, points))

    # 打印汇总
    print(f"{'sid':>3} | {'sim_actuator':<23} | {'n':>2} | {'slope':>9} | {'intercept':>9} | {'R²':>5} | {'flip':>4} | {'Δzero':>7} | warnings")
    print("-" * 110)
    for r in results:
        warn_str = "; ".join(r.warnings) if r.warnings else ""
        print(
            f"{r.servo_id:>3} | {r.sim_actuator:<23} | "
            f"{r.n_points:>2} | {r.slope:>+9.1f} | {r.intercept:>+9.1f} | "
            f"{r.r_squared:>5.3f} | "
            f"{'Y' if r.suggested_flip else 'N':>4} | "
            f"{r.zero_offset_raw:>+7.0f} | {warn_str}"
        )

    # 写 JSON 建议
    if args.output_json:
        suggestion = {
            "comment": "由 calibrate_sim_to_real.py fit 自动生成；手动合并进 servo_joint_mapping.json",
            "servo_overrides": {
                str(r.servo_id): {
                    "sim_actuator": r.sim_actuator,
                    "flip_direction": r.suggested_flip,
                    "zero_offset_raw_recommended": round(r.zero_offset_raw, 1),
                    "r_squared": round(r.r_squared, 3),
                    "n_points": r.n_points,
                    "warnings": r.warnings,
                }
                for r in results
            },
        }
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(suggestion, f, ensure_ascii=False, indent=2)
        print(f"\n建议写入：{args.output_json}")
        print("    手动合并进 servo_joint_mapping.json（注意当前 mapping 没存 flip_direction 字段，")
        print("    请按前面 todo 加上的方式写入）")
    return 0


# ============================================================================
# CLI
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibrate_sim_to_real",
        description="sim ↔ 真机 手工校准：静态检查 / 交互采样 / 自动拟合。",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inspect = sub.add_parser("inspect", help="静态检查 sim ctrlrange ↔ xdat 对齐")
    p_inspect.add_argument(
        "--xdat-dir", type=pathlib.Path,
        default=REPO_ROOT.parent / "FTServo_Python" / "参数",
    )
    p_inspect.set_defaults(func=cmd_inspect)

    p_sample = sub.add_parser("sample", help="交互采样（要 GUI + 真机）")
    p_sample.add_argument(
        "--xdat-dir", type=pathlib.Path,
        default=REPO_ROOT.parent / "FTServo_Python" / "参数",
    )
    p_sample.add_argument(
        "--output", type=pathlib.Path, default=REPO_ROOT / "calibration_data.csv",
    )
    p_sample.set_defaults(func=cmd_sample)

    p_fit = sub.add_parser("fit", help="读 CSV → 拟合 → 输出 JSON 建议")
    p_fit.add_argument("--csv", type=pathlib.Path, required=True)
    p_fit.add_argument(
        "--output-json", type=pathlib.Path,
        default=REPO_ROOT / "calibration_suggestion.json",
    )
    p_fit.set_defaults(func=cmd_fit)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())