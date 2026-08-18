#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sim → 真机一比一执行 CLI。

三种模式：
    1. demo（默认）：sim 跑随机 action，每步经 safe_deploy_action。
    2. policy：加载已训练策略 rollout，每步经 safe_deploy_action。
    3. single-pose：从 --target-json 读 17 维弧度姿态，下发一次。

公共参数：
    --xdat-dir       FTServo_Python/参数 目录（默认 ../FTServo_Python/参数）
    --hand {right,left}  碰撞过滤目标（默认 right）
    --collision-check/--no-collision-check  默认开启
    --dry-run        不连真机
    --port COMx      连真机的串口
    --baud 1000000   波特率
    --render-mode {human,rgb_array}  默认 None（无渲染）
    --seed           随机种子
    --sync-every N   policy 模式下每 N 步同步下发一次（默认 1）
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np


# ----------------------------------------------------------------------
# 启动辅助
# ----------------------------------------------------------------------
def _make_env(hand: str, render_mode: str | None, skin: bool):
    """构造 orca_sim env。hand=right → OrcaHandRight；hand=left → OrcaHandLeft。"""
    if hand == "right":
        from orca_sim.envs import OrcaHandRight
        return OrcaHandRight(version="v1", render_mode=render_mode, skin=skin)
    else:
        from orca_sim.envs import OrcaHandLeft
        return OrcaHandLeft(version="v1", render_mode=render_mode, skin=skin)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sim_to_real",
        description="orca_sim 训练结果 → FTServo 真机 17 舵机同步执行",
    )
    p.add_argument(
        "--xdat-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[3] / "FTServo_Python" / "参数",
        help="FTServo_Python/参数 目录路径（默认 ../FTServo_Python/参数）",
    )
    p.add_argument("--hand", choices=["right", "left"], default="right")
    p.add_argument("--dry-run", action="store_true", help="不连真机，用 mock backend")
    p.add_argument("--port", default=None, help="串口名（如 COM5）；非 dry-run 时必填")
    p.add_argument("--baud", type=int, default=1_000_000, help="波特率")
    p.add_argument("--render-mode", choices=[None, "human", "rgb_array"], default=None)
    p.add_argument("--collision-check", dest="collision_check", action="store_true", default=True)
    p.add_argument("--no-collision-check", dest="collision_check", action="store_false")
    p.add_argument("--seed", type=int, default=0)

    # 三种模式
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--random-steps", type=int, default=10, help="demo 模式：随机 action 步数")
    mode.add_argument("--policy-path", type=pathlib.Path, default=None, help="policy 模式：策略文件路径")
    mode.add_argument("--target-json", type=pathlib.Path, default=None, help="single-pose 模式：17 维弧度 JSON")

    p.add_argument("--sync-every", type=int, default=1, help="policy 模式下每 N 步同步下发一次")
    p.add_argument("--verbose", action="store_true", help="打印每步详细信息")
    return p


# ----------------------------------------------------------------------
# demo 模式
# ----------------------------------------------------------------------
def _run_demo(deployer, env, steps: int, verbose: bool) -> int:
    """随机 action + safe_deploy_action。返回跳过的步数。"""
    rng = np.random.default_rng(0)
    skipped = 0
    deployed = 0
    for i in range(steps):
        action = rng.uniform(env.action_low, env.action_high).astype(np.float32)
        ok, info = deployer.safe_deploy_action(action)
        if not ok:
            skipped += 1
            if verbose:
                names = [c.geom1 for c in info]
                print(f"[step {i:>3}] SKIPPED self-collision: {names}")
        else:
            deployed += 1
            if verbose:
                short = {k: v for k, v in list(info.items())[:3]}
                print(f"[step {i:>3}] DEPLOYED ({len(info)} servos): head={short}...")
        if env.render_mode == "human":
            env.render()
    print(f"[demo] steps={steps} deployed={deployed} skipped={skipped}")
    return skipped


# ----------------------------------------------------------------------
# single-pose 模式
# ----------------------------------------------------------------------
def _run_single_pose(deployer, env, pose_path: pathlib.Path, verbose: bool) -> dict[int, int]:
    """从 JSON 读 17 维弧度（数组或 {joint: rad}），下发一次。"""
    with pose_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    model = env.unwrapped.model
    nq = int(model.nq)
    if isinstance(data, list):
        qpos = np.asarray(data, dtype=np.float64)
    elif isinstance(data, dict):
        # {sim_actuator_name: rad} → 按 sim index 排序成 qpos
        from orca_sim.bridge.mapping import load_joint_mapping
        mapping = load_joint_mapping()
        qpos = np.zeros(nq, dtype=np.float64)
        # 先填 actuator 部分
        for sid_str, entry in mapping.items():
            idx = int(model.actuator(entry.sim_actuator).id)
            key_options = [entry.sim_actuator, entry.chinese, sid_str]
            chosen = None
            for k in key_options:
                if k in data:
                    chosen = data[k]
                    break
            if chosen is None:
                continue
            qpos[idx] = float(chosen)
    else:
        raise ValueError(f"--target-json must be list or dict, got {type(data)}")

    env.reset(options={"qpos": qpos})
    return deployer.deploy_qpos(env.unwrapped.data.qpos[:nq])


# ----------------------------------------------------------------------
# policy 模式
# ----------------------------------------------------------------------
def _load_policy(path: pathlib.Path):
    """加载策略；支持 numpy .npy（动作序列）或 torch .pt（callable）。"""
    if path.suffix == ".npy":
        arr = np.load(str(path))
        # 期望 shape (T, action_dim)
        if arr.ndim != 2:
            raise ValueError(f".npy policy must be 2D, got shape {arr.shape}")
        return arr
    elif path.suffix == ".pt":
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise RuntimeError(f"加载 .pt 需要 torch: {e}")
        import torch
        obj = torch.load(str(path), map_location="cpu")
        if callable(obj):
            return obj
        # 兼容：state dict → 让用户在外面包
        raise ValueError(f".pt 必须是 callable，目前是 {type(obj)}")
    else:
        raise ValueError(f"不支持的策略格式: {path.suffix}")


def _run_policy(deployer, env, policy, sync_every: int, verbose: bool) -> int:
    """运行 policy rollout；np.ndarray → 取序列；callable → 每步调用。"""
    skipped = 0
    deployed = 0
    step_idx = 0

    if isinstance(policy, np.ndarray):
        for action in policy:
            action = np.asarray(action, dtype=np.float32)
            if (step_idx % sync_every) == 0:
                ok, info = deployer.safe_deploy_action(action)
                if not ok:
                    skipped += 1
                else:
                    deployed += 1
                    if verbose:
                        print(f"[step {step_idx:>3}] DEPLOYED")
            else:
                # 仅 sim step，不下发
                env.step(action)
            if env.render_mode == "human":
                env.render()
            step_idx += 1
    elif callable(policy):
        obs, _ = env.reset()
        while True:
            try:
                action = policy(obs)
                action = np.asarray(action, dtype=np.float32)
            except StopIteration:
                break
            if (step_idx % sync_every) == 0:
                ok, info = deployer.safe_deploy_action(action)
                if not ok:
                    skipped += 1
                else:
                    deployed += 1
            else:
                env.step(action)
            obs, *_ = env.step(np.zeros_like(env.action_low))  # 占位；下一轮 policy 会再覆盖
            if env.render_mode == "human":
                env.render()
            step_idx += 1
    else:
        raise ValueError(f"未知 policy 类型: {type(policy)}")

    print(f"[policy] steps={step_idx} deployed={deployed} skipped={skipped}")
    return skipped


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def main() -> int:
    args = _build_argparser().parse_args()

    if not args.dry_run and not args.port:
        print("ERROR: 必须指定 --port 或 --dry-run", file=sys.stderr)
        return 2

    # 构造 env
    env = _make_env(args.hand, args.render_mode, skin=False)
    env.reset(seed=args.seed)

    # 构造 deployer（通过工厂）
    try:
        from orca_sim.bridge import build_deployer, load_joint_mapping
        mapping = load_joint_mapping()
        deployer, env = build_deployer(
            env_factory=lambda: _make_env(args.hand, args.render_mode, skin=False),
            xdat_dir=args.xdat_dir,
            port=args.port,
            baud=args.baud,
            mapping=mapping,
            collision_check=args.collision_check,
            dry_run=args.dry_run,
        )
        deployer.env.reset(seed=args.seed)
    except Exception as e:
        print(f"ERROR: 构造 deployer 失败: {e}", file=sys.stderr)
        return 1

    # 打印对齐报告（advisory）
    report = deployer.alignment_report()
    print("=== alignment report ===")
    for sid, name, ratio, zd in report:
        flag = "WARN" if (abs(ratio - 1.0) > 0.20 or abs(zd) > 0.10) else "ok"
        print(f"  sid={sid:>2} {name:<23} range_ratio={ratio:.3f} zero_diff={zd:+.4f}  [{flag}]")
    print()

    # 分发模式
    try:
        if args.target_json is not None:
            positions = _run_single_pose(deployer, env, args.target_json, args.verbose)
            print(f"[single-pose] deployed: {dict(list(positions.items())[:5])}...")
        elif args.policy_path is not None:
            policy = _load_policy(args.policy_path)
            _run_policy(deployer, env, policy, args.sync_every, args.verbose)
        else:
            _run_demo(deployer, env, args.random_steps, args.verbose)
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())