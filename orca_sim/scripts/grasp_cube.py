# -*- coding: utf-8 -*-
"""grasp_cube.py —— 在 orca_sim 里演示「右手张开 → 合拢握住 cube」。

零依赖：
- 不需要真机硬件
- 不需要视觉/相机
- 不需要 RL 训练

它做的事（按顺序）：
1. 构造 OrcaHandRightCubeOrientation（v1, skin=False）
2. reset 到默认姿态（cube 在手掌正上方，红色面朝下）
3. 用线性插值把 5 根手指从「全开」慢慢收到「全握」
4. 每步打印：cube 位置、是否被握住、是否触发自碰撞

运行方式（PowerShell / Git Bash 都行）：
    cd C:\\Users\\28422\\Desktop\\internship\\FT\\orca_sim
    .\\orca\\Scripts\\Activate.ps1                     # 激活 venv
    python scripts/grasp_cube.py --mode scripted      # 跑闭环
    python scripts/grasp_cube.py --mode scripted --render human  # 带可视化

可选参数：
    --steps 60            # 收拢步数（默认 60）
    --hold 30             # 收拢后保持步数（默认 30）
    --render {none,human} # 是否开 MuJoCo viewer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows 下 stdout 默认 GBK，强制 utf-8 防止中文报错
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

# 让 src/ 可以直接 import，无需 pip install
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="右手合拢抓 cube 的最小演示脚本")
    p.add_argument("--steps", type=int, default=60,
                   help="从张开到握住的过渡步数（默认 60）")
    p.add_argument("--hold", type=int, default=30,
                   help="握住后保持步数（默认 30）")
    p.add_argument("--render", choices=["none", "human"], default="human",
                   help="是否打开 MuJoCo viewer（默认 human，弹出窗口；none=离屏）")
    p.add_argument("--step-delay", type=float, default=0.1,
                   help="human 模式下每步后 sleep 多少秒（默认 0.1 = 10fps，"
                        "让 viewer 有时间看清；全过程约 (steps+hold)*delay 秒；0=全速）")
    p.add_argument("--use-collision-guard", action="store_true",
                   help="每步跑 CollisionGuard 并打印自碰撞（默认关）")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    from orca_sim.task_envs import OrcaHandRightCubeOrientation

    # render_mode='human' 弹 MuJoCo viewer 窗口；'rgb_array' 离屏
    render_mode = args.render if args.render != "none" else "rgb_array"

    env = OrcaHandRightCubeOrientation(
        version="v1",
        skin=False,
        render_mode=render_mode,
    )
    obs, info = env.reset(options=env.nominal_reset_options())

    # human 模式下窗口可能藏在任务栏，等一下让用户切过去
    if render_mode == "human":
        import time
        print("[hint] MuJoCo viewer 已启动；如果看不到窗口，请检查：")
        print("       1) 任务栏是否有 MuJoCo 图标被最小化")
        print("       2) 是否被别的窗口完全遮挡")
        print("       3) 按 Alt+Tab 切换到 MuJoCo 窗口")
        print("       （3 秒后开始仿真……）")
        time.sleep(3.0)

    # 可选：自碰撞守卫
    guard = None
    if args.use_collision_guard:
        from orca_sim.bridge import CollisionGuard
        guard = CollisionGuard(env, hand="right")

    # ===== 阶段 1：张开 → 握住（线性插值，从 action_high 收到 action_low）=====
    print("=" * 64)
    print(f"[1/2] 阶段 1：张开 → 握住（{args.steps} 步）")
    print(f"      cube 起始位置: {info['cube_pos']}")
    print("=" * 64)

    action_high = env.action_high.astype(np.float32)
    action_low = env.action_low.astype(np.float32)

    closed_contacts: list[tuple[int, float]] = []  # 记录「何时首次检测到 cube 被夹住」

    step_delay = args.step_delay if render_mode == "human" else 0.0
    if step_delay > 0:
        print(f"[hint] 每步间隔 {step_delay*1000:.0f}ms，约 {1/step_delay:.0f}fps；"
              f"全过程约 {(args.steps + args.hold) * step_delay:.1f} 秒")

    for t in range(args.steps):
        alpha = t / max(args.steps - 1, 1)  # 0 → 1
        action = (1 - alpha) * action_high + alpha * action_low

        obs, reward, terminated, truncated, info = env.step(action)

        cube_pos = info["cube_pos"]
        cube_z = float(cube_pos[2])
        n_self = len(guard.self_contacts()) if guard else 0
        flag = ""
        if cube_z < 0.08 and not closed_contacts:
            closed_contacts.append((t, cube_z))
            flag = "  <- cube 进入手掌范围"

        if t % 10 == 0 or flag:
            print(
                f"  step {t:3d}  alpha={alpha:.2f}  "
                f"cube_z={cube_z:+.4f}m  "
                f"success={info['is_success']!s:5s}  "
                f"reward={reward:+.3f}  "
                f"self_collide={n_self}{flag}"
            )
        if terminated or truncated:
            print(f"  episode 结束 @ step {t}: terminated={terminated} truncated={truncated}")
            break

        # 给 viewer 时间渲染（human 模式才有意义）
        if step_delay > 0:
            import time
            time.sleep(step_delay)

    # ===== 阶段 2：保持握姿（验证 cube 不会掉）=====
    print()
    print("=" * 64)
    print(f"[2/2] 阶段 2：保持握姿（{args.hold} 步）")
    print("=" * 64)

    hold_action = action_low  # 全握
    for t in range(args.hold):
        obs, reward, terminated, truncated, info = env.step(hold_action)
        if t % 5 == 0 or t == args.hold - 1:
            print(
                f"  hold {t:3d}/{args.hold}  "
                f"cube_z={float(info['cube_pos'][2]):+.4f}m  "
                f"dropped={info['dropped']!s:5s}  "
                f"reward={reward:+.3f}"
            )
        if info["dropped"]:
            print("  !! cube 掉了，握持失败")
            break
        if step_delay > 0:
            import time
            time.sleep(step_delay)

    # ===== 总结 =====
    print()
    print("=" * 64)
    print("总结")
    print("=" * 64)
    final_pos = info["cube_pos"]
    print(f"最终 cube 位置:    x={final_pos[0]:+.4f}  y={final_pos[1]:+.4f}  z={final_pos[2]:+.4f}")
    print(f"红面朝上对齐:      {info['red_face_up_alignment']:.3f}  (>= cos(15°)≈0.966 算成功)")
    print(f"任务成功 (is_success): {info['is_success']}")
    print(f"cube 是否掉落 (dropped): {info['dropped']}")
    if closed_contacts:
        first_t, first_z = closed_contacts[0]
        print(f"cube 首次进入手掌范围: step {first_t}, z={first_z:+.4f}m")
    if guard:
        n_self = len(guard.self_contacts())
        print(f"Final self-collision count: {n_self} (after fix, cube-floor will NOT be falsely flagged)")

    if render_mode == "human":
        print()
        print("[hint] 仿真已完成。请回到 MuJoCo viewer 窗口查看最终姿态，")
        print("       看够后回到这里按 Enter 关闭窗口。")
        try:
            input()
        except EOFError:
            pass

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())