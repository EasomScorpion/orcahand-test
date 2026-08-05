"""manual_calibrate.py —— 手动校准工具：拖滑块看 sim 手效果。

用途：让用户手动拖动每个 actuator 的滑块，观察 sim 手的真实弯曲效果，
    从而验证 ORCA v1 各关节的实际物理语义（哪个 actuator 控制哪个动作）。

用法：
    ./orca/Scripts/python.exe scripts/manual_calibrate.py
    ./orca/Scripts/python.exe scripts/manual_calibrate.py --no-render  # 不开 viewer

按键：
    q / ESC : 退出
    p       : 打印当前 17 个 actuator 值（写到 terminal）

注：
    - 滑块数量 = 17，每个 1 个 float（rad），范围 = ctrlrange
    - 用 OpenCV 的 trackbar 渲染
    - MuJoCo viewer 同步渲染 3D 手
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# 让 src/ 可直接 import
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _parse_args():
    p = argparse.ArgumentParser(description="手动校准 17 个 actuator 的 sim 手")
    p.add_argument("--no-render", action="store_true", help="不开 MuJoCo viewer")
    p.add_argument("--reset", choices=["zero", "mid", "nominal"], default="nominal",
                   help="初始值：zero=0, mid=ctrlrange 中点, nominal=qpos0")
    return p.parse_args()


# 17 个 actuator 名（必须和 orca_sim v1 right hand 顺序一致）
ACTUATOR_NAMES = [
    "wrist",          # 0
    "thumb_mcp",      # 1
    "thumb_abd",      # 2
    "thumb_pip",      # 3
    "thumb_dip",      # 4
    "index_abd",      # 5
    "index_mcp",      # 6
    "index_pip",      # 7
    "middle_abd",     # 8
    "middle_mcp",     # 9
    "middle_pip",     # 10
    "ring_abd",       # 11
    "ring_mcp",       # 12
    "ring_pip",       # 13
    "pinky_abd",      # 14
    "pinky_mcp",      # 15
    "pinky_pip",      # 16
]


def main():
    args = _parse_args()

    import cv2
    import mujoco
    from orca_sim.envs import OrcaHandRight

    env = OrcaHandRight(
        version="v1", skin=False,
        render_mode="rgb_array" if args.no_render else "human",
    )
    env.reset(seed=42)
    m = env.unwrapped.model
    d = env.unwrapped.data
    ctrlrange = m.actuator_ctrlrange.copy()

    # 初始值
    if args.reset == "zero":
        initial = np.zeros(17)
    elif args.reset == "mid":
        initial = (ctrlrange[:, 0] + ctrlrange[:, 1]) / 2
    else:  # nominal
        initial = d.qpos[:17].copy()

    # 全局变量：当前 17 个值
    state = {"qpos": initial.copy()}

    # OpenCV 窗口 + 17 个滑块
    WINDOW = "calibrate (press 'p' to print, 'q' to quit)"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 500, 800)

    # MuJoCo 同步回调：每帧把 state 写到 d.ctrl
    def sync_to_sim():
        q = state["qpos"]
        # 用 ctrl（position servo）让手跟着 qpos
        d.ctrl[:] = np.clip(q, ctrlrange[:, 0], ctrlrange[:, 1])

    def make_trackbar(i):
        name = ACTUATOR_NAMES[i]
        lo, hi = ctrlrange[i]
        # OpenCV trackbar 只支持 int，把 rad × 1000 当 int
        scale = 1000
        def cb(val):
            state["qpos"][i] = val / scale
        cv2.createTrackbar(name, WINDOW, int(initial[i] * scale), int(hi * scale), cb)
        # lo 可能 > 0 → createTrackbar 不允许负 min？查 OpenCV 文档：允许
        # 但用 setTrackbarMin/Max 设置范围（OpenCV 4.x）
        cv2.setTrackbarMin(name, WINDOW, int(lo * scale))
        cv2.setTrackbarMax(name, WINDOW, int(hi * scale))
        cv2.setTrackbarPos(name, WINDOW, int(initial[i] * scale))

    for i in range(17):
        make_trackbar(i)

    print("=" * 60)
    print("手动校准工具")
    print("=" * 60)
    print("17 个滑块对应 17 个 actuator。")
    print("拖动滑块 → sim 手实时跟随。")
    print("按 'p' 打印当前所有值到终端（方便你抄下来告诉我）。")
    print("按 'q' / ESC 退出。")
    print("=" * 60)
    print()
    print("初始值：")
    for i, name in enumerate(ACTUATOR_NAMES):
        print(f"  {i:2d} {name:12s} = {initial[i]:+.4f} rad  (range [{ctrlrange[i,0]:+.3f}, {ctrlrange[i,1]:+.3f}])")

    # 主循环
    last_print = 0
    while True:
        # 把当前 qpos 写到 ctrl，让 sim 跟
        sync_to_sim()
        # 推物理几步让 servo 收敛（kp=2.0 几步就能跟）
        for _ in range(5):
            mujoco.mj_step(m, d, nstep=1)

        # MuJoCo viewer 渲染（如果开了）
        if env.render_mode == "human":
            env.render()

        # OpenCV 显示一个状态窗口（不然 trackbar 窗口卡住）
        # 但 trackbar 已经有自己的窗口，这里只 sleep 让事件循环跑
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("p"):
            print("\n--- 当前 17 个值 ---")
            for i, name in enumerate(ACTUATOR_NAMES):
                print(f"  {i:2d} {name:12s} = {state['qpos'][i]:+.4f} rad")
            print()

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()