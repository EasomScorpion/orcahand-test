"""webcam_grasp.py —— 端到端：摄像头手势 → MediaPipe → IK → sim 17 关节角 → MuJoCo viewer。

最小可用版本（MVP），零真机依赖。要做的事：
    1. 打开默认摄像头
    2. 用 MediaPipe Hands 检测手部 21 个 3D 关键点
    3. 取 5 个指尖相对手腕的偏移 → IK → sim 17 关节角
    4. 在 OrcaHandRight env 里 step，并用 MuJoCo viewer 渲染

使用方法：
    cd orca_sim
    source orca/Scripts/activate
    python scripts/webcam_grasp.py --render human
    python scripts/webcam_grasp.py --no-webcam --recorded-tips recorded.npy  # 没摄像头时

按键：
    q / ESC  → 退出
    r        → reset sim
    c        → toggle 显示摄像头小窗口
"""

from __future__ import annotations

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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="摄像头手势 → sim 17 关节角 实时 pipeline")
    p.add_argument("--render", choices=["none", "human"], default="human",
                   help="是否打开 MuJoCo viewer（默认 human）")
    p.add_argument("--no-webcam", action="store_true",
                   help="不使用摄像头，从 --recorded-tips 读取预录指尖数据")
    p.add_argument("--recorded-tips", type=str, default=None,
                   help="N×5×3 numpy 文件，每行 5 指尖 (x,y,z) 相对手腕偏移")
    p.add_argument("--camera-index", type=int, default=0,
                   help="OpenCV 摄像头索引（默认 0 = 笔记本自带）")
    p.add_argument("--no-show-camera", action="store_true",
                   help="不弹摄像头窗口（默认会弹，显示手部骨骼 + 5 指尖高亮 + 偏移数据）")
    p.add_argument("--mp-width", type=int, default=256,
                   help="送进 MediaPipe 的帧宽（默认 256；越小越快，建议 192-320）")
    p.add_argument("--no-render", action="store_true",
                   help="不开 MuJoCo viewer（提速测 FPS 用）")
    p.add_argument("--use-ik", action="store_true",
                   help="退回用旧的 HandIKSolver 而非 CurlSolver（用于对比）")
    p.add_argument("--oe-min-cutoff", type=float, default=0.004,
                   help="OneEuroFilter min_cutoff（默认 0.004；越小越平滑）")
    p.add_argument("--oe-beta", type=float, default=0.7,
                   help="OneEuroFilter beta（默认 0.7；越大越跟手）")
    p.add_argument("--hand-scale", type=float, default=1.0,
                   help="把 MediaPipe 出来的指尖偏移放大 N 倍（人手大小不一定等于 sim 手）")
    p.add_argument("--wrist-offset", type=float, nargs=3, default=[0.0, 0.0, 0.30],
                   metavar=("X", "Y", "Z"),
                   help="手腕相对 sim base 的初始偏移（默认 +0.30m 抬高，避免穿地）")
    return p.parse_args()


def _main_loop_with_webcam(
    env, solver, render_mode: str, show_camera: bool, hand_scale: float,
    mp_width: int = 256, use_ik: bool = False,
    one_euro_min_cutoff: float = 0.004, one_euro_beta: float = 0.7,
) -> None:
    """摄像头 + 实时 IK 主循环。"""
    import cv2
    from orca_sim.retarget import MediaPipeHandTracker, OneEuroFilterND

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头 0，请检查是否被占用或加 --camera-index 1")

    tracker = MediaPipeHandTracker(model_complexity=0, process_width=mp_width)

    print("[main] 按 'q' 退出，'r' reset sim，'c' 切换摄像头窗口")
    print("[main] 请把右手伸到镜头前，张开/握拳试一下。")

    wrist_world = np.array(env.unwrapped.data.xpos[env.unwrapped.model.body("right_palm").id, :3])

    # MediaPipe Hands 21 关键点的骨架连接（标准手部骨架）
    HAND_CONNECTIONS = (
        (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),          # index
        (5, 9), (9, 10), (10, 11), (11, 12),     # middle
        (9, 13), (13, 14), (14, 15), (15, 16),   # ring
        (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
        (0, 17),                                 # palm
    )
    # 5 指尖在 21 个关键点中的索引
    FIVE_TIPS = (4, 8, 12, 16, 20)
    # 5 指尖对应的 OpenCV 颜色（BGR）：thumb=粉, index=蓝, middle=绿, ring=黄, pinky=紫
    TIP_COLORS_BGR = (
        (203, 192, 255),  # thumb: pink
        (255, 128, 0),    # index: blue
        (0, 255, 0),      # middle: green
        (0, 255, 255),    # ring: yellow
        (255, 0, 255),    # pinky: purple
    )
    TIP_NAMES = ("thumb", "index", "middle", "ring", "pinky")

    last_frame_time = time.time()
    fps_smoothed = 0.0
    # 21 landmarks × 3 维 = 63 个独立 1€ filter
    one_euro = OneEuroFilterND(
        dims=63, min_cutoff=one_euro_min_cutoff, beta=one_euro_beta,
    )
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[main] 摄像头读不到帧，退出")
            break

        frame = cv2.flip(frame, 1)  # 镜像，符合用户习惯
        pose = tracker.process(frame)

        if pose.detected:
            # 1€ filter 平滑 21 landmarks（关键抗抖动步骤）
            lms_flat = pose.landmarks_world.flatten()
            lms_smooth_flat = one_euro(lms_flat)
            landmarks_smooth = lms_smooth_flat.reshape(21, 3)
            # 求解
            qpos = solver.solve(landmarks_smooth) if not use_ik else None
            env.step(qpos.astype(np.float32))
        if show_camera and pose.image_landmarks is not None:
            lms = pose.image_landmarks
            # 画骨架连线（白色）
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, tuple(lms[a]), tuple(lms[b]), (255, 255, 255), 2, cv2.LINE_AA)
            # 画所有 21 个关键点（绿色小点）
            for i in range(21):
                cv2.circle(frame, tuple(lms[i]), 3, (0, 255, 0), -1, cv2.LINE_AA)
            # 高亮 5 指尖（大圆 + 文字标签）
            for tip_idx, color, name in zip(FIVE_TIPS, TIP_COLORS_BGR, TIP_NAMES):
                pt = tuple(lms[tip_idx])
                cv2.circle(frame, pt, 8, color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 10, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(
                    frame, name, (pt[0] + 12, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )
            # 左上角 HUD
            cv2.putText(
                frame,
                f"hand: YES  conf={pose.confidence:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            # 显示指尖相对手腕的偏移（米）
            for i, (name, color) in enumerate(zip(TIP_NAMES, TIP_COLORS_BGR)):
                tip_idx = FIVE_TIPS[i]
                lm_tip = pose.landmarks_world[tip_idx]
                lm_wrist = pose.landmarks_world[0]
                offset = lm_tip - lm_wrist
                cv2.putText(
                    frame,
                    f"{name}: ({offset[0]:+.3f}, {offset[1]:+.3f}, {offset[2]:+.3f}) m",
                    (10, 60 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        elif show_camera:
            cv2.putText(
                frame,
                "hand: NO  (把手伸到镜头前)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # 右上角 FPS（平滑后）
        if show_camera:
            cv2.putText(
                frame,
                f"FPS: {fps_smoothed:4.1f}",
                (frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # 显示摄像头窗口
        if show_camera:
            cv2.imshow("webcam + hand tracking", frame)

        # 30 fps 节流
        elapsed = time.time() - last_frame_time
        if elapsed < 1 / 30:
            time.sleep(1 / 30 - elapsed)
        # 平滑 FPS 显示（EMA，alpha=0.1）
        instant_fps = 1.0 / max(elapsed, 1e-6)
        fps_smoothed = 0.9 * fps_smoothed + 0.1 * instant_fps if fps_smoothed > 0 else instant_fps
        last_frame_time = time.time()

        # 按键
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            env.reset(options=env.nominal_reset_options())
            print("[main] sim reset")
        if key == ord("c"):
            show_camera = not show_camera
            if not show_camera:
                cv2.destroyWindow("webcam + hand tracking")

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


def _main_loop_recorded(env, ik, tips_sequence: np.ndarray, render_mode: str) -> None:
    """从预录的 (N, 5, 3) 序列回放，方便没摄像头时调试。"""
    wrist_world = np.array(env.unwrapped.data.xpos[env.unwrapped.model.body("right_palm").id, :3])

    print(f"[main] 回放 {len(tips_sequence)} 帧预录指尖轨迹")
    for i, tips in enumerate(tips_sequence):
        qpos = ik.solve(wrist_world, tips)
        env.step(qpos.astype(np.float32))
        if i % 10 == 0:
            print(f"  frame {i}/{len(tips_sequence)}  fingertip_z = {tips[:, 2]}")
        time.sleep(1 / 30)


def main() -> int:
    args = _parse_args()

    from orca_sim.envs import OrcaHandRight
    from orca_sim.retarget import (
        HandIKSolver, IKSolverConfig,
        CurlSolver, CurlSolverConfig,
    )

    render_mode = args.render if args.render != "none" else "rgb_array"

    env = OrcaHandRight(
        version="v1",
        skin=False,
        render_mode=render_mode,
    )
    obs, info = env.reset()

    # 求解器
    if args.use_ik:
        solver = HandIKSolver(
            env.unwrapped.model, env.unwrapped.data,
            IKSolverConfig(max_iterations=8),
        )
        print("[main] 使用 HandIKSolver（IK 方案）")
    else:
        solver = CurlSolver(
            env.unwrapped.model, env.unwrapped.data,
            CurlSolverConfig(),
        )
        print("[main] 使用 CurlSolver（curl-based，无 IK）")

    # 把手腕位置抬高到 +Z 方向，避免 IK 解在地里
    palm_id = env.unwrapped.model.body("right_palm").id
    wrist_world = env.unwrapped.data.xpos[palm_id, :3].copy() + np.array(args.wrist_offset)

    print(f"[main] wrist_world = {wrist_world}")

    try:
        if args.no_webcam:
            if not args.recorded_tips:
                print("[main] --no-webcam 必须配合 --recorded-tips")
                return 1
            tips_seq = np.load(args.recorded_tips)
            assert tips_seq.ndim == 3 and tips_seq.shape[1:] == (5, 3), \
                f"期待 (N, 5, 3)，实际 {tips_seq.shape}"
            _main_loop_recorded(env, ik, tips_seq, render_mode)
        else:
            _main_loop_with_webcam(
        env, solver, render_mode, not args.no_show_camera, args.hand_scale,
        mp_width=args.mp_width, use_ik=args.use_ik,
        one_euro_min_cutoff=args.oe_min_cutoff, one_euro_beta=args.oe_beta,
    )
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())