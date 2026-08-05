"""webcam_grasp.py —— 端到端：摄像头手势 → MediaPipe → BoneMatcher → sim 17 关节角 → MuJoCo viewer。

最小可用版本（MVP），零真机依赖。要做的事：
    1. 打开默认摄像头
    2. 用 MediaPipe Hands 检测手部 21 个 3D 关键点
    3. 通过 BoneMatcher（启发式初值 + 可选 IK 微调）算出 sim 17 关节角
    4. 在 OrcaHandRight env 里 step，并用 MuJoCo viewer 渲染

使用方法：
    cd orca_sim
    source orca/Scripts/activate
    python scripts/webcam_grasp.py --render human
    python scripts/webcam_grasp.py --no-webcam --recorded-tips recorded.npy  # 没摄像头时
    python scripts/webcam_grasp.py --use-curl                            # 用旧 CurlSolver（对比/兜底）

按键：
    q / ESC  → 退出
    r        → reset sim
    c        → toggle 显示摄像头小窗口
    m        → toggle BoneMatcher 启发式 vs 启发式+IK
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
    p.add_argument("--use-curl", action="store_true",
                   help="退回用旧的 CurlSolver 而非 BoneMatcher（用于对比）")
    p.add_argument("--oe-min-cutoff", type=float, default=0.004,
                   help="OneEuroFilter min_cutoff（默认 0.004；越小越平滑）")
    p.add_argument("--oe-beta", type=float, default=0.7,
                   help="OneEuroFilter beta（默认 0.7；越大越跟手）")
    p.add_argument("--hand-scale", type=float, default=1.0,
                   help="把 MediaPipe 出来的指尖偏移放大 N 倍（人手大小不一定等于 sim 手）")
    p.add_argument("--wrist-offset", type=float, nargs=3, default=[0.0, 0.0, 0.30],
                   metavar=("X", "Y", "Z"),
                   help="手腕相对 sim base 的初始偏移（默认 +0.30m 抬高，避免穿地）")
    p.add_argument("--no-ik", action="store_true",
                   help="BoneMatcher 不跑 LM IK 微调（只用启发式初值，更快但精度差）")
    p.add_argument("--frame-skip", type=int, default=20,
                   help="每帧 sim 子步数（默认 20，比默认 5 收敛更快）")
    return p.parse_args()


def _main_loop_with_webcam(
    env, solver, render_mode: str, show_camera: bool, hand_scale: float,
    mp_width: int = 256, use_curl: bool = False,
    one_euro_min_cutoff: float = 0.004, one_euro_beta: float = 0.7,
    use_ik: bool = True,
) -> None:
    """摄像头 + BoneMatcher / 旧 IK 主循环。"""
    import cv2
    from orca_sim.retarget import MediaPipeHandTracker, OneEuroFilterND

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头 0，请检查是否被占用或加 --camera-index 1")

    tracker = MediaPipeHandTracker(
        model_complexity=0,
        process_width=mp_width,
        min_detection_confidence=0.3,    # 宽松点，避免人手一动就丢失
    )

    print("[main] 按 'q' 退出，'r' reset sim，'c' 切换摄像头窗口，'m' toggle IK")
    print("[main] 请把右手伸到镜头前，张开/握拳试一下。")

    wrist_world = np.array(env.unwrapped.data.xpos[env.unwrapped.model.body("right_palm").id, :3])
    m_qpos0 = env.unwrapped.model.qpos0[:17].copy()   # 用于检测丢失后回零

    # MediaPipe Hands 21 关键点的骨架连接（标准手部骨架）
    HAND_CONNECTIONS = (
        (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),          # index
        (5, 9), (9, 10), (10, 11), (11, 12),     # middle
        (9, 13), (13, 14), (14, 15), (15, 16),   # ring
        (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
        (0, 17),                                 # palm
    )
    FIVE_TIPS = (4, 8, 12, 16, 20)
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
    prev_qpos = None
    missed_frames = 0
    last_log_t = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[main] 摄像头读不到帧，退出")
            break

        now = time.time()
        frame = cv2.flip(frame, 1)  # 镜像，符合用户习惯
        pose = tracker.process(frame)

        if pose.detected:
            # 1€ filter 平滑（传 t=now 让 cutoff 自适应真实帧率）
            lms_flat = pose.landmarks_world.flatten()
            lms_smooth_flat = one_euro(lms_flat, t=now)
            landmarks_smooth = lms_smooth_flat.reshape(21, 3)
            # 求解
            if not use_curl:
                qpos = solver.solve(
                    landmarks_smooth,
                    prev_qpos=prev_qpos,
                    data=env.unwrapped.data,
                )
            else:
                qpos = solver.solve(landmarks_smooth)
            # 直接写 ctrl 并连续 step 多次 → servo 跟得上人手变化
            qpos_f32 = qpos.astype(np.float32)
            env.unwrapped.data.ctrl[:17] = np.clip(
                qpos_f32,
                env.unwrapped.action_low,
                env.unwrapped.action_high,
            )
            # 跑 N 个 mj_step 让 servo 真的收敛（否则 frame_skip=5 视觉上基本不动）
            import mujoco as _mj
            for _ in range(8):
                _mj.mj_step(env.unwrapped.model, env.unwrapped.data, nstep=env.unwrapped.frame_skip)
            _mj.mj_forward(env.unwrapped.model, env.unwrapped.data)
            if env.unwrapped.render_mode == "human":
                env.unwrapped.render()
            prev_qpos = qpos.copy()
            missed_frames = 0
            # 调试：每 30 帧打印一次 qpos vs sim 当前 qpos
            if now - last_log_t > 1.0:
                actual_qpos = env.unwrapped.data.qpos[:17].copy()
                diff = float(np.max(np.abs(actual_qpos - qpos)))
                print(f"[main]   qpos[:3]={qpos[:3].round(2)}, sim_qpos[:3]={actual_qpos[:3].round(2)}, max_diff={diff:.3f}", flush=True)
        else:
            # MediaPipe 没检测到手——保持上一次的 qpos（不让 sim 卡死）
            missed_frames += 1
            if prev_qpos is not None:
                env.unwrapped.data.ctrl[:17] = np.clip(
                    prev_qpos.astype(np.float32),
                    env.unwrapped.action_low,
                    env.unwrapped.action_high,
                )
                import mujoco as _mj
                for _ in range(4):
                    _mj.mj_step(env.unwrapped.model, env.unwrapped.data, nstep=env.unwrapped.frame_skip)
                _mj.mj_forward(env.unwrapped.model, env.unwrapped.data)
                if env.unwrapped.render_mode == "human":
                    env.unwrapped.render()
            # 超过 30 帧 (~1s) 没检测到 → 直接 rest（人手离开镜头了）
            if missed_frames > 30 and prev_qpos is not None:
                env.unwrapped.data.ctrl[:17] = np.clip(
                    m_qpos0.astype(np.float32),
                    env.unwrapped.action_low,
                    env.unwrapped.action_high,
                )
                import mujoco as _mj
                for _ in range(8):
                    _mj.mj_step(env.unwrapped.model, env.unwrapped.data, nstep=env.unwrapped.frame_skip)
                prev_qpos = None
                missed_frames = 0
        if show_camera and pose.image_landmarks is not None:
            lms = pose.image_landmarks
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, tuple(lms[a]), tuple(lms[b]), (255, 255, 255), 2, cv2.LINE_AA)
            for i in range(21):
                cv2.circle(frame, tuple(lms[i]), 3, (0, 255, 0), -1, cv2.LINE_AA)
            for tip_idx, color, name in zip(FIVE_TIPS, TIP_COLORS_BGR, TIP_NAMES):
                pt = tuple(lms[tip_idx])
                cv2.circle(frame, pt, 8, color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 10, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(
                    frame, name, (pt[0] + 12, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )
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

        if show_camera:
            cv2.imshow("webcam + hand tracking", frame)

        elapsed = time.time() - last_frame_time
        if elapsed < 1 / 30:
            time.sleep(1 / 30 - elapsed)
        instant_fps = 1.0 / max(elapsed, 1e-6)
        fps_smoothed = 0.9 * fps_smoothed + 0.1 * instant_fps if fps_smoothed > 0 else instant_fps
        last_frame_time = time.time()

        # 每秒打印一次检测状态（便于排查 MediaPipe 检测丢失）
        if now - last_log_t > 1.0:
            status = "DETECTED" if pose.detected else f"NO_HAND (missed={missed_frames})"
            extra = ""
            if pose.detected:
                lm_now = pose.landmarks_world
                wrist = lm_now[0]
                # 各指尖相对手腕的 X 偏移（人手外展时这个值应该有明显变化）
                xs = [float(lm_now[i, 0] - lm_now[0, 0]) for i in (4, 8, 12, 16, 20)]
                dists = [float(np.linalg.norm(lm_now[i] - wrist)) for i in (4, 8, 12, 16, 20)]
                extra = (
                    f"  X(mm)={[f'{x*1000:+5.0f}' for x in xs]}"
                    f"  dists(mm)={[f'{d*1000:.0f}' for d in dists]}"
                )
            print(f"[main] FPS={fps_smoothed:4.1f}  {status}  conf={pose.confidence:.2f}{extra}", flush=True)
            last_log_t = now

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
        if key == ord("m"):
            use_ik = not use_ik
            print(f"[main] IK 微调: {'on' if use_ik else 'off'}")

        # 调试用：按数字键 0-6 强制触发 calibration 各手势 qpos（不走摄像头）
        if key in (ord("0"), ord("1"), ord("2"), ord("3"), ord("4"),
                   ord("5"), ord("6")):
            import json as _json
            cal = _json.load(open("calibration_data.json", encoding="utf-8"))
            poses = ["open_hand", "fist", "peace", "index_point",
                     "pinky_point", "thumbs_up", "spread"]
            idx = int(chr(key))
            if idx < len(poses):
                lm = np.array(cal["poses"][poses[idx]]["landmarks"])
                if not use_curl:
                    qpos = solver.solve(lm, data=env.unwrapped.data)
                else:
                    qpos = solver.solve(lm)
                env.unwrapped.data.ctrl[:17] = np.clip(
                    qpos.astype(np.float32),
                    env.unwrapped.action_low,
                    env.unwrapped.action_high,
                )
                import mujoco as _mj
                for _ in range(16):
                    _mj.mj_step(env.unwrapped.model, env.unwrapped.data, nstep=env.unwrapped.frame_skip)
                _mj.mj_forward(env.unwrapped.model, env.unwrapped.data)
                if env.unwrapped.render_mode == "human":
                    env.unwrapped.render()
                print(f"[main] *** 强制 {poses[idx]} qpos[:3]={qpos[:3].round(2)} ***", flush=True)

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
        SimSkeleton, BoneMatcher, BoneMatcherConfig,
    )

    render_mode = args.render if args.render != "none" else "rgb_array"

    env = OrcaHandRight(
        version="v1",
        skin=False,
        render_mode=render_mode,
    )
    obs, info = env.reset()

    # 加大 frame_skip → sim servo 收敛更快（默认 5 收敛太慢）
    env.unwrapped.frame_skip = args.frame_skip
    print(f"[main] sim frame_skip = {env.unwrapped.frame_skip}")

    # 求解器：默认 BoneMatcher；--use-curl 用旧 CurlSolver
    if args.use_curl:
        solver = CurlSolver(
            env.unwrapped.model, env.unwrapped.data,
            CurlSolverConfig(),
        )
        print("[main] 使用 CurlSolver（旧 curl-based 方案）")
    else:
        skel = SimSkeleton.from_model(env.unwrapped.model)
        cfg = BoneMatcherConfig(
            max_iterations=12,
            lm_damping=1e-3,
            warm_start=True,
            use_heuristic_init=True,
            enforce_limits=True,
        )
        solver = BoneMatcher(skel, env.unwrapped.model, env.unwrapped.data, cfg)
        print("[main] 使用 BoneMatcher（启发式 = CurlSolver + LM IK 微调）")

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
                mp_width=args.mp_width, use_curl=args.use_curl,
                one_euro_min_cutoff=args.oe_min_cutoff, one_euro_beta=args.oe_beta,
                use_ik=not args.no_ik,
            )
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())