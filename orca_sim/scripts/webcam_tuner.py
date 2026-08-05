"""webcam_tuner.py —— 摄像头手势 + 可交互调参面板（Tkinter）。

跟 webcam_grasp.py 一样跑摄像头 + BoneMatcher，但额外弹一个 Tkinter 滑块窗口，
让你**实时**调整以下参数并立刻看到 sim 端反应：

    CurlSolver:
        - mcp_gain        弯曲总幅度（4 指通用）
        - pip_ratio       PIP 跟 MCP 联动比例
        - thumb_pip_gain  拇指 PIP 幅度
        - thumb_mcp_ratio 拇指 MCP 联动
        - thumb_dip_ratio 拇指 DIP 联动
        - thumb_abd_y_scale 拇指外展 Y 灵敏度
        - abd_x_scale     4 指外展 X 灵敏度
        - natural_sign[index/middle/ring/pinky]  每根手指外展方向 (-1/0/+1)

    OneEuroFilter:
        - min_cutoff  平滑度（越小越平滑）
        - beta        反应速度（越大越跟手）

按键（鼠标在 OpenCV 窗口时）：
    q / ESC  退出
    r        reset sim

调好的参数可以「Save」按钮存到 ``configs/tuned_<timestamp>.json``。

用法：
    cd orca_sim
    source orca/Scripts/activate
    python scripts/webcam_tuner.py
    python scripts/webcam_tuner.py --no-camera-window    # 隐藏摄像头小窗
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import threading
import tkinter as tk
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np


# 让 src/ 可直接 import
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ============================================================
# 参数容器（主线程与 Tkinter 共享——用 tk.DoubleVar / IntVar 即可线程安全）
# ============================================================
class TunableParams:
    """所有可调参数 + Tk 变量绑定。"""

    def __init__(self, root: tk.Tk):
        # ---- CurlSolverConfig ----
        self.mcp_gain = tk.DoubleVar(value=1.0)
        self.pip_ratio = tk.DoubleVar(value=0.7)
        self.thumb_pip_gain = tk.DoubleVar(value=1.23)
        self.thumb_mcp_ratio = tk.DoubleVar(value=0.85)
        self.thumb_dip_ratio = tk.DoubleVar(value=0.65)
        self.thumb_abd_y_scale = tk.DoubleVar(value=0.04)
        self.abd_x_scale = tk.DoubleVar(value=0.04)

        # natural_signs：用 IntVar + 离散三档（-1 / 0 / +1）
        self.sign_index = tk.IntVar(value=-1)
        self.sign_middle = tk.IntVar(value=-1)
        self.sign_ring = tk.IntVar(value=-1)
        self.sign_pinky = tk.IntVar(value=+1)

        # ---- OneEuroFilter ----
        self.oe_min_cutoff = tk.DoubleVar(value=0.004)
        self.oe_beta = tk.DoubleVar(value=0.7)

        # ---- BoneMatcherConfig ----
        self.use_ik = tk.BooleanVar(value=True)

        self._root = root

    def snapshot(self) -> dict:
        """导出当前所有参数为纯 dict（用于保存）。"""
        return {
            "mcp_gain": self.mcp_gain.get(),
            "pip_ratio": self.pip_ratio.get(),
            "thumb_pip_gain": self.thumb_pip_gain.get(),
            "thumb_mcp_ratio": self.thumb_mcp_ratio.get(),
            "thumb_dip_ratio": self.thumb_dip_ratio.get(),
            "thumb_abd_y_scale": self.thumb_abd_y_scale.get(),
            "abd_x_scale": self.abd_x_scale.get(),
            "natural_signs": {
                "index": self.sign_index.get(),
                "middle": self.sign_middle.get(),
                "ring": self.sign_ring.get(),
                "pinky": self.sign_pinky.get(),
            },
            "oe_min_cutoff": self.oe_min_cutoff.get(),
            "oe_beta": self.oe_beta.get(),
            "use_ik": bool(self.use_ik.get()),
        }

    def apply_to_solver(self, solver) -> None:
        """把当前面板值写回 CurlSolver / BoneMatcher 的 config。"""
        if hasattr(solver, "_curl") and solver._curl is not None:
            cfg = solver._curl.config
        elif hasattr(solver, "config"):
            cfg = solver.config
        else:
            return

        cfg.mcp_gain = float(self.mcp_gain.get())
        cfg.pip_ratio = float(self.pip_ratio.get())
        cfg.thumb_pip_gain = float(self.thumb_pip_gain.get())
        cfg.thumb_mcp_ratio = float(self.thumb_mcp_ratio.get())
        cfg.thumb_dip_ratio = float(self.thumb_dip_ratio.get())
        cfg.thumb_abd_y_scale = float(self.thumb_abd_y_scale.get())
        cfg.abd_x_scale = float(self.abd_x_scale.get())
        cfg.natural_signs["index"] = float(self.sign_index.get())
        cfg.natural_signs["middle"] = float(self.sign_middle.get())
        cfg.natural_signs["ring"] = float(self.sign_ring.get())
        cfg.natural_signs["pinky"] = float(self.sign_pinky.get())


# ============================================================
# Tkinter 面板
# ============================================================
def _add_slider(
    parent: tk.Widget, label: str, var: tk.Variable, frm: float, to: float,
    resolution: float = 0.01, orient: tk.HORIZONTAL = tk.HORIZONTAL,
) -> tk.Scale:
    row = tk.Frame(parent)
    row.pack(fill="x", padx=4, pady=1)
    tk.Label(row, text=label, width=18, anchor="w").pack(side="left")
    s = tk.Scale(
        row, variable=var, from_=frm, to=to, resolution=resolution,
        orient=orient, length=240,
    )
    s.pack(side="left", fill="x", expand=True)
    return s


def _add_sign_selector(
    parent: tk.Widget, label: str, var: tk.IntVar,
) -> None:
    """离散三档选择器：-1 / 0 / +1（用 Radiobutton）。"""
    row = tk.Frame(parent)
    row.pack(fill="x", padx=4, pady=1)
    tk.Label(row, text=label, width=18, anchor="w").pack(side="left")
    for txt, val in (("-1", -1), ("0", 0), ("+1", +1)):
        tk.Radiobutton(
            row, text=txt, variable=var, value=val,
            width=3, indicatoron=False,
        ).pack(side="left")


def build_panel(root: tk.Tk, params: TunableParams, on_save, on_reset) -> None:
    root.title("orca_sim retarget tuner")
    root.geometry("540x720")

    # ---- Section: CurlSolver ----
    tk.Label(root, text="── CurlSolver ──", font=("TkDefaultFont", 10, "bold")).pack(
        anchor="w", padx=8, pady=(8, 2)
    )
    _add_slider(root, "mcp_gain",        params.mcp_gain,        0.3, 1.8)
    _add_slider(root, "pip_ratio",       params.pip_ratio,       0.0, 1.5)
    _add_slider(root, "thumb_pip_gain",  params.thumb_pip_gain,  0.3, 2.5)
    _add_slider(root, "thumb_mcp_ratio", params.thumb_mcp_ratio, 0.0, 1.5)
    _add_slider(root, "thumb_dip_ratio", params.thumb_dip_ratio, 0.0, 1.5)
    _add_slider(root, "thumb_abd_y",     params.thumb_abd_y_scale, 0.01, 0.10, 0.005)
    _add_slider(root, "abd_x_scale",     params.abd_x_scale,     0.01, 0.10, 0.005)

    # ---- Section: natural_signs（离散三档）----
    tk.Label(root, text="── natural_signs (外展方向) ──", font=("TkDefaultFont", 10, "bold")).pack(
        anchor="w", padx=8, pady=(8, 2)
    )
    _add_sign_selector(root, "index  sign",  params.sign_index)
    _add_sign_selector(root, "middle sign",  params.sign_middle)
    _add_sign_selector(root, "ring   sign",  params.sign_ring)
    _add_sign_selector(root, "pinky  sign",  params.sign_pinky)

    # ---- Section: OneEuroFilter ----
    tk.Label(root, text="── OneEuroFilter ──", font=("TkDefaultFont", 10, "bold")).pack(
        anchor="w", padx=8, pady=(8, 2)
    )
    _add_slider(root, "oe_min_cutoff", params.oe_min_cutoff, 0.001, 0.05, 0.001)
    _add_slider(root, "oe_beta",       params.oe_beta,       0.0,   3.0,  0.05)

    # ---- Section: BoneMatcher ----
    tk.Label(root, text="── BoneMatcher ──", font=("TkDefaultFont", 10, "bold")).pack(
        anchor="w", padx=8, pady=(8, 2)
    )
    tk.Checkbutton(root, text="use_ik (LM microadjust)", variable=params.use_ik).pack(
        anchor="w", padx=12
    )

    # ---- Buttons ----
    btn_row = tk.Frame(root)
    btn_row.pack(fill="x", padx=8, pady=12)
    tk.Button(btn_row, text="Save → configs/tuned_*.json", command=on_save, width=24).pack(
        side="left", padx=4
    )
    tk.Button(btn_row, text="Reset to defaults", command=on_reset, width=18).pack(
        side="left", padx=4
    )


DEFAULTS = {
    "mcp_gain": 1.0,
    "pip_ratio": 0.7,
    "thumb_pip_gain": 1.23,
    "thumb_mcp_ratio": 0.85,
    "thumb_dip_ratio": 0.65,
    "thumb_abd_y_scale": 0.04,
    "abd_x_scale": 0.04,
    "natural_signs": {"index": -1, "middle": -1, "ring": -1, "pinky": 1},
    "oe_min_cutoff": 0.004,
    "oe_beta": 0.7,
    "use_ik": True,
}


def reset_to_defaults(params: TunableParams) -> None:
    params.mcp_gain.set(DEFAULTS["mcp_gain"])
    params.pip_ratio.set(DEFAULTS["pip_ratio"])
    params.thumb_pip_gain.set(DEFAULTS["thumb_pip_gain"])
    params.thumb_mcp_ratio.set(DEFAULTS["thumb_mcp_ratio"])
    params.thumb_dip_ratio.set(DEFAULTS["thumb_dip_ratio"])
    params.thumb_abd_y_scale.set(DEFAULTS["thumb_abd_y_scale"])
    params.abd_x_scale.set(DEFAULTS["abd_x_scale"])
    params.sign_index.set(DEFAULTS["natural_signs"]["index"])
    params.sign_middle.set(DEFAULTS["natural_signs"]["middle"])
    params.sign_ring.set(DEFAULTS["natural_signs"]["ring"])
    params.sign_pinky.set(DEFAULTS["natural_signs"]["pinky"])
    params.oe_min_cutoff.set(DEFAULTS["oe_min_cutoff"])
    params.oe_beta.set(DEFAULTS["oe_beta"])
    params.use_ik.set(DEFAULTS["use_ik"])


# ============================================================
# 主体
# ============================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="摄像头 + Tkinter 调参面板")
    p.add_argument("--no-camera-window", action="store_true", help="隐藏摄像头小窗")
    p.add_argument("--mp-width", type=int, default=256)
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--frame-skip", type=int, default=20)
    p.add_argument("--use-curl", action="store_true", help="退回用 CurlSolver（无 IK）")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    from orca_sim.envs import OrcaHandRight
    from orca_sim.retarget import (
        MediaPipeHandTracker, OneEuroFilterND,
        CurlSolver, CurlSolverConfig,
        SimSkeleton, BoneMatcher, BoneMatcherConfig,
    )
    import cv2
    import mujoco as mj

    # ---- env + solver ----
    env = OrcaHandRight(version="v1", skin=False, render_mode="human")
    env.reset()
    env.unwrapped.frame_skip = args.frame_skip

    if args.use_curl:
        solver = CurlSolver(env.unwrapped.model, env.unwrapped.data, CurlSolverConfig())
        print("[main] 使用 CurlSolver")
    else:
        skel = SimSkeleton.from_model(env.unwrapped.model)
        cfg = BoneMatcherConfig(max_iterations=12, warm_start=True, use_heuristic_init=True)
        solver = BoneMatcher(skel, env.unwrapped.model, env.unwrapped.data, cfg)
        print("[main] 使用 BoneMatcher")

    # ---- Tkinter 面板（独立线程）----
    root = tk.Tk()
    params = TunableParams(root)

    saved_log: list[str] = []

    def on_save() -> None:
        snap = params.snapshot()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).resolve().parent.parent / "configs"
        out_dir.mkdir(exist_ok=True)
        out = out_dir / f"tuned_{ts}.json"
        out.write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
        msg = f"[main] saved → {out}"
        print(msg, flush=True)
        saved_log.append(msg)

    def on_reset() -> None:
        reset_to_defaults(params)
        print("[main] tuner reset to defaults", flush=True)

    build_panel(root, params, on_save=on_save, on_reset=on_reset)

    def _tick():
        # 把滑块值实时写回 solver config
        params.apply_to_solver(solver)
        root.after(100, _tick)

    _tick()

    # Tk 在子线程跑（OpenCV 主循环不被阻塞）
    tk_thread = threading.Thread(target=root.mainloop, daemon=True)
    tk_thread.start()

    # ---- webcam 主循环 ----
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"[main] 无法打开摄像头 {args.camera_index}")
        return 1

    tracker = MediaPipeHandTracker(
        model_complexity=0,
        process_width=args.mp_width,
        min_detection_confidence=0.3,
    )

    one_euro = OneEuroFilterND(dims=63, min_cutoff=0.004, beta=0.7)
    prev_qpos: np.ndarray | None = None
    missed_frames = 0
    last_log_t = 0.0
    last_frame_time = time.time()
    fps = 0.0
    use_ik_flag = True

    print("[main] 调参面板已开（独立窗口）。滑块实时生效；按 'q' 退出；'r' reset sim。")
    print("[main] 自然 sign 用三档（-1/0/+1）；保存按钮会写 configs/tuned_<ts>.json。")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        frame = cv2.flip(frame, 1)
        pose = tracker.process(frame)

        # 把 OneEuro 参数同步到 filter（实时）
        one_euro.min_cutoff = float(params.oe_min_cutoff.get())
        one_euro.beta = float(params.oe_beta.get())

        # IK toggle
        use_ik_flag = bool(params.use_ik.get())

        if pose.detected:
            lms_smooth_flat = one_euro(pose.landmarks_world.flatten(), t=now)
            landmarks_smooth = lms_smooth_flat.reshape(21, 3)

            if not args.use_curl and use_ik_flag:
                qpos = solver.solve(landmarks_smooth, prev_qpos=prev_qpos, data=env.unwrapped.data)
            elif not args.use_curl:
                # 临时关 IK——启发式 init 但跳过 LM 微调
                cfg = solver.config
                old_iter = cfg.max_iterations
                cfg.max_iterations = 0
                qpos = solver.solve(landmarks_smooth, prev_qpos=prev_qpos, data=env.unwrapped.data)
                cfg.max_iterations = old_iter
            else:
                qpos = solver.solve(landmarks_smooth)

            env.unwrapped.data.ctrl[:17] = np.clip(
                qpos.astype(np.float32),
                env.unwrapped.action_low,
                env.unwrapped.action_high,
            )
            for _ in range(8):
                mj.mj_step(env.unwrapped.model, env.unwrapped.data, nstep=env.unwrapped.frame_skip)
            mj.mj_forward(env.unwrapped.model, env.unwrapped.data)
            if env.unwrapped.render_mode == "human":
                env.unwrapped.render()

            prev_qpos = qpos.copy()
            missed_frames = 0
        else:
            missed_frames += 1
            if prev_qpos is not None:
                env.unwrapped.data.ctrl[:17] = np.clip(
                    prev_qpos.astype(np.float32),
                    env.unwrapped.action_low,
                    env.unwrapped.action_high,
                )
                for _ in range(4):
                    mj.mj_step(env.unwrapped.model, env.unwrapped.data, nstep=env.unwrapped.frame_skip)
                mj.mj_forward(env.unwrapped.model, env.unwrapped.data)
                if env.unwrapped.render_mode == "human":
                    env.unwrapped.render()

        # 摄像头小窗
        if not args.no_camera_window:
            if pose.detected and pose.image_landmarks is not None:
                cv2.putText(
                    frame,
                    f"FPS {fps:4.1f}  HAND  conf={pose.confidence:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame,
                    f"FPS {fps:4.1f}  NO HAND  missed={missed_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
                )
            cv2.imshow("webcam + tuner", frame)

        # FPS
        elapsed = time.time() - last_frame_time
        if elapsed < 1 / 30:
            time.sleep(1 / 30 - elapsed)
        instant = 1.0 / max(elapsed, 1e-6)
        fps = 0.9 * fps + 0.1 * instant if fps > 0 else instant
        last_frame_time = time.time()

        # 每秒日志
        if now - last_log_t > 1.0:
            extra = ""
            if pose.detected:
                xs = [float(pose.landmarks_world[i, 0] - pose.landmarks_world[0, 0]) for i in (4, 8, 12, 16, 20)]
                extra = f"  X(mm)={[f'{x*1000:+4.0f}' for x in xs]}"
                extra += f"  signs=[idx:{params.sign_index.get():+d} mid:{params.sign_middle.get():+d} ring:{params.sign_ring.get():+d} pinky:{params.sign_pinky.get():+d}]"
            print(f"[main] FPS={fps:4.1f}{extra}", flush=True)
            last_log_t = now

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            env.reset(options=env.nominal_reset_options())
            print("[main] sim reset")

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())