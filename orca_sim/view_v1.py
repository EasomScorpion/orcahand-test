"""view_v1.py — Open the MuJoCo viewer for an ORCA hand environment.

The simplest way to see the v1 hand without writing any policy code.
Defaults to the v1 right hand; supports left, combined, and the
"extended" variants that include the U2D2 board, fans and camera mount.

Examples
--------
    python view_v1.py                                # v1 right hand, static pose
    python view_v1.py --env left --mode demo         # v1 left hand, open/close cycle
    python view_v1.py --env combined                 # v1 both hands
    python view_v1.py --env right_extended           # v1 right hand + mounted hardware
    python view_v1.py --env right_cube_orientation --version v2
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


try:
    import numpy as np
    from orca_sim import (
        OrcaHandLeft,
        OrcaHandLeftExtended,
        OrcaHandRight,
        OrcaHandRightExtended,
        OrcaHandCombined,
        OrcaHandCombinedExtended,
        OrcaHandRightCubeOrientation,
    )
except ModuleNotFoundError as exc:
    if exc.name in {"gymnasium", "mujoco", "numpy"}:
        raise SystemExit(
            "Missing runtime dependency "
            f"'{exc.name}'. Activate your environment and run `uv pip install -e .`."
        ) from exc
    raise


ENV_BUILDERS = {
    "left": OrcaHandLeft,
    "left_extended": OrcaHandLeftExtended,
    "right": OrcaHandRight,
    "right_extended": OrcaHandRightExtended,
    "combined": OrcaHandCombined,
    "combined_extended": OrcaHandCombinedExtended,
    "right_cube_orientation": OrcaHandRightCubeOrientation,
}

# Indices for finger flexion joints (MCP + PIP/DIP) in the 17-DoF action vector.
FLEX_IDX = [3, 4, 7, 10, 13, 16]   # thumb_pip, thumb_dip, index/middle/ring/pinky pip
MCP_IDX = [6, 9, 12, 15]            # index/middle/ring/pinky mcp


def _make_fist(open_pose: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    """Push flexion joints toward the 'fist' end of their control range."""
    target = open_pose.copy()
    for i in FLEX_IDX:
        # action_high already encodes the flexion direction (positive for both
        # left and right hand PIPs in v1).
        target[i] = action_high[i] * 0.7
    for i in MCP_IDX:
        target[i] = action_high[i] * 0.4
    return target.astype(np.float32)


def _static_loop(env) -> None:
    """Render + step loop.

    Slider drags in the MuJoCo Control panel write ``data.ctrl``; we step
    physics after each ``sync()`` so the fingers actually follow the slider.
    """
    print("Static mode — drag sliders in the Control panel to move each finger.")
    print("Mouse: left=rotate, right=pan, wheel=zoom.")
    try:
        while True:
            env.render()  # sync(): pulls Control-panel slider edits into data.ctrl
            mujoco.mj_step(env.model, env.data, nstep=env.frame_skip)
            time.sleep(1.0 / env.metadata["render_fps"])
    except KeyboardInterrupt:
        pass


def _demo_loop(env) -> None:
    """Open ↔ fist cycle, 2 seconds per direction at render_fps."""
    obs, _ = env.reset()
    naction = env.action_space.shape[0]
    open_pose = env.data.qpos[:naction].astype(np.float32)
    fist_pose = _make_fist(open_pose, env.action_high)
    fps = env.metadata["render_fps"]
    period = int(fps * 2)
    print(f"Demo mode — cycling open ↔ fist every {period / fps:.1f}s.")
    step = 0
    try:
        while True:
            phase = (step % (2 * period)) / period            # 0 .. 2
            alpha = 1.0 - abs(phase - 1.0)                    # triangle wave 0→1→0
            action = open_pose * (1.0 - alpha) + fist_pose * alpha
            env.step(action.astype(np.float32))
            step += 1
    except KeyboardInterrupt:
        pass


def _random_loop(env, max_steps: int) -> None:
    msg = f"{max_steps} steps" if max_steps else "Ctrl+C"
    print(f"Random mode — uniform action sampling for {msg}.")
    step = 0
    try:
        while max_steps == 0 or step < max_steps:
            env.step(env.action_space.sample())
            step += 1
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=sorted(ENV_BUILDERS), default="right")
    parser.add_argument("--version", default="v1", help="Embodiment version, e.g. 'v1' or 'v2'.")
    parser.add_argument("--mode", choices=["static", "demo", "random"], default="static")
    parser.add_argument("--steps", type=int, default=0, help="Random mode only: 0 = until Ctrl+C.")
    parser.add_argument(
        "--no-skin",
        action="store_true",
        help="Load the scene with all *_skin meshes/geoms stripped (matches a hand without printed pads).",
    )
    args = parser.parse_args()

    env_cls = ENV_BUILDERS[args.env]
    env = env_cls(render_mode="human", version=args.version, skin=not args.no_skin)
    obs, info = env.reset()

    print(f"env       = {args.env}")
    print(f"version   = {env.version}")
    print(f"obs.shape = {obs.shape}")
    print(f"act.shape = {env.action_space.shape}")
    print(f"act.low   = {np.array2string(env.action_low, precision=3, suppress_small=True)}")
    print(f"act.high  = {np.array2string(env.action_high, precision=3, suppress_small=True)}")

    if args.mode == "static":
        _static_loop(env)
    elif args.mode == "demo":
        _demo_loop(env)
    else:
        _random_loop(env, args.steps)

    env.close()


if __name__ == "__main__":
    main()