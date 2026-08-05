"""retarget — 手势识别 → 17 关节角重定向。

子模块：
    - fingertip_helper : sim 端 5 指尖世界坐标读取
    - mediapipe_hand   : MediaPipe Hands 21 关键点 → 5 指尖目标（手腕局部系）
    - ik_solver        : 5 指尖 3D 目标 → sim 17 关节角（MuJoCo 原生 IK）

依赖（额外安装）：
    uv pip install mediapipe opencv-python
    或：
    uv pip install -e ".[teleop]"
"""

from orca_sim.retarget.fingertip_helper import (
    FINGERTIP_BODY_NAMES,
    fingertip_positions,
    fingertip_body_ids,
    wrist_position,
)
from orca_sim.retarget.mediapipe_hand import MediaPipeHandTracker
from orca_sim.retarget.ik_solver import HandIKSolver, IKSolverConfig
from orca_sim.retarget.curl_solver import CurlSolver, CurlSolverConfig
from orca_sim.retarget.one_euro_filter import OneEuroFilter, OneEuroFilterND

__all__ = [
    "FINGERTIP_BODY_NAMES",
    "fingertip_body_ids",
    "fingertip_positions",
    "wrist_position",
    "MediaPipeHandTracker",
    "HandIKSolver",
    "IKSolverConfig",
    "CurlSolver",
    "CurlSolverConfig",
    "OneEuroFilter",
    "OneEuroFilterND",
]