"""retarget/conftest.py — retarget/ 测试共享 fixture。

提供 v1 右手 env + SimSkeleton + 干净 data，不污染调用方状态。
"""

import sys
from pathlib import Path


def _ensure_src_on_path():
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_on_path()


import numpy as np
import pytest


@pytest.fixture()
def env_v1_right():
    """OrcaHandRight v1 skin=False，rgb_array 模式。"""
    from orca_sim.envs import OrcaHandRight
    env = OrcaHandRight(version="v1", skin=False, render_mode="rgb_array")
    env.reset(seed=42)
    yield env
    env.close()


@pytest.fixture()
def skel(env_v1_right):
    """从 env 的 model 抽 SimSkeleton。"""
    from orca_sim.retarget import SimSkeleton
    return SimSkeleton.from_model(env_v1_right.unwrapped.model)


@pytest.fixture()
def skel_with_data(env_v1_right, skel):
    """(skel, data) —— 用于 forward_kinematics / BoneMatcher 测试。

    data 来自 env，**就地修改**调用方请负责复原。
    """
    return skel, env_v1_right.unwrapped.data
