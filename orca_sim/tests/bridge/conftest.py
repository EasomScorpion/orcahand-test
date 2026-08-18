"""共用 fixture：构造 orca_sim v1 右手 skin=False env。"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def env_v1_right_skin_false():
    """OrcaHandRight(version='v1', skin=False)；session-scoped 以加速测试。"""
    from orca_sim.envs import OrcaHandRight
    e = OrcaHandRight(version="v1", skin=False)
    e.reset(seed=42)
    yield e
    e.close()


@pytest.fixture(scope="module")
def mapping():
    """加载默认 17 舵机映射 JSON。"""
    from orca_sim.bridge import load_joint_mapping
    return load_joint_mapping()


@pytest.fixture(scope="module")
def env_v1_right_cube_skin_false():
    """OrcaHandRightCubeOrientation(version='v1', skin=False)。

    这个 fixture 才会构造 cube + floor 的 geom，
    用于测试「过滤非 hand geom」等场景。
    """
    from orca_sim.task_envs import OrcaHandRightCubeOrientation
    e = OrcaHandRightCubeOrientation(version="v1", skin=False)
    e.reset(seed=42)
    yield e
    e.close()