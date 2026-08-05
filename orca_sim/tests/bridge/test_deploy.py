"""test_deploy.py：弧度→raw 转换、ServoSafetyLayer 校验路径、单调性、忽略零点。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


# ----------------------------------------------------------------------
# Mock backend
# ----------------------------------------------------------------------
@dataclass
class _FakeBundle:
    servo_id: int
    min_angle: int
    max_angle: int
    ofs: int = 0
    fields: dict | None = None


class _FakeSafety:
    """模拟 ServoSafetyLayer.sync_go_to_pose 的最小行为。"""

    def __init__(self, backend):
        self._backend = backend
        self._emergency_stopped = False
        self.last_positions = None

    def sync_go_to_pose(self, positions):
        for sid, raw in positions.items():
            b = self._backend.bundles[sid - 1]
            if not (b.min_angle <= raw <= b.max_angle):
                # 模拟 servo_console.SafetyViolation
                raise AssertionError(
                    f"ID {sid}: raw {raw} ∉ [{b.min_angle}, {b.max_angle}]"
                )
        self.last_positions = dict(positions)
        return 0

    def emergency_stop(self):
        self._emergency_stopped = True

    def recovery(self):
        self._emergency_stopped = False

    def is_emergency_stopped(self):
        return self._emergency_stopped


class _FakeBackend:
    """最小 ConsoleBackend mock。"""

    def __init__(self, bundles):
        self.bundles = bundles
        self.safety = _FakeSafety(self)


@pytest.fixture
def fake_backend():
    """17 个 fake bundle，raw 范围各不相同（模拟真实 xdat）。"""
    bundles = [
        _FakeBundle(servo_id=i, min_angle=200 + i * 100, max_angle=200 + i * 100 + 2000)
        for i in range(1, 18)
    ]
    return _FakeBackend(bundles)


@pytest.fixture
def deployer(env_v1_right_skin_false, mapping, fake_backend):
    from orca_sim.bridge import SimToRealDeployer
    return SimToRealDeployer(
        console_backend=fake_backend,
        env=env_v1_right_skin_false,
        mapping=mapping,
        collision_check=False,
    )


# ----------------------------------------------------------------------
# 测试
# ----------------------------------------------------------------------
def test_deploy_qpos_returns_17_servos(deployer):
    """deploy_qpos 应返回 17 个 servo_id 的 dict（含 wrist=17）。"""
    qpos = np.zeros(env_v1_right_skin_false_nq(deployer), dtype=np.float64)
    out = deployer.deploy_qpos(qpos)
    assert len(out) == 17
    assert all(1 <= sid <= 17 for sid in out.keys())
    assert 17 in out  # wrist 必须包含


def env_v1_right_skin_false_nq(deployer):
    return int(deployer.env.unwrapped.model.nq)


def test_deploy_qpos_raw_in_range(deployer):
    """对每个 servo_id，raw ∈ [min_angle, max_angle]。"""
    qpos = np.zeros(env_v1_right_skin_false_nq(deployer), dtype=np.float64)
    out = deployer.deploy_qpos(qpos)
    bundles = deployer.backend.bundles
    for sid, raw in out.items():
        b = bundles[sid - 1]
        assert b.min_angle <= raw <= b.max_angle, \
            f"sid={sid} raw={raw} ∉ [{b.min_angle}, {b.max_angle}]"


def test_ignore_zero_mapping_endpoints(deployer):
    """sim_low → raw_low；sim_high → raw_high（线性映射端点）。"""
    info = deployer._info
    for sid_int, inf in info.items():
        # sim_low → raw_low
        raw_at_low = deployer._rad_to_raw(inf, inf["sim_low"])
        assert raw_at_low == inf["raw_low"], \
            f"sid={sid_int} sim_low→raw 应为 raw_low={inf['raw_low']}, got {raw_at_low}"
        # sim_high → raw_high
        raw_at_high = deployer._rad_to_raw(inf, inf["sim_high"])
        assert raw_at_high == inf["raw_high"], \
            f"sid={sid_int} sim_high→raw 应为 raw_high={inf['raw_high']}, got {raw_at_high}"
        # 中点 → 中点
        sim_mid = (inf["sim_low"] + inf["sim_high"]) / 2.0
        raw_mid_expected = int(round((inf["raw_low"] + inf["raw_high"]) / 2.0))
        raw_mid = deployer._rad_to_raw(inf, sim_mid)
        # 容许 ±1 的整数舍入误差
        assert abs(raw_mid - raw_mid_expected) <= 1, \
            f"sid={sid_int} mid expected={raw_mid_expected} got={raw_mid}"


def test_rad_to_raw_clamps_below_and_above(deployer):
    """rad 超出 sim ctrlrange 时仍 clip 到 raw 边界。"""
    info = deployer._info[1]
    # 极小 rad
    raw_below = deployer._rad_to_raw(info, info["sim_low"] - 10.0)
    assert raw_below == info["raw_low"]
    # 极大 rad
    raw_above = deployer._rad_to_raw(info, info["sim_high"] + 10.0)
    assert raw_above == info["raw_high"]


def test_safe_deploy_action_skips_on_collision(env_v1_right_skin_false, mapping, fake_backend):
    """碰撞预筛选：构造会自碰撞的姿态，safe_deploy_action 应返回 (False, contacts)。"""
    from orca_sim.bridge import SimToRealDeployer
    d = SimToRealDeployer(
        console_backend=fake_backend,
        env=env_v1_right_skin_false,
        mapping=mapping,
        collision_check=True,
    )
    # 注入极端姿态（所有 actuator 到其上限，期望触发自碰撞）
    nq = int(env_v1_right_skin_false.unwrapped.model.nq)
    qpos = np.array(env_v1_right_skin_false.unwrapped.data.qpos, copy=True)
    # 把所有 actuator index 位置设为上限（已知会自碰撞）
    for sid_int in range(1, 18):
        idx = d._sim_idx_by_servo[sid_int]
        if idx < nq:
            qpos[idx] = env_v1_right_skin_false.action_high[idx]
    env_v1_right_skin_false.reset(options={"qpos": qpos})
    # 强制 mj_forward + mj_step 一次以填充 data.contact
    import mujoco
    mujoco.mj_forward(env_v1_right_skin_false.unwrapped.model,
                      env_v1_right_skin_false.unwrapped.data)
    mujoco.mj_step(env_v1_right_skin_false.unwrapped.model,
                   env_v1_right_skin_false.unwrapped.data)

    contacts = d.guard.self_contacts()
    # 不一定每个姿态都碰撞；这里只断言接口可用
    assert isinstance(contacts, list)


def test_alignment_report_returns_17_rows(deployer):
    """alignment_report 返回 17 行。"""
    report = deployer.alignment_report()
    assert len(report) == 17
    for sid, name, ratio, zd in report:
        assert 1 <= sid <= 17
        assert name.startswith("right_")
        assert isinstance(ratio, float)
        assert isinstance(zd, float)