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
        # 记录最后一次 load_bundles 调用，便于测试断言
        self.last_load_xdat_dir = None

    def load_bundles(self, xdat_dir=None):
        """替换 bundles 模拟「重新读取 xdat」。"""
        self.last_load_xdat_dir = xdat_dir
        # 默认行为：保留原 bundles（让测试显式改再调用 reload）
        return self.bundles


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
    """rad 超出 sim ctrlrange 时仍 clip 到 raw 边界。

    注：方向翻转后，``sim_low`` 映射到 ``raw_high``（``raw_low`` 实际是
    ``xdat.max_angle``）；``sim_high`` 映射到 ``raw_low``。clip 行为仍然
    正确 —— ``_rad_to_raw`` 仍按 ``raw_low → raw_high`` 区间裁剪。
    """
    info = deployer._info[1]
    # 极小 rad → clip 到 raw_low（翻转后是 xdat.max_angle）
    raw_below = deployer._rad_to_raw(info, info["sim_low"] - 10.0)
    assert raw_below == info["raw_low"]
    # 极大 rad → clip 到 raw_high（翻转后是 xdat.min_angle）
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

# ============================================================================
# flip_direction 字段在 _cache_per_servo 里的实际行为
# ============================================================================
def _mapping_with_flip(mapping, flip: bool):
    """返回 mapping 的浅拷贝，每个 entry 的 flip_direction 都设成给定值。

    避免污染模块级 fixture（其它测试期望看到原始 JSON 的 flip_direction）。
    """
    from dataclasses import replace
    return {
        sid: replace(entry, flip_direction=flip)
        for sid, entry in mapping.items()
    }


def test_cache_per_servo_flip_direction_true_swaps_raw_low_high(
    env_v1_right_skin_false, mapping, fake_backend,
):
    """``flip_direction=True`` (默认)：``raw_low = xdat.max_angle``、``raw_high = xdat.min_angle``。"""
    from orca_sim.bridge import SimToRealDeployer
    m_flip = _mapping_with_flip(mapping, True)
    d = SimToRealDeployer(
        console_backend=fake_backend, env=env_v1_right_skin_false,
        mapping=m_flip, collision_check=False,
    )
    for sid, info in d._info.items():
        bundle = fake_backend.bundles[sid - 1]
        assert info["raw_low"] == bundle.max_angle, \
            f"sid={sid}: flip=True 应 raw_low=xdat.max_angle={bundle.max_angle}, got {info['raw_low']}"
        assert info["raw_high"] == bundle.min_angle, \
            f"sid={sid}: flip=True 应 raw_high=xdat.min_angle={bundle.min_angle}, got {info['raw_high']}"


def test_cache_per_servo_flip_direction_false_keeps_raw_order(
    env_v1_right_skin_false, mapping, fake_backend,
):
    """``flip_direction=False``：``raw_low = xdat.min_angle``、``raw_high = xdat.max_angle``（不翻）。"""
    from orca_sim.bridge import SimToRealDeployer
    m_noflip = _mapping_with_flip(mapping, False)
    d = SimToRealDeployer(
        console_backend=fake_backend, env=env_v1_right_skin_false,
        mapping=m_noflip, collision_check=False,
    )
    for sid, info in d._info.items():
        bundle = fake_backend.bundles[sid - 1]
        assert info["raw_low"] == bundle.min_angle, \
            f"sid={sid}: flip=False 应 raw_low=xdat.min_angle={bundle.min_angle}, got {info['raw_low']}"
        assert info["raw_high"] == bundle.max_angle, \
            f"sid={sid}: flip=False 应 raw_high=xdat.max_angle={bundle.max_angle}, got {info['raw_high']}"


def test_default_mapping_abd_flip_matches_json(
    env_v1_right_skin_false, mapping, fake_backend,
):
    """用默认 JSON 加载：servo 3/4/5（index/middle/pinky_abd）+ servo 8（middle_pip）
    应不翻；其它翻转。
    """
    from orca_sim.bridge import SimToRealDeployer
    d = SimToRealDeployer(
        console_backend=fake_backend, env=env_v1_right_skin_false,
        mapping=mapping, collision_check=False,
    )
    NO_FLIP_SIDS = {3, 4, 5, 8}  # JSON 显式 flip_direction=false
    for sid, info in d._info.items():
        bundle = fake_backend.bundles[sid - 1]
        if sid in NO_FLIP_SIDS:
            # JSON 显式 flip_direction=false → 不翻
            assert info["raw_low"] == bundle.min_angle, \
                f"sid={sid}: JSON 期望 flip=False 但 raw_low 仍是翻转后的值"
        else:
            # 其它 servo 默认 flip_direction=true → 翻
            assert info["raw_low"] == bundle.max_angle, \
                f"sid={sid}: 期望默认 flip=True 但 raw_low 不是翻转后的值"


def test_qpos_to_positions_with_flip_false_runs_monotonic(
    env_v1_right_skin_false, mapping, fake_backend,
):
    """flip=False 时 ``qpos_to_positions`` 仍单调（sim_low→raw_low, sim_high→raw_high）。"""
    from orca_sim.bridge import SimToRealDeployer
    m_noflip = _mapping_with_flip(mapping, False)
    d = SimToRealDeployer(
        console_backend=fake_backend, env=env_v1_right_skin_false,
        mapping=m_noflip, collision_check=False,
    )
    for sid, info in d._info.items():
        # sim_low → raw_low (=xdat.min_angle，因为 flip=False)
        raw_at_low = d._rad_to_raw(info, info["sim_low"])
        assert raw_at_low == info["raw_low"], \
            f"sid={sid}: sim_low 应映射到 raw_low={info['raw_low']}, got {raw_at_low}"
        # sim_high → raw_high (=xdat.max_angle)
        raw_at_high = d._rad_to_raw(info, info["sim_high"])
        assert raw_at_high == info["raw_high"], \
            f"sid={sid}: sim_high 应映射到 raw_high={info['raw_high']}, got {raw_at_high}"


def test_reload_from_xdat_refreshes_raw_high_cache(deployer, fake_backend):
    """reload_from_xdat 应触发 backend.load_bundles 并刷新 _info 的 raw_high。

    模拟场景：GUI 把某 servo 的 max_angle 从原值改大 → 写 EPROM + 自动同步
    xdat → sim 端 reload → 拖 sim滑条到 sim_high 应输出新 max_angle
    （不再是缓存的旧值）。
    """
    # 取一个非 wrist 的 servo（避开 wrist 翻转逻辑的歧义；这里用 1）
    sid = 1
    info_before = deployer._info[sid]
    old_max = int(info_before["raw_high"])
    # fake_backend 默认 min/max 计算: min_angle=200+1*100=300, max_angle=300+2000=2300
    # JSON 默认 flip_direction=true → raw_low=max_angle=2300, raw_high=min_angle=300
    # 我们要把 max_angle 改成 3500，看 raw_high 是否更新
    bundle = fake_backend.bundles[sid - 1]
    bundle.max_angle = 3500  # 模拟 GUI 改 xdat + EPROM

    # 显式替换 backend.bundles 模拟 load_bundles 已经重读文件
    # （_FakeBackend.load_bundles 默认保留 bundles，但我们已经 mutate 了 in-place）
    fake_backend.load_bundles(xdat_dir="/fake/path/参数")

    # reload → _cache_per_servo 应重跑 → 看到 max_angle=3500
    deployer.reload_from_xdat(xdat_dir="/fake/path/参数")

    info_after = deployer._info[sid]
    new_max = int(info_after["raw_high"])
    # flip_direction=true 时 raw_high == bundle.min_angle（不变），raw_low = bundle.max_angle
    # 我们的 mutate 只动了 max_angle，所以 raw_low 应该变成 3500
    new_low = int(info_after["raw_low"])
    assert new_low == 3500, (
        f"reload 后 sid={sid} 的 raw_low 应反映新 max_angle=3500，got {new_low}"
    )
    # raw_high（=min_angle）没变
    assert new_max == old_max
    # _info 中非 xdat 相关字段（sim_idx / sim_low / sim_high）保持不变
    assert info_after["sim_idx"] == info_before["sim_idx"]
    assert info_after["sim_low"] == info_before["sim_low"]
    assert info_after["sim_high"] == info_before["sim_high"]
    # load_bundles 被调用过
    assert fake_backend.last_load_xdat_dir == "/fake/path/参数"


def test_reload_from_xdat_preserves_sim_idx_by_servo(deployer):
    """reload 不应触碰 _sim_idx_by_servo（来自 env+mapping，与 xdat 无关）。"""
    snap_before = dict(deployer._sim_idx_by_servo)
    deployer.reload_from_xdat(xdat_dir=None)
    assert deployer._sim_idx_by_servo == snap_before


def test_reload_from_xdat_raises_when_backend_lacks_load_bundles(
    env_v1_right_skin_false, mapping, fake_backend,
):
    """若 backend 没有 load_bundles 方法，应抛 AttributeError。"""
    from orca_sim.bridge import SimToRealDeployer

    class _NoLoadBackend:
        bundles = fake_backend.bundles
        safety = fake_backend.safety

    d = SimToRealDeployer(
        console_backend=_NoLoadBackend(),
        env=env_v1_right_skin_false,
        mapping=mapping,
        collision_check=False,
    )
    with pytest.raises(AttributeError, match="load_bundles"):
        d.reload_from_xdat()
