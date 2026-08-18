"""retarget/test_sim_skeleton.py — SimSkeleton 单元测试。

覆盖：
    1. from_model 抽到 17 个 body，命名 / 轴向 / range 与 MJCF 一致
    2. forward_kinematics：rest / hi / lo 三种 qpos 下，5 指尖 world xpos 单调正确
    3. fingertip_local_lengths：每个手指的链长 > 0
    4. fingertip_body_ids 顺序正确（thumb → pinky）
    5. 数据污染防护：forward_kinematics 不应改调用方原 qpos
"""

import numpy as np
import pytest


# ----------------------------------------------------------------------
# 1. from_model 基本结构
# ----------------------------------------------------------------------
def test_skel_has_17_bones(skel):
    assert len(skel.bones) == 17


def test_skel_bones_match_known_names(skel):
    expected = {
        "right_palm",
        "right_thumb_mp", "right_thumb_pp", "right_thumb_ip", "right_thumb_dp",
        "right_index_mp", "right_index_pp", "right_index_ip",
        "right_middle_mp", "right_middle_pp", "right_middle_ip",
        "right_ring_mp", "right_ring_pp", "right_ring_ip",
        "right_pinky_mp", "right_pinky_pp", "right_pinky_ip",
    }
    assert set(skel.bones.keys()) == expected


def test_skel_palm_is_root(skel):
    """palm 的 parent 应该是 right_tower（无 joint 的固定 body）。"""
    palm = skel.get_bone("right_palm")
    assert palm.body_name == "right_palm"
    # palm 的 parent body id 应是 1（right_tower）
    assert palm.parent_body_id > 0


def test_skel_fingertip_order(skel):
    assert skel.fingertip_body_names == (
        "right_thumb_dp",
        "right_index_ip",
        "right_middle_ip",
        "right_ring_ip",
        "right_pinky_ip",
    )


def test_skel_index_ip_axis(skel):
    """index_ip 的 joint axis 必须是 (0, -1, 0)，range = (-0.349, +1.885)。"""
    b = skel.get_bone("right_index_ip")
    np.testing.assert_allclose(b.joint_axis_local, [0.0, -1.0, 0.0], atol=1e-6)
    assert b.joint_range == pytest.approx((-0.34907, 1.88496), abs=1e-4)
    assert b.joint_qpos_idx == 7


def test_skel_thumb_abd_axis_is_tilted(skel):
    """thumb_abd 的 axis 是斜轴 (0, 0.342, 0.940)，不是单轴。"""
    b = skel.get_bone("right_thumb_pp")
    np.testing.assert_allclose(b.joint_axis_local, [0.0, 0.34202086, 0.93969236], atol=1e-4)
    # 单边 range
    assert b.joint_range[1] == pytest.approx(0.0, abs=1e-6)
    assert b.joint_range[0] == pytest.approx(-1.08211, abs=1e-4)
    # ref 偏置
    assert b.joint_ref == pytest.approx(-0.73304, abs=1e-4)


def test_skel_index_abd_is_single_sided(skel):
    """index_abd 是单边偏负：range (-1.04577, +0.24577)。"""
    b = skel.get_bone("right_index_mp")
    assert b.joint_range == pytest.approx((-1.04577, 0.24577), abs=1e-4)


def test_skel_pinky_abd_is_single_sided(skel):
    """pinky_abd 是单边偏正：range (-0.122, +1.169)。"""
    b = skel.get_bone("right_pinky_mp")
    assert b.joint_range == pytest.approx((-0.12244, 1.1691), abs=1e-4)


# ----------------------------------------------------------------------
# 2. forward_kinematics
# ----------------------------------------------------------------------
def test_fk_rest_xpos_matches_known(skel_with_data, env_v1_right):
    """rest 时 5 指尖 xpos 与 env.unwrapped 直接读的对得上（差 < 1mm）。"""
    skel, data = skel_with_data
    import mujoco
    mujoco.mj_forward(data.model, data)
    expected = {
        n: data.xpos[skel.get_bone(n).body_id, :3].copy()
        for n in skel.fingertip_body_names
    }

    qpos = data.qpos[:17].copy()
    got = skel.forward_kinematics(qpos, data)
    for n in skel.fingertip_body_names:
        np.testing.assert_allclose(got[n], expected[n], atol=1e-3, rtol=0,
                                   err_msg=f"{n} mismatch")


def test_fk_hi_moves_fingertips(skel_with_data):
    """ctrl=hi 时所有指尖位置应与 rest 不同（>=5mm）。"""
    skel, data = skel_with_data
    import mujoco
    mujoco.mj_forward(data.model, data)
    qpos_rest = data.qpos[:17].copy()
    qpos_hi = data.model.actuator_ctrlrange[:, 1].copy()
    rest_xpos = skel.forward_kinematics(qpos_rest, data)
    hi_xpos = skel.forward_kinematics(qpos_hi, data)
    for n in skel.fingertip_body_names:
        d = np.linalg.norm(hi_xpos[n] - rest_xpos[n])
        assert d > 5e-3, f"{n} moved only {d*1000:.2f}mm at ctrl=hi"


def test_fk_pinky_abd_lo_moves_pinky(skel_with_data):
    """pinky_abd=lo（最小值，应比 rest 更靠外展）和 rest 比较，pinky 应该有位移。"""
    skel, data = skel_with_data
    import mujoco
    mujoco.mj_forward(data.model, data)
    qpos_rest = data.qpos[:17].copy()
    qpos = qpos_rest.copy()
    qpos[14] = data.model.actuator_ctrlrange[14, 0]   # pinky_abd lo
    rest_xpos = skel.forward_kinematics(qpos_rest, data)
    lo_xpos = skel.forward_kinematics(qpos, data)
    d = np.linalg.norm(lo_xpos["right_pinky_ip"] - rest_xpos["right_pinky_ip"])
    assert d > 1e-3, f"pinky_abd=lo should move pinky, moved {d*1000:.2f}mm"


def test_fk_does_not_mutate_caller_qpos(skel_with_data):
    """forward_kinematics 不应改调用方传入的 qpos 数组。"""
    skel, data = skel_with_data
    qpos_in = data.model.actuator_ctrlrange[:, 1].copy()
    qpos_copy = qpos_in.copy()
    skel.forward_kinematics(qpos_in, data)
    np.testing.assert_array_equal(qpos_in, qpos_copy)


# ----------------------------------------------------------------------
# 3. fingertip_local_lengths
# ----------------------------------------------------------------------
def test_fingertip_lengths_positive(skel):
    lens = skel.fingertip_local_lengths()
    assert set(lens.keys()) == {"thumb", "index", "middle", "ring", "pinky"}
    for finger, length in lens.items():
        assert length > 0, f"{finger} chain length must be > 0"


def test_fingertip_lengths_in_realistic_range(skel):
    """人手 sim 手指长度应在 20-100mm 之间。"""
    lens = skel.fingertip_local_lengths()
    for finger, length in lens.items():
        assert 0.02 < length < 0.10, f"{finger} length {length*1000:.1f}mm 超出合理范围"


# ----------------------------------------------------------------------
# 4. fingertip_body_ids
# ----------------------------------------------------------------------
def test_fingertip_body_ids_are_int64(skel):
    ids = skel.fingertip_body_ids()
    assert ids.dtype == np.int64
    assert ids.shape == (5,)


def test_fingertip_body_ids_match_xpos(skel_with_data):
    skel, data = skel_with_data
    import mujoco
    mujoco.mj_forward(data.model, data)
    ids = skel.fingertip_body_ids()
    for i, name in enumerate(skel.fingertip_body_names):
        expected = data.xpos[skel.get_bone(name).body_id, :3]
        np.testing.assert_allclose(data.xpos[ids[i], :3], expected, atol=1e-9)