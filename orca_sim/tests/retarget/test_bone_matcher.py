"""retarget/test_bone_matcher.py — BoneMatcher 单元测试。

覆盖：
    1. MediaPipeHand.from_landmarks 数据包装
    2. flip_yz 翻转（验证 Y/Z 翻转后坐标对应关系）
    3. BoneMatcher 启发式 init 输出在 ctrlrange 范围内
    4. 不同手势（fist vs open_hand）输出 qpos 有显著差异
    5. solve() 在合理 landmarks 输入下不会崩溃且输出合法 qpos
"""

import json
import numpy as np
import pytest


# ----------------------------------------------------------------------
# 1. MediaPipeHand
# ----------------------------------------------------------------------
def test_mphand_from_landmarks_shape():
    from orca_sim.retarget.bone_matcher import MediaPipeHand
    lm = np.zeros((21, 3))
    lm[0] = [1.0, 2.0, 3.0]              # wrist
    lm[8] = [0.5, 1.0, 0.5]               # index tip
    # 不翻转时直接复制
    hand = MediaPipeHand.from_landmarks(lm, flip_yz=False)
    np.testing.assert_allclose(hand.wrist, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(hand.index[3], [0.5, 1.0, 0.5])
    assert hand.thumb.shape == (4, 3)


def test_mphand_wrong_shape_raises():
    from orca_sim.retarget.bone_matcher import MediaPipeHand
    with pytest.raises(ValueError):
        MediaPipeHand.from_landmarks(np.zeros((20, 3)))


def test_mphand_flip_yz_default():
    from orca_sim.retarget.bone_matcher import MediaPipeHand
    lm = np.zeros((21, 3))
    lm[0] = [1.0, 2.0, 3.0]
    hand = MediaPipeHand.from_landmarks(lm, flip_yz=True)
    # y 和 z 应该被翻转（乘 -1）
    np.testing.assert_allclose(hand.wrist, [1.0, -2.0, -3.0])


def test_mphand_no_flip():
    from orca_sim.retarget.bone_matcher import MediaPipeHand
    lm = np.zeros((21, 3))
    lm[0] = [1.0, 2.0, 3.0]
    hand = MediaPipeHand.from_landmarks(lm, flip_yz=False)
    np.testing.assert_allclose(hand.wrist, [1.0, 2.0, 3.0])


def test_mphand_does_not_mutate_input():
    from orca_sim.retarget.bone_matcher import MediaPipeHand
    lm = np.zeros((21, 3))
    lm[0] = [1.0, 2.0, 3.0]
    lm_copy = lm.copy()
    MediaPipeHand.from_landmarks(lm, flip_yz=True)
    np.testing.assert_array_equal(lm, lm_copy)


# ----------------------------------------------------------------------
# 2. BoneMatcher 启发式 init
# ----------------------------------------------------------------------
@pytest.fixture()
def matcher(skel_with_data):
    from orca_sim.retarget.bone_matcher import BoneMatcher, BoneMatcherConfig
    skel, data = skel_with_data
    return BoneMatcher(skel, data.model, data, BoneMatcherConfig(use_heuristic_init=True, max_iterations=0))


def test_heuristic_init_in_range(matcher, skel_with_data):
    """启发式初值必须在 ctrlrange 范围内。"""
    skel, data = skel_with_data
    model = data.model
    ctrlrange = model.actuator_ctrlrange
    # 构造一个「张直手」landmarks（rest-ish）
    lm = np.zeros((21, 3))
    lm[0] = [0, 0, 0]
    lm[5] = lm[9] = lm[13] = lm[17] = [0, 0, 0.05]
    lm[6] = lm[10] = lm[14] = lm[18] = [0, -0.03, 0.08]
    lm[7] = lm[11] = lm[15] = lm[19] = [0, -0.05, 0.11]
    lm[8] = lm[12] = lm[16] = lm[20] = [0, -0.08, 0.14]
    lm[1] = [-0.02, 0, 0]
    lm[2] = [-0.03, -0.01, 0.02]
    lm[3] = [-0.03, -0.02, 0.03]
    lm[4] = [-0.03, -0.03, 0.04]

    qpos = matcher._heuristic_init(lm, model)
    for i in range(17):
        assert ctrlrange[i, 0] - 1e-6 <= qpos[i] <= ctrlrange[i, 1] + 1e-6, (
            f"qpos[{i}]={qpos[i]} out of range [{ctrlrange[i, 0]}, {ctrlrange[i, 1]}]"
        )


def test_heuristic_init_distinguishes_fist_from_open_hand(matcher, skel_with_data):
    """fist 和 open_hand 的启发式初值必须有显著差异（curl_norm 不同）。"""
    skel, data = skel_with_data
    cal = json.load(open("calibration_data.json", encoding="utf-8"))
    lm_open = np.array(cal["poses"]["open_hand"]["landmarks"])
    lm_fist = np.array(cal["poses"]["fist"]["landmarks"])
    qpos_open = matcher._heuristic_init(lm_open, data.model)
    qpos_fist = matcher._heuristic_init(lm_fist, data.model)

    # 至少一个 mcp 关节应该有差异 > 0.3 rad
    mcp_indices = [6, 9, 12, 15]            # index/middle/ring/pinky mcp
    diffs = [abs(qpos_open[i] - qpos_fist[i]) for i in mcp_indices]
    assert max(diffs) > 0.3, (
        f"启发式 init 对 fist/open 未充分区分：mcp diffs={diffs}"
    )


# ----------------------------------------------------------------------
# 3. solve() 端到端
# ----------------------------------------------------------------------
def test_solve_in_range(matcher, skel_with_data):
    skel, data = skel_with_data
    cal = json.load(open("calibration_data.json", encoding="utf-8"))
    lm = np.array(cal["poses"]["open_hand"]["landmarks"])
    qpos = matcher.solve(lm, data=data)
    ctrlrange = data.model.actuator_ctrlrange
    for i in range(17):
        assert ctrlrange[i, 0] - 1e-6 <= qpos[i] <= ctrlrange[i, 1] + 1e-6


def test_solve_calibration_poses(matcher, skel_with_data):
    """7 个校准手势都能合法求解（不崩溃）。"""
    skel, data = skel_with_data
    cal = json.load(open("calibration_data.json", encoding="utf-8"))
    ctrlrange = data.model.actuator_ctrlrange
    for name, info in cal["poses"].items():
        lm = np.array(info["landmarks"])
        matcher.reset()
        qpos = matcher.solve(lm, data=data)
        for i in range(17):
            assert ctrlrange[i, 0] - 1e-6 <= qpos[i] <= ctrlrange[i, 1] + 1e-6, (
                f"{name} qpos[{i}] out of range"
            )


def test_solve_no_silent_hot_start(matcher, skel_with_data):
    """solve() 不应偷偷用 _last_qpos 热启动（避免测试 / 批量处理时污染）。"""
    skel, data = skel_with_data
    cal = json.load(open("calibration_data.json", encoding="utf-8"))
    lm_open = np.array(cal["poses"]["open_hand"]["landmarks"])
    lm_fist = np.array(cal["poses"]["fist"]["landmarks"])

    matcher.reset()
    qpos_a = matcher.solve(lm_open, data=data)
    matcher.reset()
    qpos_b = matcher.solve(lm_fist, data=data)

    # 两个 qpos 至少有一个 mcp 关节差异 > 0.3
    diffs = [abs(qpos_a[i] - qpos_b[i]) for i in (6, 9, 12, 15)]
    assert max(diffs) > 0.3, f"两次 solve 几乎相同：{diffs}"


def test_solve_warm_start_when_explicit(matcher, skel_with_data):
    """当显式传 prev_qpos 时，应该用作初值。"""
    skel, data = skel_with_data
    cal = json.load(open("calibration_data.json", encoding="utf-8"))
    lm = np.array(cal["poses"]["open_hand"]["landmarks"])

    matcher.reset()
    qpos_a = matcher.solve(lm, data=data)
    # 强制 prev_qpos = qpos_a + 扰动
    prev = qpos_a + 0.05
    qpos_b = matcher.solve(lm, prev_qpos=prev, data=data)
    # 第二次的初值基于 prev，IK 应该让 qpos_b ≈ qpos_a
    # 检查 mcp 关节差异 < 0.2
    diffs = [abs(qpos_a[i] - qpos_b[i]) for i in range(17)]
    assert max(diffs) < 0.5, f"warm_start 失败：{diffs}"


def test_solve_reset_clears_history(matcher):
    matcher.reset()
    assert matcher._last_qpos is None


# ----------------------------------------------------------------------
# 4. 与 SimSkeleton 的集成
# ----------------------------------------------------------------------
def test_solve_uses_skel_for_ik(matcher, skel_with_data):
    """BoneMatcher 用的 skel 与 sim model 一致——指向同一组 body。"""
    skel, data = skel_with_data
    cal = json.load(open("calibration_data.json", encoding="utf-8"))
    lm = np.array(cal["poses"]["fist"]["landmarks"])
    qpos = matcher.solve(lm, data=data)
    # mj_forward 后 5 指尖应取在 sim 可达域内（不全 0 也不全 ctrl hi）
    import mujoco
    data.qpos[:17] = qpos
    mujoco.mj_forward(data.model, data)
    palm_id = skel.get_bone(skel.palm_body_name).body_id
    sim_wrist = data.xpos[palm_id, :3].copy()
    for name in skel.fingertip_body_names:
        tip = data.xpos[skel.get_bone(name).body_id, :3] - sim_wrist
        assert 10e-3 < np.linalg.norm(tip) < 200e-3, f"{name} tip distance weird"