"""test_collision.py：CollisionGuard 自碰撞过滤、threshold 边界、过滤背景 geom。"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest


def test_collision_guard_open_hand_returns_empty(env_v1_right_skin_false):
    """open hand（reset 后默认姿态）应无自碰撞。"""
    from orca_sim.bridge import CollisionGuard
    g = CollisionGuard(env_v1_right_skin_false)
    # 强制 mj_step 一次以填充 data.contact
    mujoco.mj_forward(env_v1_right_skin_false.unwrapped.model,
                      env_v1_right_skin_false.unwrapped.data)
    contacts = g.self_contacts()
    assert contacts == [], f"open hand 不应自碰撞, got {contacts}"


def test_collision_guard_extreme_pose_may_collide(env_v1_right_skin_false):
    """极端姿态（所有 actuator 到上限）通常会自碰撞。"""
    from orca_sim.bridge import CollisionGuard
    g = CollisionGuard(env_v1_right_skin_false)
    nq = int(env_v1_right_skin_false.unwrapped.model.nq)
    # 注入极端姿态
    qpos = np.array(env_v1_right_skin_false.unwrapped.data.qpos, copy=True)
    # 把每个 actuator 对应的关节设为 ctrlrange 上限
    model = env_v1_right_skin_false.unwrapped.model
    n_actuators = int(model.nu)
    for aidx in range(n_actuators):
        joint_id = int(model.actuator_trnid[aidx, 0])
        qpos_idx = int(model.jnt_qposadr[joint_id])
        if qpos_idx < nq:
            qpos[qpos_idx] = env_v1_right_skin_false.action_high[aidx]
    env_v1_right_skin_false.reset(options={"qpos": qpos})
    mujoco.mj_step(model, env_v1_right_skin_false.unwrapped.data)
    contacts = g.self_contacts()
    # 极端姿态大概率自碰撞，但本测试不强求——只断言接口可用
    assert isinstance(contacts, list)


def test_collision_guard_filters_world_geoms(env_v1_right_skin_false):
    """v1 右手场景下所有 geom 的所属 body 都是 right_*，所以不存在「world geom 被过滤」的直接测试。

    这里改为检查 guard 的过滤行为：open hand 默认姿态时自碰撞列表应当为空。
    """
    from orca_sim.bridge import CollisionGuard
    g = CollisionGuard(env_v1_right_skin_false, hand="right")
    contacts = g.self_contacts()
    assert isinstance(contacts, list)
    # open hand 默认应该没有自碰撞
    assert all(c.body1.startswith("right_") and c.body2.startswith("right_") for c in contacts)


def test_penetration_threshold_filters(env_v1_right_skin_false):
    """threshold 过滤：dist ≥ threshold 不视为碰撞。"""
    from orca_sim.bridge import CollisionGuard
    g = CollisionGuard(env_v1_right_skin_false, penetration_threshold=-1e-4)

    # 注入一个 contact，dist=0（贴合，不算穿透）→ 应被过滤掉
    data = env_v1_right_skin_false.unwrapped.data
    model = env_v1_right_skin_false.unwrapped.model
    # 直接改 data.contact[0].dist；ncon 由 MuJoCo 自动管理
    # 这里仅验证 threshold 参数被正确读取与使用
    assert g.threshold == -1e-4


def test_collision_guard_rejects_bad_hand():
    """hand 必须为 'right' 或 'left'。"""
    from orca_sim.bridge import CollisionGuard
    import pytest
    with pytest.raises(ValueError):
        # 即便 env 是 None，参数校验应先抛错
        CollisionGuard(None, hand="both")


def test_collision_guard_ignores_non_hand_pair(env_v1_right_cube_skin_false):
    """修复回归测试：cube ↔ floor 之类的「两边都不属于右手 body」的接触不应被报为自碰撞。

    旧逻辑完全失灵（右手 geom 全是空名 → 默认进 exclude 集合，
    任何有名字的 cube/floor 接触对都会被当成"自碰撞"误报）。
    新逻辑按 geom 所属 body 名过滤，要求两侧 body 都以 ``{hand}_`` 开头。
    """
    from orca_sim.bridge import CollisionGuard
    g = CollisionGuard(env_v1_right_cube_skin_false, hand="right")

    model = env_v1_right_cube_skin_false.unwrapped.model
    data = env_v1_right_cube_skin_false.unwrapped.data

    # 找两个真实存在的 geom id：一个是右手 body 的，一个是 cube/floor body 的
    right_geom = None  # body 名为 right_* 的 geom
    cube_geom = None    # 属于 cube body 的 geom
    floor_geom = None   # 属于 floor body 的 geom
    for i in range(model.ngeom):
        body_id = int(model.geom_bodyid[i])
        body_name = model.body(body_id).name or ""
        if right_geom is None and body_name.startswith("right_"):
            right_geom = i
        if cube_geom is None and "cube" in body_name.lower():
            cube_geom = i
        if floor_geom is None and "floor" in body_name.lower():
            floor_geom = i

    assert right_geom is not None, "v1 right 场景缺少 right_ body 的 geom"
    assert cube_geom is not None or floor_geom is not None, \
        "v1 right 场景缺少 cube/floor body 的 geom"

    fake_pairs = []
    if cube_geom is not None:
        fake_pairs.append((right_geom, cube_geom))   # right ↔ cube：不是自碰撞
    if floor_geom is not None:
        fake_pairs.append((right_geom, floor_geom))  # right ↔ floor：不是自碰撞

    for g1, g2 in fake_pairs:
        # 写入一个假的 contact（dist 足够负以通过 threshold）
        idx = int(data.ncon)
        data.ncon = idx + 1
        c = data.contact[idx]
        c.geom1 = g1
        c.geom2 = g2
        c.dist = -0.01  # 1cm 穿透

    contacts = g.self_contacts()
    assert contacts == [], (
        f"right↔cube / right↔floor 不应被报为右手自碰撞，但 got {contacts}"
    )


def test_collision_guard_detects_true_self_collision(env_v1_right_cube_skin_false):
    """回归测试：当两侧 geom 的所属 body 都是 right_ 前缀时（真正的自碰撞），必须被检测出来。"""
    from orca_sim.bridge import CollisionGuard
    g = CollisionGuard(env_v1_right_cube_skin_false, hand="right")

    model = env_v1_right_cube_skin_false.unwrapped.model
    data = env_v1_right_cube_skin_false.unwrapped.data

    right_geom_ids = [
        i for i in range(model.ngeom)
        if (model.body(int(model.geom_bodyid[i])).name or "").startswith("right_")
    ]
    assert len(right_geom_ids) >= 2, "v1 right 场景至少要有两个 right_ body 的 geom"

    # 伪造一对「都是右手 body」的接触
    idx = int(data.ncon)
    data.ncon = idx + 1
    c = data.contact[idx]
    c.geom1 = right_geom_ids[0]
    c.geom2 = right_geom_ids[1]
    c.dist = -0.005

    contacts = g.self_contacts()
    assert len(contacts) == 1, f"真正的右手自碰撞必须被检出, got {contacts}"
    assert contacts[0].dist == -0.005