"""test_mapping.py：JSON 加载、servo_id ↔ sim_actuator 双向查表、缺失/越界报错。"""

from __future__ import annotations

import json
import pathlib

import pytest


def test_load_default_mapping_has_17_entries(mapping):
    """默认 JSON 应包含 1..17 共 17 条。"""
    assert len(mapping) == 17
    for sid in range(1, 18):
        assert str(sid) in mapping


def test_mapping_entries_have_required_fields(mapping):
    """每个 entry 含 chinese / sim_actuator / joint_role。"""
    for sid_str, entry in mapping.items():
        assert entry.chinese, f"sid={sid_str} missing chinese"
        assert entry.sim_actuator.startswith("right_"), \
            f"sid={sid_str} actuator={entry.sim_actuator!r} 应以 right_ 开头"
        assert entry.joint_role in ("pip", "mcp", "abd", "wrist")


def test_sid_17_is_wrist(mapping):
    """用户确认 wrist = servo_id 17 → right_wrist。"""
    assert mapping["17"].chinese == "腕关节"
    assert mapping["17"].sim_actuator == "right_wrist"
    assert mapping["17"].joint_role == "wrist"


def test_sim_actuator_for_servo(mapping):
    """反向查表：servo_id → sim actuator 名。"""
    from orca_sim.bridge.mapping import sim_actuator_for_servo
    assert sim_actuator_for_servo(mapping, 1) == "right_index_pip"
    assert sim_actuator_for_servo(mapping, 17) == "right_wrist"


def test_unknown_servo_id_raises(mapping):
    from orca_sim.bridge.mapping import sim_actuator_for_servo
    with pytest.raises(KeyError):
        sim_actuator_for_servo(mapping, 18)
    with pytest.raises(KeyError):
        sim_actuator_for_servo(mapping, 0)


def test_resolve_sim_indices(env_v1_right_skin_false, mapping):
    """动态解析 sim actuator index（不依赖硬编码）。"""
    from orca_sim.bridge.mapping import resolve_sim_indices
    indices = resolve_sim_indices(mapping, env_v1_right_skin_false.unwrapped.model)
    assert len(indices) == 17
    # wrist 应是 sim index 0（v1 right.mjcf 声明顺序）
    assert indices[17] == 0


def test_invalid_json_missing_field(tmp_path):
    """缺字段的 JSON 应抛 ValueError。"""
    from orca_sim.bridge import load_joint_mapping
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"servo_id_to_actuator": {
        "1": {"chinese": "x", "sim_actuator": "right_index_pip"},  # 缺 joint_role
    }}))
    with pytest.raises(ValueError, match="missing field"):
        load_joint_mapping(path=bad)


def test_invalid_json_out_of_range(tmp_path):
    """servo_id 越界应抛 ValueError。"""
    from orca_sim.bridge import load_joint_mapping
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"servo_id_to_actuator": {
        "0": {"chinese": "x", "sim_actuator": "x", "joint_role": "x"},
    }}))
    with pytest.raises(ValueError, match="out of range"):
        load_joint_mapping(path=bad)


def test_invalid_json_missing_sid(tmp_path):
    """缺 servo_id 应抛 ValueError。"""
    from orca_sim.bridge import load_joint_mapping
    bad = tmp_path / "bad.json"
    # 只有 1..16，缺 17
    entries = {
        str(i): {"chinese": str(i), "sim_actuator": f"r{i}", "joint_role": "pip"}
        for i in range(1, 17)
    }
    bad.write_text(json.dumps({"servo_id_to_actuator": entries}))
    with pytest.raises(ValueError, match="missing servo_ids"):
        load_joint_mapping(path=bad)


def test_missing_file(tmp_path):
    from orca_sim.bridge import load_joint_mapping
    with pytest.raises(FileNotFoundError):
        load_joint_mapping(path=tmp_path / "nonexistent.json")

# ============================================================================
# flip_direction 字段
# ============================================================================
def test_flip_direction_defaults_true_when_missing(mapping):
    """JSON 缺 ``flip_direction`` 字段时默认为 True（保持向后兼容）。"""
    # 默认 JSON 现在显式给某些 servo 写了 false；但其余 servo 没写，应为 True
    FLIP_FALSE_SIDS = {"3", "4", "5", "8"}  # 用户实测 sim 与真机方向一致的关节
    for sid_str, entry in mapping.items():
        # 已显式标记的应反映为 False
        if sid_str in FLIP_FALSE_SIDS:
            assert entry.flip_direction is False, \
                f"sid={sid_str} 期望 flip=False（实测 sim 与真机方向一致）"
        else:
            # 其它 servo 默认 True
            assert entry.flip_direction is True, \
                f"sid={sid_str} 期望默认 flip=True"


def test_flip_direction_loader_accepts_explicit_value(tmp_path):
    """自定义 JSON：显式 ``flip_direction: true/false`` 都能被正确读出。"""
    custom = {
        "version": "v1", "hand": "right", "skin": False,
        "servo_id_to_actuator": {
            str(sid): {
                "chinese": f"servo{sid}", "sim_actuator": f"right_joint{sid}",
                "joint_role": "pip",
                "flip_direction": (sid % 2 == 0),  # 偶数不翻、奇数翻
            } for sid in range(1, 18)
        },
    }
    p = tmp_path / "custom.json"
    p.write_text(json.dumps(custom, ensure_ascii=False), encoding="utf-8")
    from orca_sim.bridge import load_joint_mapping
    m = load_joint_mapping(path=p)
    for sid in range(1, 18):
        expected = (sid % 2 == 0)
        assert m[str(sid)].flip_direction is expected, \
            f"sid={sid} expected flip={expected}"
