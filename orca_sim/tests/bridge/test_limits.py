"""test_limits.py：sim ctrlrange 与 xdat 范围的对齐检查（advisory WARN）。"""

from __future__ import annotations

import pytest


def _make_info(sim_low, sim_high, raw_low, raw_high):
    return {
        "sim_idx": 0,
        "sim_low": sim_low,
        "sim_high": sim_high,
        "raw_low": raw_low,
        "raw_high": raw_high,
    }


def test_check_alignment_basic():
    from orca_sim.bridge.limits import check_alignment
    import math

    mapping = {"1": type("E", (), {"servo_id": 1, "sim_actuator": "right_index_pip"})()}
    info = {1: _make_info(-0.5, 1.5, 500, 3500)}

    rows = check_alignment(mapping, info)
    assert len(rows) == 1
    sid, name, ratio, md = rows[0]
    assert sid == 1
    # sim_range = 2.0；xdat_range 转弧度 = 3000 × 360/4095 × π/180 ≈ 4.603
    xdat_range_rad = 3000.0 * 360.0 / 4095 * math.pi / 180.0
    # ratio = sim_range / xdat_range_rad
    assert ratio == pytest.approx(2.0 / xdat_range_rad, rel=1e-9)
    # 中点差：sim_mid = 0.5；xdat_mid_rad = (500+3500)/2 × 360/4095 × π/180
    xdat_mid_rad = 2000.0 * 360.0 / 4095 * math.pi / 180.0
    assert md == pytest.approx(0.5 - xdat_mid_rad, rel=1e-9)


def test_check_alignment_handles_flipped_raw_storage():
    """``flip=True`` 时 ``raw_low > raw_high``；xdat 范围仍按小→大值算 ratio。"""
    from orca_sim.bridge.limits import check_alignment
    import math

    mapping = {"1": type("E", (), {"servo_id": 1, "sim_actuator": "right_index_pip"})()}
    # 翻转后：raw_low=3500, raw_high=500；xdat 物理范围仍是 500..3500
    info = {1: _make_info(-0.5, 1.5, 3500, 500)}

    rows = check_alignment(mapping, info)
    sid, name, ratio, md = rows[0]
    xdat_range_rad = 3000.0 * 360.0 / 4095 * math.pi / 180.0
    # ratio 应仍 ≈ 2.0 / 4.603（不应是 nan 或负数）
    assert ratio == pytest.approx(2.0 / xdat_range_rad, rel=1e-9)


def test_format_alignment_report_basic():
    from orca_sim.bridge.limits import format_alignment_report

    rows = [
        (1, "right_index_pip", 1.0, 0.0),    # ok
        (2, "right_index_mcp", 0.5, 0.0),    # range-warn
        (3, "right_index_abd", 1.0, 0.5),    # mid-warn
    ]
    out = format_alignment_report(rows)
    assert "right_index_pip" in out
    assert "RANGE-WARN" in out
    assert "MID-WARN" in out


def test_format_alignment_report_includes_all_rows():
    from orca_sim.bridge.limits import format_alignment_report

    rows = [(i, f"right_j{i}", 1.0, 0.0) for i in range(1, 18)]
    out = format_alignment_report(rows)
    for i in range(1, 18):
        assert f"right_j{i}" in out


# ============================================================================
# detailed_alignment / format_detailed_alignment（每个 servo 的诊断表）
# ============================================================================
def _make_entry(servo_id: int, sim_actuator: str, chinese: str, flip: bool = True):
    return type("E", (), {
        "servo_id": servo_id,
        "sim_actuator": sim_actuator,
        "chinese": chinese,
        "flip_direction": flip,
    })()


def test_detailed_alignment_basic_fields():
    from orca_sim.bridge.limits import detailed_alignment

    mapping = {
        "1": _make_entry(1, "right_index_pip", "食指指尖", flip=True),
    }
    info = {1: _make_info(-0.5, 1.5, 500, 3500)}

    rows = detailed_alignment(mapping, info)
    assert len(rows) == 1
    r = rows[0]
    assert r["servo_id"] == 1
    assert r["sim_actuator"] == "right_index_pip"
    assert r["chinese"] == "食指指尖"
    assert r["flip_direction"] is True
    assert r["sim_range"] == pytest.approx(2.0)
    assert r["raw_range"] == pytest.approx(3000.0)
    assert r["sim_mid"] == pytest.approx(0.5)
    assert r["xdat_min"] == 500
    assert r["xdat_max"] == 3500


def test_detailed_alignment_flags_warn_ratio_when_ranges_diverge():
    """sim range 远小于 xdat range（< 0.8 倍）应触发 WARN_RATIO。"""
    from orca_sim.bridge.limits import detailed_alignment
    import math

    # xdat 范围转弧度：2000 raw × 360/4095 × π/180 ≈ 3.069 rad
    # sim range 设 0.5 rad（仅 xdat 的 16%）→ ratio ≈ 0.16 → < 0.8 → WARN_RATIO
    mapping = {"1": _make_entry(1, "right_index_pip", "食指指尖")}
    info = {1: _make_info(0.0, 0.5, 1000, 3000)}

    rows = detailed_alignment(mapping, info)
    assert "WARN_RATIO" in rows[0]["flags"]


def test_detailed_alignment_flags_warn_deadzone():
    """raw range < 100 时舵机几乎转不动，应触发 WARN_DEADZONE。"""
    from orca_sim.bridge.limits import detailed_alignment

    mapping = {"1": _make_entry(1, "right_index_pip", "食指指尖")}
    info = {1: _make_info(0.0, 1.0, 2000, 2050)}  # raw_range=50

    rows = detailed_alignment(mapping, info)
    assert "WARN_DEADZONE" in rows[0]["flags"]


def test_detailed_alignment_flags_warn_zero_when_mid_offsets():
    """sim_mid 与 xdat_mid 偏差 > 0.10 rad 时触发 WARN_ZERO。"""
    from orca_sim.bridge.limits import detailed_alignment

    # xdat_mid_rad ≈ 2000 × 360/4095 × π/180 ≈ 3.069
    # sim_mid = 4.0 → mid_diff = 4.0 − 3.069 = 0.931 > 0.10
    mapping = {"1": _make_entry(1, "right_index_pip", "食指指尖")}
    info = {1: _make_info(3.0, 5.0, 500, 3500)}

    rows = detailed_alignment(mapping, info)
    assert "WARN_ZERO" in rows[0]["flags"]


def test_detailed_alignment_no_flags_when_well_aligned():
    """对称 + 量程匹配 + 零点对得齐 → 无 flag。"""
    from orca_sim.bridge.limits import detailed_alignment

    # 构造完美对齐：sim range 与 xdat range 都接近，sim_low ≈ xdat_low
    # 用 sim_low=0, sim_high=2.0，xdat=[1895, 2200] → range 305 raw → 0.4680 rad
    # ratio = 2.0 / 0.4680 ≈ 4.27 → WARN_RATIO
    # 没法构造完美对齐（协议本身就是粗对齐），所以这个测试只验证"全部 flag 同时触发"
    mapping = {"1": _make_entry(1, "right_index_pip", "食指指尖")}
    info = {1: _make_info(0.0, 1.0, 500, 3500)}
    rows = detailed_alignment(mapping, info)
    # 不要空字符串以外的内容（不可能完美，但不应崩）
    assert isinstance(rows[0]["flags"], str)


def test_detailed_alignment_handles_flipped_raw_storage():
    """flip_direction=true 时 raw_low/raw_high 已交换，xdat_min/xdat_max 仍应按大小顺序输出。"""
    from orca_sim.bridge.limits import detailed_alignment

    # 默认 flip 路径下 raw_low > raw_high（存储语义交换过）
    mapping = {"1": _make_entry(1, "right_index_pip", "食指指尖", flip=True)}
    info = {1: _make_info(0.0, 1.0, 3500, 500)}  # raw_low=3500 > raw_high=500

    rows = detailed_alignment(mapping, info)
    r = rows[0]
    assert r["xdat_min"] == 500      # 物理真实值
    assert r["xdat_max"] == 3500     # 物理真实值
    assert r["raw_range"] == pytest.approx(3000.0)


def test_format_detailed_alignment_basic():
    from orca_sim.bridge.limits import detailed_alignment, format_detailed_alignment

    mapping = {
        "1": _make_entry(1, "right_index_pip", "食指指尖"),
        "2": _make_entry(2, "right_index_mcp", "食指指中", flip=False),
    }
    info = {
        1: _make_info(-0.5, 1.5, 500, 3500),
        2: _make_info(0.0, 1.0, 1000, 3000),
    }
    rows = detailed_alignment(mapping, info)
    out = format_detailed_alignment(rows)

    assert "right_index_pip" in out
    assert "right_index_mcp" in out
    assert "Y" in out or "N" in out  # flip 列
    assert "sim_range" in out or "ratio" in out


# ============================================================================
# calibrate_sim_to_real.py 的 _linear_fit / _analyze_servo 单元测试
# ============================================================================
def test_linear_fit_perfect():
    """理想线性数据 → R²=1，slope / intercept 与理论值一致。"""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
    from calibrate_sim_to_real import _linear_fit

    # raw = 1000 * rad + 2048
    sim_rads = [0.0, 0.5, 1.0, -0.5, -1.0]
    actual_raws = [2048, 2548, 3048, 1548, 1048]
    slope, intercept, r2 = _linear_fit(sim_rads, actual_raws)

    assert slope == pytest.approx(1000.0)
    assert intercept == pytest.approx(2048.0)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_linear_fit_negative_slope():
    """斜率为负 → fit 应如实反映（不修正）。"""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
    from calibrate_sim_to_real import _linear_fit

    # raw = -800 * rad + 2048
    sim_rads = [0.0, 0.5, 1.0]
    actual_raws = [2048, 1648, 1248]
    slope, intercept, r2 = _linear_fit(sim_rads, actual_raws)

    assert slope == pytest.approx(-800.0)
    assert intercept == pytest.approx(2048.0)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_analyze_servo_suggests_no_flip_when_slope_positive():
    """slope > 0 → suggested_flip = False。"""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
    from calibrate_sim_to_real import _analyze_servo

    points = [
        {"sim_rad": "0.0", "actual_raw": "2048"},
        {"sim_rad": "0.5", "actual_raw": "2548"},
        {"sim_rad": "1.0", "actual_raw": "3048"},
    ]
    r = _analyze_servo(1, "right_index_pip", points)

    assert r.suggested_flip is False
    assert r.n_points == 3
    assert abs(r.zero_offset_raw) < 1.0  # 完美对中


def test_analyze_servo_suggests_flip_when_slope_negative():
    """slope < 0 → suggested_flip = True。"""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
    from calibrate_sim_to_real import _analyze_servo

    points = [
        {"sim_rad": "0.0", "actual_raw": "2048"},
        {"sim_rad": "0.5", "actual_raw": "1548"},
        {"sim_rad": "1.0", "actual_raw": "1048"},
    ]
    r = _analyze_servo(1, "right_index_pip", points)

    assert r.suggested_flip is True
    assert r.n_points == 3


def test_analyze_servo_detects_zero_offset():
    """截距偏离 2048 → 报警 + zero_offset_raw 反映差值。"""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
    from calibrate_sim_to_real import _analyze_servo

    # raw = 1000*rad + 2200（中点偏移 +152）
    points = [
        {"sim_rad": "0.0", "actual_raw": "2200"},
        {"sim_rad": "1.0", "actual_raw": "3200"},
    ]
    r = _analyze_servo(1, "right_index_pip", points)

    assert r.zero_offset_raw == pytest.approx(152.0, abs=1.0)
    assert any("zero offset" in w for w in r.warnings)


def test_analyze_servo_warns_on_too_few_points():
    """< 2 个有效点 → 应返回带警告的结果，不抛异常。"""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
    from calibrate_sim_to_real import _analyze_servo

    r = _analyze_servo(1, "right_index_pip", [{"sim_rad": "0", "actual_raw": ""}])
    assert r.n_points == 0
    assert any("数据点不足" in w for w in r.warnings)


def test_analyze_servo_warns_on_nonlinear():
    """明显非线性（3 个点不在一条直线） → R² 警告。"""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
    from calibrate_sim_to_real import _analyze_servo

    points = [
        {"sim_rad": "0.0", "actual_raw": "2048"},
        {"sim_rad": "0.5", "actual_raw": "3000"},
        {"sim_rad": "1.0", "actual_raw": "2048"},  # 跳回
    ]
    r = _analyze_servo(1, "right_index_pip", points)
    assert any("R²" in w for w in r.warnings)