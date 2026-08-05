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
    sid, name, ratio, zd = rows[0]
    assert sid == 1
    # sim_range = 2.0；xdat_range 转弧度 = 3000 × 360/4095 × π/180 ≈ 4.603
    xdat_range_rad = 3000.0 * 360.0 / 4095 * math.pi / 180.0
    # ratio = sim_range / xdat_range_rad
    assert ratio == pytest.approx(2.0 / xdat_range_rad, rel=1e-9)
    # 零差：sim_low=−0.5；xdat_low 转弧度 = 500 × 360/4095 × π/180 ≈ 0.7674
    xdat_low_rad = 500.0 * 360.0 / 4095 * math.pi / 180.0
    assert zd == pytest.approx(-0.5 - xdat_low_rad, rel=1e-9)


def test_format_alignment_report_basic():
    from orca_sim.bridge.limits import format_alignment_report

    rows = [
        (1, "right_index_pip", 1.0, 0.0),    # ok
        (2, "right_index_mcp", 0.5, 0.0),    # range-warn
        (3, "right_index_abd", 1.0, 0.5),    # zero-warn
    ]
    out = format_alignment_report(rows)
    assert "right_index_pip" in out
    assert "RANGE-WARN" in out
    assert "ZERO-WARN" in out


def test_format_alignment_report_includes_all_rows():
    from orca_sim.bridge.limits import format_alignment_report

    rows = [(i, f"right_j{i}", 1.0, 0.0) for i in range(1, 18)]
    out = format_alignment_report(rows)
    for i in range(1, 18):
        assert f"right_j{i}" in out