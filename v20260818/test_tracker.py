#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手势追踪 — 纯视觉测试工具 (无需串口 / 舵机)

启动摄像头，实时显示：
  - 手部骨架叠加画面
  - 17 个关节的屈曲角度 & 对应舵机位置
  - 当前手势状态

运行:
    python test_tracker.py              # 内置摄像头
    python test_tracker.py --camera 1   # USB 摄像头

依赖:
    pip install -r gesture_recognition/requirements.txt
    pip install PyQt5
"""

import sys
import os
import time
import json
import argparse
import threading
from pathlib import Path

# 确保可以 import gesture_recognition 和 hand_tracker
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(THIS_DIR))  # FTServo_Python_visual_recognization/
sys.path.insert(0, ROOT_DIR)

import cv2

from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QStatusBar,
)
from PyQt5.QtGui import QImage, QPixmap, QColor

from gesture_recognition.gesture_rps import (
    create_detector, detect_hand, draw_landmarks_on_image,
)
from hand_tracker import HandTracker, SERVO_CONFIG


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------
def _compute_lateral(lateral_ids, samples):
    """Compute per-servo offset and range from raw (signed) angle samples.

    Uses abs values internally — the direction-agnostic formula is:
        effective = abs(abs(raw) - offset)
    """
    offsets = {}
    ranges = {}
    for sid in lateral_ids:
        vals = samples.get(sid, [])
        if not vals:
            offsets[sid] = 0.0
            ranges[sid] = (0.0, 15.0)
            continue
        abs_vals = [abs(v) for v in vals]
        offset = round(min(abs_vals), 1)
        eff_range = round(max(abs_vals) - min(abs_vals), 1)
        offsets[sid] = offset
        ranges[sid] = (0.0, eff_range)
    return offsets, ranges


def _print_lateral_table(lateral_ids, samples):
    """Print a per-servo lateral calibration table from raw (signed) samples.

    Shows raw min/max (signed, as seen in UI) alongside abs-derived offset & range.
    """
    print(f"{'SID':>4} {'Name':<10} {'Samples':>8} {'Raw min':>8} {'Raw max':>8} "
          f"{'Range':>8} {'Offset':>8}")
    print(f"{'---':>4} {'----':<10} {'-------':>8} {'-------':>8} {'-------':>8} "
          f"{'-----':>8} {'------':>8}")
    for sid in lateral_ids:
        vals = samples.get(sid, [])
        if not vals:
            print(f"{sid:4d} {'—':<10} {'(no data)':>8}")
            continue
        name = SERVO_CONFIG[sid]["name"]
        raw_min = round(min(vals), 1)
        raw_max = round(max(vals), 1)
        abs_vals = [abs(v) for v in vals]
        offset = round(min(abs_vals), 1)
        eff_range = round(max(abs_vals) - min(abs_vals), 1)
        print(f"{sid:4d} {name:<10} {len(vals):8d} {raw_min:8.1f} {raw_max:8.1f} "
              f"{eff_range:8.1f} {offset:8.1f}")


# ---------------------------------------------------------------------------
# 后台引擎
# ---------------------------------------------------------------------------
class TrackerEngine(QObject):
    """摄像头 + MediaPipe + HandTracker 后台线程。"""

    frame_ready   = pyqtSignal(object)   # BGR frame
    angles_ready  = pyqtSignal(object)   # dict[servo_id, angle_deg]
    pos_ready     = pyqtSignal(object)   # dict[servo_id, position]
    state_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self._thread = None
        self._camera_index = 0

    def start(self, camera_index=0):
        if self._running:
            return
        self._camera_index = camera_index
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.state_changed.emit("STOPPED")

    @property
    def running(self):
        return self._running

    def _open_camera(self):
        target = self._camera_index
        idx_list = [target] + [i for i in (0, 1, 2, 3) if i != target]
        for idx in idx_list:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return cap, idx
            cap.release()
        return None, None

    def _run(self):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("GLOG_logtostderr", "0")
        os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

        cap, idx = self._open_camera()
        if cap is None:
            self.error_occurred.emit("无法打开摄像头 (尝试了 index 0/1/2/3)")
            self._running = False
            return

        self.state_changed.emit(f"TRACKING (camera {idx})")
        detector = create_detector(num_hands=1)
        self._tracker = HandTracker(alpha=0.35)
        tracker = self._tracker

        frame_idx = 0
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    self.error_occurred.emit("摄像头读取失败")
                    break
                frame = cv2.flip(frame, 1)
                frame_idx += 1

                all_landmarks = detect_hand(detector, frame)

                if all_landmarks:
                    lm = all_landmarks[0]
                    positions, angles = tracker.process(lm)
                    draw_landmarks_on_image(frame, lm)

                    if frame_idx % 3 == 0:  # ~10 Hz
                        self.pos_ready.emit(positions)
                        self.angles_ready.emit(angles)

                if frame_idx % 3 == 0:
                    self.frame_ready.emit(frame)

        finally:
            cap.release()
            detector.close()
            self._running = False
            self.state_changed.emit("STOPPED")


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------
class TrackerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手势追踪 — 视觉测试")
        self.resize(1100, 680)

        self._engine = TrackerEngine()
        self._recording = False
        self._record_buffer: list = []  # [(t, servo_id, angle), ...]
        self._record_count = 0          # number of completed recordings
        self._logs_dir = Path(__file__).parent / "logs"
        self._logs_dir.mkdir(exist_ok=True)
        self._build_ui()
        self._wire()

    # ─── UI ───────────────────────────────────────────────────────────
    def _build_ui(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("就绪 — 点击「开始追踪」")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # ─ 左: 摄像头画面 ─
        left = QWidget()
        left_layout = QVBoxLayout(left)

        ctrl_row = QHBoxLayout()
        self.start_btn = QPushButton("▶ 开始追踪")
        self.start_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 10px; font-size: 12pt;"
        )
        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.setStyleSheet(
            "background-color: #f44336; color: white; "
            "font-weight: bold; padding: 10px; font-size: 12pt;"
        )
        self.stop_btn.setEnabled(False)
        self.record_btn = QPushButton("⏺ 记录")
        self.record_btn.setStyleSheet(
            "background-color: #607D8B; color: white; "
            "font-weight: bold; padding: 10px; font-size: 12pt;"
        )
        self.record_btn.setEnabled(False)
        ctrl_row.addWidget(self.start_btn)
        ctrl_row.addWidget(self.stop_btn)
        ctrl_row.addWidget(self.record_btn)
        self.record_label = QLabel("(0/5)")
        self.record_label.setStyleSheet(
            "color: #888; font-weight: bold; padding: 4px;"
        )
        ctrl_row.addWidget(self.record_label)
        self.compute_btn = QPushButton("📊 计算校准")
        self.compute_btn.setStyleSheet(
            "background-color: #FF9800; color: white; "
            "font-weight: bold; padding: 10px; font-size: 12pt;"
        )
        self.compute_btn.setEnabled(False)
        ctrl_row.addWidget(self.compute_btn)
        ctrl_row.addStretch()
        ctrl_row.addWidget(QLabel("摄像头:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["0", "1", "2", "3"])
        ctrl_row.addWidget(self.camera_combo)
        left_layout.addLayout(ctrl_row)

        self.state_label = QLabel("状态: 未启动")
        self.state_label.setStyleSheet(
            "padding: 6px; font-weight: bold; font-size: 11pt; color: gray;"
        )
        left_layout.addWidget(self.state_label)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: #000; color: #888; border: 1px solid #444;"
        )
        self.video_label.setText("（摄像头画面）")
        left_layout.addWidget(self.video_label)

        layout.addWidget(left, stretch=3)

        # ─ 右: 角度表 ─
        right = QWidget()
        right_layout = QVBoxLayout(right)

        table_group = QGroupBox("关节数据")
        table_layout = QVBoxLayout(table_group)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["舵机ID", "关节名称", "屈曲角度 (°)", "映射位置", "状态"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.table.setMinimumWidth(420)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)

        # 初始化 17 行
        self.table.setRowCount(17)
        for i, sid in enumerate(range(1, 18)):
            cfg = SERVO_CONFIG[sid]
            self.table.setItem(i, 0, QTableWidgetItem(str(sid)))
            self.table.setItem(i, 1, QTableWidgetItem(cfg["name"]))
            self.table.setItem(i, 2, QTableWidgetItem("—"))
            self.table.setItem(i, 3, QTableWidgetItem("—"))
            self.table.setItem(i, 4, QTableWidgetItem("—"))

        table_layout.addWidget(self.table)
        right_layout.addWidget(table_group)

        # 图例
        legend = QLabel(
            "屈曲角度: 0° = 完全伸展  ·  100° = 完全弯曲\n"
            "肘关节/横向展开: 0° = 并拢  ·  30° = 最大展开"
        )
        legend.setStyleSheet("color: gray; padding: 4px;")
        legend.setWordWrap(True)
        right_layout.addWidget(legend)

        layout.addWidget(right, stretch=2)

    # ─── 信号 ─────────────────────────────────────────────────────────
    def _wire(self):
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.record_btn.clicked.connect(self._on_record_toggle)
        self.compute_btn.clicked.connect(self._compute_calibration)
        self._latest_positions: dict[int, int] = {}
        self._engine.frame_ready.connect(self._on_frame)
        self._engine.angles_ready.connect(self._on_angles)
        self._engine.pos_ready.connect(self._on_positions)
        self._engine.state_changed.connect(self._on_state)
        self._engine.error_occurred.connect(self._on_error)

    # ─── 槽 ───────────────────────────────────────────────────────────
    def _on_start(self):
        camera_idx = int(self.camera_combo.currentText())
        self._engine.start(camera_index=camera_idx)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.record_btn.setEnabled(True)
        self._record_count = 0
        self._update_record_label()
        self.record_btn.setText("⏺ 记录")
        self.compute_btn.setEnabled(False)
        self._recording = False
        self._record_buffer.clear()
        self.statusbar.showMessage("追踪中… 连续记录 5 次后点击「计算校准」")

    def _on_stop(self):
        if self._recording:
            self._finish_recording()
        self._engine.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        self.record_btn.setText("⏺ 记录")
        self._recording = False
        self.video_label.setText("（摄像头画面）")
        self.video_label.setPixmap(QPixmap())
        self.state_label.setText("状态: 未启动")
        self.state_label.setStyleSheet(
            "padding: 6px; font-weight: bold; font-size: 11pt; color: gray;"
        )
        # 清空角度表
        for i in range(17):
            self.table.item(i, 2).setText("—")
            self.table.item(i, 3).setText("—")
            self.table.item(i, 4).setText("—")
        self.statusbar.showMessage("已停止")

    def _on_state(self, state):
        self.state_label.setText(f"状态: {state}")
        color = {"TRACKING": "#4CAF50",
                 "STOPPED": "gray"}.get(
            state.split()[0] if state.startswith("TRACKING") else state, "gray"
        )
        self.state_label.setStyleSheet(
            f"padding: 6px; font-weight: bold; font-size: 11pt; color: {color};"
        )

    def _on_frame(self, bgr_frame):
        try:
            h, w = bgr_frame.shape[:2]
            rgb = bgr_frame[:, :, ::-1].copy()
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            target_w = self.video_label.width()
            target_h = self.video_label.height()
            if target_w > 0 and target_h > 0:
                pix = QPixmap.fromImage(qimg).scaled(
                    target_w, target_h,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
                self.video_label.setPixmap(pix)
        except Exception:
            pass

    def _on_positions(self, positions):
        self._latest_positions = positions

    def _on_angles(self, angles):
        if self._recording:
            t = time.time()
            splay_mode = self._engine._tracker.splay_mode if hasattr(self._engine, '_tracker') else {}
            tip_flex = self._engine._tracker.tip_flexion if hasattr(self._engine, '_tracker') else {}
            for sid, angle in angles.items():
                mode = splay_mode.get(sid, 'A')
                tf = tip_flex.get(sid, -1.0)
                self._record_buffer.append((t, sid, angle, mode, tf))

        for i, sid in enumerate(range(1, 18)):
            if sid not in angles:
                self.table.item(i, 2).setText("—")
                self.table.item(i, 3).setText("—")
                self.table.item(i, 4).setText("—")
                continue

            angle = angles[sid]
            self.table.item(i, 2).setText(f"{angle:.1f}")

            pos = self._latest_positions.get(sid)
            if pos is not None:
                self.table.item(i, 3).setText(str(pos))

            cfg = SERVO_CONFIG[sid]
            jtype = cfg["type"]
            from hand_tracker import ANGLE_RANGES, LATERAL_MAP, SERVO_ANGLE_RANGE
            lm = LATERAL_MAP.get(sid)
            if lm:
                mode = self._engine._tracker.splay_mode.get(sid, 'A') if hasattr(self._engine, '_tracker') else 'A'
                params = lm.get(mode, lm['A'])
                if mode == 'B':
                    rng = params["pos_range"]
                    ratio = abs(angle) / rng if rng > 0 else 0
                else:
                    deviation = angle - params["mid_angle"]
                    rng = params["pos_range"] if deviation >= 0 else params["neg_range"]
                    ratio = abs(deviation) / rng if rng > 0 else 0
                ratio = max(0.0, min(1.0, ratio))
            else:
                a = abs(angle) if jtype in ("bottom", "thumb_lateral") else angle
                min_a, max_a = (SERVO_ANGLE_RANGE.get(sid) or
                                ANGLE_RANGES.get(jtype, (0.0, 90.0)))
                ratio = (a - min_a) / (max_a - min_a) if max_a > min_a else 0
                ratio = max(0.0, min(1.0, ratio))

            if jtype in ("bottom", "thumb_lateral"):
                mode = self._engine._tracker.splay_mode.get(sid, 'A') if hasattr(self._engine, '_tracker') else 'A'
                if ratio < 0.1:
                    status = f"并拢({mode})"
                    color = QColor(76, 175, 80)
                elif ratio > 0.8:
                    status = f"展开({mode})"
                    color = QColor(244, 67, 54)
                else:
                    status = f"中间({mode})"
                    color = QColor(255, 152, 0)
            elif ratio < 0.1:
                status = "伸展"
                color = QColor(76, 175, 80)
            elif ratio > 0.8:
                status = "弯曲"
                color = QColor(244, 67, 54)
            else:
                status = "半屈"
                color = QColor(255, 152, 0)
            item = QTableWidgetItem(status)
            item.setForeground(color)
            self.table.setItem(i, 4, item)

    def _update_record_label(self):
        self.record_label.setText(f"({self._record_count}/5)")

    def _on_record_toggle(self):
        if self._recording:
            self._finish_recording()
        else:
            self._recording = True
            self._record_buffer.clear()
            self.record_btn.setText("⏹ 停止记录")
            self.record_btn.setStyleSheet(
                "background-color: #f44336; color: white; "
                "font-weight: bold; padding: 10px; font-size: 12pt;"
            )
            self.compute_btn.setEnabled(False)
            self.statusbar.showMessage(
                f"⏺ 记录 {self._record_count + 1}/5 … 移动手指从伸展到握拳"
            )

    def _finish_recording(self):
        self._recording = False
        self.record_btn.setText("⏺ 记录")
        self.record_btn.setStyleSheet(
            "background-color: #607D8B; color: white; "
            "font-weight: bold; padding: 10px; font-size: 12pt;"
        )
        if not self._record_buffer:
            self.statusbar.showMessage("记录为空，未保存")
            return

        # Compute per-joint min/max from recorded angles
        joint_data: dict[int, list[float]] = {}
        for _t, sid, angle, _mode, _tf in self._record_buffer:
            joint_data.setdefault(sid, []).append(angle)

        stats = {}
        for sid in sorted(joint_data):
            vals = joint_data[sid]
            cfg = SERVO_CONFIG[sid]
            stats[str(sid)] = {
                "name": cfg["name"],
                "type": cfg["type"],
                "min": round(min(vals), 2),
                "max": round(max(vals), 2),
                "range": round(max(vals) - min(vals), 2),
                "samples": len(vals),
            }

        # Save to numbered log file
        self._record_count += 1
        out_path = self._logs_dir / f"joint_range_log_{self._record_count}.json"
        payload = {
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session": self._record_count,
            "total_samples": len(self._record_buffer),
            "joints": stats,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # Also save per-sample data for dual-strategy calibration
        samples_path = self._logs_dir / f"joint_samples_log_{self._record_count}.json"
        samples_payload = [
            {"t": t, "sid": sid, "angle": angle, "mode": mode, "tip_flex": tf}
            for t, sid, angle, mode, tf in self._record_buffer
        ]
        with open(samples_path, "w", encoding="utf-8") as f:
            json.dump(samples_payload, f, ensure_ascii=False)

        self._update_record_label()
        self._record_buffer.clear()

        if self._record_count >= 5:
            self.compute_btn.setEnabled(True)
            self.compute_btn.setStyleSheet(
                "background-color: #4CAF50; color: white; "
                "font-weight: bold; padding: 10px; font-size: 12pt;"
            )
            self.statusbar.showMessage(
                f"已保存 {self._record_count}/5 条记录 → 点击「计算校准」"
            )
        else:
            self.statusbar.showMessage(
                f"已保存第 {self._record_count} 条记录 → {out_path.name}"
            )

    def _compute_calibration(self):
        """Read all log files, compute per-type and per-servo calibration."""
        range_logs = sorted(self._logs_dir.glob("joint_range_log_*.json"))
        sample_logs = sorted(self._logs_dir.glob("joint_samples_log_*.json"))
        if len(range_logs) < 2:
            self.statusbar.showMessage("需要至少 2 条记录才能计算均值")
            return

        # Aggregate per-servo min/max across all range logs (flexion joints)
        servo_mins: dict[int, list[float]] = {}
        servo_maxs: dict[int, list[float]] = {}
        for log_path in range_logs:
            with open(log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for sid_str, v in data.get("joints", {}).items():
                sid = int(sid_str)
                servo_mins.setdefault(sid, []).append(v["min"])
                servo_maxs.setdefault(sid, []).append(v["max"])

        # ── Per-type calibration (flexion joints) ──
        type_mins: dict[str, list[float]] = {}
        type_maxs: dict[str, list[float]] = {}
        for sid in sorted(servo_mins):
            cfg = SERVO_CONFIG[sid]
            jtype = cfg["type"]
            if jtype in ("bottom", "thumb_lateral"):
                continue  # handled separately below
            mean_min = sum(servo_mins[sid]) / len(servo_mins[sid])
            mean_max = sum(servo_maxs[sid]) / len(servo_maxs[sid])
            type_mins.setdefault(jtype, []).append(mean_min)
            type_maxs.setdefault(jtype, []).append(mean_max)

        calibration: dict[str, tuple[float, float]] = {}
        for jtype in sorted(type_mins):
            mins = type_mins[jtype]
            maxs = type_maxs[jtype]
            cal_min = round(sum(mins) / len(mins), 1)
            cal_max = round(sum(maxs) / len(maxs), 1)
            calibration[jtype] = (cal_min, cal_max)

        print(f"\n{'='*65}")
        print(f"Calibration from {len(range_logs)} logs — paper → fist")
        print(f"{'='*65}")

        # ── Per-type ANGLE_RANGES ──
        print(f"\n{'Type':<16} {'Cal Min':>8} {'Cal Max':>8}")
        print(f"{'-'*16} {'-'*8} {'-'*8}")
        for jtype in sorted(calibration):
            lo, hi = calibration[jtype]
            print(f"{jtype:<16} {lo:8.1f} {hi:8.1f}")

        print("\n# --- ANGLE_RANGES (flexion joints) ---")
        print("ANGLE_RANGES = {")
        for jtype in sorted(calibration):
            lo, hi = calibration[jtype]
            print(f'    "{jtype}": ({lo:.1f}, {hi:.1f}),')
        print("}")

        # ── Per-servo lateral calibration (dual-strategy from sample logs) ──
        lateral_ids = [sid for sid in sorted(servo_mins)
                       if SERVO_CONFIG[sid]["type"] in ("bottom", "thumb_lateral")]
        if lateral_ids and sample_logs:
            # Collect per-strategy samples from all sample logs
            samples_a: dict[int, list[float]] = {}  # servo → raw angles (Strategy A)
            samples_b: dict[int, list[float]] = {}  # servo → raw angles (Strategy B)
            for sp in sample_logs:
                with open(sp, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                for e in entries:
                    sid = e["sid"]
                    if sid not in lateral_ids:
                        continue
                    mode = e.get("mode", "A")
                    angle = e["angle"]
                    if mode == "B":
                        samples_b.setdefault(sid, []).append(angle)
                    else:
                        samples_a.setdefault(sid, []).append(angle)

            # ── Strategy A (extended) ──
            print(f"\n{'='*65}")
            print("Strategy A — Extended fingers (wrist→MCP vs MCP→PIP)")
            print(f"{'='*65}")
            _print_lateral_table(lateral_ids, samples_a)

            splay_offsets_a, splay_ranges_a = _compute_lateral(lateral_ids, samples_a)
            print("\n# --- SPLAY_OFFSET_A ---")
            print("SPLAY_OFFSET_A: dict[int, float] = {")
            for sid in sorted(splay_offsets_a):
                print(f"    {sid}: {splay_offsets_a[sid]:.1f},")
            print("}")
            print("\n# --- SPLAY_RANGE_A ---")
            print("SPLAY_RANGE_A: dict[int, tuple[float, float]] = {")
            for sid in sorted(splay_ranges_a):
                lo, hi = splay_ranges_a[sid]
                print(f"    {sid}: ({lo:.1f}, {hi:.1f}),")
            print("}")

            # ── Strategy B (bent) ──
            if any(samples_b.values()):
                print(f"\n{'='*65}")
                print("Strategy B — Bent fingers (relative MCP→PIP)")
                print(f"{'='*65}")
                _print_lateral_table(lateral_ids, samples_b)

                splay_offsets_b, splay_ranges_b = _compute_lateral(lateral_ids, samples_b)
                print("\n# --- SPLAY_OFFSET_B ---")
                print("SPLAY_OFFSET_B: dict[int, float] = {")
                for sid in sorted(splay_offsets_b):
                    if sid == 14:
                        continue  # thumb always uses Strategy A
                    print(f"    {sid}: {splay_offsets_b[sid]:.1f},")
                print("}")
                print("\n# --- SPLAY_RANGE_B ---")
                print("SPLAY_RANGE_B: dict[int, tuple[float, float]] = {")
                for sid in sorted(splay_ranges_b):
                    if sid == 14:
                        continue
                    lo, hi = splay_ranges_b[sid]
                    print(f"    {sid}: ({lo:.1f}, {hi:.1f}),")
                print("}")
            else:
                print("\n# No Strategy B samples — record bent-finger sessions for SPLAY_OFFSET_B / SPLAY_RANGE_B")

        print()

        self.statusbar.showMessage(
            f"校准完成 — 基于 {len(range_logs)} 条记录，结果已输出到终端"
        )

    def _on_error(self, msg):
        self.statusbar.showMessage(f"错误: {msg}")
        self._on_stop()

    def closeEvent(self, event):
        self._engine.stop()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="手势追踪视觉测试")
    ap.add_argument("--camera", type=int, default=0, help="摄像头索引 (默认 0)")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 跨平台统一深色风格

    win = TrackerWindow()
    # 启动时自动开始追踪
    win._engine._camera_index = args.camera
    win._on_start()
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
