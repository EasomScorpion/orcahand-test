#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STS3215 × 17 舵机控制台
========================

一个 PyQt5 桌面控制台，用于 17 个 STS3215 舵机的日常调试。

功能:
    - 🎮 控制台 Tab: 串口连接 / 单舵机滑块控制 / 复位到 2048 / 紧急停止
    - 📋 参数浏览器 Tab: 复刻 FT 调试软件，显示所有 EPROM+SRAM 寄存器
                        支持"保存到 EPROM"和"导出到 .xdat 文件"

安全底线（不可被任何上层功能覆盖）:
    - MAX_TORQUE     = 100
    - PROTECT_TORQUE = 20
    - OVERLOAD_TORQUE= 20
    - PROTECT_TIME   = 20
    - 角度限位       = 每个舵机的 参数/N.xdat 里的 最小/最大角度限制
"""

import sys
import os
import threading
import time
import json
from dataclasses import dataclass

# ---------- 路径设置 ----------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(THIS_DIR))  # FTServo_Python_visual_recognization/
sys.path.insert(0, ROOT_DIR)     # 用于 import xdat_tool, 读取 参数/
sys.path.insert(0, "..")          # 用于 import scservo_sdk

from scservo_sdk import *          # noqa: E402
import xdat_tool                   # noqa: E402

from PyQt5.QtCore import Qt, QObject, pyqtSignal, QAbstractTableModel, QModelIndex, QVariant  # noqa: E402
from PyQt5.QtWidgets import (                                            # noqa: E402
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QSlider, QSpinBox, QLineEdit,
    QTableView, QHeaderView, QFileDialog, QMessageBox, QStatusBar,
    QGroupBox, QSizePolicy, QInputDialog,
    QListWidget, QListWidgetItem, QPlainTextEdit,
)
from PyQt5.QtGui import QColor, QImage, QPixmap  # noqa: E402


# ===========================================================================
# 常量
# ===========================================================================
PARAM_DIR = os.path.join(ROOT_DIR, "参数")
JOINT_MAP_FILE = os.path.join(THIS_DIR, "servo_joints.json")
POSES_FILE = os.path.join(THIS_DIR, "saved_poses.json")
DEFAULT_BAUDRATE = 1_000_000
POLL_INTERVAL = 0.2


def list_serial_ports():
    """Return a list of likely serial port names for this OS."""
    import glob
    import platform
    system = platform.system()
    ports = []
    if system == "Darwin":
        ports = [p for p in glob.glob("/dev/cu.*") if any(
            kw in p for kw in ["usbserial", "usbmodem", "SLAB_USB", "wchusbserial",
                               "ch340", "cp210", "ftdi", "pl2303"])]
    elif system == "Linux":
        ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    elif system == "Windows":
        try:
            import serial.tools.list_ports
            ports = [p.device for p in serial.tools.list_ports.comports()]
        except ImportError:
            ports = [f"COM{i}" for i in range(1, 13)]
    return sorted(ports) if ports else []


# ===========================================================================
# 数据类
# ===========================================================================
@dataclass(frozen=True)
class SafetyLimits:
    """硬底线安全参数（上限/下限，不直接写入舵机）。

    写入舵机的实际值从 xdat 参数文件读取，不在代码中硬编码。
    """
    # 复位动作
    RESET_POSITION:  int = 2048
    RESET_SPEED:     int = 100
    RESET_ACC:       int = 10


@dataclass
class ServoBundle:
    """单个舵机的参数集合（来自 参数/N.xdat）。"""
    servo_id: int
    min_angle: int          # 最小角度限制（来自 xdat，机械手物理极限）
    max_angle: int          # 最大角度限制
    ofs: int                # 位置偏移
    fields: dict            # xdat 的所有 EPROM 字段（key=字段中文名）


@dataclass
class Pose:
    """单个姿态记录：17 个舵机的目标位置集合。"""
    name: str
    positions: dict         # {servo_id: position}
    created_at: str = ""    # ISO 时间戳

    def to_dict(self):
        return {
            "name": self.name,
            "positions": {str(k): int(v) for k, v in self.positions.items()},
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d):
        positions = {int(k): int(v) for k, v in d["positions"].items()}
        return cls(
            name=d["name"],
            positions=positions,
            created_at=d.get("created_at", ""),
        )


class PoseLibrary:
    """
    已保存姿态的容器，支持 JSON 文件自动持久化。

    用法:
        lib = PoseLibrary("poses.json")
        lib.add(Pose(name="石头", positions={1: 1370, 2: 1560, ...}))
        lib.get("石头")        # → Pose
        lib.remove("石头")     # 删除
        lib.save_to_file()    # 显式保存（构造函数会自动持久化）
    """

    def __init__(self, filepath=None):
        self.poses: list = []
        self.filepath = filepath
        self.load_from_file()

    def add(self, pose):
        """新增姿态。返回 True 成功；False 表示同名已存在。"""
        if self.get(pose.name):
            return False
        self.poses.append(pose)
        self._autosave()
        return True

    def remove(self, name):
        """按名字删除姿态。"""
        before = len(self.poses)
        self.poses = [p for p in self.poses if p.name != name]
        if len(self.poses) != before:
            self._autosave()
            return True
        return False

    def rename(self, old_name, new_name):
        """按旧名重命名为新名。返回 True 成功。"""
        for p in self.poses:
            if p.name == old_name:
                p.name = new_name
                self._autosave()
                return True
        return False

    def get(self, name):
        """按名字查找姿态，找不到返回 None。"""
        for p in self.poses:
            if p.name == name:
                return p
        return None

    def names(self):
        return [p.name for p in self.poses]

    def clear(self):
        self.poses = []
        self._autosave()

    def save_to_file(self, path=None):
        path = path or self.filepath
        if not path:
            return
        data = {"version": 1, "poses": [p.to_dict() for p in self.poses]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_file(self, path=None):
        path = path or self.filepath
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.poses = [Pose.from_dict(p) for p in data.get("poses", [])]
        except Exception:
            # 文件损坏时不要清空，给用户保留机会
            pass

    def _autosave(self):
        if self.filepath:
            try:
                self.save_to_file()
            except Exception:
                pass


class SafetyViolation(Exception):
    """违反安全底线时抛出的异常。"""


# ===========================================================================
# 石头剪刀布游戏引擎（独立于 Qt 主线程）
# ===========================================================================
class RPSGameEngine(QObject):
    """
    背景线程：摄像头 + MediaPipe + RPS 分类器 + 舵机联动。

    信号:
        state_changed(str)        当前状态: IDLE / DETECTING / EXECUTING / RESETTING
        gesture_visible(str)      当前帧识别到的稳定手势
        frame_ready(np.ndarray)   BGR 帧（用于 UI 显示）
        action_triggered(str)     确认触发 → 派发的 winning pose 名
        action_finished(str)      机械手动作完成
        timeout_reset()           2 秒无手势，自动复位
        score_updated(int, int)   (用户回合数, 机械手回合数)
        error_occurred(str)       错误信息
    """

    state_changed    = pyqtSignal(str)
    gesture_visible  = pyqtSignal(str)
    frame_ready      = pyqtSignal(object)
    action_triggered = pyqtSignal(str)
    action_finished  = pyqtSignal(str)
    timeout_reset    = pyqtSignal()
    score_updated    = pyqtSignal(int, int)
    error_occurred   = pyqtSignal(str)

    # 胜出映射: 用户出 X → 机械手出 Y (Y beats X)
    DEFAULT_WIN_MAP = {
        "rock":     "布",       # 用户石头 → 机械手出布（包住石头）
        "paper":    "剪刀",     # 用户布 → 机械手出剪刀（剪开布）
        "scissors": "石头",     # 用户剪刀 → 机械手出石头（砸坏剪刀）
    }
    # 出拳字符串 → 中文显示
    ZH = {"rock": "石头", "paper": "布", "scissors": "剪刀", None: "—"}

    def __init__(self, backend, pose_lib, win_map=None):
        super().__init__()
        self._backend = backend
        self._pose_lib = pose_lib
        self._win_map = dict(win_map or self.DEFAULT_WIN_MAP)
        self._running = False
        self._thread = None
        self._user_score = 0
        self._bot_score = 0
        self._stable_since = None
        self._stable_label = None
        self._last_gesture_time = None
        # 机械手忙：True 时主循环跳过识别，但摄像头照常推帧
        self._hand_busy = False
        self._confirmed_label = None   # 最近一次确认的手势（用于画面叠加）
        self._confirmed_zh = None       # 中文化，用于叠加文字

    # ─── 启动 / 停止 ────────────────────────────────────────────────
    def start(self, camera_index=0):
        if self._running:
            return
        self._camera_index = camera_index
        self._running = True
        self._stable_since = None
        self._stable_label = None
        self._last_gesture_time = time.time()
        self._hand_busy = False
        self._confirmed_label = None
        self._confirmed_zh = None
        self._thread = threading.Thread(
            target=self._run, daemon=True
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.5)
            self._thread = None
        self.state_changed.emit("已停止")

    def set_win_map(self, win_map):
        self._win_map = dict(win_map)

    @property
    def running(self):
        return self._running

    # ─── 主循环 ─────────────────────────────────────────────────────
    def _run(self):
        # 静默 MediaPipe / TensorFlow / absl 的启动噪音（必须 import 前设置）
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("GLOG_logtostderr", "0")
        os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
        # 延迟导入 cv2 / mediapipe — 让主 UI 不依赖这些包也能跑
        try:
            import cv2  # noqa: F401
            from gesture_recognition.gesture_rps import (
                RPSClassifier, create_detector, detect_hand,
            )
        except ImportError as e:
            self.error_occurred.emit(
                f"缺少依赖: {e}\n请运行: "
                f"pip install -r gesture_recognition/requirements.txt"
            )
            self._running = False
            return

        cap, idx = self._open_camera()
        if cap is None:
            self.error_occurred.emit("无法打开摄像头 (尝试了 index 0/1/2/3)")
            self._running = False
            return

        self.state_changed.emit("IDLE")
        detector = create_detector(num_hands=1)
        clf = RPSClassifier(smooth_window=8)

        STABLE_DURATION = 1.0     # 持续 1 秒才确认
        NO_GESTURE_TIMEOUT = 2.0  # 2 秒无手势自动复位
        frame_emit_every = 3      # 每 3 帧向 UI 推一次画面（约 10 fps）

        frame_idx = 0
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    self.error_occurred.emit("摄像头读取失败")
                    break
                frame = cv2.flip(frame, 1)
                frame_idx += 1

                # 在画面叠加"已确认: X"水印（机械手忙时）
                if self._hand_busy and self._confirmed_zh:
                    self._draw_confirm_overlay(frame, self._confirmed_zh)

                if frame_idx % frame_emit_every == 0:
                    self.frame_ready.emit(frame)

                # ── 机械手忙时：跳过识别但照常读帧、推送画面 ──
                if self._hand_busy:
                    # 继续吃帧（不浪费 cap.read 队列），但不识别
                    self.gesture_visible.emit(
                        f"已确认 {self._confirmed_zh}，机械手动作中…"
                    )
                    continue

                all_landmarks = detect_hand(detector, frame)

                label = None
                if all_landmarks:
                    lm = all_landmarks[0]
                    label = clf.classify(lm)
                self.gesture_visible.emit(self.ZH.get(label, "—"))

                # ─── 状态机 ───
                if label in ("rock", "paper", "scissors"):
                    self._last_gesture_time = time.time()
                    if label != self._stable_label:
                        self._stable_label = label
                        self._stable_since = time.time()
                        self.state_changed.emit("DETECTING")
                    elif (time.time() - self._stable_since) >= STABLE_DURATION:
                        # 确认！置忙标志、派发 winning、起子线程等机械手
                        zh = self.ZH[label]
                        self._confirmed_label = label
                        self._confirmed_zh = zh
                        self._hand_busy = True
                        # 先发指令（短阻塞，~50ms）
                        self._execute_winning(label)
                        # 重置检测状态
                        self._stable_label = None
                        self._stable_since = None
                        clf.reset()
                        # 起子线程等机械手完成，完成后清忙标志
                        wait_thread = threading.Thread(
                            target=self._wait_and_clear_busy,
                            daemon=True,
                        )
                        wait_thread.start()
                else:
                    if self._stable_label is not None:
                        # 之前在 tracking 但现在手不见了
                        self._stable_label = None
                        self._stable_since = None
                        clf.reset()
                    if (self._last_gesture_time is not None and
                            time.time() - self._last_gesture_time > NO_GESTURE_TIMEOUT):
                        # 2 秒没识别到任何手势 → 复位
                        self.state_changed.emit("RESETTING")
                        self._do_reset()
                        self.timeout_reset.emit()
                        self._last_gesture_time = time.time()
                        self.state_changed.emit("IDLE")
        finally:
            cap.release()
            detector.close()
            self._running = False
            self.state_changed.emit("已停止")

    def _draw_confirm_overlay(self, frame, zh):
        """在帧上画红色"已确认"水印 + 用户出的手势，避免画面与状态脱钩。"""
        try:
            import cv2
            h, w = frame.shape[:2]
            # 顶部红条
            cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 180), -1)
            cv2.putText(frame, f"CONFIRMED: {zh}",
                        (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        except Exception:
            pass

    def _wait_and_clear_busy(self):
        """子线程：轮询 ReadMoving 等机械手完成，保持姿势 3 秒，然后清忙标志。"""
        self._wait_for_hand_finished()
        self.state_changed.emit("HOLD")
        hold = 1.0
        deadline = time.time() + hold
        while time.time() < deadline and self._running:
            time.sleep(0.1)
        self._hand_busy = False
        self._confirmed_label = None
        self._confirmed_zh = None
        self._stable_since = None
        self._stable_label = None
        self.state_changed.emit("IDLE")
        self.action_finished.emit("done")

    # ─── 机械手动作 ──────────────────────────────────────────────────
    def _execute_winning(self, user_label):
        winning = self._win_map.get(user_label)
        if not winning:
            self.error_occurred.emit(
                f"未配置用户手势 '{self.ZH[user_label]}' 的胜出姿态"
            )
            return
        pose = self._pose_lib.get(winning)
        if pose is None:
            self.error_occurred.emit(
                f"姿态库缺少「{winning}」，请先在动作库 Tab 录制"
            )
            return
        if not self._backend.connected:
            self.error_occurred.emit("串口未连接，无法控制机械手")
            return

        self.action_triggered.emit(winning)
        try:
            self._backend.safety.sync_go_to_pose(pose.positions, speed=250)
        except SafetyViolation as e:
            self.error_occurred.emit(f"安全限制: {e}")
            return
        # 简化计分：每轮都算机械手胜
        self._bot_score += 1
        self.score_updated.emit(self._user_score, self._bot_score)
        self.action_finished.emit(winning)

    def _do_reset(self):
        if not self._backend.connected:
            return
        try:
            self._backend.safety.sync_reset()
        except Exception:
            pass

    def _wait_for_hand_finished(self):
        """轮询 ReadMoving 等所有舵机停止。在子线程中运行。"""
        if not self._backend.connected or not self._backend.packet_handler:
            time.sleep(1.5)  # fallback
            return
        deadline = time.time() + 8.0
        poll_interval = 0.05
        # 等 200ms 给舵机真正开始动
        time.sleep(0.2)
        while time.time() < deadline and self._running:
            all_stopped = True
            for b in self._backend.bundles:
                try:
                    moving, comm, _ = self._backend.packet_handler.ReadMoving(
                        b.servo_id
                    )
                    if comm == COMM_SUCCESS and moving == 1:
                        all_stopped = False
                        break
                except Exception:
                    pass
            if all_stopped:
                return
            time.sleep(poll_interval)

    # ─── 摄像头打开（同 gesture_rps._open_camera） ────────────────────
    def _open_camera(self):
        try:
            import cv2
        except ImportError:
            return None, None
        if sys.platform.startswith("win"):
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]
        target = getattr(self, '_camera_index', 0)
        idx_list = [target] + [i for i in (0, 1, 2, 3) if i != target]
        for idx in idx_list:
            for backend in backends:
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    return cap, idx
                cap.release()
        return None, None


# ===========================================================================
# 手势追踪引擎（摄像头 → MediaPipe → ORCA 手实时镜像）
# ===========================================================================
class HandTrackingEngine(QObject):
    """
    背景线程：摄像头 + MediaPipe + HandTracker 连续映射 + 舵机实时镜像。

    信号:
        state_changed(str)         当前状态: TRACKING / PAUSED / STOPPED
        frame_ready(np.ndarray)    BGR 帧（含骨架叠加，用于 UI 显示）
        angles_updated(dict)       {servo_id: angle_degrees}
        error_occurred(str)        错误信息
    """

    state_changed   = pyqtSignal(str)
    frame_ready     = pyqtSignal(object)
    angles_updated  = pyqtSignal(object)   # dict[int, float]
    error_occurred  = pyqtSignal(str)

    def __init__(self, backend):
        super().__init__()
        self._backend = backend       # ConsoleBackend (has .safety, .connected, .packet_handler)
        self._running = False
        self._paused = False
        self._thread = None
        self._speed = 200
        self._update_hz = 15
        self._camera_index = 0
        self._tracker = None          # HandTracker instance, created in _run()
        self._calibrate_requested = False

    # ─── 启动 / 停止 ────────────────────────────────────────────────
    def start(self, camera_index=0, speed=200, update_hz=15):
        if self._running:
            return
        self._camera_index = camera_index
        self._speed = speed
        self._update_hz = update_hz
        self._running = True
        self._paused = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.5)
            self._thread = None
        self.state_changed.emit("STOPPED")

    def pause(self):
        self._paused = True
        self.state_changed.emit("PAUSED")

    def resume(self):
        self._paused = False
        self.state_changed.emit("TRACKING")

    def set_speed(self, speed):
        self._speed = int(speed)

    def set_update_hz(self, hz):
        self._update_hz = int(hz)

    @property
    def running(self):
        return self._running

    @property
    def paused(self):
        return self._paused

    # ─── 主循环 ─────────────────────────────────────────────────────
    def _run(self):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("GLOG_logtostderr", "0")
        os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
        try:
            import cv2
            from gesture_recognition.gesture_rps import (
                create_detector, detect_hand, draw_landmarks_on_image,
            )
            from realtime_mirroring.hand_tracker import HandTracker
        except ImportError as e:
            self.error_occurred.emit(
                f"缺少依赖: {e}\n请运行: "
                f"pip install -r gesture_recognition/requirements.txt"
            )
            self._running = False
            return

        cap, idx = _open_camera(self._camera_index)
        if cap is None:
            self.error_occurred.emit("无法打开摄像头 (尝试了 index 0/1/2/3)")
            self._running = False
            return

        self.state_changed.emit("TRACKING")
        detector = create_detector(num_hands=1)
        tracker = HandTracker(alpha=0.35)
        self._tracker = tracker

        frame_idx = 0
        frame_emit_every = 3
        last_write_time = 0.0
        last_positions: dict[int, int] = {}
        no_hand_since = None

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    self.error_occurred.emit("摄像头读取失败")
                    break
                frame = cv2.flip(frame, 1)
                frame_idx += 1

                # Handle calibration request from UI
                if self._calibrate_requested:
                    all_landmarks_cal = detect_hand(detector, frame)
                    if all_landmarks_cal:
                        tracker.set_neutral(all_landmarks_cal[0])
                        tracker.calibrate_splay(all_landmarks_cal[0])
                    self._calibrate_requested = False

                all_landmarks = detect_hand(detector, frame)

                if all_landmarks and not self._paused:
                    lm = all_landmarks[0]
                    positions, angles = tracker.process(lm)
                    no_hand_since = None

                    # Draw skeleton
                    draw_landmarks_on_image(frame, lm)

                    # Throttle servo writes
                    now = time.time()
                    interval = 1.0 / max(self._update_hz, 1)
                    if now - last_write_time >= interval:
                        if _positions_changed(last_positions, positions):
                            if self._backend.connected and self._backend.safety:
                                try:
                                    self._backend.safety.sync_go_to_pose(
                                        positions, speed=self._speed, acc=10,
                                    )
                                except Exception:
                                    pass  # safety layer handles emergency stop
                            last_positions = positions.copy()
                        last_write_time = now

                    # Emit angles for UI display (throttled to ~10 Hz)
                    if frame_idx % 3 == 0:
                        self.angles_updated.emit(angles)
                elif not all_landmarks:
                    # No hand detected — hold last position
                    if no_hand_since is None:
                        no_hand_since = time.time()
                    # After 2s of no hand, fade wrist toward center
                    if time.time() - no_hand_since > 2.0 and self._tracker:
                        self._tracker.reset()
                        last_positions.clear()
                        no_hand_since = time.time()  # prevent repeated resets

                # Emit frame to UI
                if frame_idx % frame_emit_every == 0:
                    self.frame_ready.emit(frame)

        finally:
            cap.release()
            detector.close()
            self._tracker = None
            self._running = False
            self.state_changed.emit("STOPPED")


def _positions_changed(old: dict, new: dict, threshold: int = 8) -> bool:
    """True if any servo position changed more than *threshold* units."""
    if not old:
        return True
    for sid in new:
        if abs(new.get(sid, 0) - old.get(sid, 0)) > threshold:
            return True
    return False


def _open_camera(target_index=0):
    """Open first working camera, trying *target_index* first."""
    try:
        import cv2
    except ImportError:
        return None, None
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]
    idx_list = [target_index] + [i for i in (0, 1, 2, 3) if i != target_index]
    for idx in idx_list:
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return cap, idx
            cap.release()
    return None, None


# ===========================================================================
# 安全层（所有写入的唯一入口）
# ===========================================================================
class ServoSafetyLayer:
    """
    所有写舵机的代码必须经过本类。
    UI 层应只持有 ServoSafetyLayer 引用，不直接持有 sms_sts。

    防护层次:
        1) 紧急停止开关（_emergency_stopped）
        2) 角度限位（取自 xdat 的机械手物理极限）
        3) 速度/加速度上限
        4) EPROM 写入时自动 unLock/lock
    """

    def __init__(self, ph, bundles, limits=None):
        self._ph = ph
        self._bundles = {b.servo_id: b for b in bundles}
        self._limits = limits or SafetyLimits()
        self._emergency_stopped = False

    # ─── 将 xdat 参数写入舵机固件 ─────────────────────────────────────
    def write_xdat_to_firmware(self, scs_id):
        """将单个舵机的 xdat 参数写入其 EEPROM（仅当用户明确请求时调用）。

        写入的字段: 最大扭矩, 保护扭矩, 过载扭矩, 保护时间,
        最小角度限制, 最大角度限制。
        所有值均来自该舵机的 xdat 参数文件。
        """
        bundle = self._bundles[scs_id]
        fields = bundle.fields
        self._ph.unLockEprom(scs_id)
        try:
            # 安全参数 — 值来自 xdat
            max_torque = fields.get("最大扭矩", 50)
            protect_torque = fields.get("保护扭矩", 10)
            overload_torque = fields.get("过载扭矩", 10)
            protect_time = fields.get("保护时间", 10)
            min_angle = fields.get("最小角度限制", 0)
            max_angle = fields.get("最大角度限制", 4095)

            self._ph.write2ByteTxRx(scs_id, SMS_STS_MAX_TORQUE_L, max_torque)
            self._ph.write1ByteTxRx(scs_id, SMS_STS_PROTECT_TORQUE, protect_torque)
            self._ph.write1ByteTxRx(scs_id, SMS_STS_OVERLOAD_TORQUE, overload_torque)
            self._ph.write1ByteTxRx(scs_id, SMS_STS_PROTECT_TIME, protect_time)
            # 角度限位
            self._ph.write2ByteTxRx(scs_id, 9, min_angle)
            self._ph.write2ByteTxRx(scs_id, 11, max_angle)
        finally:
            self._ph.write1ByteTxRx(scs_id, SMS_STS_LOCK, 1)

    # ─── 位置写入（含角度限位检查） ───────────────────────────────────
    def write_pos_ex(self, scs_id, position, speed, acc, time=0):
        """time=0 → 按 GOAL_SPEED 走（默认）;
        time>0 → 强制在 time 毫秒内到达（覆盖速度，最快的提速方式）。"""
        self._assert_not_emergency_stopped()
        bundle = self._bundles[scs_id]
        if not (bundle.min_angle <= position <= bundle.max_angle):
            raise SafetyViolation(
                f"ID {scs_id}: position {position} ∉ "
                f"[{bundle.min_angle}, {bundle.max_angle}]"
            )
        speed = max(0, min(int(speed), 2400))
        acc   = max(0, min(int(acc),   254))
        # SDK passes time directly to goal_time register (regs 44-45),
        # which uses 10ms units. Convert from ms.
        time_reg = max(0, min(int(time) // 10, 65535))
        return self._ph.WritePosEx(scs_id, position, speed, acc, time_reg)

    # ─── 同步复位（17 个舵机 → 2048） ──────────────────────────────────
    def sync_reset(self):
        self._assert_not_emergency_stopped()
        for sid in self._bundles:
            self._ph.RegWritePosEx(
                sid,
                self._limits.RESET_POSITION,
                self._limits.RESET_SPEED,
                self._limits.RESET_ACC,
            )
        return self._ph.RegAction()

    # ─── 同步到指定姿态（动作库用） ───────────────────────────────────
    def sync_go_to_pose(self, positions, speed=None, acc=None, time=None):
        """
        同步移动到保存的姿态。使用 RegWritePosEx + RegAction 确保所有舵机
        严格同时启动。

        参数:
            positions: {servo_id: position} 字典
            speed: 速度（默认走 SafetyLimits.RESET_SPEED = 100）
            acc:   加速度（默认走 SafetyLimits.RESET_ACC = 10）
            time:  目标时间（毫秒）；None=按 speed/acc，0=同 None；非 0=强制覆盖
        """
        self._assert_not_emergency_stopped()
        speed = self._limits.RESET_SPEED if speed is None else speed
        acc   = self._limits.RESET_ACC   if acc   is None else acc
        speed = max(0, min(int(speed), 2400))
        acc   = max(0, min(int(acc),   254))
        # SDK passes time directly to goal_time register (regs 44-45),
        # which uses 10ms units. Convert from ms.
        time_reg = 0 if time is None else max(0, min(int(time) // 10, 65535))

        # 第一遍：校验所有位置都在合法范围内，任一失败则整体不下发
        for sid, pos in positions.items():
            if sid not in self._bundles:
                raise SafetyViolation(f"姿态包含未知舵机 ID: {sid}")
            bundle = self._bundles[sid]
            if not (bundle.min_angle <= pos <= bundle.max_angle):
                raise SafetyViolation(
                    f"ID {sid}: 目标位置 {pos} ∉ "
                    f"[{bundle.min_angle}, {bundle.max_angle}]"
                )

        # 第二遍：实际下发（RegWritePosEx + RegAction）
        for sid, pos in positions.items():
            self._ph.RegWritePosEx(sid, pos, speed, acc, time_reg)
        return self._ph.RegAction()

    # ─── 参数浏览器的 EPROM 字段写入 ──────────────────────────────────
    def write_eprom_register(self, scs_id, addr, size, value):
        """参数浏览器「💾 保存选中项到 EPROM」入口。自动 unLock/lock EPROM。"""
        self._assert_not_emergency_stopped()
        self._ph.unLockEprom(scs_id)
        try:
            if size == 1:
                self._ph.write1ByteTxRx(scs_id, addr, value)
            elif size == 2:
                self._ph.write2ByteTxRx(scs_id, addr, value)
            else:
                raise ValueError(f"不支持的寄存器长度: {size}")
        finally:
            self._ph.write1ByteTxRx(scs_id, SMS_STS_LOCK, 1)

    # ─── 紧急停止 / 恢复 ──────────────────────────────────────────────
    def emergency_stop(self):
        """切断所有扭矩并冻结后续写入。"""
        self._emergency_stopped = True
        try:
            # 用广播 ID 一次性关闭所有舵机扭矩
            self._ph.write1ByteTxRx(BROADCAST_ID, SMS_STS_TORQUE_ENABLE, 0)
        except Exception:
            pass  # 总线异常也允许本地状态保持急停

    def recovery(self):
        """解除急停冻结状态（扭矩需用户在参数浏览器里手动打开）。"""
        self._emergency_stopped = False

    def is_emergency_stopped(self):
        return self._emergency_stopped

    # ─── 内部 ─────────────────────────────────────────────────────────
    def _assert_not_emergency_stopped(self):
        if self._emergency_stopped:
            raise SafetyViolation("急停已激活，请先点击「✅ 恢复控制」按钮")


# ===========================================================================
# SRAM 寄存器表（参数浏览器使用）
# ===========================================================================
# 格式: (addr, name, size, access, unit, description)
SRAM_REGISTERS = [
    (40, "力矩输出",     1, "RW", "开关", "0=关闭, 1=开启"),
    (41, "加速度",       1, "RW", "",     "0~254"),
    (42, "目标位置",     2, "RW", "编码", "0~4095"),
    (44, "目标时间",     2, "RW", "",     ""),
    (46, "目标速度",     2, "RW", "",     ""),
    (48, "目标扭矩限制", 2, "RW", "‰",    ""),
    (55, "EPROM 锁",     1, "RW", "开关", "0=解锁, 1=上锁"),
    (56, "当前位置",     2, "RO", "编码", ""),
    (58, "当前速度",     2, "RO", "",     ""),
    (60, "当前负载",     2, "RO", "",     ""),
    (62, "当前电压",     1, "RO", "0.1V", ""),
    (63, "当前温度",     1, "RO", "℃",    ""),
    (66, "运动标志",     1, "RO", "开关", "0=停止, 1=运动"),
    (69, "当前电流",     2, "RO", "",     ""),
]


# ===========================================================================
# 后端：串口、polling 线程、信号
# ===========================================================================
class ConsoleBackend(QObject):
    position_updated = pyqtSignal(dict)            # {id: present_pos}
    connection_changed = pyqtSignal(bool, str)     # (connected, message)

    def __init__(self):
        super().__init__()
        self._portHandler = None
        self._packetHandler = None
        self._bundles = []
        self._safety = None
        self._running = False
        self._poll_thread = None

    # ─── 加载 xdat ─────────────────────────────────────────────────────
    def load_bundles(self):
        """从 参数/ 目录加载 17 个 xdat 文件。"""
        bundles = []
        for i in range(1, 18):
            path = os.path.join(PARAM_DIR, f"{i}.xdat")
            if not os.path.exists(path):
                raise FileNotFoundError(f"缺少参数文件: {path}")
            fields = xdat_tool.read_xdat(path)
            bundle = ServoBundle(
                servo_id=fields["ID"],
                min_angle=fields["最小角度限制"],
                max_angle=fields["最大角度限制"],
                ofs=fields["位置偏移"],
                fields=fields,
            )
            bundles.append(bundle)
        bundles.sort(key=lambda b: b.servo_id)
        self._bundles = bundles
        return bundles

    # ─── 连接 / 断开 ──────────────────────────────────────────────────
    def connect(self, port_name, baud_rate):
        # 任何异常都转成 connection_changed 信号，绝不抛给 Qt 槽函数
        try:
            if self._portHandler:
                self.disconnect()
            self._portHandler = PortHandler(port_name)
            self._packetHandler = sms_sts(self._portHandler)
            if not self._portHandler.openPort():
                self.connection_changed.emit(False,
                    f"无法打开串口 {port_name}（设备不存在或被占用）")
                self._portHandler = None
                self._packetHandler = None
                return False
            if not self._portHandler.setBaudRate(baud_rate):
                self.connection_changed.emit(False,
                    f"无法设置波特率 {baud_rate}")
                try:
                    self._portHandler.closePort()
                except Exception:
                    pass
                self._portHandler = None
                self._packetHandler = None
                return False
            self._safety = ServoSafetyLayer(self._packetHandler, self._bundles)
            self._start_polling()
            self.connection_changed.emit(True,
                f"已连接 {port_name} @ {baud_rate}")
            return True
        except Exception as e:
            # openPort() 内部 pyserial.Serial() 构造器会抛 SerialException
            # （端口不存在 / 被占用 / 权限不足等），这里兜底
            self.connection_changed.emit(False,
                f"连接失败: {type(e).__name__}: {e}")
            self._portHandler = None
            self._packetHandler = None
            self._safety = None
            return False

    def disconnect(self):
        self._stop_polling()
        if self._portHandler:
            try:
                self._portHandler.closePort()
            except Exception:
                pass
        self._portHandler = None
        self._packetHandler = None
        self._safety = None
        self.connection_changed.emit(False, "未连接")

    # ─── Polling ───────────────────────────────────────────────────────
    def _start_polling(self):
        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def _stop_polling(self):
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=1.0)
            self._poll_thread = None

    def _poll_loop(self):
        while self._running and self._packetHandler:
            try:
                gsr = GroupSyncRead(self._packetHandler,
                                    SMS_STS_PRESENT_POSITION_L, 2)
                for b in self._bundles:
                    gsr.addParam(b.servo_id)
                comm = gsr.txRxPacket()
                states = {}
                for b in self._bundles:
                    ok, _ = gsr.isAvailable(b.servo_id,
                                            SMS_STS_PRESENT_POSITION_L, 2)
                    if ok:
                        pos = gsr.getData(b.servo_id,
                                          SMS_STS_PRESENT_POSITION_L, 2)
                        states[b.servo_id] = pos
                if states:
                    self.position_updated.emit(states)
            except Exception:
                pass
            time.sleep(POLL_INTERVAL)

    # ─── 访问器 ────────────────────────────────────────────────────────
    @property
    def safety(self):
        return self._safety

    @property
    def bundles(self):
        return self._bundles

    @property
    def packet_handler(self):
        return self._packetHandler

    @property
    def connected(self):
        return self._packetHandler is not None


# ===========================================================================
# 参数浏览器表格模型
# ===========================================================================
class ParamTableModel(QAbstractTableModel):
    HEADERS = ["区域", "名称", "地址", "当前值", "默认值", "读写", "单位", "说明"]

    def __init__(self):
        super().__init__()
        self._rows = []
        self._init_rows()

    # ─── 行结构 ────────────────────────────────────────────────────────
    def _init_rows(self):
        """初始化行结构：所有 EPROM 字段 + 所有 SRAM 字段。"""
        self._rows = []
        # EPROM
        for addr, name, default, size, region in xdat_tool.REGISTERS:
            if name == "预留":
                continue
            self._rows.append({
                "region": region,
                "name":   name,
                "addr":   addr,
                "size":   size,
                "access": "RW",
                "unit":   self._unit_for(name),
                "desc":   self._desc_for(name),
                "default": default,
                "current": default,
            })
        # SRAM
        for addr, name, size, access, unit, desc in SRAM_REGISTERS:
            self._rows.append({
                "region":  "SRAM",
                "name":    name,
                "addr":    addr,
                "size":    size,
                "access":  access,
                "unit":    unit,
                "desc":    desc,
                "default": "—",
                "current": 0,
            })

    def reset_to_xdat(self, fields):
        """切换舵机时调用：用 xdat 字段重置 EPROM 行的当前值，SRAM 行清零。"""
        for row in self._rows:
            if row["region"] == "EPROM" and row["name"] in fields:
                row["current"] = fields[row["name"]]
            elif row["region"] == "SRAM":
                row["current"] = 0
        if self._rows:
            top = self.index(0, 3)
            bot = self.index(len(self._rows) - 1, 3)
            self.dataChanged.emit(top, bot)

    def update_values(self, values):
        """从设备读回所有寄存器后更新当前值。"""
        for row in self._rows:
            if row["name"] in values:
                row["current"] = values[row["name"]]
        if self._rows:
            top = self.index(0, 3)
            bot = self.index(len(self._rows) - 1, 3)
            self.dataChanged.emit(top, bot)

    def to_xdat_fields(self):
        """导出 EPROM 行为 xdat 字段字典。"""
        return {row["name"]: row["current"]
                for row in self._rows if row["region"] == "EPROM"}

    # ─── 字段显示辅助 ──────────────────────────────────────────────────
    @staticmethod
    def _unit_for(name):
        if "角度" in name or "偏移" in name:
            return "编码"
        if "电压" in name:
            return "0.1V"
        if "扭矩" in name:
            return "‰"
        if "时间" in name and "保护" in name:
            return "×10ms"
        if "温度" in name:
            return "℃"
        if "电流" in name and "保护" in name:
            return "mA"
        if "波特率" in name:
            return "索引"
        if "分辨率" in name:
            return ""
        return ""

    @staticmethod
    def _desc_for(name):
        m = {
            "固件主版本": "只读",
            "固件次版本": "只读",
            "舵机主版本": "只读",
            "舵机次版本": "只读",
            "ID":         "1~252",
            "波特率":     "0~7 (索引)",
            "返回延时":   "0~254 (×2µs)",
            "应答状态级别": "0=仅 Ping/读, 1=全部",
            "最小角度限制": "0~4095 (机械手物理下限)",
            "最大角度限制": "0~4095 (机械手物理上限)",
            "最高温度上限": "℃",
            "最高电压":   "×0.1V",
            "最低电压":   "×0.1V",
            "最大扭矩":   "0~1000 (硬底线: 50)",
            "最小启动力": "0~1000",
            "保护电流":   "mA",
            "保护扭矩":   "0~100 (硬底线: 20)",
            "保护时间":   "×10ms (硬底线: 20)",
            "过载扭矩":   "0~100 (硬底线: 20)",
            "过流保护时间": "ms",
        }
        return m.get(name, "")

    # ─── Qt 模型接口 ───────────────────────────────────────────────────
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        col = index.column()
        if role in (Qt.DisplayRole, Qt.EditRole):
            if col == 0: return row["region"]
            if col == 1: return row["name"]
            if col == 2: return row["addr"]
            if col == 3: return str(row["current"])
            if col == 4: return str(row["default"])
            if col == 5: return row["access"]
            if col == 6: return row["unit"]
            if col == 7: return row["desc"]
        if role == Qt.TextAlignmentRole and col in (2, 3, 4):
            return Qt.AlignCenter
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole or index.column() != 3:
            return False
        try:
            v = int(value)
        except (TypeError, ValueError):
            return False
        # 范围检查
        row = self._rows[index.row()]
        max_v = (1 << (8 * row["size"])) - 1
        if v < 0 or v > max_v:
            return False
        row["current"] = v
        self.dataChanged.emit(index, index)
        return True

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() == 3:
            row = self._rows[index.row()]
            if row["access"] == "RW":
                return base | Qt.ItemIsEditable
        return base

    def get_row(self, row_idx):
        return self._rows[row_idx] if 0 <= row_idx < len(self._rows) else None


# ===========================================================================
# 手势追踪角度表 Model
# ===========================================================================
class AngleTableModel(QAbstractTableModel):
    HEADERS = ["舵机ID", "关节名称", "角度 (°)", "位置", "模式"]

    def __init__(self):
        super().__init__()
        self._angles: dict[int, float] = {}
        self._positions: dict[int, int] = {}
        self._splay_mode: dict[int, str] = {}
        from realtime_mirroring.hand_tracker import SERVO_CONFIG
        self._servo_config = SERVO_CONFIG

    def update_data(self, angles: dict[int, float], positions: dict[int, int],
                    splay_mode: dict[int, str] | None = None):
        self._angles = dict(angles)
        self._positions = dict(positions)
        self._splay_mode = dict(splay_mode or {})
        top_left = self.index(0, 0)
        bottom_right = self.index(16, len(self.HEADERS) - 1)
        self.dataChanged.emit(top_left, bottom_right)

    def rowCount(self, parent=QModelIndex()):
        return 17

    def columnCount(self, parent=QModelIndex()):
        return len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            if 0 <= section < len(self.HEADERS):
                return self.HEADERS[section]
        return QVariant()

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return QVariant()
        sid = index.row() + 1
        cfg = self._servo_config.get(sid, {})
        col = index.column()

        if role == Qt.DisplayRole:
            if col == 0:
                return str(sid)
            elif col == 1:
                return cfg.get("name", "")
            elif col == 2:
                ang = self._angles.get(sid)
                return f"{ang:.1f}" if ang is not None else "—"
            elif col == 3:
                pos = self._positions.get(sid)
                return str(pos) if pos is not None else "—"
            elif col == 4:
                jtype = cfg.get("type", "")
                if jtype in ("bottom", "thumb_lateral"):
                    return self._splay_mode.get(sid, "A")
                return "—"
        return QVariant()


# ===========================================================================
# 主窗口
# ===========================================================================
class ConsoleWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔧 舵机控制台 — STS3215 × 17")
        self.resize(1000, 720)

        self._backend = ConsoleBackend()
        self._joint_map = self._load_joint_map()
        self._pose_lib = PoseLibrary(POSES_FILE)
        self._rps_engine = RPSGameEngine(self._backend, self._pose_lib)
        self._hand_engine = HandTrackingEngine(self._backend)

        self._build_ui()
        self._wire_backend()
        self._wire_rps()
        self._wire_hand_tracking()

        # 启动时立即加载 xdat（即使未连接也允许浏览参数）
        try:
            bundles = self._backend.load_bundles()
            self._populate_servo_selectors(bundles)
            self._status(f"已加载 {len(bundles)} 个舵机的参数（来自 {PARAM_DIR}）")
        except FileNotFoundError as e:
            QMessageBox.critical(self, "参数文件缺失", str(e))
            sys.exit(1)

    # ─── 启动辅助 ──────────────────────────────────────────────────────
    def _load_joint_map(self):
        if os.path.exists(JOINT_MAP_FILE):
            try:
                with open(JOINT_MAP_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    # ─── 构造 UI ───────────────────────────────────────────────────────
    def _build_ui(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        tabs = QTabWidget()
        tabs.addTab(self._build_control_tab(), "🎮 控制台")
        tabs.addTab(self._build_param_tab(), "📋 参数浏览器")
        tabs.addTab(self._build_pose_tab(), "🎭 动作库")
        tabs.addTab(self._build_rps_tab(), "✊ 石头剪刀布")
        tabs.addTab(self._build_hand_tracking_tab(), "🖐 手势追踪")
        self.setCentralWidget(tabs)

        self._set_controls_enabled(False)

    def _build_control_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # ─ 连接 ─
        conn_group = QGroupBox("连接")
        conn_layout = QHBoxLayout(conn_group)
        conn_layout.addWidget(QLabel("串口:"))
        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        self.port_combo.setMinimumWidth(220)
        self._refresh_port_combo()
        conn_layout.addWidget(self.port_combo)
        self.port_refresh_btn = QPushButton("⟳")
        self.port_refresh_btn.setToolTip("刷新串口列表")
        self.port_refresh_btn.setFixedWidth(40)
        conn_layout.addWidget(self.port_refresh_btn)
        conn_layout.addWidget(QLabel("波特率:"))
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(
            ["1000000", "500000", "250000", "115200", "57600", "9600"]
        )
        self.baud_combo.setCurrentText("1000000")
        conn_layout.addWidget(self.baud_combo)
        self.connect_btn = QPushButton("连接")
        self.disconnect_btn = QPushButton("断开")
        conn_layout.addWidget(self.connect_btn)
        conn_layout.addWidget(self.disconnect_btn)
        layout.addWidget(conn_group)

        # ─ 状态 ─
        self.status_label = QLabel("状态: 未连接")
        self.status_label.setStyleSheet("padding: 6px; font-weight: bold;")
        layout.addWidget(self.status_label)

        # ─ 复位 ─
        action_layout = QHBoxLayout()
        self.write_xdat_btn = QPushButton("写入 xdat 参数到舵机固件")
        self.write_xdat_btn.setStyleSheet(
            "background-color: #FF9800; color: white; "
            "font-weight: bold; padding: 10px;"
        )
        self.write_xdat_btn.setToolTip(
            "⚠ 将 xdat 参数写入全部 17 个舵机的 EEPROM，\n"
            "不是只写当前选中的舵机。\n"
            "仅在需要批量恢复/同步舵机固件设置时使用。"
        )
        self.reset_btn = QPushButton("复位到 2048 (原点)")
        self.reset_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 10px;"
        )
        action_layout.addWidget(self.write_xdat_btn)
        action_layout.addWidget(self.reset_btn)
        layout.addLayout(action_layout)

        # ─ 单舵机控制 ─
        sel_group = QGroupBox("单舵机控制")
        sel_layout = QVBoxLayout(sel_group)

        servo_row = QHBoxLayout()
        servo_row.addWidget(QLabel("选择舵机:"))
        self.servo_combo = QComboBox()
        servo_row.addWidget(self.servo_combo)
        sel_layout.addLayout(servo_row)

        self.limit_label = QLabel("当前角度限制: —")
        sel_layout.addWidget(self.limit_label)

        # ─ 角度限制编辑 ─
        limit_row = QHBoxLayout()
        limit_row.addWidget(QLabel("最小角度:"))
        self.min_angle_spin = QSpinBox()
        self.min_angle_spin.setRange(0, 4095)
        limit_row.addWidget(self.min_angle_spin)
        limit_row.addWidget(QLabel("最大角度:"))
        self.max_angle_spin = QSpinBox()
        self.max_angle_spin.setRange(0, 4095)
        limit_row.addWidget(self.max_angle_spin)
        self.apply_limit_btn = QPushButton("应用限制")
        self.apply_limit_btn.setToolTip(
            "将最小/最大角度限制写入 EEPROM (寄存器 9, 11)\n"
            "断电后仍然有效"
        )
        self.apply_limit_btn.setStyleSheet(
            "background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 4px 10px;"
        )
        limit_row.addWidget(self.apply_limit_btn)
        limit_row.addStretch()
        sel_layout.addLayout(limit_row)
        self.current_pos_label = QLabel("当前实际位置: —")
        sel_layout.addWidget(self.current_pos_label)

        # ─ 中位校准 ─
        mid_row = QHBoxLayout()
        self.set_mid_btn = QPushButton("📍 设当前位置为中位 (2048)")
        self.set_mid_btn.setToolTip(
            "将舵机当前物理位置设为新的 2048 中位\n"
            "会写入 EEPROM 寄存器 31 (位置偏移)，断电不丢失"
        )
        self.set_mid_btn.setStyleSheet(
            "background-color: #FF9800; color: white; "
            "font-weight: bold; padding: 6px;"
        )
        mid_row.addWidget(self.set_mid_btn)
        self.offset_label = QLabel("当前偏移: —")
        mid_row.addWidget(self.offset_label)
        mid_row.addStretch()
        sel_layout.addLayout(mid_row)

        # 单舵机固件写入
        fw_row = QHBoxLayout()
        self.write_one_xdat_btn = QPushButton("写入 xdat 到当前舵机")
        self.write_one_xdat_btn.setToolTip(
            "仅将 xdat 参数写入当前选中的这个舵机的 EEPROM"
        )
        self.write_one_xdat_btn.setStyleSheet(
            "background-color: #FF9800; color: white; "
            "font-weight: bold; padding: 4px 10px;"
        )
        fw_row.addWidget(self.write_one_xdat_btn)
        fw_row.addStretch()
        sel_layout.addLayout(fw_row)

        slider_row = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(4095)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(50)
        self.slider.setTracking(True)
        slider_row.addWidget(self.slider)
        self.slider_value_label = QLabel("0")
        self.slider_value_label.setMinimumWidth(60)
        self.slider_value_label.setAlignment(Qt.AlignCenter)
        self.slider_value_label.setStyleSheet(
            "font-family: monospace; font-weight: bold;"
        )
        slider_row.addWidget(self.slider_value_label)
        sel_layout.addLayout(slider_row)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("速度:"))
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(0, 2400)
        self.speed_spin.setValue(100)
        param_row.addWidget(self.speed_spin)
        param_row.addWidget(QLabel("加速度:"))
        self.acc_spin = QSpinBox()
        self.acc_spin.setRange(0, 254)
        self.acc_spin.setValue(10)
        param_row.addWidget(self.acc_spin)
        param_row.addWidget(QLabel("目标时间(ms):"))
        self.time_spin = QSpinBox()
        self.time_spin.setRange(0, 65535)
        self.time_spin.setValue(0)
        self.time_spin.setSpecialValueText("速度模式")
        self.time_spin.setToolTip(
            "0 = 按「速度」走（默认）\n"
            "非 0 = 强制在指定毫秒内到达（覆盖速度，最快的提速方式）\n"
            "建议: 100~500 ms"
        )
        param_row.addWidget(self.time_spin)
        self.send_btn = QPushButton("下发到滑块位置")
        param_row.addWidget(self.send_btn)
        sel_layout.addLayout(param_row)

        layout.addWidget(sel_group)

        # 弹性占位
        layout.addStretch(1)

        # ─ 紧急停止 ─
        estop_group = QGroupBox("紧急控制")
        estop_layout = QVBoxLayout(estop_group)
        self.estop_btn = QPushButton("⚠  紧 急 停 止  ⚠")
        self.estop_btn.setStyleSheet(
            "background-color: #f44336; color: white; "
            "font-weight: bold; font-size: 18pt; padding: 18px;"
        )
        self.estop_btn.setMinimumHeight(70)
        estop_layout.addWidget(self.estop_btn)
        self.estop_hint = QLabel("（按下后立即关闭所有扭矩、冻结所有写入）")
        self.estop_hint.setAlignment(Qt.AlignCenter)
        self.estop_hint.setStyleSheet("color: gray;")
        estop_layout.addWidget(self.estop_hint)
        layout.addWidget(estop_group)

        # ─ 信号连接 ─
        self.connect_btn.clicked.connect(self._on_connect)
        self.disconnect_btn.clicked.connect(self._on_disconnect)
        self.port_refresh_btn.clicked.connect(self._on_refresh_ports)
        self.write_xdat_btn.clicked.connect(self._on_write_xdat_to_firmware)
        self.reset_btn.clicked.connect(self._on_reset)
        self.servo_combo.currentIndexChanged.connect(self._on_servo_changed)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.send_btn.clicked.connect(self._on_send_position)
        self.write_one_xdat_btn.clicked.connect(self._on_write_one_to_firmware)
        self.set_mid_btn.clicked.connect(self._on_set_mid_position)
        self.apply_limit_btn.clicked.connect(self._on_apply_limits)
        self.estop_btn.clicked.connect(self._on_estop_toggle)

        return widget

    def _build_param_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("选择舵机:"))
        self.param_servo_combo = QComboBox()
        toolbar.addWidget(self.param_servo_combo)
        self.refresh_btn = QPushButton("⟳ 从舵机刷新")
        toolbar.addWidget(self.refresh_btn)
        self.save_btn = QPushButton("💾 保存选中项到 EPROM")
        toolbar.addWidget(self.save_btn)
        toolbar.addStretch()
        self.export_btn = QPushButton("📤 导出当前舵机参数到 .xdat 文件…")
        toolbar.addWidget(self.export_btn)
        layout.addLayout(toolbar)

        self.param_model = ParamTableModel()
        self.param_view = QTableView()
        self.param_view.setModel(self.param_model)
        self.param_view.setSelectionBehavior(QTableView.SelectRows)
        self.param_view.setSelectionMode(QTableView.SingleSelection)
        hdr = self.param_view.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        layout.addWidget(self.param_view)

        self.param_servo_combo.currentIndexChanged.connect(
            self._on_param_servo_changed
        )
        self.refresh_btn.clicked.connect(self._on_param_refresh)
        self.save_btn.clicked.connect(self._on_param_save)
        self.export_btn.clicked.connect(self._on_param_export)

        return widget

    def _build_pose_tab(self):
        """🎭 动作库 Tab：记录 / 列出 / 跳转 / 删除已保存的 17 舵机姿态。"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # ─ 记录当前姿态 ─
        rec_group = QGroupBox("记录当前姿态")
        rec_layout = QVBoxLayout(rec_group)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("姿态名称:"))
        self.pose_name_edit = QLineEdit()
        self.pose_name_edit.setPlaceholderText(
            "例如: 石头 / 剪刀 / 布 / 张开 / 握拳 / 安全位"
        )
        name_row.addWidget(self.pose_name_edit)
        rec_layout.addLayout(name_row)

        btn_row = QHBoxLayout()
        self.record_pose_btn = QPushButton("📸 记录当前 17 舵机位置")
        self.record_pose_btn.setStyleSheet(
            "background-color: #FF9800; color: white; "
            "font-weight: bold; padding: 8px;"
        )
        btn_row.addWidget(self.record_pose_btn)
        btn_row.addStretch()
        rec_layout.addLayout(btn_row)

        self.pose_record_hint = QLabel(
            "提示: 用「🎮 控制台」的滑块手动调整 17 个舵机到目标姿态，"
            "然后在此命名并记录。"
        )
        self.pose_record_hint.setStyleSheet("color: gray;")
        self.pose_record_hint.setWordWrap(True)
        rec_layout.addWidget(self.pose_record_hint)

        layout.addWidget(rec_group)

        # ─ 已保存姿态库（可折叠） ─
        self.pose_library_group = QGroupBox("已保存姿态库")
        self.pose_library_group.setCheckable(True)
        self.pose_library_group.setChecked(True)
        library_layout = QVBoxLayout(self.pose_library_group)

        self.pose_list = QListWidget()
        self.pose_list.setSelectionMode(QListWidget.SingleSelection)
        self.pose_list.setStyleSheet(
            "QListWidget { font-size: 11pt; }"
            "QListWidget::item { padding: 6px; }"
        )
        library_layout.addWidget(self.pose_list)

        action_row = QHBoxLayout()
        self.goto_pose_btn = QPushButton("▶ 到达选中姿态")
        self.goto_pose_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px;"
        )
        self.rename_pose_btn = QPushButton("✏ 重命名")
        self.delete_pose_btn = QPushButton("🗑 删除选中")
        self.delete_pose_btn.setStyleSheet(
            "background-color: #f44336; color: white; "
            "font-weight: bold; padding: 8px;"
        )
        action_row.addWidget(self.goto_pose_btn)
        action_row.addWidget(self.rename_pose_btn)
        action_row.addWidget(self.delete_pose_btn)
        action_row.addStretch()
        library_layout.addLayout(action_row)

        # ─ 移动参数 ─
        move_param_row = QHBoxLayout()
        move_param_row.addWidget(QLabel("移动速度:"))
        self.pose_speed_spin = QSpinBox()
        self.pose_speed_spin.setRange(0, 2400)
        self.pose_speed_spin.setValue(100)
        self.pose_speed_spin.setToolTip("目标速度（time=0 时生效）")
        move_param_row.addWidget(self.pose_speed_spin)
        move_param_row.addWidget(QLabel("加速度:"))
        self.pose_acc_spin = QSpinBox()
        self.pose_acc_spin.setRange(0, 254)
        self.pose_acc_spin.setValue(10)
        move_param_row.addWidget(self.pose_acc_spin)
        move_param_row.addWidget(QLabel("目标时间(ms):"))
        self.pose_time_spin = QSpinBox()
        self.pose_time_spin.setRange(0, 65535)
        self.pose_time_spin.setValue(0)
        self.pose_time_spin.setSpecialValueText("速度模式")
        self.pose_time_spin.setToolTip(
            "0 = 按左边速度/加速度走\n"
            "非 0 = 强制在指定毫秒内到达（所有舵机同时到位）"
        )
        move_param_row.addWidget(self.pose_time_spin)
        move_param_row.addStretch()
        library_layout.addLayout(move_param_row)

        file_row = QHBoxLayout()
        self.export_pose_btn = QPushButton("💾 导出姿态库到文件…")
        self.import_pose_btn = QPushButton("📂 从文件导入姿态库…")
        file_row.addWidget(self.export_pose_btn)
        file_row.addWidget(self.import_pose_btn)
        file_row.addStretch()
        self.pose_lib_path_label = QLabel(f"自动保存到: {POSES_FILE}")
        self.pose_lib_path_label.setStyleSheet("color: gray; font-size: 9pt;")
        file_row.addWidget(self.pose_lib_path_label)
        library_layout.addLayout(file_row)

        layout.addWidget(self.pose_library_group)
        layout.addStretch(1)

        # ─ 信号连接 ─
        self.record_pose_btn.clicked.connect(self._on_record_pose)
        self.goto_pose_btn.clicked.connect(self._on_goto_pose)
        self.rename_pose_btn.clicked.connect(self._on_rename_pose)
        self.delete_pose_btn.clicked.connect(self._on_delete_pose)
        self.export_pose_btn.clicked.connect(self._on_export_pose_lib)
        self.import_pose_btn.clicked.connect(self._on_import_pose_lib)
        self.pose_list.itemDoubleClicked.connect(self._on_goto_pose)

        # 启动时加载已有的姿态列表
        self._refresh_pose_list()

        return widget

    # ─── 填充下拉框 ────────────────────────────────────────────────────
    def _populate_servo_selectors(self, bundles):
        for combo in (self.servo_combo, self.param_servo_combo):
            combo.blockSignals(True)
            combo.clear()
            for b in bundles:
                label = f"ID {b.servo_id}"
                joint = self._joint_map.get(str(b.servo_id))
                if joint:
                    label += f" — {joint}"
                combo.addItem(label, b.servo_id)
            combo.blockSignals(False)
        if bundles:
            self._on_servo_changed(0)
            self._on_param_servo_changed(0)

    def _wire_backend(self):
        self._backend.position_updated.connect(self._on_position_updated)
        self._backend.connection_changed.connect(self._on_connection_changed)

    # ─── 连接 / 断开 ──────────────────────────────────────────────────
    def _on_connect(self):
        port = self.port_combo.currentText().strip()
        if not port:
            QMessageBox.warning(self, "错误", "请先输入串口名")
            return
        try:
            baud = int(self.baud_combo.currentText())
        except ValueError:
            QMessageBox.warning(self, "错误", "波特率无效")
            return
        # 双保险：即便 backend.connect() 真的抛了（不应该了）也不会让程序崩
        try:
            self._backend.connect(port, baud)
        except Exception as e:
            QMessageBox.critical(
                self, "连接失败",
                f"无法连接到 {port} @ {baud}:\n\n{type(e).__name__}: {e}"
            )
            self._status(f"❌ 连接异常: {e}")

    def _on_disconnect(self):
        self._backend.disconnect()

    def _refresh_port_combo(self):
        """Refresh the serial port dropdown with currently available ports."""
        current = self.port_combo.currentText().strip()
        self.port_combo.blockSignals(True)
        self.port_combo.clear()
        ports = list_serial_ports()
        if ports:
            self.port_combo.addItems(ports)
            # Restore previous selection if still available
            if current and current in ports:
                self.port_combo.setCurrentText(current)
        else:
            # Fallback hints for manual entry
            self.port_combo.addItem("/dev/cu.wchusbserial...")
            self.port_combo.addItem("/dev/cu.usbserial...")
            self.port_combo.addItem("/dev/cu.usbmodem...")
            if current:
                self.port_combo.setCurrentText(current)
        self.port_combo.blockSignals(False)

    def _on_refresh_ports(self):
        """Handle refresh ports button click."""
        self._refresh_port_combo()
        self._status(f"已刷新串口列表 ({len(list_serial_ports())} 个可用)")

    def _on_connection_changed(self, connected, msg):
        self._status(msg)
        if connected:
            self.status_label.setText(f"状态: ● {msg}")
            self.status_label.setStyleSheet(
                "padding: 6px; font-weight: bold; color: green;"
            )
            self._set_controls_enabled(True)
        else:
            # 区分"主动断开"和"连接失败"：只有"未连接"是主动断开，
            # 其他消息（"无法打开…"、"连接失败: SerialException:…"等）
            # 都是连接失败，弹一个错误框提醒用户
            is_passive = (msg == "未连接")
            self.status_label.setText(f"状态: ● {msg}")
            color = "gray" if is_passive else "red"
            self.status_label.setStyleSheet(
                f"padding: 6px; font-weight: bold; color: {color};"
            )
            self._set_controls_enabled(False)
            if not is_passive:
                QMessageBox.warning(self, "连接失败", msg)
        # 刷新滑块初值（连接后立即读一次 PresentPos）
        if connected:
            idx = self.servo_combo.currentIndex()
            if idx >= 0:
                self._on_servo_changed(idx)

    # ─── 写入 xdat 参数到舵机固件 ──────────────────────────────────────
    def _on_write_xdat_to_firmware(self):
        """将 xdat 参数写入所有舵机 EEPROM（仅用户明确点击时触发）。"""
        if not self._backend.connected:
            QMessageBox.warning(self, "未连接", "请先连接串口再写入固件参数。")
            return

        reply = QMessageBox.question(
            self, "确认写入固件 — 全部 17 个舵机",
            "⚠ 这将把 xdat 参数写入 全部 17 个舵机 的 EEPROM，\n"
            "不是只写当前选中的舵机。\n\n"
            "写入内容:\n"
            "  - 最大扭矩 (reg 16)\n"
            "  - 保护扭矩 (reg 34)\n"
            "  - 过载扭矩 (reg 36)\n"
            "  - 保护时间 (reg 35)\n"
            "  - 最小/最大角度限制 (reg 9, 11)\n\n"
            "这些值来自 参数/*.xdat 文件，断电后仍然有效。\n"
            "确定继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            errors = []
            for b in self._backend.bundles:
                try:
                    self._backend.safety.write_xdat_to_firmware(b.servo_id)
                except Exception as e:
                    errors.append(f"ID {b.servo_id}: {e}")
            if errors:
                QMessageBox.warning(self, "部分失败",
                                    "以下舵机写入失败:\n\n" +
                                    "\n".join(errors))
                self._status(f"⚠ xdat 参数写入部分失败: {len(errors)} 个")
            else:
                self._status(
                    "✅ 已将 xdat 参数写入全部 17 个舵机固件"
                )
        finally:
            QApplication.restoreOverrideCursor()

    def _on_reset(self):
        if not self._backend.connected:
            return
        try:
            self._backend.safety.sync_reset()
            self._status(
                "✅ 已下发复位指令 (17 舵机 → 2048, speed=100, acc=10)"
            )
        except SafetyViolation as e:
            QMessageBox.warning(self, "无法复位", str(e))

    # ─── 单舵机滑块 ────────────────────────────────────────────────────
    def _on_servo_changed(self, idx):
        scs_id = self.servo_combo.itemData(idx)
        if scs_id is None:
            return
        bundle = next((b for b in self._backend.bundles
                       if b.servo_id == scs_id), None)
        if not bundle:
            return

        # 从舵机固件读取角度限位和当前实际位置
        if self._backend.connected:
            ph = self._backend.packet_handler
            # EEPROM 默认锁定，需先解锁才能读 reg 9/11/31
            ph.unLockEprom(scs_id)
            time.sleep(0.01)  # 等待解锁生效
            # 读 EEPROM 角度限位 (reg 9 = min, reg 11 = max)
            min_angle, comm_min, _ = ph.read2ByteTxRx(scs_id, 9)
            max_angle, comm_max, _ = ph.read2ByteTxRx(scs_id, 11)
            # 读位置偏移 (reg 31)
            ofs, comm_ofs, _ = ph.read2ByteTxRx(scs_id, 31)
            # 重新锁定 EEPROM
            ph.LockEprom(scs_id)

            if comm_min == COMM_SUCCESS and comm_max == COMM_SUCCESS:
                self.limit_label.setText(
                    f"当前角度限制: min={min_angle}  max={max_angle}  (来自舵机固件)"
                )
            else:
                min_angle = bundle.min_angle
                max_angle = bundle.max_angle
                self.limit_label.setText(
                    f"当前角度限制: min={min_angle}  max={max_angle}  (读取失败，使用 xdat)"
                )

            if comm_ofs == COMM_SUCCESS:
                self.offset_label.setText(f"当前偏移: {ofs}")
            else:
                self.offset_label.setText(f"当前偏移: {bundle.ofs} (xdat)")

            # 读当前实际位置
            pos, comm_pos, _ = ph.ReadPos(scs_id)
            if comm_pos == COMM_SUCCESS:
                self.current_pos_label.setText(f"当前实际位置: {pos}")
                init_pos = pos
            else:
                init_pos = 2048
                self.current_pos_label.setText(
                    f"当前实际位置: 读取失败 (默认 {init_pos})"
                )
        else:
            min_angle = bundle.min_angle
            max_angle = bundle.max_angle
            self.limit_label.setText(
                f"当前角度限制: min={min_angle}  max={max_angle}  (xdat，未连接)"
            )
            self.offset_label.setText(f"当前偏移: {bundle.ofs} (xdat)")
            init_pos = 2048
            self.current_pos_label.setText(
                f"当前实际位置: 未连接 (默认 {init_pos})"
            )

        # 同步 min/max 编辑框
        self.min_angle_spin.blockSignals(True)
        self.min_angle_spin.setValue(min_angle)
        self.min_angle_spin.blockSignals(False)
        self.max_angle_spin.blockSignals(True)
        self.max_angle_spin.setValue(max_angle)
        self.max_angle_spin.blockSignals(False)

        # 设置滑块范围 + 初值（blockSignals 防触发）
        self.slider.blockSignals(True)
        self.slider.setRange(min_angle, max_angle)
        self.slider.setValue(init_pos)
        self.slider_value_label.setText(str(init_pos))
        self.slider.blockSignals(False)

    def _on_slider_changed(self, value):
        """拖动过程中只更新数字显示，不发送串口命令。"""
        self.slider_value_label.setText(str(value))

    def _on_slider_released(self):
        """松手时才真正下发一次。"""
        if not self._backend.connected or not self._backend.safety:
            return
        idx = self.servo_combo.currentIndex()
        scs_id = self.servo_combo.itemData(idx)
        if scs_id is None:
            return
        value = self.slider.value()
        speed = self.speed_spin.value()
        acc = self.acc_spin.value()
        time = self.time_spin.value()
        try:
            self._backend.safety.write_pos_ex(scs_id, value, speed, acc, time)
            mode = f"time={time}ms" if time > 0 else f"speed={speed}, acc={acc}"
            self._status(
                f"✅ ID {scs_id} → {value} ({mode})"
            )
        except SafetyViolation as e:
            QMessageBox.warning(self, "安全限制", str(e))
            bundle = next(b for b in self._backend.bundles
                          if b.servo_id == scs_id)
            mid = (bundle.min_angle + bundle.max_angle) // 2
            self.slider.blockSignals(True)
            self.slider.setValue(mid)
            self.slider_value_label.setText(str(mid))
            self.slider.blockSignals(False)

    def _on_send_position(self):
        """「下发到滑块位置」按钮 = 等价于滑块松手。"""
        self._on_slider_released()

    def _on_position_updated(self, states):
        """polling 线程更新当前位置标签（不动滑块）。"""
        idx = self.servo_combo.currentIndex()
        scs_id = self.servo_combo.itemData(idx)
        if scs_id is not None and scs_id in states:
            self.current_pos_label.setText(
                f"当前实际位置: {states[scs_id]}  "
                f"(polling @ {POLL_INTERVAL*1000:.0f}ms)"
            )

    # ─── 中位校准 ──────────────────────────────────────────────────────
    def _on_write_one_to_firmware(self):
        """将 xdat 参数写入当前选中的单个舵机 EEPROM。"""
        if not self._backend.connected:
            QMessageBox.warning(self, "未连接", "请先连接串口再写入固件参数。")
            return
        idx = self.servo_combo.currentIndex()
        scs_id = self.servo_combo.itemData(idx)
        if scs_id is None:
            return

        bundle = next((b for b in self._backend.bundles
                       if b.servo_id == scs_id), None)
        if not bundle:
            return

        reply = QMessageBox.question(
            self, f"确认写入固件 — 仅舵机 {scs_id}",
            f"将把 xdat 参数写入 仅舵机 {scs_id} 的 EEPROM。\n\n"
            "写入内容: 最大扭矩, 保护扭矩, 过载扭矩, 保护时间,\n"
            "最小/最大角度限制\n\n"
            "这些值来自 参数/*.xdat 文件，断电后仍然有效。\n"
            "确定继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            self._backend.safety.write_xdat_to_firmware(scs_id)
            self._status(
                f"✅ 已将 xdat 参数写入舵机 {scs_id} 固件"
            )
            # 刷新显示
            self._on_servo_changed(idx)
        except Exception as e:
            QMessageBox.warning(self, "写入失败", str(e))

    def _on_set_mid_position(self):
        """将当前物理位置设为该舵机的新 2048 中位（写入 reg 31 位置偏移）。"""
        if not self._backend.connected or not self._backend.safety:
            QMessageBox.warning(self, "未连接", "请先连接串口")
            return
        idx = self.servo_combo.currentIndex()
        scs_id = self.servo_combo.itemData(idx)
        if scs_id is None:
            return
        # 读当前实际位置
        pos, comm, _ = self._backend.packet_handler.ReadPos(scs_id)
        if comm != COMM_SUCCESS:
            QMessageBox.warning(self, "读取失败",
                                f"无法读取 ID {scs_id} 当前位置")
            return
        bundle = next((b for b in self._backend.bundles
                       if b.servo_id == scs_id), None)
        old_offset = bundle.ofs if bundle else 0
        new_offset = old_offset + pos - 2048

        if QMessageBox.question(
            self, "确认中位校准",
            f"ID {scs_id} 当前物理位置: {pos}\n\n"
            f"将写入 EEPROM 寄存器 31 (位置偏移):\n"
            f"  当前偏移: {old_offset}\n"
            f"  新偏移:   {new_offset}  (= {old_offset} + {pos} - 2048)\n\n"
            f"此后，舵机当前物理位置 {pos} 将被视为 2048 (中位)。\n"
            f"断电后仍然有效。确定继续？",
            QMessageBox.Yes | QMessageBox.No,
        ) != QMessageBox.Yes:
            return

        try:
            self._backend.safety.write_eprom_register(
                scs_id, 31, 2, new_offset
            )
            # 更新内存中的 bundle 缓存
            bundle.ofs = new_offset
            bundle.fields["位置偏移"] = new_offset
            self.offset_label.setText(f"当前偏移: {new_offset}")
            # 移动到新中位 2048 并更新滑块
            self._backend.safety.write_pos_ex(scs_id, 2048, 100, 10)
            self.slider.blockSignals(True)
            self.slider.setValue(2048)
            self.slider_value_label.setText("2048")
            self.slider.blockSignals(False)
            self.current_pos_label.setText(f"当前实际位置: 2048 (新中位)")
            self._status(
                f"✅ ID {scs_id} 中位校准完成: "
                f"位置偏移 {old_offset} → {new_offset}, "
                f"物理 {pos} → 逻辑 2048"
            )
        except SafetyViolation as e:
            QMessageBox.warning(self, "急停中", str(e))
        except Exception as e:
            QMessageBox.warning(self, "写入失败", str(e))

    def _on_apply_limits(self):
        """将编辑框中的最小/最大角度限制写入 EEPROM (寄存器 9, 11)。"""
        if not self._backend.connected or not self._backend.safety:
            QMessageBox.warning(self, "未连接", "请先连接串口")
            return
        idx = self.servo_combo.currentIndex()
        scs_id = self.servo_combo.itemData(idx)
        if scs_id is None:
            return
        new_min = self.min_angle_spin.value()
        new_max = self.max_angle_spin.value()
        if new_min >= new_max:
            QMessageBox.warning(self, "范围错误",
                                f"最小角度 ({new_min}) 必须小于最大角度 ({new_max})")
            return
        bundle = next((b for b in self._backend.bundles
                       if b.servo_id == scs_id), None)
        if not bundle:
            return

        if QMessageBox.question(
            self, "确认修改角度限制",
            f"ID {scs_id} 角度限制:\n"
            f"  最小: {bundle.min_angle} → {new_min}\n"
            f"  最大: {bundle.max_angle} → {new_max}\n\n"
            f"将写入 EEPROM 寄存器 9 (最小) 和 11 (最大)。\n"
            f"断电后仍然有效。确定继续？",
            QMessageBox.Yes | QMessageBox.No,
        ) != QMessageBox.Yes:
            return

        try:
            self._backend.safety.write_eprom_register(scs_id, 9, 2, new_min)
            self._backend.safety.write_eprom_register(scs_id, 11, 2, new_max)
            # 更新内存缓存
            bundle.min_angle = new_min
            bundle.max_angle = new_max
            bundle.fields["最小角度限制"] = new_min
            bundle.fields["最大角度限制"] = new_max
            self.limit_label.setText(
                f"当前角度限制: min={new_min}  max={new_max}  (已保存)"
            )
            # 更新滑块范围
            self.slider.blockSignals(True)
            self.slider.setRange(new_min, new_max)
            self.slider.blockSignals(False)
            self._status(
                f"✅ ID {scs_id} 角度限制已更新: "
                f"min={new_min}, max={new_max}"
            )
        except SafetyViolation as e:
            QMessageBox.warning(self, "急停中", str(e))
        except Exception as e:
            QMessageBox.warning(self, "写入失败", str(e))

    # ─── 紧急停止 ──────────────────────────────────────────────────────
    def _on_estop_toggle(self):
        if not self._backend.safety:
            return
        if self._backend.safety.is_emergency_stopped():
            self._backend.safety.recovery()
            self.estop_btn.setText("⚠  紧 急 停 止  ⚠")
            self.estop_btn.setStyleSheet(
                "background-color: #f44336; color: white; "
                "font-weight: bold; font-size: 18pt; padding: 18px;"
            )
            self.status_label.setText(
                "状态: 已恢复（扭矩需要到参数浏览器手动打开）"
            )
            self.status_label.setStyleSheet(
                "padding: 6px; font-weight: bold; color: orange;"
            )
            self._set_controls_enabled(self._backend.connected)
        else:
            self._backend.safety.emergency_stop()
            self.estop_btn.setText("✅  恢 复 控 制")
            self.estop_btn.setStyleSheet(
                "background-color: #FF9800; color: white; "
                "font-weight: bold; font-size: 18pt; padding: 18px;"
            )
            self.status_label.setText(
                "🚨 急停激活：所有扭矩已切断，所有写入已冻结"
            )
            self.status_label.setStyleSheet(
                "padding: 6px; font-weight: bold; color: red;"
            )
            self._set_controls_enabled(False)

    # ─── 参数浏览器 ────────────────────────────────────────────────────
    def _on_param_servo_changed(self, idx):
        scs_id = self.param_servo_combo.itemData(idx)
        if scs_id is None:
            return
        bundle = next((b for b in self._backend.bundles
                       if b.servo_id == scs_id), None)
        if not bundle:
            return
        self.param_model.reset_to_xdat(bundle.fields)

    def _on_param_refresh(self):
        if not self._backend.connected:
            QMessageBox.information(self, "提示", "请先连接串口")
            return
        idx = self.param_servo_combo.currentIndex()
        scs_id = self.param_servo_combo.itemData(idx)
        if scs_id is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            values = {}
            # EPROM
            for addr, name, default, size, region in xdat_tool.REGISTERS:
                if name == "预留":
                    continue
                try:
                    if size == 1:
                        v, comm, _ = self._backend.packet_handler.read1ByteTxRx(
                            scs_id, addr)
                    else:
                        v, comm, _ = self._backend.packet_handler.read2ByteTxRx(
                            scs_id, addr)
                    if comm == COMM_SUCCESS:
                        values[name] = v
                except Exception:
                    pass
            # SRAM
            for addr, name, size, access, unit, desc in SRAM_REGISTERS:
                try:
                    if size == 1:
                        v, comm, _ = self._backend.packet_handler.read1ByteTxRx(
                            scs_id, addr)
                    else:
                        v, comm, _ = self._backend.packet_handler.read2ByteTxRx(
                            scs_id, addr)
                    if comm == COMM_SUCCESS:
                        values[name] = v
                except Exception:
                    pass
            self.param_model.update_values(values)
            self._status(f"✅ 已从 ID {scs_id} 刷新 {len(values)} 个字段")
        finally:
            QApplication.restoreOverrideCursor()

    def _on_param_save(self):
        if not self._backend.connected or not self._backend.safety:
            return
        sel = self.param_view.selectionModel().selectedRows()
        if not sel:
            QMessageBox.information(self, "提示", "请先在表格中选择一行")
            return
        row_idx = sel[0].row()
        row = self.param_model.get_row(row_idx)
        if not row or row["region"] != "EPROM":
            QMessageBox.information(
                self, "提示",
                "只能保存 EPROM 字段。SRAM 字段读写请使用专门写指令。"
            )
            return
        idx = self.param_servo_combo.currentIndex()
        scs_id = self.param_servo_combo.itemData(idx)
        if scs_id is None:
            return
        value = row["current"]
        if not QMessageBox.question(
            self, "确认保存",
            f"将 ID {scs_id} 的 {row['name']} 写入 EPROM，新值 = {value}？",
            QMessageBox.Yes | QMessageBox.No,
        ) == QMessageBox.Yes:
            return
        try:
            self._backend.safety.write_eprom_register(
                scs_id, row["addr"], row["size"], value
            )
            self._status(
                f"✅ ID {scs_id} {row['name']} = {value} 已写入 EPROM"
            )
        except SafetyViolation as e:
            QMessageBox.warning(self, "急停中", str(e))
        except Exception as e:
            QMessageBox.warning(self, "写入失败", str(e))

    def _on_param_export(self):
        idx = self.param_servo_combo.currentIndex()
        scs_id = self.param_servo_combo.itemData(idx)
        if scs_id is None:
            return
        default_name = f"{scs_id}.xdat"
        path, _ = QFileDialog.getSaveFileName(
            self, "导出参数到 .xdat",
            os.path.join(PARAM_DIR, default_name),
            "xdat files (*.xdat)",
        )
        if not path:
            return
        try:
            fields = self.param_model.to_xdat_fields()
            fields["ID"] = scs_id
            xdat_tool.write_xdat(path, fields)
            self._status(f"✅ 已导出到 {path}")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))

    # ─── 动作库 ────────────────────────────────────────────────────────
    def _refresh_pose_list(self):
        """刷新姿态列表显示。"""
        self.pose_list.clear()
        for pose in self._pose_lib.poses:
            label = f"{pose.name}    ({len(pose.positions)} 个舵机"
            if pose.created_at:
                label += f", {pose.created_at}"
            label += ")"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, pose.name)
            self.pose_list.addItem(item)

    def _on_record_pose(self):
        """记录当前 17 舵机位置为新姿态。"""
        name = self.pose_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "需要名称",
                                "请先在「姿态名称」框里输入一个名字")
            return
        if not self._backend.connected:
            QMessageBox.warning(self, "未连接",
                                "请先连接串口，再记录姿态")
            return

        # 检查同名
        existing = self._pose_lib.get(name)
        if existing:
            if QMessageBox.question(
                self, "姿态已存在",
                f"姿态「{name}」已存在，要覆盖吗？",
                QMessageBox.Yes | QMessageBox.No,
            ) != QMessageBox.Yes:
                return
            self._pose_lib.remove(name)

        # 用 GroupSyncRead 一次性读全部 17 舵机当前位
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            gsr = GroupSyncRead(self._backend.packet_handler,
                                SMS_STS_PRESENT_POSITION_L, 2)
            for b in self._backend.bundles:
                gsr.addParam(b.servo_id)
            comm = gsr.txRxPacket()
            positions = {}
            for b in self._backend.bundles:
                ok, _ = gsr.isAvailable(b.servo_id,
                                        SMS_STS_PRESENT_POSITION_L, 2)
                if ok:
                    pos = gsr.getData(b.servo_id,
                                      SMS_STS_PRESENT_POSITION_L, 2)
                    positions[b.servo_id] = pos

            missing = [b.servo_id for b in self._backend.bundles
                       if b.servo_id not in positions]
            if missing:
                QMessageBox.warning(
                    self, "部分舵机读取失败",
                    f"以下舵机未读到位置: {missing}\n"
                    f"已读到 {len(positions)} 个，建议先检查总线再记录。"
                )
                return

            pose = Pose(
                name=name,
                positions=positions,
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            self._pose_lib.add(pose)
            self._refresh_pose_list()
            self.pose_name_edit.clear()
            self._status(
                f"✅ 已记录姿态「{name}」({len(positions)} 个舵机)"
            )
        except Exception as e:
            QMessageBox.warning(self, "记录失败", str(e))
        finally:
            QApplication.restoreOverrideCursor()

    def _on_goto_pose(self, *args):
        """到达选中的姿态（双击列表项或点按钮）。"""
        item = self.pose_list.currentItem()
        if not item:
            QMessageBox.information(self, "提示",
                                    "请先在列表中选择一个姿态")
            return
        name = item.data(Qt.UserRole)
        pose = self._pose_lib.get(name)
        if not pose:
            return
        if not self._backend.connected:
            QMessageBox.warning(self, "未连接",
                                "请先连接串口")
            return
        try:
            speed = self.pose_speed_spin.value()
            acc = self.pose_acc_spin.value()
            time_ms = self.pose_time_spin.value()
            self._backend.safety.sync_go_to_pose(
                pose.positions, speed=speed, acc=acc, time=time_ms
            )
            if time_ms > 0:
                self._status(
                    f"✅ 已下发姿态「{name}」(time={time_ms}ms, 17舵机同时到位)"
                )
            else:
                self._status(
                    f"✅ 已下发姿态「{name}」(speed={speed}, acc={acc})"
                )
        except SafetyViolation as e:
            QMessageBox.warning(self, "安全限制", str(e))
        except Exception as e:
            QMessageBox.warning(self, "执行失败", str(e))

    def _on_rename_pose(self):
        """重命名选中的姿态。"""
        item = self.pose_list.currentItem()
        if not item:
            QMessageBox.information(self, "提示",
                                    "请先在列表中选择一个姿态")
            return
        old_name = item.data(Qt.UserRole)
        new_name, ok = QInputDialog.getText(
            self, "重命名姿态", "新名称:", text=old_name
        )
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name or new_name == old_name:
            return
        if self._pose_lib.get(new_name):
            QMessageBox.warning(self, "重名", f"姿态「{new_name}」已存在")
            return
        if self._pose_lib.rename(old_name, new_name):
            self._refresh_pose_list()
            self._status(f"✅ 已重命名: {old_name} → {new_name}")

    def _on_delete_pose(self):
        """删除选中的姿态。"""
        item = self.pose_list.currentItem()
        if not item:
            QMessageBox.information(self, "提示",
                                    "请先在列表中选择一个姿态")
            return
        name = item.data(Qt.UserRole)
        if QMessageBox.question(
            self, "确认删除",
            f"删除姿态「{name}」？\n此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        if self._pose_lib.remove(name):
            self._refresh_pose_list()
            self._status(f"✅ 已删除姿态「{name}」")

    def _on_export_pose_lib(self):
        """导出姿态库到 JSON 文件（用户选路径）。"""
        if not self._pose_lib.poses:
            QMessageBox.information(self, "空", "当前姿态库为空")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出姿态库",
            os.path.join(THIS_DIR, "poses_export.json"),
            "JSON files (*.json)",
        )
        if not path:
            return
        try:
            self._pose_lib.save_to_file(path)
            self._status(f"✅ 已导出姿态库到 {path}")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))

    def _on_import_pose_lib(self):
        """从 JSON 文件导入姿态库（追加或覆盖）。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "导入姿态库",
            THIS_DIR,
            "JSON files (*.json)",
        )
        if not path:
            return
        mode = QMessageBox.question(
            self, "导入方式",
            "选择导入方式:\n「是」= 追加(保留已有)\n「否」= 覆盖(清空后再导入)",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
        )
        if mode == QMessageBox.Cancel:
            return
        try:
            if mode == QMessageBox.No:
                self._pose_lib.clear()
            new_lib = PoseLibrary()
            new_lib.load_from_file(path)
            added = 0
            skipped = 0
            for pose in new_lib.poses:
                if self._pose_lib.add(pose):
                    added += 1
                else:
                    skipped += 1
            self._refresh_pose_list()
            self._status(
                f"✅ 导入完成: 新增 {added} 个, 跳过 {skipped} 个同名"
            )
        except Exception as e:
            QMessageBox.warning(self, "导入失败", str(e))

    # ─── RPS 石头剪刀布游戏 Tab ────────────────────────────────────────
    def _build_rps_tab(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # ─ 左: 摄像头画面 + 状态 ─
        left = QWidget()
        left_layout = QVBoxLayout(left)

        ctrl_row = QHBoxLayout()
        self.rps_start_btn = QPushButton("🎬 启动游戏")
        self.rps_start_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 10px; font-size: 12pt;"
        )
        self.rps_stop_btn = QPushButton("⏹ 停止游戏")
        self.rps_stop_btn.setStyleSheet(
            "background-color: #f44336; color: white; "
            "font-weight: bold; padding: 10px; font-size: 12pt;"
        )
        self.rps_stop_btn.setEnabled(False)
        ctrl_row.addWidget(self.rps_start_btn)
        ctrl_row.addWidget(self.rps_stop_btn)
        ctrl_row.addStretch()
        ctrl_row.addWidget(QLabel("摄像头:"))
        self.rps_camera_combo = QComboBox()
        self.rps_camera_combo.addItems(["0", "1", "2", "3"])
        self.rps_camera_combo.setCurrentIndex(0)
        ctrl_row.addWidget(self.rps_camera_combo)
        left_layout.addLayout(ctrl_row)

        self.rps_state_label = QLabel("状态: 未启动")
        self.rps_state_label.setStyleSheet(
            "padding: 6px; font-weight: bold; font-size: 11pt;"
        )
        left_layout.addWidget(self.rps_state_label)

        self.rps_gesture_label = QLabel("当前识别: —")
        self.rps_gesture_label.setStyleSheet(
            "padding: 6px; font-weight: bold; font-size: 14pt; "
            "color: #FF9800;"
        )
        self.rps_gesture_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.rps_gesture_label)

        # 摄像头画面
        self.rps_video_label = QLabel()
        self.rps_video_label.setMinimumSize(640, 480)
        self.rps_video_label.setAlignment(Qt.AlignCenter)
        self.rps_video_label.setStyleSheet(
            "background-color: #000; color: #888; "
            "border: 1px solid #444;"
        )
        self.rps_video_label.setText("（摄像头画面）")
        left_layout.addWidget(self.rps_video_label)

        # 战绩
        score_row = QHBoxLayout()
        self.rps_score_label = QLabel("累计: 你 0  :  机械手 0")
        self.rps_score_label.setStyleSheet(
            "padding: 6px; font-weight: bold; font-size: 12pt;"
        )
        self.rps_score_label.setAlignment(Qt.AlignCenter)
        score_row.addWidget(self.rps_score_label)
        left_layout.addLayout(score_row)

        layout.addWidget(left, stretch=2)

        # ─ 右: 配置 + 日志 ─
        right = QWidget()
        right_layout = QVBoxLayout(right)

        # 胜出映射配置
        map_group = QGroupBox("胜出映射配置")
        map_layout = QFormLayout(map_group)

        self.rps_map_combos = {}
        for user_label, default_winning in [
            ("rock",     "布"),
            ("paper",    "剪刀"),
            ("scissors", "石头"),
        ]:
            zh = RPSGameEngine.ZH[user_label]
            combo = QComboBox()
            combo.setEditable(True)
            # 用当前姿态库的名字填充
            for name in self._pose_lib.names():
                combo.addItem(name)
            combo.setCurrentText(default_winning)
            self.rps_map_combos[user_label] = combo
            map_layout.addRow(f"你出 【{zh}】 → 机械手出:", combo)

        map_hint = QLabel(
            "姿态名必须与「🎭 动作库」Tab 中已保存的姿态一致。"
        )
        map_hint.setStyleSheet("color: gray;")
        map_hint.setWordWrap(True)
        map_layout.addRow(map_hint)

        right_layout.addWidget(map_group)

        # 日志
        log_group = QGroupBox("游戏日志")
        log_layout = QVBoxLayout(log_group)
        self.rps_log = QPlainTextEdit()
        self.rps_log.setReadOnly(True)
        self.rps_log.setMaximumBlockCount(200)
        self.rps_log.setStyleSheet(
            "background-color: #1e1e1e; color: #ddd; "
            "font-family: monospace;"
        )
        log_layout.addWidget(self.rps_log)
        right_layout.addWidget(log_group, stretch=1)

        layout.addWidget(right, stretch=1)

        # ─ 信号 ─
        self.rps_start_btn.clicked.connect(self._on_rps_start)
        self.rps_stop_btn.clicked.connect(self._on_rps_stop)

        return widget

    def _build_hand_tracking_tab(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # ─ 左: 摄像头画面 + 控制 ─
        left = QWidget()
        left_layout = QVBoxLayout(left)

        ctrl_row = QHBoxLayout()
        self.ht_start_btn = QPushButton("▶ 开始追踪")
        self.ht_start_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 10px; font-size: 12pt;"
        )
        self.ht_pause_btn = QPushButton("⏸ 暂停")
        self.ht_pause_btn.setStyleSheet(
            "background-color: #FF9800; color: white; "
            "font-weight: bold; padding: 10px; font-size: 12pt;"
        )
        self.ht_pause_btn.setEnabled(False)
        self.ht_stop_btn = QPushButton("⏹ 停止")
        self.ht_stop_btn.setStyleSheet(
            "background-color: #f44336; color: white; "
            "font-weight: bold; padding: 10px; font-size: 12pt;"
        )
        self.ht_stop_btn.setEnabled(False)
        ctrl_row.addWidget(self.ht_start_btn)
        ctrl_row.addWidget(self.ht_pause_btn)
        ctrl_row.addWidget(self.ht_stop_btn)
        ctrl_row.addStretch()
        ctrl_row.addWidget(QLabel("摄像头:"))
        self.ht_camera_combo = QComboBox()
        self.ht_camera_combo.addItems(["0", "1", "2", "3"])
        self.ht_camera_combo.setCurrentIndex(0)
        ctrl_row.addWidget(self.ht_camera_combo)
        left_layout.addLayout(ctrl_row)

        self.ht_state_label = QLabel("状态: 未启动")
        self.ht_state_label.setStyleSheet(
            "padding: 6px; font-weight: bold; font-size: 11pt;"
        )
        left_layout.addWidget(self.ht_state_label)

        # 摄像头画面
        self.ht_video_label = QLabel()
        self.ht_video_label.setMinimumSize(640, 480)
        self.ht_video_label.setAlignment(Qt.AlignCenter)
        self.ht_video_label.setStyleSheet(
            "background-color: #000; color: #888; "
            "border: 1px solid #444;"
        )
        self.ht_video_label.setText("（摄像头画面）")
        left_layout.addWidget(self.ht_video_label)

        layout.addWidget(left, stretch=2)

        # ─ 右: 参数 + 角度表 + 安全 ─
        right = QWidget()
        right_layout = QVBoxLayout(right)

        # 更新频率 & 速度
        settings_group = QGroupBox("追踪参数")
        settings_layout = QFormLayout(settings_group)
        rate_row = QHBoxLayout()
        self.ht_rate_combo = QComboBox()
        self.ht_rate_combo.addItems(["5 Hz", "10 Hz", "15 Hz", "20 Hz"])
        self.ht_rate_combo.setCurrentIndex(2)  # 15 Hz default
        rate_row.addWidget(self.ht_rate_combo)
        settings_layout.addRow("更新频率:", rate_row)
        speed_row = QHBoxLayout()
        self.ht_speed_slider = QSlider(Qt.Horizontal)
        self.ht_speed_slider.setRange(10, 2400)
        self.ht_speed_slider.setValue(200)
        self.ht_speed_label = QLabel("200")
        self.ht_speed_slider.valueChanged.connect(
            lambda v: self.ht_speed_label.setText(str(v))
        )
        speed_row.addWidget(self.ht_speed_slider)
        speed_row.addWidget(self.ht_speed_label)
        settings_layout.addRow("舵机速度:", speed_row)
        right_layout.addWidget(settings_group)

        # 关节角度表
        angles_group = QGroupBox("关节角度")
        angles_layout = QVBoxLayout(angles_group)
        self.ht_angles_table = QTableView()
        self.ht_angles_table.setMinimumHeight(200)
        self.ht_angles_table.horizontalHeader().setStretchLastSection(True)
        self.ht_angles_model = AngleTableModel()
        self.ht_angles_table.setModel(self.ht_angles_model)
        self.ht_angles_table.setSelectionBehavior(QTableView.SelectRows)
        self.ht_angles_table.setAlternatingRowColors(True)
        self.ht_angles_table.verticalHeader().setVisible(False)
        # Set column widths: ID narrow, name stretches, angle/pos/mode compact
        hdr = self.ht_angles_table.horizontalHeader()
        hdr.resizeSection(0, 50)   # ID
        hdr.resizeSection(2, 80)   # Angle
        hdr.resizeSection(3, 70)   # Position
        hdr.resizeSection(4, 50)   # Mode
        angles_layout.addWidget(self.ht_angles_table)
        right_layout.addWidget(angles_group, stretch=1)

        # 安全按钮
        safety_group = QGroupBox("安全")
        safety_layout = QVBoxLayout(safety_group)
        self.ht_calibrate_btn = QPushButton("📐 校准中性位 (手指并拢后点击)")
        self.ht_calibrate_btn.setStyleSheet(
            "background-color: #FF9800; color: white; "
            "font-weight: bold; padding: 8px;"
        )
        self.ht_calibrate_btn.setEnabled(False)
        self.ht_reset_neutral_btn = QPushButton("↩ 复位到中性位")
        self.ht_reset_neutral_btn.setStyleSheet(
            "background-color: #607D8B; color: white; "
            "font-weight: bold; padding: 8px;"
        )
        self.ht_estop_btn = QPushButton("⚠ 紧急停止")
        self.ht_estop_btn.setStyleSheet(
            "background-color: #f44336; color: white; "
            "font-weight: bold; padding: 12px; font-size: 14pt;"
        )
        safety_layout.addWidget(self.ht_calibrate_btn)
        safety_layout.addWidget(self.ht_reset_neutral_btn)
        safety_layout.addWidget(self.ht_estop_btn)
        right_layout.addWidget(safety_group)

        layout.addWidget(right, stretch=1)

        # ─ 信号 ─
        self.ht_start_btn.clicked.connect(self._on_ht_start)
        self.ht_stop_btn.clicked.connect(self._on_ht_stop)
        self.ht_pause_btn.clicked.connect(self._on_ht_pause)
        self.ht_calibrate_btn.clicked.connect(self._on_ht_calibrate)
        self.ht_reset_neutral_btn.clicked.connect(self._on_ht_reset_neutral)
        self.ht_estop_btn.clicked.connect(self._on_ht_estop)

        return widget

    def _wire_hand_tracking(self):
        self._hand_engine.state_changed.connect(self._on_ht_state)
        self._hand_engine.frame_ready.connect(self._on_ht_frame)
        self._hand_engine.angles_updated.connect(self._on_ht_angles)
        self._hand_engine.error_occurred.connect(self._on_ht_error)

    # ─── 手势追踪 槽函数 ─────────────────────────────────────────────
    def _on_ht_start(self):
        try:
            camera_text = self.ht_camera_combo.currentText()
            camera_idx = int(camera_text.split()[0])
            speed = self.ht_speed_slider.value()
            rate_text = self.ht_rate_combo.currentText()
            update_hz = int(rate_text.split()[0])

            # Mutual exclusion: stop RPS if running
            if self._rps_engine.running:
                self._on_rps_stop()

            self._hand_engine.start(
                camera_index=camera_idx, speed=speed, update_hz=update_hz,
            )
            self.ht_start_btn.setEnabled(False)
            self.ht_pause_btn.setEnabled(True)
            self.ht_stop_btn.setEnabled(True)
            self.ht_calibrate_btn.setEnabled(True)
            self._status("🖐 手势追踪已启动")
        except Exception as e:
            QMessageBox.critical(
                self, "启动失败",
                f"无法启动手势追踪:\n\n{type(e).__name__}: {e}\n\n"
                f"请检查摄像头依赖 (pip install -r gesture_recognition/requirements.txt)"
            )
            self.ht_start_btn.setEnabled(True)
            self.ht_pause_btn.setEnabled(False)
            self.ht_stop_btn.setEnabled(False)

    def _on_ht_stop(self):
        try:
            self._hand_engine.stop()
            self.ht_start_btn.setEnabled(True)
            self.ht_pause_btn.setEnabled(False)
            self.ht_stop_btn.setEnabled(False)
            self.ht_calibrate_btn.setEnabled(False)
            self.ht_video_label.setText("（摄像头画面）")
            self.ht_video_label.setPixmap(QPixmap())
            self.ht_state_label.setText("状态: 未启动")
            self.ht_state_label.setStyleSheet(
                "padding: 6px; font-weight: bold; font-size: 11pt;"
            )
            self._status("⏹ 手势追踪已停止")
        except Exception as e:
            QMessageBox.critical(
                self, "停止失败",
                f"停止手势追踪时出错:\n\n{type(e).__name__}: {e}"
            )
            self.ht_start_btn.setEnabled(True)
            self.ht_pause_btn.setEnabled(False)
            self.ht_stop_btn.setEnabled(False)

    def _on_ht_pause(self):
        if self._hand_engine.paused:
            self._hand_engine.resume()
            self.ht_pause_btn.setText("⏸ 暂停")
            self.ht_pause_btn.setStyleSheet(
                "background-color: #FF9800; color: white; "
                "font-weight: bold; padding: 10px; font-size: 12pt;"
            )
        else:
            self._hand_engine.pause()
            self.ht_pause_btn.setText("▶ 恢复")
            self.ht_pause_btn.setStyleSheet(
                "background-color: #2196F3; color: white; "
                "font-weight: bold; padding: 10px; font-size: 12pt;"
            )

    def _on_ht_state(self, state):
        self.ht_state_label.setText(f"状态: {state}")
        color = {
            "TRACKING": "#4CAF50",
            "PAUSED":    "#FF9800",
            "STOPPED":   "gray",
        }.get(state, "black")
        self.ht_state_label.setStyleSheet(
            f"padding: 6px; font-weight: bold; font-size: 11pt; color: {color};"
        )

    def _on_ht_frame(self, bgr_frame):
        try:
            h, w = bgr_frame.shape[:2]
            rgb = bgr_frame[:, :, ::-1].copy()
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            target_w = self.ht_video_label.width()
            target_h = self.ht_video_label.height()
            pix = QPixmap.fromImage(qimg).scaled(
                target_w, target_h,
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self.ht_video_label.setPixmap(pix)
        except Exception:
            pass

    def _on_ht_angles(self, angles):
        positions = {}
        splay_mode = {}
        if self._hand_engine._tracker is not None:
            positions = self._hand_engine._tracker.last_positions
            splay_mode = self._hand_engine._tracker.splay_mode
        self.ht_angles_model.update_data(angles, positions, splay_mode)

    def _on_ht_calibrate(self):
        if not self._hand_engine.running:
            return
        tracker = self._hand_engine._tracker
        if tracker is None:
            return
        # Calibration happens on next process() call in the engine thread.
        # Store a flag for the engine to pick up.
        self._hand_engine._calibrate_requested = True
        self._status("📐 已请求校准中性位")

    def _on_ht_reset_neutral(self):
        tracker = self._hand_engine._tracker
        if tracker:
            tracker.clear_neutral()
            tracker.clear_splay()
        if not self._backend.connected:
            QMessageBox.warning(self, "未连接", "请先连接串口")
            return
        try:
            self._backend.safety.sync_reset()
            self._status("↩ 校准已清除，机械手已复位到中性位")
        except Exception as e:
            QMessageBox.critical(self, "复位失败", str(e))

    def _on_ht_estop(self):
        if self._backend.safety and not self._backend.safety.is_emergency_stopped():
            try:
                self._backend.safety.emergency_stop()
                self._hand_engine.pause()
                self.ht_estop_btn.setText("✅ 恢复控制")
                self.ht_estop_btn.setStyleSheet(
                    "background-color: #FF9800; color: white; "
                    "font-weight: bold; padding: 12px; font-size: 14pt;"
                )
                self._status("⚠ 紧急停止！所有舵机扭矩已切断")
            except Exception as e:
                QMessageBox.critical(self, "急停失败", str(e))
        else:
            if self._backend.safety:
                try:
                    self._backend.safety.recovery()
                except Exception:
                    pass
            self.ht_estop_btn.setText("⚠ 紧急停止")
            self.ht_estop_btn.setStyleSheet(
                "background-color: #f44336; color: white; "
                "font-weight: bold; padding: 12px; font-size: 14pt;"
            )
            self._status("✅ 急停已解除")

    def _on_ht_error(self, msg):
        # 严重错误时停止追踪
        if "缺少依赖" in msg or "无法打开摄像头" in msg:
            self._on_ht_stop()
        QMessageBox.critical(self, "手势追踪错误", msg)

    def _wire_rps(self):
        self._rps_engine.state_changed.connect(self._on_rps_state)
        self._rps_engine.gesture_visible.connect(self._on_rps_gesture_visible)
        self._rps_engine.frame_ready.connect(self._on_rps_frame)
        self._rps_engine.action_triggered.connect(self._on_rps_action_triggered)
        self._rps_engine.action_finished.connect(self._on_rps_action_finished)
        self._rps_engine.timeout_reset.connect(self._on_rps_timeout_reset)
        self._rps_engine.score_updated.connect(self._on_rps_score_updated)
        self._rps_engine.error_occurred.connect(self._on_rps_error)

    # ─── RPS 槽函数 ─────────────────────────────────────────────────
    def _on_rps_start(self):
        try:
            if not self._backend.connected:
                QMessageBox.warning(
                    self, "未连接",
                    "请先在「🎮 控制台」Tab 连接串口，否则机械手不会响应"
                )
                # 仍然允许启动游戏（用户可能只想看识别）
            # 应用最新映射
            win_map = {k: c.currentText().strip()
                       for k, c in self.rps_map_combos.items()}
            self._rps_engine.set_win_map(win_map)
            self.rps_log.clear()
            camera_text = self.rps_camera_combo.currentText()
            camera_idx = int(camera_text.split()[0])
            self.rps_log.appendPlainText(
                f"[{time.strftime('%H:%M:%S')}] 游戏启动 (摄像头 {camera_idx}), "
                f"胜出映射: {win_map}"
            )
            self._rps_engine.start(camera_index=camera_idx)
            self.rps_start_btn.setEnabled(False)
            self.rps_stop_btn.setEnabled(True)
            self._status("🎮 石头剪刀布游戏已启动")
        except Exception as e:
            # 任何异常都不能让窗口消失
            QMessageBox.critical(
                self, "启动失败",
                f"无法启动游戏:\n\n{type(e).__name__}: {e}\n\n"
                f"请检查摄像头依赖 (pip install -r gesture_recognition/requirements.txt)"
            )
            self.rps_start_btn.setEnabled(True)
            self.rps_stop_btn.setEnabled(False)

    def _on_rps_stop(self):
        try:
            self._rps_engine.stop()
            self.rps_start_btn.setEnabled(True)
            self.rps_stop_btn.setEnabled(False)
            self.rps_video_label.setText("（摄像头画面）")
            self.rps_video_label.setPixmap(QPixmap())  # 清空
            self.rps_log.appendPlainText(
                f"[{time.strftime('%H:%M:%S')}] 游戏停止"
            )
            self._status("⏹ 石头剪刀布游戏已停止")
        except Exception as e:
            QMessageBox.critical(
                self, "停止失败",
                f"停止游戏时出错:\n\n{type(e).__name__}: {e}"
            )
            self.rps_start_btn.setEnabled(True)
            self.rps_stop_btn.setEnabled(False)

    def _on_rps_state(self, state):
        self.rps_state_label.setText(f"状态: {state}")
        # 不同状态不同颜色
        color = {
            "IDLE":      "gray",
            "DETECTING": "#FF9800",
            "EXECUTING": "#2196F3",
            "RESETTING": "#9C27B0",
            "HOLD":      "#4CAF50",
            "已停止":    "gray",
        }.get(state, "black")
        self.rps_state_label.setStyleSheet(
            f"padding: 6px; font-weight: bold; font-size: 11pt; color: {color};"
        )

    def _on_rps_gesture_visible(self, gesture_zh):
        self.rps_gesture_label.setText(f"当前识别: {gesture_zh}")

    def _on_rps_frame(self, bgr_frame):
        """从 BGR frame 转 QPixmap 显示。"""
        try:
            h, w = bgr_frame.shape[:2]
            # BGR → RGB
            rgb = bgr_frame[:, :, ::-1].copy()
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            # 缩放到 label 大小，保持比例
            target_w = self.rps_video_label.width()
            target_h = self.rps_video_label.height()
            pix = QPixmap.fromImage(qimg).scaled(
                target_w, target_h,
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self.rps_video_label.setPixmap(pix)
        except Exception as e:
            # 静默忽略渲染错误（避免刷屏）
            pass

    def _on_rps_action_triggered(self, winning):
        self.rps_log.appendPlainText(
            f"[{time.strftime('%H:%M:%S')}] ▶ 触发 → 机械手出【{winning}】"
        )

    def _on_rps_action_finished(self, winning):
        self.rps_log.appendPlainText(
            f"[{time.strftime('%H:%M:%S')}] ✅ 机械手完成【{winning}】，等待下一轮"
        )

    def _on_rps_timeout_reset(self):
        self.rps_log.appendPlainText(
            f"[{time.strftime('%H:%M:%S')}] ⏰ 2 秒无手势，机械手自动复位"
        )

    def _on_rps_score_updated(self, user, bot):
        self.rps_score_label.setText(f"累计: 你 {user}  :  机械手 {bot}")

    def _on_rps_error(self, msg):
        self.rps_log.appendPlainText(
            f"[{time.strftime('%H:%M:%S')}] ❌ {msg}"
        )
        # 严重错误时停止游戏
        if "缺少依赖" in msg or "无法打开摄像头" in msg:
            self._on_rps_stop()

    # ─── 控件可用性管理 ───────────────────────────────────────────────
    def _set_controls_enabled(self, enabled):
        # 控制台 Tab
        self.write_xdat_btn.setEnabled(enabled)
        self.reset_btn.setEnabled(enabled)
        self.servo_combo.setEnabled(enabled)
        self.slider.setEnabled(enabled)
        self.speed_spin.setEnabled(enabled)
        self.acc_spin.setEnabled(enabled)
        self.time_spin.setEnabled(enabled)
        self.send_btn.setEnabled(enabled)
        self.write_one_xdat_btn.setEnabled(enabled)
        self.set_mid_btn.setEnabled(enabled)
        self.min_angle_spin.setEnabled(enabled)
        self.max_angle_spin.setEnabled(enabled)
        self.apply_limit_btn.setEnabled(enabled)
        # 参数浏览器 Tab
        self.param_servo_combo.setEnabled(enabled)
        self.refresh_btn.setEnabled(enabled)
        self.save_btn.setEnabled(enabled)
        self.export_btn.setEnabled(enabled)
        # 动作库 Tab
        self.record_pose_btn.setEnabled(enabled)
        self.goto_pose_btn.setEnabled(enabled)
        self.rename_pose_btn.setEnabled(enabled)
        self.delete_pose_btn.setEnabled(enabled)
        # RPS Tab: 启动按钮始终可用（即使未连接也能看识别），停止按钮由 running 状态控制
        self.rps_map_combos["rock"].setEnabled(True)
        self.rps_map_combos["paper"].setEnabled(True)
        self.rps_map_combos["scissors"].setEnabled(True)
        # 连接按钮互斥
        self.connect_btn.setEnabled(not enabled)
        self.disconnect_btn.setEnabled(enabled)
        # 急停按钮：只在连接后才允许
        self.estop_btn.setEnabled(enabled or self._backend.safety is not None)

    def _status(self, msg):
        self.statusbar.showMessage(msg)

    def closeEvent(self, event):
        try:
            self._hand_engine.stop()
        except Exception:
            pass
        try:
            self._rps_engine.stop()
        except Exception:
            pass
        try:
            self._backend.disconnect()
        except Exception:
            pass
        super().closeEvent(event)


# ===========================================================================
# 入口
# ===========================================================================
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = ConsoleWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()