#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rock-Paper-Scissors 手势识别 (基于 MediaPipe Hands + 规则判定)
=============================================================
启动笔记本摄像头，实时识别石头/剪刀/布。

运行:
    pip install -r requirements.txt
    python gesture_rps.py

键位:
    Q    退出
    S    截图保存 (写到当前目录 rps_<时间戳>.png)
    1/2/3  手动覆盖当前帧为 rock / scissors / paper

用法 (作为模块):
    from gesture_rps import RPSClassifier
    clf = RPSClassifier()
    result = clf.classify(hand_landmarks)   # -> 'rock' | 'paper' | 'scissors' | None
"""

import time
import sys
import argparse
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# 中文字体渲染 (cv2.putText 不支持中文, 用 PIL 绕一圈)
# ---------------------------------------------------------------------------
_FONTS: dict = {}

def _load_cn_font(size: int = 22):
    """加载一个支持中文的 TrueType, 找不到就用 PIL default."""
    if size in _FONTS:
        return _FONTS[size]
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc",                       # macOS
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",           # Linux
    ]
    font = None
    for p in candidates:
        if Path(p).exists():
            try:
                font = ImageFont.truetype(p, size)
                break
            except (IOError, OSError):
                continue
    if font is None:
        font = ImageFont.load_default()
    _FONTS[size] = font
    return font


def put_cn(img, text, pos, color=(0, 255, 255), size=22):
    """在 OpenCV BGR frame 上写中文文本."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    draw.text(pos, text, font=_load_cn_font(size), fill=tuple(color[::-1]))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# MediaPipe 句柄
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

LABEL_ZH = {
    "rock":     "石头",
    "paper":    "布",
    "scissors": "剪刀",
    None:       "未识别",
}


# -----------------------------------------------------------------------------
# 几何判定
# -----------------------------------------------------------------------------
def _vec(a, b):
    """生成从 a 指向 b 的归一化向量"""
    v = np.array([b.x - a.x, b.y - a.y, b.z - a.z], dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.zeros(3)


def fingers_state(landmarks):
    """根据 21 个关键点返回 5 指是否伸直 [thumb, idx, mid, ring, pinky].

    拇指判定:
      - 如果拇指远节 (4) 偏离掌心平面 (wrist → index_mcp 的连线) 较远 → 伸直
    其余 4 指:
      - tip.y < pip.y (屏幕坐标系, y 越小越靠上) → 伸直
    """
    states = []

    # 拇指: 用 wrist→index_mcp 向量的法向与拇指方向比较
    palm_v = _vec(landmarks[0], landmarks[5])   # wrist → index_mcp
    thumb_v = _vec(landmarks[2], landmarks[4])  # thumb mcp → thumb tip
    # 拇指与手掌大致同向 (夹角 < 60°) → 收起
    cos = float(np.dot(palm_v, thumb_v))
    states.append(cos < 0.4)                   # 阈值偏小更宽松

    # 其余 4 指: tip.y < pip.y
    for tip_i, pip_i in zip((8, 12, 16, 20), (6, 10, 14, 18)):
        states.append(landmarks[tip_i].y < landmarks[pip_i].y)
    return states


def classify_rps(fingers):
    """根据 5 指状态判定 RPS.

    fingers: [thumb, index, middle, ring, pinky]  bool 列表
    返回: 'rock' | 'paper' | 'scissors' | None
    """
    _, idx, mid, ring, pinky = fingers
    four_extended = sum([idx, mid, ring, pinky])

    # 石头: 4 指全部收
    if four_extended == 0:
        return "rock"
    # 布: 4 指全部伸
    if four_extended == 4:
        return "paper"
    # 剪刀: 只有 食指 + 中指 伸
    if idx and mid and not ring and not pinky:
        return "scissors"
    # 兜底
    return None


# -----------------------------------------------------------------------------
# 分类器封装
# -----------------------------------------------------------------------------
class RPSClassifier:
    """单手 RPS 分类器,自带滑动窗口平滑."""

    def __init__(self, smooth_window: int = 8):
        self.smooth_window = smooth_window
        self.history = deque(maxlen=smooth_window)

    def classify(self, hand_landmarks, also_extend_thumb: bool = False):
        """对一组 MediaPipe landmarks 分类.

        手势稳定 <smooth_window> 帧后输出, 否则输出 None.
        """
        if hand_landmarks is None:
            return None
        fingers = fingers_state(hand_landmarks.landmark)
        # 拇指锁定 (拇指不影响 RPS 结果, 但对 "石头 vs 握拳但拇指在上" 略放宽)
        if not also_extend_thumb:
            fingers[0] = False
        g = classify_rps(fingers)
        self.history.append(g)
        if len(self.history) < self.smooth_window:
            return None
        most = Counter(self.history).most_common(1)[0][0]
        return most

    def reset(self):
        self.history.clear()


# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------
def open_camera(index: int = 0):
    """打开摄像头.

    - 不强行设分辨率, 避免 max_camera_resolution 与 cap 不匹配触发
      `_step >= minstep` 断言 (cv2 4.8 在内置摄像头上的常见 bug)
    - Windows 优先尝试 DirectShow (CAP_DSHOW), 比默认 MSMF 稳
    """
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        idx_list = [index] + [i for i in (1, 2, 3) if i != index]
    else:
        backends = [cv2.CAP_ANY]
        idx_list = [index] + [i for i in (1, 2, 3) if i != index]

    for idx in idx_list:
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                print(f"   分辨率: {w}x{h}  (backend={int(backend)})")
                return cap, idx
            cap.release()
    return None, None


def draw_overlay(frame, fps, current, fingers=None):
    h, w = frame.shape[:2]

    # 顶部状态条
    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    put_cn(frame, f"识别: {LABEL_ZH[current]}", (w - 360, 14),
          color=(0, 255, 255), size=26)

    # 底部键位提示
    cv2.rectangle(frame, (0, h - 30), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, "[Q] quit  [S] snapshot  [1/2/3] force=rock/scissors/paper",
                (10, h - 8), cv2.FONT_HERSHEY_PLAIN, 1.0, (180, 180, 180), 1)

    if fingers is not None:
        flag = " ".join("Y" if f else "N" for f in fingers)
        cv2.putText(frame, f"fingers [{flag}]",
                    (w // 2 - 80, 38), cv2.FONT_HERSHEY_PLAIN, 1.2, (200, 200, 200), 1)


def annotate_hand(frame, hand_landmarks, current):
    """在手上方画标签 + 状态灯."""
    h, w = frame.shape[:2]
    xs = [p.x for p in hand_landmarks.landmark]
    ys = [p.y for p in hand_landmarks.landmark]
    x_min, x_max = int(min(xs) * w), int(max(xs) * w)
    y_min, y_max = int(min(ys) * h), int(max(ys) * h)

    # 方框
    cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10),
                  (0, 255, 0), 2)
    # 标签条
    cv2.rectangle(frame, (x_min - 10, y_min - 60), (x_max + 10, y_min - 10),
                  (0, 0, 0), -1)
    put_cn(frame, LABEL_ZH[current], (x_min, y_min - 50),
          color=(0, 255, 255), size=28)


def main():
    ap = argparse.ArgumentParser(description="Rock-Paper-Scissors 手势识别")
    ap.add_argument("--camera", type=int, default=0,
                    help="摄像头 index (默认 0, 即笔记本内置)")
    ap.add_argument("--no-flip", action="store_true",
                    help="不水平镜像画面 (镜像更直觉)")
    ap.add_argument("--snapshot-dir", default=".",
                    help="截图保存目录")
    args = ap.parse_args()

    cap, idx = open_camera(args.camera)
    if cap is None:
        print("❌ 摄像头打不开, 请检查是否被其它程序占用或加上 --camera 1")
        return

    print(f"✓ 摄像头已打开 (index={idx})")
    print("操作: Q=退出, S=截图, 1/2/3=强制设定 rock/scissors/paper")

    Path(args.snapshot_dir).mkdir(parents=True, exist_ok=True)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    clf = RPSClassifier(smooth_window=8)
    force_label = None
    last_t = time.time()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if not args.no_flip:
            frame = cv2.flip(frame, 1)            # 镜像, 像照镜子
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        current_label = force_label
        last_fingers = None
        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame, lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 200, 0), thickness=2),
            )
            raw = classify_rps(fingers_state(lm.landmark))
            clf.history.append(raw)
            if len(clf.history) >= clf.smooth_window:
                cur = Counter(clf.history).most_common(1)[0][0]
                if force_label is None:
                    current_label = cur
            last_fingers = fingers_state(lm.landmark)
            if current_label is not None and not force_label:
                annotate_hand(frame, lm, current_label)

        now = time.time()
        fps = 1 / (now - last_t) if now > last_t else 0
        last_t = now
        draw_overlay(frame, fps, current_label, last_fingers)

        cv2.imshow("RPS Gesture Recognition  (Q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            out = Path(args.snapshot_dir) / f"rps_{ts}.png"
            cv2.imwrite(str(out), frame)
            print(f"📸 saved: {out}")
        if key == ord("1"):
            force_label = "rock"
        elif key == ord("2"):
            force_label = "scissors"
        elif key == ord("3"):
            force_label = "paper"
        elif key == ord("0"):
            force_label = None
            clf.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
