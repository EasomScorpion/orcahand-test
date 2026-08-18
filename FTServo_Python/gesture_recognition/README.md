# Gesture Recognition — Rock-Paper-Scissors

> 笔记本摄像头 + MediaPipe Hands → 实时识别 石头 / 剪刀 / 布。
> 本目录独立于仓库其他部分，可单独运行。

## 1. 安装

Python ≥ 3.9, Windows / macOS / Linux 均可。

```bash
pip install -r requirements.txt
```

> 仓库其余部分（舵机控制、`xdat_tool.py`）用不到这个目录，互不干扰。

## 2. 运行

```bash
python gesture_rps.py
```

可选参数：

| 参数 | 默认 | 作用 |
|------|------|------|
| `--camera N`   | 0   | 摄像头 index, 默认笔记本内置；外接一般填 1 |
| `--no-flip`    | 关  | 关闭水平镜像（默认镜像更像照镜子） |
| `--snapshot-dir PATH` | `.` | 截图保存目录 |

成功后会弹出窗口：

```
+--------------------------------------+
| FPS: 28.3         识别: 石头 🪨       |
|                                  ... |
| [Q] quit  [S] snapshot  [1/2/3] force=rock/scissors/paper   ← 键位提示
+--------------------------------------+
```

### 键位

| 键 | 作用 |
|----|------|
| `Q` / `Esc` | 退出 |
| `S` | 把当前帧截图到 `rps_YYYYMMDD_HHMMSS.png` |
| `1` / `2` / `3` | **强制** 把识别结果锁定为 rock / scissors / paper （用于联调） |
| `0` | 解除强制，回到自动识别 |

## 3. 识别原理

> 训练数据 / 模型都不用，纯几何规则。

### 21 个关键点

MediaPipe Hands 对每只手返回 21 个 3D 关键点：

```
       4 (thumb tip)
       |
    3──2
    |  |     8        12       16       20
    1  4     |         |        |        |
    |  |     6───5    10───9   14──13   18──17
    0──────────0─────────────────────────────────
   wrist      index    middle   ring     pinky
```

### 手指伸直判定

- **拇指**：用 `wrist→index_mcp` 作为「手掌基准方向」，拇指尖端向量与它的夹角余弦 < 0.4 视为伸直。
- **其余 4 指**：`tip.y < pip.y` （屏幕坐标里 y 越小越靠上）。

### RPS 映射

| 手势 | 食指 | 中指 | 无名指 | 小指 | → |
|------|------|------|--------|------|----|
| 石头 🪨   | 收   | 收   | 收     | 收   | `rock` |
| 剪刀 ✌️ | 伸   | 伸   | 收     | 收   | `scissors` |
| 布   📄 | 伸   | 伸   | 伸     | 伸   | `paper` |

### 平滑

8 帧滑动窗口取众数，避免单帧抖动导致 "石头→剪刀→石头" 闪屏。

## 4. 作为 Python 模块用

```python
from gesture_rps import RPSClassifier, classify_rps, fingers_state
import mediapipe as mp

clf = RPSClassifier(smooth_window=8)

hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1)
# ... 循环读帧 ...
result = hands.process(rgb_frame)
if result.multi_hand_landmarks:
    label = clf.classify(result.multi_hand_landmarks[0])
    print(label)                # 'rock' / 'paper' / 'scissors' / None
```

## 5. 跟舵机联动

跟这个仓库 `scservo_sdk` 一起用：

```python
import cv2, mediapipe as mp
from gesture_rps import RPSClassifier
from scservo_sdk import PortHandler, sms_sts
# ... 打开端口 ...
from xdat_tool import read_xdat

# 例: 石头 → 舵机 5 转到 0°; 剪刀 → 90°; 布 → 180°
angle_map = {"rock": 0, "scissors": 2048, "paper": 4095}

clf = RPSClassifier(smooth_window=8)
while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if res.multi_hand_landmarks:
        label = clf.classify(res.multi_hand_landmarks[0])
        if label in angle_map:
            sms.WritePosEx(1, angle_map[label], speed=1500, acc=50)
    cv2.imshow("RPS", frame)
    if cv2.waitKey(1) == 27: break
```

## 6. 故障排查

| 现象 | 解决 |
|------|------|
| `❌ 摄像头打不开` | Win: 设置 → 隐私 → 摄像头 → 允许桌面应用访问。Mac: 系统设置 → 隐私与安全 → 摄像头 |
| 一直识别为「未识别」 | 把手放远一些 (50–80 cm)，避免遮挡；光线不要太暗 |
| 全程识别为「剪刀」 | 可能是拇指被遮；把手完全露出，掌心朝镜头 |
| 卡顿 / FPS 低 | `--camera 0` + `--no-flip`；或 `--camera 1` 用外接 USB 摄像头 |
| 已识别但闪动 | 把 `--smooth-window` 调大 (默认 8 已很稳) |

## 7. 文件

| 文件 | 作用 |
|------|------|
| `gesture_rps.py` | 主程序：摄像头采集 + 关键点 + RPS 分类 |
| `requirements.txt` | 三依赖：opencv-python / mediapipe / numpy |
| `README.md` | 就是你正在读的这份 |
