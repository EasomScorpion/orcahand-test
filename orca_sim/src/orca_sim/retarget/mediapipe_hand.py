"""mediapipe_hand — MediaPipe Hands 21 关键点 → 21 landmarks 3D 世界坐标。

MediaPipe Hands landmarks 索引：
    0  WRIST
    1-4   THUMB  (CMC, MCP, IP, TIP)
    5-8   INDEX  (MCP, PIP, DIP, TIP)
    9-12  MIDDLE (MCP, PIP, DIP, TIP)
    13-16 RING   (MCP, PIP, DIP, TIP)
    17-20 PINKY  (MCP, PIP, DIP, TIP)

MediaPipe 的 ``world_landmarks`` 输出以「手腕」附近的右手系三维坐标（米），
本模块把它整个 dump 出来；下游 :class:`CurlSolver` 自行消费。

API：
    tracker = MediaPipeHandTracker()
    pose = tracker.process(frame)
    pose.landmarks_world  # shape=(21, 3)，米，右手系
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# MediaPipe landmark 索引常量（保留供外部引用）
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20


@dataclass
class HandPose:
    """单帧手势识别结果。"""

    # 21 个关键点在 MediaPipe 右手世界系下的 3D 坐标（米）
    # shape=(21, 3)，索引对应 MediaPipe HAND_LANDMARKS 顺序：
    #   0=WRIST, 1-4=THUMB(CMC/MCP/IP/TIP), 5-8=INDEX(MCP/PIP/DIP/TIP),
    #   9-12=MIDDLE, 13-16=RING, 17-20=PINKY
    landmarks_world: np.ndarray

    # MediaPipe 给的整体检测置信度 [0, 1]
    confidence: float

    # 是否检测到有效手（hand_landmarks != None）
    detected: bool

    # 21 个关键点在**像素**坐标系下的 (x, y) 坐标
    # shape=(21, 2)，dtype=int。None 表示未检测到手
    image_landmarks: np.ndarray | None

    # 帧宽高（用于把归一化坐标转像素）
    image_width: int
    image_height: int


class MediaPipeHandTracker:
    """从摄像头帧（或任意 BGR 图像）输出 5 指尖相对手腕的 3D 目标。

    使用 MediaPipe Hands 的 ``world_landmarks`` + ``image_landmarks``：
        - ``world_landmarks``：以手腕为原点的右手系三维坐标（米）→ 用于 IK 目标
        - ``image_landmarks``：归一化 [0, 1] 的像素坐标 → 用于在画面上画骨架

    Parameters
    ----------
    max_num_hands : int
        最多检测几只手（默认 1）
    model_complexity : int
        0 = lite（更快，推荐实时），1 = full（默认，更准）
    min_detection_confidence : float
        媒体管线的检测阈值（默认 0.5）
    mirror_x : bool
        是否镜像 X 轴（把左手坐标系翻成右手坐标系，默认 True）
    process_width : int
        送进 MediaPipe 的帧宽（默认 320）。越小越快，建议 320 或 256。
        最终骨架坐标会用此宽高转回原图，所以画面上的标注位置会正确。
    """

    def __init__(
        self,
        *,
        max_num_hands: int = 1,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.5,
        mirror_x: bool = True,
        process_width: int = 320,
    ) -> None:
        try:
            import mediapipe as mp  # noqa: WPS433  (延迟导入：未装 mediapipe 时友好报错)
        except ImportError as exc:
            raise ImportError(
                "MediaPipe 未安装。请先运行：\n"
                "    uv pip install -e \".[teleop]\"\n"
                "或：\n"
                "    uv pip install mediapipe opencv-python"
            ) from exc

        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
        )
        self.mirror_x = mirror_x
        self._process_width = int(process_width)
        self._last: Optional[HandPose] = None

    def process(self, frame_bgr: np.ndarray) -> HandPose:
        """从 BGR 帧中检测手，返回 ``HandPose``。

        内部会把帧缩到 ``process_width`` 宽再送 MediaPipe（提速），但返回的
        ``image_landmarks`` 坐标已经换算回**原图**像素。

        Parameters
        ----------
        frame_bgr : np.ndarray
            OpenCV 格式 BGR 图像，shape=(H, W, 3)，dtype=uint8

        Returns
        -------
        HandPose
            ``detected=False`` 表示没检测到手（此时 landmarks_world 是 0 矩阵）
        """
        h, w = frame_bgr.shape[:2]
        # 缩到 _process_width 提速
        if w != self._process_width:
            scale = self._process_width / w
            new_w = self._process_width
            new_h = int(h * scale)
            small = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            new_h, new_w = h, w
            scale = 1.0
            small = frame_bgr
        # BGR → RGB
        small_rgb = small[:, :, ::-1].copy()
        result = self._hands.process(small_rgb)

        if result is None or not result.multi_hand_landmarks or not result.multi_hand_world_landmarks:
            empty = np.zeros((21, 3), dtype=np.float64)
            pose = HandPose(
                landmarks_world=empty,
                confidence=0.0,
                detected=False,
                image_landmarks=None,
                image_width=w,
                image_height=h,
            )
            self._last = pose
            return pose

        wl = result.multi_hand_world_landmarks[0]
        il = result.multi_hand_landmarks[0]

        # 21 landmarks 的世界坐标（米）
        # MediaPipe 默认输出左手系 → 翻 X、Z 变右手系，与 sim 一致
        lms_world = np.zeros((21, 3), dtype=np.float64)
        for i in range(21):
            lm = wl.landmark[i]
            lms_world[i, 0] = lm.x
            lms_world[i, 1] = lm.y
            lms_world[i, 2] = lm.z
        if self.mirror_x:
            lms_world[:, 0] *= -1.0
            lms_world[:, 2] *= -1.0

        # 21 个关键点像素坐标 → 换算回原图
        img_lms = np.zeros((21, 2), dtype=np.int32)
        for i in range(21):
            lm = il.landmark[i]
            img_lms[i, 0] = int(lm.x * new_w / scale)
            img_lms[i, 1] = int(lm.y * new_h / scale)

        conf = 1.0
        if result.multi_handedness:
            conf = float(result.multi_handedness[0].classification[0].score)

        pose = HandPose(
            landmarks_world=lms_world,
            confidence=conf,
            detected=True,
            image_landmarks=img_lms,
            image_width=w,
            image_height=h,
        )
        self._last = pose
        return pose

    def close(self) -> None:
        """释放 MediaPipe 资源。"""
        self._hands.close()