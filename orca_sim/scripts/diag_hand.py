"""diag_hand.py —— 诊断 MediaPipe Hands 在你摄像头下的检测情况。

跑 30 帧，打印每帧的 detected / confidence / 5 指尖 (米)。
如果 detected=False 占比 > 50%，说明摄像头 / 光线 / 手势有问题。
如果 detected=True 但指尖数据抖动 > 1cm，说明要加平滑。
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orca_sim.retarget import MediaPipeHandTracker


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("FAIL: 摄像头打不开")
        return

    # 用原图分辨率，不缩（最准确的诊断模式）
    tracker = MediaPipeHandTracker(model_complexity=1, process_width=640)

    print("=" * 60)
    print("诊断：把右手伸到镜头前，张开手掌")
    print("30 帧后自动结束")
    print("=" * 60)

    N = 30
    detected = 0
    all_tips = []  # 用于算抖动

    for i in range(N):
        ok, frame = cap.read()
        if not ok:
            print("FAIL: 读不到帧")
            break
        frame = cv2.flip(frame, 1)

        pose = tracker.process(frame)
        if pose.detected:
            detected += 1
            all_tips.append(pose.fingertip_targets_wrist_frame.copy())
            t = pose.fingertip_targets_wrist_frame
            print(
                f"  frame {i:2d}  conf={pose.confidence:.2f}  "
                f"thumb=({t[0,0]:+.3f},{t[0,1]:+.3f},{t[0,2]:+.3f})  "
                f"index=({t[1,0]:+.3f},{t[1,1]:+.3f},{t[1,2]:+.3f})",
                flush=True,
            )
        else:
            print(f"  frame {i:2d}  --- NO HAND DETECTED ---", flush=True)
        time.sleep(0.1)

    print()
    print("=" * 60)
    print(f"检测到手的帧数: {detected}/{N} ({detected*100/N:.0f}%)")
    if detected >= 2:
        tips_arr = np.stack(all_tips, axis=0)  # (T, 5, 3)
        # 帧间抖动：相邻帧指尖位置差
        jitter = np.linalg.norm(np.diff(tips_arr, axis=0), axis=2)  # (T-1, 5)
        print(f"帧间平均抖动: {jitter.mean()*100:.2f} cm")
        print(f"帧间最大抖动: {jitter.max()*100:.2f} cm")
        print("(如果抖动 > 1cm，需要加指数平滑)")
    print("=" * 60)

    cap.release()
    tracker.close()


if __name__ == "__main__":
    main()