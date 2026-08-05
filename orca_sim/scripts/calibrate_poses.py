"""calibrate_poses.py — 手势校准工具。

用途：
    摄像头采集一组常见手势，每个手势：
      - 输入：用户做出手势的名字（key/数字选）+ 实时 MediaPipe 21 landmarks
      - 记录：landmarks_world (21, 3) + sim 17 关节角 + 时间戳
      - 存到：calibration_data.json

操作流程：
    1. 启动程序，看到 webcam 预览
    2. 选择手势（按数字键 0-9，或自定义输入名字）
    3. 做手势 2 秒（程序会采集 30 帧平均）
    4. 程序打印：landmarks + sim 关节角 → 写入 JSON
    5. 下一手势，或按 q 退出

按数字键 0-9 直接选预设手势：
    0: open_hand      张直手
    1: fist           握拳
    2: peace          食指+中指张，其他握
    3: thumbs_up      大拇指竖起
    4: index_point    食指指，其他握
    5: pinky_point    小拇指指，其他握
    6: spread         五指全部张开
    7: relaxed        自然放松
    8-9: custom_<n>   自定义

按 'c' 进入自定义名字模式（输入名字 + Enter）。
按 's' 显示当前已采集的手势列表。
按 'q' 退出并保存。
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import cv2

from orca_sim.envs import OrcaHandRight
from orca_sim.retarget import MediaPipeHandTracker, CurlSolver


PRESET_POSES = [
    ("0", "open_hand",   "张直手"),
    ("1", "fist",        "握拳"),
    ("2", "peace",       "食指+中指张，其他握（V）"),
    ("3", "thumbs_up",   "大拇指竖起"),
    ("4", "index_point", "食指指，其他握"),
    ("5", "pinky_point", "小拇指指，其他握"),
    ("6", "spread",      "五指全部最大张开"),
    ("7", "relaxed",     "自然放松"),
    ("8", "custom_8",    "自定义 #8"),
    ("9", "custom_9",    "自定义 #9"),
]


def collect_pose_samples(tracker, cap, solver, n_frames=30):
    """采集 n_frames 帧，返回 (mean_landmarks, mean_qpos, per_frame)。"""
    samples_land = []
    samples_qpos = []
    per_frame = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            continue
        pose = tracker.process(frame)
        if pose.detected and pose.landmarks_world is not None:
            q = solver.solve(pose.landmarks_world)
            samples_land.append(pose.landmarks_world.copy())
            samples_qpos.append(q.copy())
            per_frame.append({
                "landmarks": pose.landmarks_world.tolist(),
                "qpos": q.tolist(),
            })
    if not samples_land:
        return None, None, []
    mean_land = np.mean(samples_land, axis=0)
    mean_qpos = np.mean(samples_qpos, axis=0)
    return mean_land, mean_qpos, per_frame


def save_data(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[saved] {path}  ({len(data['poses'])} poses)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="calibration_data.json", help="输出 JSON 路径")
    parser.add_argument("--no-render", action="store_true", help="不开 MuJoCo viewer")
    parser.add_argument("--frames", type=int, default=30, help="每个手势采集帧数")
    args = parser.parse_args()

    # 启动 sim（rgb_array 模式即可）
    env = OrcaHandRight(
        version="v1", skin=False,
        render_mode="rgb_array" if args.no_render else "human",
    )
    env.reset(seed=42)
    m = env.unwrapped.model
    solver = CurlSolver(m, env.unwrapped.data)

    # MediaPipe tracker
    tracker = MediaPipeHandTracker(model_complexity=0, process_width=320)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        sys.exit(1)

    # 已采集数据
    data = {
        "version": "1.0",
        "ctrlrange": m.actuator_ctrlrange.tolist(),
        "actuator_names": [
            "wrist","th_mcp","th_abd","th_pip","th_dip",
            "i_abd","i_mcp","i_pip",
            "m_abd","m_mcp","m_pip",
            "r_abd","r_mcp","r_pip",
            "p_abd","p_mcp","p_pip",
        ],
        "poses": {},  # name -> {landmarks, qpos, samples_count, timestamp}
    }

    print("=" * 60)
    print("手势校准工具")
    print("=" * 60)
    print("按数字 0-9 选预设手势，做手势 2 秒（自动采集 30 帧）")
    print("按 'c' 自定义手势名")
    print("按 's' 显示已采集列表")
    print("按 'd' 删除最后一个")
    print("按 'q' 退出保存")
    print("=" * 60)
    print()
    for k, name, desc in PRESET_POSES:
        print(f"  [{k}] {name:14s} - {desc}")
    print()

    cv2.namedWindow("calibration (webcam)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("calibration (webcam)", 640, 480)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.imshow("calibration (webcam)", frame)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                print()
                print(f"已采集 {len(data['poses'])} 个手势：")
                for name, info in data["poses"].items():
                    print(f"  - {name:14s} ({info['samples_count']} 帧, t={info['timestamp']})")
                print()
            elif key == ord("d"):
                if data["poses"]:
                    last = list(data["poses"].keys())[-1]
                    del data["poses"][last]
                    print(f"[deleted] {last}")
            elif key == ord("c"):
                print("输入自定义手势名（英文/数字）：")
                name = input("> ").strip()
                if not name:
                    continue
                print(f"做 '{name}' 手势，2 秒后开始采集...")
                time.sleep(0.5)
                mean_land, mean_qpos, per_frame = collect_pose_samples(
                    tracker, cap, solver, n_frames=args.frames)
                if mean_land is not None:
                    data["poses"][name] = {
                        "landmarks": mean_land.tolist(),
                        "qpos": mean_qpos.tolist(),
                        "samples_count": len(per_frame),
                        "timestamp": time.time(),
                    }
                    print(f"  [ok] 采集到 {name}: 17 关节 = {mean_qpos.round(3).tolist()}")
                else:
                    print(f"  [fail] 没检测到手")
            elif chr(key) in [p[0] for p in PRESET_POSES]:
                k = chr(key)
                pose_info = next(p for p in PRESET_POSES if p[0] == k)
                name = pose_info[1]
                print(f"做 '{pose_info[2]}' 手势，2 秒后开始采集...")
                time.sleep(0.5)
                mean_land, mean_qpos, per_frame = collect_pose_samples(
                    tracker, cap, solver, n_frames=args.frames)
                if mean_land is not None:
                    data["poses"][name] = {
                        "landmarks": mean_land.tolist(),
                        "qpos": mean_qpos.tolist(),
                        "samples_count": len(per_frame),
                        "timestamp": time.time(),
                    }
                    print(f"  [ok] {name}: 17 关节 = {mean_qpos.round(3).tolist()}")
                else:
                    print(f"  [fail] 没检测到手")
    finally:
        save_data(data, args.out)
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()