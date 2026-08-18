# v20260818

Major thumb refactor. Simplified servo 13 from tilted-axis geometry to direct MCP flexion, added dual-strategy splay for thumb lateral (servo 14), and adjusted thumb mid/tip to use MCP-IP-TIP flexion.

## What this folder is

A versioned snapshot of the ORCA-hand desktop console + vision code. It is **not
standalone** — it must live inside the main `FTServo_Python` repository (the
folder that already contains `scservo_sdk/`, `xdat_tool.py`, `参数/`,
`gesture_recognition/`). Keep it at `<main_folder>/versions/v20260818/`: the
scripts locate the main folder as *two levels up* from their own location.

## Files

| File | What it does |
|------|--------------|
| `servo_console.py` | Main PyQt5 desktop console (entry point). Serial control, parameter browser, pose library, Rock-Paper-Scissors game, and real-time hand mirroring. |
| `hand_tracker.py` | Core mapping: MediaPipe 21 landmarks → 17 joint angles → 17 servo positions (`HandTracker` class + `SERVO_CONFIG` limits). Pure math + numpy, no camera/serial. |
| `test_tracker.py` | Vision-only test tool: shows the hand skeleton plus the 17 computed angles/target positions on screen. No serial/servo required. |
| `servo_joints.json` | Servo ID → joint name (Chinese) table, used for UI labels. |
| `saved_poses.json` | Saved pose library (e.g. `fist`), recorded via the 动作库 tab. |

## Script features

`servo_console.py` has 5 tabs:

1. **🎮 控制台 (Control)** — open/close the serial port, drive a single servo
   with a slider, reset all servos to 2048, and emergency-stop.
2. **📋 参数浏览器 (Parameter browser)** — reads/writes each servo's EPROM +
   SRAM registers (like the FT debug tool); can "save to EPROM" and export the
   current servo to a `.xdat` file.
3. **🎭 动作库 (Pose library)** — record, play, import and export full-hand
   poses (all 17 servo positions).
4. **✊ 石头剪刀布 (Rock-Paper-Scissors)** — visual recognition.
5. **🖐 手势追踪 (Hand tracking)** — real-time hand mirroring.

### Visual recognition (✊ Rock-Paper-Scissors)

- Camera + MediaPipe **HandLandmarker** (`gesture_recognition/hand_landmarker.task`)
  detect a single hand (21 landmarks).
- `RPSClassifier` (sliding-window majority vote, `smooth_window=8`) classifies
  the gesture as **rock / scissors / paper**.
- A state machine confirms a gesture only after it is held stable for 1 s, then
  triggers the robot hand's pre-recorded *winning* pose, holds it, and returns
  to idle. Two seconds with no hand auto-resets the hand.

### Hand mirroring (🖐 Hand tracking)

The core "my hand controls the ORCA hand" feature:

1. `gesture_recognition.gesture_rps.detect_hand()` returns the 21 MediaPipe
   landmarks each frame.
2. `hand_tracker.HandTracker.process(landmarks)` maps them to 17 joint angles
   using a 3D orthonormal palm frame, weighted MCP+PIP+DIP flexion, signed
   splay angles about the palm normal, thumb via the metacarpal axis, and wrist
   from palm orientation. EMA smoothing (`alpha=0.35`) removes jitter.
3. The 17 target positions are clamped to a ±50-unit margin inside each servo's
   `.xdat` limits and written through `ServoSafetyLayer.sync_go_to_pose`
   (synchronized writes, ~15 Hz, only when a position actually moved).

The 17 servo ↔ joint mapping is in `servo_joints.json` and in the
"Current calibration state" table below.

## What you need to download

Download the main `FTServo_Python` repo (which already contains the environment
files), then place this folder at `<main_folder>/versions/v20260818/` so the
relative paths resolve:

```
FTServo_Python_visual_recognization/       ← main folder (from our repo)
├── scservo_sdk/          Feetech STS3215 SDK (serial protocol)
├── xdat_tool.py          .xdat parameter read/write helper
├── 参数/                 17 × {1..17}.xdat servo parameter files (angle limits, torque, …)
├── gesture_recognition/  RPS + hand-landmarker module
│   ├── gesture_rps.py
│   └── hand_landmarker.task   (~7.5 MB MediaPipe model)
└── versions/
    └── v20260818/        ← THIS folder
```

Python dependencies (install once):

```bash
pip install -r gesture_recognition/requirements.txt   # opencv-python, mediapipe 0.10.x, numpy
pip install PyQt5 pyserial Pillow
```

- `gesture_recognition/requirements.txt` → `opencv-python`, `mediapipe (>=0.10,<1.0)`, `numpy`
- `PyQt5` + `pyserial` for the console UI and serial port
- `Pillow` for the on-screen Chinese-font overlay in `gesture_rps.py`

### Run

```bash
# from the main folder:
python versions/v20260818/servo_console.py    # full console (needs serial for the servos)
python versions/v20260818/test_tracker.py     # vision only — no serial/servo needed
```

Connect the 17 STS3215 servos over the serial port, then use **🖐 手势追踪** for
mirroring or **✊ 石头剪刀布** for the RPS game. The console loads
`参数/{1..17}.xdat` on startup and exits with an error if those files are
missing.

## ⚠ Maintenance note — keep servo position limits in sync

`SERVO_CONFIG` in `hand_tracker.py` (this folder) hardcodes each servo's
`min`/`max` position. These **must match** the `最小角度限制` / `最大角度限制`
values in `参数/{servo_id}.xdat`.

If they drift (e.g. after editing angle limits via the console "应用限制" button
or directly in the `.xdat` files), `ServoSafetyLayer.sync_go_to_pose` validates
every target against the `.xdat` limits and raises `SafetyViolation` — which the
hand-tracking loop swallows silently (`except Exception: pass`). The result is
the hand freezing at extreme poses (wide-open / tight fist): a single
out-of-range servo aborts the whole 17-servo write.

Whenever you change a servo's angle limits, re-sync `SERVO_CONFIG` min/max (and
the lateral `LATERAL_MAP` min_pos/max_pos) to match.

## Changes from v20260806

### Thumb Root (servo 13)
- **Replaced** complex 30°-tilted-axis `abs(signed_angle(...))` with simple MCP flexion: `flexion(∠ CMC-MCP-IP)`
- Range: 0°–45°, mid point 10°
- Max position reduced from 2522 to 2230

### Thumb Lateral (servo 14)
- **Added dual-strategy splay** matching the four-finger approach:
  - **A** (mid ≤ 35°): `signed_angle(proj(wrist→CMC), proj(MCP→IP), palm_z)` — spread in palm plane, mid=10°, neg=20°, pos=30°
  - **B** (mid > 35°): `signed_angle(wrist→CMC, MCP→CMC, palm_z)` — raw 3D vectors, mid=-166°, neg=6°, pos=6°
- Added to LATERAL_MAP with both A/B entries
- `invert: True`

### Thumb Mid/Tip (servos 16, 15)
- Angle source changed from `∠ CMC-MCP-IP` to `∠ MCP-IP-TIP` (IP flexion at landmark 3)
- Mid: `raw * 0.9`, range 3°–75°
- Tip: `raw * 0.8`, range 3°–65°

### Angle-to-position mapping
- B-mode now checks for `mid_angle` in params — if present (thumb), uses signed deviation like A; if absent (fingers), uses original abs/pos_range with 0° neutral
- Clamping now uses `cfg.get("max_offset", 50)` instead of hardcoded 10

## Current calibration state

| Servo | Joint | Range | Mapping |
|-------|-------|-------|---------|
| 1 | Index tip | — | distal flexion |
| 2 | Index mid | 2°–85° | proximal flexion |
| 3 | Index bottom | A: mid=3°, neg=9°, pos=16.5° | lateral, 1800–2380 |
| 4 | Middle bottom | A: mid=0.5°, neg=5.5°, pos=14.5° | lateral, 1870–2250 |
| 5 | Pinky bottom | A: mid=-3°, neg=5°, pos=24° | lateral, 1790–2270 |
| 6 | Pinky tip | — | distal flexion |
| 7 | Pinky mid | — | proximal flexion |
| 8 | Middle tip | — | distal flexion, invert |
| 9 | Middle mid | — | proximal flexion |
| 10 | Ring mid | — | proximal flexion |
| 11 | Ring tip | — | distal flexion |
| 12 | Ring bottom | A: mid=-0.5°, neg=2.5°, pos=10.5° | lateral, 1870–2150, invert |
| 13 | Thumb big | 0°–45° | MCP flexion, invert: False |
| 14 | Thumb lateral | A: mid=10°, neg=20°, pos=30° / B: mid=-166°, neg=6°, pos=6° | lateral, 1615–2120, invert: True |
| 15 | Thumb tip | 3°–65° | coupled (mid × 0.8) |
| 16 | Thumb mid | 3°–75° | IP flexion × 0.9 |
| 17 | Wrist | -16° – -1° | palm orientation, invert: True |
