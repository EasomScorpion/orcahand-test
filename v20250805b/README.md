# v20250805b — Dual-Strategy Splay + Asymmetric Lateral Mapping

Real-time camera → MediaPipe hand tracking → ORCA robotic hand mirroring (17 × STS3215 servos).

---

## Quick Start

### 1. Install Dependencies

```
pip install mediapipe opencv-python numpy PyQt5 pyserial
```

### 2. Verify Servo Parameters

Ensure `参数/1.xdat` through `参数/17.xdat` exist with correct min/max angle limits.

### 3. Run

```bash
# Main console (serial + mirroring)
python test/servo_console.py

# Visual test tool (no serial needed, calibration only)
python realtime_mirroring/test_tracker.py
```

### 4. Calibrate for a New Hand

1. Run `test_tracker.py`
2. Record 5 sessions covering: extended fingers together↔spread, fist together↔spread
3. Click "计算校准" — copy the printed `SPLAY_OFFSET_A`, `SPLAY_RANGE_A`, `SPLAY_OFFSET_B`, `SPLAY_RANGE_B` into `hand_tracker.py`
4. Run `servo_console.py` → "手势追踪" tab → "校准中性位" with fingers together

---

## Architecture

```
Camera → MediaPipe HandLandmarker (21 landmarks)
              ↓
       hand_tracker.py
       landmarks_to_angles()
              ↓
       17 joint angles (°)
              ↓
       angles_to_positions()
              ↓
       17 servo positions (0–4095)
              ↓
       ServoSafetyLayer .sync_go_to_pose()
              ↓
       STS3215 servos ×17
```

### Palm Frame (3D orthonormal)

```
palm_x = unit(index_mcp − pinky_mcp)     # left → right
palm_y = unit(wrist → middle_mcp)        # re-orthogonalized
palm_z = cross(palm_x, palm_y)           # palm normal (outward)
```

All vectors are projected into this frame before angle computation.

### Flexion (mid + tip servos)

Weighted blend of MCP, PIP, DIP interior angles converted to flexion:

| Servo type | MCP | PIP | DIP |
|------------|-----|-----|-----|
| mid (proximal) | 0.45 | 0.45 | 0.10 |
| tip (distal) | 0.10 | 0.45 | 0.45 |

### Splay — Dual Strategy

**Strategy A** (extended, PIP flexion ≤ 60°):
```
angle = signed_angle(wrist→MCP, MCP→PIP) about palm_z
```
Per-finger absolute lateral deviation.

**Strategy B** (bent, PIP flexion > 60°):
```
angle = signed_angle(ref_MCP→PIP, finger_MCP→PIP) about palm_z
```
Relative MCP→PIP angle between nearest bent-finger neighbors. Reference found by scanning `FINGER_ORDER = [3, 4, 12, 5]` outward from the current finger; left neighbor preferred at equal distance.

### Lateral Position Mapping

Asymmetric: signed deviation from `mid_angle`, each side scaled independently to a position range centered at 2048.

```
deviation = raw_angle − mid_angle

if deviation ≥ 0:   pos = 2048 + ratio × (max_pos − 2048)
if deviation < 0:   pos = 2048 − ratio × (2048 − min_pos)
```

Strategy B: `mid_angle` is always 0° (MCP→PIP vectors parallel at neutral).

The `invert` flag in `SERVO_CONFIG` flips the direction.

### EMA Smoothing

α = 0.35, applied per-servo on the ratio (0–1) before position conversion.

---

## Calibration State

### Strategy A (Extended)

| Servo | Finger | mid_angle | neg_range | pos_range | min_pos | max_pos |
|-------|--------|-----------|-----------|-----------|---------|---------|
| 3 | index | 3.0° | 9.0° | 16.5° | 1800 | 2380 |
| 4 | middle | 0.5° | 5.5° | 14.5° | 1870 | 2250 |
| 5 | pinky | −3.0° | 5.0° | 24.0° | 1790 | 2290 |
| 12 | ring | −0.5° | 2.5° | 10.5° | 1870 | 2150 |

### Strategy B (Bent — mid_angle = 0°)

| Servo | pos_range | max_pos |
|-------|-----------|---------|
| 3 | 10.0° | 2380 |
| 4 | 10.0° | 2250 |
| 5 | 7.0° | 2290 |
| 12 | 3.0° | 2150 |

### Invert Flags

| Servo | Joint | Invert |
|-------|-------|--------|
| 3 | index bottom | False |
| 8 | middle tip | True |
| 12 | ring bottom | True |
| 17 | wrist | True |

### Per-Servo Angle Overrides

| Servo | Joint | Range | Reason |
|-------|-------|-------|--------|
| 2 | index mid | (2.0, 85.0) | Camera max ≈ 85° |

---

## File Reference

| File | Description |
|------|-------------|
| `realtime_mirroring/hand_tracker.py` | Core: SERVO_CONFIG, ANGLE_RANGES, LATERAL_MAP, `landmarks_to_angles()`, `angles_to_positions()`, smoothing, calibration |
| `realtime_mirroring/test_tracker.py` | Visual test tool. Per-sample recording with splay mode, dual-strategy calibration computation |
| `test/servo_console.py` | Main Qt5 console: serial control, param browser, pose library, RPS game, hand tracking mirroring tab |
| `gesture_recognition/gesture_rps.py` | MediaPipe detector wrapper (`create_detector`, `detect_hand`, `draw_landmarks_on_image`) |
| `参数/*.xdat` | Per-servo mechanical limits and EPROM defaults |
| `scservo_sdk/` | Feetech STS3215 protocol SDK |

---

## Safety

| Parameter | Value |
|-----------|-------|
| MAX_TORQUE | 50 (5%) |
| PROTECT_TORQUE | 20 |
| OVERLOAD_TORQUE | 20 |
| PROTECT_TIME | 20 |

All position writes go through `ServoSafetyLayer` which enforces:
1. Emergency stop flag (cuts torque, freezes all writes)
2. Angle limits from xdat files
3. Speed/acceleration caps

Emergency stop button is always accessible in the console.

---

## Key Changes from v20250805

- Dual-strategy splay (A: extended, B: bent with nearest-neighbor pairing)
- Asymmetric lateral mapping centered at 2048
- Per-servo angle range overrides (`SERVO_ANGLE_RANGE`)
- Splay mode displayed in console and test tool (A/B column)
- Per-sample calibration logging with strategy tracking
- PIP flexion (not DIP) used for strategy switching threshold
