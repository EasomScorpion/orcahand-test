# v20250805b — Dual-Strategy Splay + Asymmetric Lateral Mapping

## Key Changes from v20250805

- **Dual-strategy splay**: Strategy A (extended, PIP ≤ 60°) using wrist→MCP vs MCP→PIP; Strategy B (bent, PIP > 60°) using relative MCP→PIP between nearest bent neighbors
- **Asymmetric lateral mapping**: signed deviation from mid_angle, each side scaled independently to position range centered at 2048
- **Per-servo angle range overrides**: `SERVO_ANGLE_RANGE` for individual joint flexibility differences
- **Splay mode display**: console and test tool show A/B mode per lateral joint

## Calibration State

### Strategy A (extended)

| Servo | Name | mid_angle | neg_range | pos_range | min_pos | max_pos |
|-------|------|-----------|-----------|-----------|---------|---------|
| 3 | index bottom | 3.0° | 9.0° | 16.5° | 1800 | 2380 |
| 4 | middle bottom | 0.5° | 5.5° | 14.5° | 1870 | 2250 |
| 5 | pinky bottom | -3.0° | 5.0° | 24.0° | 1790 | 2290 |
| 12 | ring bottom | -0.5° | 2.5° | 10.5° | 1870 | 2150 |

### Strategy B (bent, parallel=0°)

| Servo | pos_range | max_pos |
|-------|-----------|---------|
| 3 | 10.0° | 2380 |
| 4 | 10.0° | 2250 |
| 5 | 7.0° | 2290 |
| 12 | 3.0° | 2150 |

### Invert Flags

| Servo | Invert |
|-------|--------|
| 3 (index bottom) | False |
| 8 (middle tip) | True |
| 12 (ring bottom) | True |
| 17 (wrist) | True |

### Per-Servo Angle Range

| Servo | Range |
|-------|-------|
| 2 (index mid) | (2.0, 85.0) |

## Files

| File | Description |
|------|-------------|
| `hand_tracker.py` | Core: landmarks → angles → servo positions |
| `test_tracker.py` | Visual test tool with dual-strategy calibration |
| `servo_console.py` | Main console with splay mode display |
