# v20250806

Major thumb refactor. Simplified servo 13 from tilted-axis geometry to direct MCP flexion, added dual-strategy splay for thumb lateral (servo 14), and adjusted thumb mid/tip to use MCP-IP-TIP flexion.

## Changes from v20250805b

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
- Clamping now uses `cfg.get("max_offset", 10)` instead of hardcoded 10

## Current calibration state

| Servo | Joint | Range | Mapping |
|-------|-------|-------|---------|
| 1 | Index tip | — | distal flexion |
| 2 | Index mid | 2°–85° | proximal flexion |
| 3 | Index bottom | A: mid=3°, neg=9°, pos=16.5° | lateral, 1800–2380 |
| 4 | Middle bottom | A: mid=0.5°, neg=5.5°, pos=14.5° | lateral, 1870–2250 |
| 5 | Pinky bottom | A: mid=-3°, neg=5°, pos=24° | lateral, 1790–2290 |
| 6 | Pinky tip | — | distal flexion |
| 7 | Pinky mid | — | proximal flexion |
| 8 | Middle tip | — | distal flexion, invert |
| 9 | Middle mid | — | proximal flexion |
| 10 | Ring mid | — | proximal flexion |
| 11 | Ring tip | — | distal flexion |
| 12 | Ring bottom | A: mid=-0.5°, neg=2.5°, pos=10.5° | lateral, 1870–2150, invert |
| 13 | Thumb big | 0°–45° | MCP flexion, invert: False |
| 14 | Thumb lateral | A: mid=10°, neg=20°, pos=30° / B: mid=-166°, neg=6°, pos=6° | lateral, 1641–2156, invert: True |
| 15 | Thumb tip | 3°–65° | coupled (mid × 0.8) |
| 16 | Thumb mid | 3°–75° | IP flexion × 0.9 |
| 17 | Wrist | -16° – -1° | palm orientation, invert: True |
