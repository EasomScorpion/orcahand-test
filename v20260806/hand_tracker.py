#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continuous hand-joint → servo-position mapping for ORCA hand mirroring.

Takes MediaPipe HandLandmarker 21-landmark output and produces 17 servo
target positions, with EMA smoothing and visibility gating.

Adapted from coworker's camera3d_hand_to_api.py approach:
  - 3D orthonormal palm frame
  - Weighted MCP+PIP+DIP → proximal/distal flexion blending
  - Signed splay angles about palm normal
  - Thumb root via tilted metacarpal axis
  - Wrist from palm orientation vs camera

Usage:
    from realtime_mirroring.hand_tracker import HandTracker
    tracker = HandTracker()
    positions, angles = tracker.process(landmarks)
"""

import math
from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# Servo configuration — authoritative limits from "servo to joint table"
# (servo_id, min_pos, max_pos, joint_type, is_inverted, joint_name_zh)
# ---------------------------------------------------------------------------
SERVO_CONFIG: dict[int, dict] = {
    1:  {"min": 1370, "max": 2468, "type": "tip",    "finger": "index",  "invert": False, "name": "食指指尖"},
    2:  {"min": 1560, "max": 2615, "type": "mid",    "finger": "index",  "invert": False, "name": "食指中节"},
    3:  {"min": 1800, "max": 2452, "type": "bottom", "finger": "index",  "invert": False, "name": "食指根部"},
    4:  {"min": 1642, "max": 2300, "type": "bottom", "finger": "middle", "invert": False, "name": "中指根部"},
    5:  {"min": 1723, "max": 2453, "type": "bottom", "finger": "pinky",  "invert": False, "name": "小指根部"},
    6:  {"min": 1600, "max": 2655, "type": "tip",    "finger": "pinky",  "invert": False, "name": "小指指尖"},
    7:  {"min": 1500, "max": 2700, "type": "mid",    "finger": "pinky",  "invert": False, "name": "小指中节"},
    8:  {"min": 1780, "max": 2827, "type": "tip",    "finger": "middle", "invert": True,  "name": "中指指尖"},
    9:  {"min": 1470, "max": 2536, "type": "mid",    "finger": "middle", "invert": False, "name": "中指中节"},
    10: {"min": 1400, "max": 2370, "type": "mid",    "finger": "ring",   "invert": False, "name": "无名指中节"},
    11: {"min": 1423, "max": 2600, "type": "tip",    "finger": "ring",   "invert": False, "name": "无名指指尖"},
    12: {"min": 1668, "max": 2333, "type": "bottom", "finger": "ring",   "invert": True,  "name": "无名指根部"},
    13: {"min": 1700, "max": 2230, "type": "thumb_big",     "finger": "thumb",  "invert": False, "name": "拇指大根部"},
    14: {"min": 1641, "max": 2156, "type": "thumb_lateral", "finger": "thumb",  "invert": True, "name": "拇指小根部"},
    15: {"min": 1506, "max": 2559, "type": "thumb_tip",     "finger": "thumb",  "invert": False, "name": "拇指指尖"},
    16: {"min": 1329, "max": 2400, "type": "thumb_mid",     "finger": "thumb",  "invert": False, "name": "拇指中节"},
    17: {"min": 1300, "max": 2858, "type": "wrist",  "finger": "wrist",  "invert": True,  "name": "手腕"},
}

# MediaPipe landmark indices
LM = {
    "wrist": 0,
    "thumb_cmc": 1, "thumb_mcp": 2, "thumb_ip": 3, "thumb_tip": 4,
    "index_mcp": 5, "index_pip": 6, "index_dip": 7, "index_tip": 8,
    "middle_mcp": 9, "middle_pip": 10, "middle_dip": 11, "middle_tip": 12,
    "ring_mcp": 13, "ring_pip": 14, "ring_dip": 15, "ring_tip": 16,
    "pinky_mcp": 17, "pinky_pip": 18, "pinky_dip": 19, "pinky_tip": 20,
}

# Expected joint angle ranges in degrees — trimmed mean from 5 logs
ANGLE_RANGES = {
    "mid":            (3.0, 90.0),    # weighted prox flexion (MCP+PIP+DIP blend)
    "tip":            (3.0, 109.0),   # weighted dist flexion (MCP+PIP+DIP blend)
    "bottom":         (0.0, 25.0),    # splay from neutral — fallback, see PER_SERVO
    "thumb_big":      (0.0, 45.0),   # thumb MCP flexion (∠ CMC-MCP-IP)
    "thumb_lateral":  (25.0, 62.0),   # thumb splay in palm plane (abs)
    "thumb_mid":      (3.0, 75.0),    # thumb MCP-IP-TIP flexion (×0.9)
    "thumb_tip":      (3.0, 65.0),    # thumb tip (×0.8)
    "wrist":          (-16.0, -1.0),  # palm orientation vs camera
}

# Per-servo overrides for ANGLE_RANGES (e.g. limited finger flexibility)
SERVO_ANGLE_RANGE: dict[int, tuple[float, float]] = {
    2: (2.0, 85.0),  # index mid: camera max ~85°
}

# ── Dual-strategy splay calibration ────────────────────────────
# Strategy A (extended, tip ≤ 60°): angle(wrist→MCP, MCP→PIP) per finger
# Strategy B (bent, tip > 60°):     angle(ref_MCP→PIP, finger_MCP→PIP)
# Thumb (servo 14) always uses Strategy A.

SPLAY_OFFSET_A: dict[int, float] = {
    3: 3.0, 4: 0.5, 5: 3.0, 12: 0.5, 14: 0.0,
}
SPLAY_RANGE_A: dict[int, tuple[float, float]] = {
    3: (0.0, 16.5), 4: (0.0, 14.5), 5: (0.0, 18.0), 12: (0.0, 9.5),
}

SPLAY_OFFSET_B: dict[int, float] = {
    3: 0.0, 4: 0.0, 5: 0.0, 12: 0.0,
}
SPLAY_RANGE_B: dict[int, tuple[float, float]] = {
    3: (0.0, 15.0), 4: (0.0, 15.0), 5: (0.0, 15.0), 12: (0.0, 15.0),
}

# Finger adjacency order (servo IDs, left → right)
FINGER_ORDER: list[int] = [3, 4, 12, 5]

# Per-servo ORCA position sub-range for lateral joints.
# Overrides the servo's full mechanical [min, max] when the user can't comfortably
# reach the extremes. Neutral maps to neutral_pos, max splay maps to spread_pos.
POSITION_RANGE: dict[int, tuple[int, int]] = {
    4:  (1930, 2190),  # middle bottom: comfortable sub-range within [1642, 2300]
    12: (1870, 2150),  # ring bottom:   comfortable sub-range within [1668, 2333]
}

# ── Asymmetric lateral mapping ──────────────────────────────────
# Strategy A (extended): signed deviation from mid_angle, each side
# scaled independently to a position range centred at MID_POS.
#
#   raw < mid_angle  →  pos = MID_POS - ratio_neg * (MID_POS - min_pos)
#   raw > mid_angle  →  pos = MID_POS + ratio_pos * (max_pos - MID_POS)
#
# Strategy B (bent): angle between MCP→PIP vectors. Parallel = 0°
# is the neutral, maps to MID_POS. No neg_range — spread only goes one way.
#
#   raw >= 0         →  pos = MID_POS + ratio * (max_pos - MID_POS)
#
MID_POS = 2048

LATERAL_MAP: dict[int, dict] = {
    3:  {"A": {"mid_angle": 3.0, "neg_range": 9.0, "pos_range": 16.5,
               "min_pos": 1800, "max_pos": 2380},
         "B": {"pos_range": 10.0, "min_pos": 1800, "max_pos": 2380}},
    4:  {"A": {"mid_angle": 0.5, "neg_range": 5.5, "pos_range": 14.5,
               "min_pos": 1870, "max_pos": 2250},
         "B": {"pos_range": 10.0, "min_pos": 1930, "max_pos": 2190}},
    5:  {"A": {"mid_angle": -3.0, "neg_range": 5.0, "pos_range": 24.0,
               "min_pos": 1790, "max_pos": 2290},
         "B": {"pos_range": 7.0, "min_pos": 1790, "max_pos": 2290}},
    12: {"A": {"mid_angle": -0.5, "neg_range": 2.5, "pos_range": 10.5,
               "min_pos": 1870, "max_pos": 2150},
         "B": {"pos_range": 3.0, "min_pos": 1870, "max_pos": 2150}},
    14: {"A": {"mid_angle": 10.0, "neg_range": 20.0, "pos_range": 30.0,
               "min_pos": 1641, "max_pos": 2156},
         "B": {"mid_angle": -166.0, "neg_range": 6.0, "pos_range": 6.0,
               "min_pos": 1641, "max_pos": 2156}},
}


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------
def _unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _angle_at_rad(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle ABC in radians."""
    ba = a - b
    bc = c - b
    ba_u = _unit(ba)
    bc_u = _unit(bc)
    dot = float(np.clip(np.dot(ba_u, bc_u), -1.0, 1.0))
    return float(math.acos(dot))


def _flexion_from_angle(angle_abc: float) -> float:
    """Convert interior angle to flexion. Straight ≈ π → flexion ≈ 0."""
    return max(0.0, math.pi - float(angle_abc))


def _flexion_deg(angle_abc: float) -> float:
    """Flexion in degrees."""
    return math.degrees(_flexion_from_angle(angle_abc))


def _signed_angle_about_axis(v_from: np.ndarray, v_to: np.ndarray,
                              axis: np.ndarray) -> float:
    """Signed angle from v_from to v_to about axis (radians)."""
    a = _unit(axis)
    v1 = _unit(v_from)
    v2 = _unit(v_to)
    cross = np.cross(v1, v2)
    sin_term = float(np.dot(a, cross))
    cos_term = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(math.atan2(sin_term, cos_term))


def _landmark_ok(lm, threshold=0.5) -> bool:
    """Check if a landmark has sufficient presence/visibility."""
    presence = getattr(lm, "presence", None)
    visibility = getattr(lm, "visibility", None)
    if presence is not None:
        return presence >= threshold
    if visibility is not None:
        return visibility >= threshold
    return True


def _find_nearest_bent(sid: int, bent: set[int]) -> int | None:
    """Find the nearest other bent finger by adjacency in FINGER_ORDER.

    Scans outward from *sid*'s position; prefers left neighbour at equal distance.
    Returns None when fewer than 2 bent fingers exist.
    """
    if len(bent) < 2:
        return None
    idx = FINGER_ORDER.index(sid)
    for dist in range(1, len(FINGER_ORDER)):
        left = FINGER_ORDER[idx - dist] if idx - dist >= 0 else None
        right = FINGER_ORDER[idx + dist] if idx + dist < len(FINGER_ORDER) else None
        if left is not None and left in bent:
            return left
        if right is not None and right in bent:
            return right
    return None


# ---------------------------------------------------------------------------
# HandTracker
# ---------------------------------------------------------------------------
class HandTracker:
    """Continuous landmark → servo-position mapper with EMA smoothing."""

    def __init__(self, alpha: float = 0.35, angle_ranges: dict | None = None):
        self.alpha = alpha
        self._ranges = dict(angle_ranges or ANGLE_RANGES)
        self._smoothed: dict[int, float] = {}        # servo_id → smoothed ratio
        self._neutral: dict[int, float] = {}         # servo_id → calibration offset
        self._splay_mode: dict[int, str] = {}           # servo_id → 'A' | 'B'
        self._tip_flexion: dict[int, float] = {}     # servo_id → tip flexion deg (splay fingers)
        self._last_positions: dict[int, int] = {}    # servo_id → last written position

    # ------------------------------------------------------------------
    # Landmarks → angles
    # ------------------------------------------------------------------
    def landmarks_to_angles(self, landmarks) -> dict[int, float]:
        """Compute 17 raw joint angles (degrees) from 21 MediaPipe landmarks.

        Returns dict[servo_id, angle_degrees]. Keys absent if landmarks
        have low visibility.
        """
        angles: dict[int, float] = {}

        # Convert landmarks to numpy arrays
        pts = [np.array([lm.x, lm.y, lm.z], dtype=float) for lm in landmarks]

        # ── 3D orthonormal palm frame ──────────────────────────────
        # palm_x: pinky_mcp → index_mcp (left-right)
        # palm_y: wrist → middle_mcp (forward, re-orthogonalized)
        # palm_z: palm normal (outward from palm)
        palm_x = _unit(pts[5] - pts[17])
        palm_y = _unit(pts[9] - pts[0])
        palm_z = _unit(np.cross(palm_x, palm_y))
        palm_y = _unit(np.cross(palm_z, palm_x))

        palm_ok = all(_landmark_ok(landmarks[i]) for i in (0, 5, 9, 17))

        # ── Flexion helpers ────────────────────────────────────────
        def flex_mcp(mcp_i, pip_i):
            return _flexion_deg(_angle_at_rad(pts[0], pts[mcp_i], pts[pip_i]))

        def flex_pip(mcp_i, pip_i, dip_i):
            return _flexion_deg(_angle_at_rad(pts[mcp_i], pts[pip_i], pts[dip_i]))

        def flex_dip(pip_i, dip_i, tip_i):
            return _flexion_deg(_angle_at_rad(pts[pip_i], pts[dip_i], pts[tip_i]))

        # ── Four fingers: weighted MCP+PIP+DIP → mid(prox)/tip(dist) ──
        pw = (0.45, 0.45, 0.10)  # proximal weights → mid servo
        dw = (0.10, 0.45, 0.45)  # distal weights  → tip servo

        fingers = {
            "index":  (5, 6, 7, 8,   {"tip": 1, "mid": 2, "bottom": 3}),
            "middle": (9, 10, 11, 12, {"tip": 8, "mid": 9, "bottom": 4}),
            "ring":   (13, 14, 15, 16, {"tip": 11, "mid": 10, "bottom": 12}),
            "pinky":  (17, 18, 19, 20, {"tip": 6, "mid": 7, "bottom": 5}),
        }

        for _name, (mcp_i, pip_i, dip_i, tip_i, sv) in fingers.items():
            if not all(_landmark_ok(landmarks[i])
                       for i in (mcp_i, pip_i, dip_i, tip_i)):
                continue

            f_mcp = flex_mcp(mcp_i, pip_i)
            f_pip = flex_pip(mcp_i, pip_i, dip_i)
            f_dip = flex_dip(pip_i, dip_i, tip_i)

            angles[sv["mid"]] = pw[0]*f_mcp + pw[1]*f_pip + pw[2]*f_dip
            angles[sv["tip"]] = dw[0]*f_mcp + dw[1]*f_pip + dw[2]*f_dip

        # ── Splay (bottom joints) — dual strategy ─────────────────
        # Strategy A (extended): angle(wrist→MCP, MCP→PIP) — absolute per finger.
        # Strategy B (bent):     angle(ref_MCP→PIP, finger_MCP→PIP) — relative
        #                        to nearest other bent finger. Switches at tip > 60°.
        if palm_ok:
            def _proj(v):
                return v - palm_z * float(np.dot(v, palm_z))

            def _splay_deg(a, b):
                return math.degrees(_signed_angle_about_axis(a, b, palm_z))

            # Per-finger: (mcp, pip, dip, tip) landmark indices
            splay_fingers = {
                3:  (5, 6, 7, 8),      # index
                4:  (9, 10, 11, 12),    # middle
                12: (13, 14, 15, 16),  # ring
                5:  (17, 18, 19, 20),   # pinky
            }

            # ── Pass 1: compute per-finger data ──
            mcp_pip = {}      # servo_id → unit MCP→PIP vector
            wrist_mcp = {}    # servo_id → unit wrist→MCP vector
            tip_flex = {}     # servo_id → tip flexion degrees
            visible = set()   # servos with all landmarks OK

            for sid, (mcp_i, pip_i, dip_i, tip_i) in splay_fingers.items():
                if all(_landmark_ok(landmarks[i]) for i in (0, mcp_i, pip_i, dip_i, tip_i)):
                    visible.add(sid)
                    mcp_pip[sid] = _unit(_proj(pts[pip_i] - pts[mcp_i]))
                    wrist_mcp[sid] = _unit(_proj(pts[mcp_i] - pts[0]))
                    tip_flex[sid] = flex_pip(mcp_i, pip_i, dip_i)

            # ── Pass 2: determine strategy and compute splay ──
            self._tip_flexion = dict(tip_flex)
            bent = {sid for sid in visible if tip_flex.get(sid, 0) > 60}
            self._splay_mode.clear()

            for sid in visible:
                if sid in bent:
                    ref = _find_nearest_bent(sid, bent)
                    if ref is not None:
                        self._splay_mode[sid] = 'B'
                        angles[sid] = _splay_deg(mcp_pip[ref], mcp_pip[sid])
                        continue

                # Strategy A (default)
                self._splay_mode[sid] = 'A'
                angles[sid] = _splay_deg(wrist_mcp[sid], mcp_pip[sid])

        # ── Thumb ──────────────────────────────────────────────────
        thumb_ok = all(_landmark_ok(landmarks[i]) for i in (0, 1, 2, 3, 4))
        if thumb_ok and palm_ok:
            # Root (servo 13): MCP flexion ∠ CMC-MCP-IP
            angles[13] = _flexion_deg(_angle_at_rad(pts[1], pts[2], pts[3]))

            # Mid (servo 16): MCP-IP-TIP flexion (2→3→4)
            thumb_flex = _flexion_deg(_angle_at_rad(pts[2], pts[3], pts[4]))
            angles[16] = thumb_flex * 0.9

            # Tip (servo 15): coupled from mid
            angles[15] = thumb_flex * 0.8

            # Lateral splay (servo 14): dual-strategy, like fingers
            # Strategy A (thumb mid ≤ 35°): angle(wrist→CMC, MCP→IP) in palm plane
            # Strategy B (thumb mid > 35°): angle(wrist→CMC, MCP→CMC) raw 3D
            wrist_cmc_p = _unit(_proj(pts[1] - pts[0]))
            if angles.get(16, 0) > 35:
                self._splay_mode[14] = 'B'
                wrist_cmc_3d = _unit(pts[1] - pts[0])
                mcp_cmc_3d = _unit(pts[1] - pts[2])
                angles[14] = math.degrees(
                    _signed_angle_about_axis(wrist_cmc_3d, mcp_cmc_3d, palm_z))
            else:
                self._splay_mode[14] = 'A'
                mcp_ip_p = _unit(_proj(pts[3] - pts[2]))
                angles[14] = math.degrees(
                    _signed_angle_about_axis(wrist_cmc_p, mcp_ip_p, palm_z))

        # ── Wrist ──────────────────────────────────────────────────
        if palm_ok:
            camera_z = np.array([0.0, 0.0, 1.0], dtype=float)
            axis = palm_x
            ref = camera_z - axis * float(np.dot(camera_z, axis))
            val = palm_z - axis * float(np.dot(palm_z, axis))
            nr = float(np.linalg.norm(ref))
            nv = float(np.linalg.norm(val))
            if nr > 1e-8 and nv > 1e-8:
                angles[17] = math.degrees(
                    _signed_angle_about_axis(ref, val, axis))

        return angles

    # ------------------------------------------------------------------
    # Angles → positions
    # ------------------------------------------------------------------
    def angles_to_positions(self, angles: dict[int, float]) -> dict[int, int]:
        """Convert joint angles to servo positions (0-4095).

        Returns all 17 servo positions. Servos not in *angles* hold last value.
        Lateral joints use asymmetric mapping: signed deviation from mid_angle,
        each side scaled independently to a position range centred at 2048.
        """
        positions: dict[int, int] = {}

        for sid, cfg in SERVO_CONFIG.items():
            jtype = cfg["type"]

            if sid in angles:
                raw_angle = angles[sid]

                # ── Lateral joints: asymmetric signed-deviation mapping ──
                if jtype in ("bottom", "thumb_lateral") and sid in LATERAL_MAP:
                    mode = self._splay_mode.get(sid, 'A')
                    lm = LATERAL_MAP[sid]
                    params = lm.get(mode, lm['A'])

                    if mode == 'B' and "mid_angle" not in params:
                        # Finger B: angle between MCP→PIP vectors.
                        # Parallel (raw ≈ 0°) is neutral → MID_POS.
                        rng = params["pos_range"]
                        ratio = abs(raw_angle) / rng if rng > 0 else 0.0
                        ratio = max(0.0, min(1.0, ratio))
                        deviation_sign = 1  # spread only goes one way
                    else:
                        # Strategy A / Thumb B: signed deviation from mid_angle
                        deviation = raw_angle - params["mid_angle"]
                        if deviation >= 0:
                            rng = params["pos_range"]
                            ratio = deviation / rng if rng > 0 else 0.0
                        else:
                            rng = params["neg_range"]
                            ratio = abs(deviation) / rng if rng > 0 else 0.0
                        ratio = max(0.0, min(1.0, ratio))
                        deviation_sign = 1 if deviation >= 0 else -1

                    if sid in self._smoothed:
                        ratio = (self.alpha * ratio +
                                 (1.0 - self.alpha) * self._smoothed[sid])
                    self._smoothed[sid] = ratio

                    # Invert flag flips the spread direction
                    if cfg["invert"]:
                        deviation_sign = -deviation_sign

                    if deviation_sign >= 0:
                        pos = MID_POS + int(ratio * (params["max_pos"] - MID_POS) + 0.5)
                    else:
                        pos = MID_POS - int(ratio * (MID_POS - params["min_pos"]) + 0.5)

                    pos = max(cfg["min"] + 10, min(cfg["max"] - cfg.get("max_offset", 10), pos))
                    positions[sid] = pos
                    self._last_positions[sid] = pos
                    continue

                # ── Flexion / wrist joints ──
                min_a, max_a = (SERVO_ANGLE_RANGE.get(sid) or
                                self._ranges.get(jtype, (0.0, 90.0)))

                ratio = ((raw_angle - min_a) / (max_a - min_a)
                         if max_a > min_a else 0.0)
                ratio = max(0.0, min(1.0, ratio))

                if sid in self._neutral:
                    ratio = max(0.0, min(1.0, ratio - self._neutral[sid]))

                if sid in self._smoothed:
                    ratio = (self.alpha * ratio +
                             (1.0 - self.alpha) * self._smoothed[sid])
                self._smoothed[sid] = ratio
            else:
                ratio = self._smoothed.get(sid, 0.5)
                if jtype == "wrist":
                    ratio = self._smoothed.get(sid, 0.5) * 0.95

            # Ratio → position (flexion / wrist / non-mapped lateral)
            span = cfg["max"] - cfg["min"]
            flexion = jtype in ("mid", "tip", "thumb_mid", "thumb_tip", "thumb_big")

            if flexion:
                if cfg["invert"]:
                    pos = cfg["min"] + int(ratio * span + 0.5)
                else:
                    pos = cfg["max"] - int(ratio * span + 0.5)
            else:
                if cfg["invert"]:
                    pos = cfg["max"] - int(ratio * span + 0.5)
                else:
                    pos = cfg["min"] + int(ratio * span + 0.5)

            pos = max(cfg["min"] + 10, min(cfg["max"] - cfg.get("max_offset", 10), pos))
            positions[sid] = pos
            self._last_positions[sid] = pos

        return positions

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def process(self, landmarks) -> tuple[dict[int, int], dict[int, float]]:
        """Run full pipeline: landmarks → angles → smoothed positions."""
        angles = self.landmarks_to_angles(landmarks)
        positions = self.angles_to_positions(angles)
        return positions, angles

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def set_neutral(self, landmarks) -> dict[int, float]:
        """Record current angles as neutral baseline. Returns the offsets."""
        angles = self.landmarks_to_angles(landmarks)
        self._neutral.clear()
        for sid in SERVO_CONFIG:
            if sid not in angles:
                continue
            jtype = SERVO_CONFIG[sid]["type"]
            min_a, max_a = (SERVO_ANGLE_RANGE.get(sid) or
                            self._ranges.get(jtype, (0.0, 90.0)))
            if max_a > min_a:
                ratio = (angles[sid] - min_a) / (max_a - min_a)
                self._neutral[sid] = max(0.0, min(1.0, ratio))
        return dict(self._neutral)

    def clear_neutral(self):
        """Remove calibration offset."""
        self._neutral.clear()

    def calibrate_splay(self, landmarks):
        """Record current splay angles as mid_angle for Strategy A.

        Call with fingers straight and together. Updates LATERAL_MAP['A']
        mid_angle and the neg/pos ranges for each side.
        """
        angles = self.landmarks_to_angles(landmarks)
        for sid, cfg in SERVO_CONFIG.items():
            if sid in angles and cfg["type"] in ("bottom", "thumb_lateral"):
                lm = LATERAL_MAP.get(sid)
                if lm is None:
                    continue
                raw = angles[sid]
                params = lm["A"]
                old_mid = params.get("_prev_mid", raw)
                params["mid_angle"] = round(raw, 1)
                params["neg_range"] = round(params["neg_range"] + (old_mid - raw), 1)
                params["pos_range"] = round(params["pos_range"] - (old_mid - raw), 1)
                params["_prev_mid"] = raw
        for sid in sorted(LATERAL_MAP):
            p = LATERAL_MAP[sid]["A"]
            print(f"[splay cal] servo {sid}: mid={p['mid_angle']:.1f}  "
                  f"neg_range={p['neg_range']:.1f}  pos_range={p['pos_range']:.1f}")

    def clear_splay(self):
        """Reset LATERAL_MAP['A'] mid_angle to defaults."""
        defaults = {3: 3.0, 4: 0.5, 5: -3.0, 12: -0.5}
        for sid, mid in defaults.items():
            lm = LATERAL_MAP.get(sid)
            if lm:
                lm["A"]["mid_angle"] = mid

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def reset(self):
        """Clear all smoothing and calibration state."""
        self._smoothed.clear()
        self._neutral.clear()
        self._last_positions.clear()

    @property
    def last_positions(self) -> dict[int, int]:
        return dict(self._last_positions)

    @property
    def splay_mode(self) -> dict[int, str]:
        return dict(self._splay_mode)

    @property
    def tip_flexion(self) -> dict[int, float]:
        return dict(self._tip_flexion)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    class MockLM:
        def __init__(self, x, y, z, presence=1.0):
            self.x = x; self.y = y; self.z = z
            self.presence = presence

    open_hand = [MockLM(x, y, 0.0) for x, y in [
        (0.5, 0.8),  # 0: wrist
        (0.55, 0.7), (0.58, 0.58), (0.60, 0.48), (0.62, 0.40),  # 1-4: thumb
        (0.48, 0.55), (0.46, 0.38), (0.44, 0.25), (0.42, 0.15),  # 5-8: index
        (0.50, 0.53), (0.50, 0.35), (0.50, 0.22), (0.50, 0.12),  # 9-12: middle
        (0.52, 0.55), (0.54, 0.38), (0.56, 0.26), (0.58, 0.17),  # 13-16: ring
        (0.54, 0.57), (0.58, 0.42), (0.60, 0.32), (0.62, 0.24),  # 17-20: pinky
    ]]

    tracker = HandTracker()
    pos, angles = tracker.process(open_hand)

    print("Open hand → servo positions:")
    for sid in sorted(pos):
        cfg = SERVO_CONFIG[sid]
        ang = angles.get(sid, None)
        ang_str = f"{ang:.1f}" if ang is not None else "—"
        print(f"  Servo {sid:2d} ({cfg['name']:6s}): pos={pos[sid]:4d}  "
              f"[{cfg['min']}-{cfg['max']}]  angle={ang_str}")
