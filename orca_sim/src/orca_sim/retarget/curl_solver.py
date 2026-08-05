"""curl_solver — MediaPipe 21 landmarks → ORCA 17 关节角（无 IK）。

ORCA v1 右手关节物理语义（实测，参考 orca_sim/models/v1/right.mjcf）：

非拇指手指（index/middle/ring/pinky）：
    - xxx_abd：根部侧向。abd 变小 = 远离中指；abd 变大 = 朝中指
    - xxx_mcp：中段弯曲
        * 关键：人手完全伸直 → sim mcp_lo（手完全伸直）
        * 人手完全握拳 → sim mcp_hi
    - xxx_pip：指节弯曲
        * 伸直 → pip_lo；握拳 → pip_hi

MediaPipe → ORCA 的映射（直接几何，不用 lookup 表）：
    1. mcp：curl_norm=0（人手完全伸直）→ sim mcp_lo
       curl_norm=1（握拳）→ sim mcp_hi
    2. pip 跟 mcp 联动
    3. abd 用"手掌局部系 X"测度（手掌旋转不变）
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


# MediaPipe landmark 索引
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


@dataclass
class CurlSolverConfig:
    """CurlSolver 配置。"""

    # 非拇指：mcp_angle = mcp_lo + curl_eff × (mcp_hi - mcp_lo)
    # 张直(curl_eff=0) → mcp_lo（人手伸直）；握拳(curl_eff=1) → mcp_hi
    # mcp_gain=1.0 时刚好用满 ctrlrange；不要再放大
    mcp_gain: float = 1.0

    # pip 跟 mcp 联动比例
    # 单独 pip 在 ORCA v1 几乎不动，但 mcp+pip 联动时 pip 影响 ip 段最后位置
    pip_ratio: float = 0.7

    # 拇指：thumb_pip_gain 决定 thumb_pip 最大值
    # thumb_pip ctrlrange 上限 = +1.23 → gain=1.23 时 curl_eff=1 刚好到 hi
    # 不要再放大，否则 curl=0.5 就 saturate 到 hi，丢失中段区分度
    thumb_pip_gain: float = 1.23
    thumb_mcp_ratio: float = 0.85       # thumb_mcp 联动更强
    thumb_dip_ratio: float = 0.65

    # thumb_abd：thumb_tip.y 映射
    thumb_abd_y_scale: float = 0.04     # 0.08 → 0.04 (更灵敏)

    # 非拇指 abd：TIP 相对当前手指 MCP 的横向偏移（不是相对 middle_mcp）
    #   extra = (tip.x - mcp.x) / 单手指"自然张开方向"
    #   index 自然外展方向 = +X（远离中指）；外展 extra > 0 → abd_lo
    #   pinky 自然外展方向 = -X；外展 extra < 0 → abd_lo
    # 各手指 natural sign:
    #   index: +1, middle: 0, ring: -1, pinky: -1
    # abd_x_scale = 0.04（额外外展 40mm → 达到 abd_lo）
    abd_x_scale: float = 0.04

    # 输出限位
    enforce_limits: bool = True


class CurlSolver:
    """Curl-based 手势重定向（直接几何，不用 lookup 表）。

    原理：
      - curl_norm = 1 - |TIP-MCP| / 手指伸直长度 ∈ [0, 1]
        张直 → 0；握拳 → 1
      - mcp_angle = curl_norm × mcp_gain
      - pip_angle = mcp_angle × pip_ratio
      - abd 从手指侧向偏移估
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: CurlSolverConfig | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.config = config or CurlSolverConfig()
        self._ctrlrange = model.actuator_ctrlrange.copy()

        # actuator 索引（按 v1 right hand 声明顺序）：
        #   0: wrist
        #   1: thumb_mcp  2: thumb_abd  3: thumb_pip  4: thumb_dip
        #   5: index_abd  6: index_mcp  7: index_pip
        #   8: middle_abd 9: middle_mcp 10: middle_pip
        #   11: ring_abd 12: ring_mcp 13: ring_pip
        #   14: pinky_abd 15: pinky_mcp 16: pinky_pip

    # ------------------------------------------------------------------
    # 核心：curl ratio → mcp angle
    # ------------------------------------------------------------------
    @staticmethod
    def _curl_norm(lm_mcp: np.ndarray, lm_tip: np.ndarray,
                   finger_length: float) -> float:
        """算 MediaPipe 手指 curl 程度，0=张直，1=握拳。

        用 TIP-MCP 距离 / 手指伸直参考长度。
        手指参考长度 ≈ MCP→PIP 距离 + PIP→DIP 距离 + DIP→TIP 距离 ≈ 0.07m
        （典型人手）或外部传入的 finger_length（已归一化）。

        真实 MediaPipe 输出：
          - 张直手：TIP 远离 MCP，距离 ≈ finger_length → curl_norm ≈ 0
          - 握拳：  TIP 折回到 MCP 附近，距离 ≈ 0 → curl_norm ≈ 1

        注：不能直接用 |TIP-MCP| / finger_length，因为握拳时这个比是 0。
        用 1 - 这个比：
          curl_norm = 1 - clamp(|TIP-MCP| / finger_length, 0, 1)
        """
        d = float(np.linalg.norm(lm_tip - lm_mcp))
        if finger_length < 1e-6:
            return 0.0
        norm = d / finger_length
        return float(np.clip(1.0 - norm, 0.0, 1.0))

    def _finger_flexion(
        self,
        lm_mcp: np.ndarray,
        lm_pip: np.ndarray,
        lm_dip: np.ndarray,
        lm_tip: np.ndarray,
        mcp_aidx: int,
        pip_aidx: int,
    ) -> tuple[float, float]:
        """返回 (mcp_angle, pip_angle) for 非拇指手指。

        关键修正：人手完全伸直时，sim 手指也要完全伸直。
            curl_norm=0 (完全伸直) → sim mcp_lo
            curl_norm=1 (握拳)     → sim mcp_hi
            （线性映射，sqrt 曲线补偿人手 curl_norm 上限 ≈ 0.67）

        mcp_angle = mcp_lo + curl_eff × (mcp_hi - mcp_lo)
        pip_angle = mcp_lo + curl_eff × (pip_hi - pip_lo) × pip_ratio
        """
        seg1 = float(np.linalg.norm(lm_pip - lm_mcp))
        seg2 = float(np.linalg.norm(lm_dip - lm_pip))
        seg3 = float(np.linalg.norm(lm_tip - lm_dip))
        finger_length = seg1 + seg2 + seg3
        if finger_length < 0.03:
            finger_length = 0.08

        d = float(np.linalg.norm(lm_tip - lm_mcp))
        curl_norm = float(np.clip(1.0 - d / finger_length, 0.0, 1.0))
        # 不再用 piecewise 阈值：保留真实 curl_norm（小差异直接反映到 mcp 微调）
        # 这样 ring 与 middle 即使 curl=0.05 与 0.07 也会有差异
        if curl_norm < 0.05:
            curl_eff = 0.0
        else:
            # 缩放到 [0, 1] 区间；保留小 curl 让 mcp 有细微差异
            scaled = (curl_norm - 0.05) / 0.95
            curl_eff = float(np.sqrt(max(scaled, 0.0)))  # sqrt 加速

        mcp_lo, mcp_hi = self._ctrlrange[mcp_aidx]
        # 伸直(curl_eff=0) → mcp_lo；握拳(curl_eff=1) → mcp_hi
        mcp_angle = mcp_lo + curl_eff * (mcp_hi - mcp_lo) * self.config.mcp_gain

        pip_lo, pip_hi = self._ctrlrange[pip_aidx]
        # pip 跟 mcp 联动：伸直 → pip_lo，握拳 → pip_hi × pip_ratio
        pip_angle = pip_lo + curl_eff * (pip_hi - pip_lo) * self.config.pip_ratio

        return mcp_angle, pip_angle

    # ------------------------------------------------------------------
    # abd 角度（侧展）
    # ------------------------------------------------------------------
    def _abd_angle(
        self,
        finger_name: str,
        lm_mcp: np.ndarray,
        lm_tip: np.ndarray,
        lm_middle_mcp: np.ndarray,
    ) -> float:
        """非拇指手指 abd：测 TIP 相对 MCP 在'手掌局部系'的左右偏移。

        关键修正（用户需求）：
            手掌整体顺时针/逆时针旋转时，每根手指相对手掌的左右角度**不变**。
            所以 abd 不能用世界系 X 偏移（tip.x - mcp.x），
            要用"手掌局部系"的左右偏移。

        步骤：
            1. 手掌主轴 = wrist→middle_mcp，归一化为 axis
            2. 手指相对主轴的"垂直"方向 = world_up - (world_up · axis) × axis
               （取世界 Y 与主轴正交的横向分量）
            3. extra = (tip - mcp) 在垂直方向上的投影
               → 手掌旋转时，axis 变，垂直方向也跟着转，投影自动保持
            4. 按 natural_sign 决定每根手指"外展"对应的方向
        """
        import numpy as _np

        # 手腕位置（landmark 0）
        lm_wrist = self._last_wrist  # 调用前要 set

        # 主轴方向（手掌前后方向，wrist→middle_mcp）
        palm_axis = _np.asarray(lm_middle_mcp, dtype=float) - _np.asarray(lm_wrist, dtype=float)
        norm = _np.linalg.norm(palm_axis)
        # "手掌左右"方向 = 人手局部 X = 主轴 × 世界 up
        # 这个人手局部 X 跟着手掌旋转，与人手手指的真实"左右"对齐
        if norm < 1e-6:
            extra_raw = float(lm_tip[0]) - float(lm_mcp[0])
        else:
            palm_axis = palm_axis / norm
            world_up = _np.array([0.0, 1.0, 0.0])
            # 局部 X = up × palm_axis（让食指（+X 侧）外展时投影为正）
            local_x = _np.cross(world_up, palm_axis)
            lnorm = _np.linalg.norm(local_x)
            if lnorm < 1e-6:
                # 主轴几乎垂直 → 用世界 X 投影
                world_horiz = _np.array([1.0, 0.0, 0.0])
                local_x = world_horiz - _np.dot(world_horiz, palm_axis) * palm_axis
                lnorm = _np.linalg.norm(local_x)
                if lnorm < 1e-6:
                    extra_raw = float(lm_tip[0]) - float(lm_mcp[0])
                else:
                    local_x = local_x / lnorm
                    v = _np.asarray(lm_tip, dtype=float) - _np.asarray(lm_mcp, dtype=float)
                    extra_raw = float(_np.dot(v, local_x))
            else:
                local_x = local_x / lnorm
                v = _np.asarray(lm_tip, dtype=float) - _np.asarray(lm_mcp, dtype=float)
                extra_raw = float(_np.dot(v, local_x))

        # 沿用之前的 natural_sign 配置（不动，按用户要求）
        # sim 4 指 abd 旋转方向不一致：
        #   index_abd / middle_abd: range 单边偏负，extra_raw > 0（朝 +X 拇指侧）→ abd_lo
        #   ring_abd   / pinky_abd: range 单边偏正，extra_raw > 0              → abd_hi
        natural_signs = {"index": -1.0, "middle": -1.0, "ring": +1.0, "pinky": +1.0}
        ns = natural_signs[finger_name]
        extra = extra_raw * ns

        abd_aidx = {"index": 5, "middle": 8, "ring": 11, "pinky": 14}[finger_name]
        abd_lo, abd_hi = self._ctrlrange[abd_aidx]

        # 缩小 scale 让小幅横向偏移就能产生明显 abd 变化（人手外展 5mm 就该有反应）
        # 之前 0.04 太宽——人手外展 1cm 只产生 25% 范围
        abd_scale = self.config.abd_x_scale * 0.6   # 0.04 → 0.024
        norm_v = float(_np.clip(extra / abd_scale, -1, 1))
        abd_value = abd_lo + 0.5 * (norm_v + 1) * (abd_hi - abd_lo)
        return float(_np.clip(abd_value, abd_lo, abd_hi))

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def solve(self, landmarks_world: np.ndarray) -> np.ndarray:
        """从 21 landmarks 算出 17 关节角（按 actuator 顺序）。"""
        qpos = np.zeros(17, dtype=np.float64)
        lm_middle_mcp = landmarks_world[MIDDLE_MCP]
        # 把 wrist 缓存给 _abd_angle 用（手掌局部系计算）
        self._last_wrist = landmarks_world[WRIST].copy()

        # === 4 根非拇指手指 ===
        specs = [
            # name,   MCP,    PIP,    DIP,    TIP,    abd, mcp, pip
            ("index",  INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP,  5, 6, 7),
            ("middle", MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, 8, 9, 10),
            ("ring",   RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP,   11, 12, 13),
            ("pinky",  PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP,  14, 15, 16),
        ]
        for name, mcp_lm, pip_lm, dip_lm, tip_lm, abd_aidx, mcp_aidx, pip_aidx in specs:
            mcp_angle, pip_angle = self._finger_flexion(
                landmarks_world[mcp_lm],
                landmarks_world[pip_lm],
                landmarks_world[dip_lm],
                landmarks_world[tip_lm],
                mcp_aidx,
                pip_aidx,
            )
            qpos[mcp_aidx] = mcp_angle
            qpos[pip_aidx] = pip_angle
            qpos[abd_aidx] = self._abd_angle(
                name,
                landmarks_world[mcp_lm],
                landmarks_world[tip_lm],
                lm_middle_mcp,
            )

        # === 拇指（4 关节）===
        # 拇指：用 CMC→MCP→IP→TIP 链长
        seg1 = float(np.linalg.norm(landmarks_world[THUMB_MCP] - landmarks_world[THUMB_CMC]))
        seg2 = float(np.linalg.norm(landmarks_world[THUMB_IP] - landmarks_world[THUMB_MCP]))
        seg3 = float(np.linalg.norm(landmarks_world[THUMB_TIP] - landmarks_world[THUMB_IP]))
        thumb_len = seg1 + seg2 + seg3
        if thumb_len < 0.03:
            thumb_len = 0.06

        thumb_curl_norm = self._curl_norm(
            landmarks_world[THUMB_MCP],
            landmarks_world[THUMB_TIP],
            thumb_len,
        )
        # 与 4 指一致：阈值降 + sqrt 加速，让小幅度 thumb 弯曲也能区分
        if thumb_curl_norm < 0.05:
            thumb_curl_eff = 0.0
        else:
            thumb_curl_eff = float(np.sqrt(max((thumb_curl_norm - 0.05) / 0.95, 0.0)))
        thumb_pip_angle = thumb_curl_eff * self.config.thumb_pip_gain
        thumb_mcp_angle = thumb_pip_angle * self.config.thumb_mcp_ratio
        thumb_dip_angle = thumb_pip_angle * self.config.thumb_dip_ratio

        qpos[1] = thumb_mcp_angle    # thumb_mcp
        qpos[3] = thumb_pip_angle    # thumb_pip
        qpos[4] = thumb_dip_angle    # thumb_dip

        # thumb_abd：thumb_tip.y 越大（手背方向）→ 拇指向"上" → abd 越大
        # MediaPipe 标系：手背方向是 -Y（Y 越小 = 越向上）
        # 所以要反转符号
        tip_y = float(landmarks_world[THUMB_TIP][1])
        abd_lo, abd_hi = self._ctrlrange[2]
        norm_y = float(np.clip(-tip_y / self.config.thumb_abd_y_scale, -1, 1))
        qpos[2] = abd_lo + 0.5 * (norm_y + 1) * (abd_hi - abd_lo)

        # === wrist：保持 rest ===
        qpos[0] = 0.5 * (self._ctrlrange[0, 0] + self._ctrlrange[0, 1])

        # 限位
        if self.config.enforce_limits:
            qpos = np.clip(qpos, self._ctrlrange[:, 0], self._ctrlrange[:, 1])

        return qpos.astype(np.float64)