"""one_euro_filter — 时域低通滤波，专为带噪声的实时信号设计（如 MediaPipe landmarks）。

原理：对每个坐标轴做 1D 二阶低通滤波。截止频率随信号速度自适应：
    - 静止时 → 低截止频率 → 强平滑
    - 运动时 → 截止频率被速度项抬高 → 跟手

References:
    - Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems", CHI 2012
    - https://cristal.univ-lille.fr/~casiez/1euro/

用法：
    f = OneEuroFilter(min_cutoff=0.004, beta=0.7)
    for x in stream:
        y = f(x)            # x,y 都是标量或 shape=(D,) 的 numpy
"""

from __future__ import annotations

import numpy as np


def _smoothing_factor(te: float, cutoff: float) -> float:
    """计算 1D low-pass 的平滑系数 alpha。

    te  : 采样周期（秒）
    cutoff : 截止频率（Hz）
    """
    r = 2 * np.pi * cutoff * te
    return r / (r + 1)


class OneEuroFilter:
    """对一个标量信号做 1€ filter。

    Parameters
    ----------
    min_cutoff : float
        最小截止频率（Hz）。越小越平滑。
        MediaPipe landmarks 推荐 0.003~0.01。
    beta : float
        速度项权重。越大越跟手。
        MediaPipe landmarks 推荐 0.5~1.0。
    d_cutoff : float
        对速度信号本身的截止频率，默认 1.0。
    """

    def __init__(
        self,
        min_cutoff: float = 0.004,
        beta: float = 0.7,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def __call__(self, x: float, t: float | None = None) -> float:
        """对一个新样本滤波。"""
        if self._t_prev is None:
            # 第一帧：直接返回原值，建立初值
            self._x_prev = float(x)
            self._t_prev = t
            return float(x)

        if t is None:
            # 默认 60Hz
            te = 1.0 / 60.0
        else:
            te = max(t - self._t_prev, 1e-6)
        self._t_prev = t

        # 速度的低通
        dx = (x - self._x_prev) / te
        a_d = _smoothing_factor(te, self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        self._dx_prev = dx_hat

        # 自适应截止频率
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(te, cutoff)

        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev = x_hat
        return float(x_hat)


class OneEuroFilterND:
    """对 shape=(D,) 的向量信号，每一维独立做 1€ filter。

    Parameters
    ----------
    dims : int
        向量维度
    min_cutoff, beta, d_cutoff : 同 OneEuroFilter
    """

    def __init__(
        self,
        dims: int,
        min_cutoff: float = 0.004,
        beta: float = 0.7,
        d_cutoff: float = 1.0,
    ) -> None:
        self._filters = [
            OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
            for _ in range(dims)
        ]

    def __call__(self, x: np.ndarray, t: float | None = None) -> np.ndarray:
        out = np.empty_like(x, dtype=np.float64)
        for i, f in enumerate(self._filters):
            out[i] = f(float(x[i]), t=t)
        return out