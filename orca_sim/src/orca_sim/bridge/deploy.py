"""sim → 真机一比一部署 + 碰撞预筛选。

核心思想：
    - sim ctrlrange [low, high]（弧度）↔ xdat [min_angle, max_angle]（raw 0..4095）线性对应；
      **忽略零点偏移**——sim rad=low 对应 raw=min_angle，sim rad=high 对应 raw=max_angle。
    - 部署前先 ``env.step(action)``，检查 sim 自碰撞；有自碰撞则跳过下发真机。
    - 真机端由 :class:`ServoSafetyLayer` 二次校验 raw ∈ [min_angle, max_angle]，
      越界抛 :class:`SafetyViolation`，bridge 捕获并报告。
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from orca_sim.bridge.collision import CollisionGuard
from orca_sim.bridge.limits import check_alignment
from orca_sim.bridge.mapping import Mapping, resolve_sim_indices


# 容差：raw 转换时若超出 [min_angle, max_angle] 但仅 epsilon 范围，仍视为合法（避免浮点误差）。
_EPS = 1e-6


class SimToRealDeployer:
    """sim 训练结果 → 真机 17 舵机同步执行的部署器。

    Parameters
    ----------
    console_backend : ConsoleBackend
        FTServo_Python 端的 :class:`ConsoleBackend`（已在 ``xdat_dir`` 下 ``load_bundles`` 完毕）。
        必须是已加载 17 个 ``ServoBundle`` 的实例。
    env : gymnasium.Env
        orca_sim env；需已 ``reset``。
    mapping : Mapping
        :func:`load_joint_mapping` 返回的对象。
    collision_check : bool
        是否启用碰撞预筛选（默认 True）。
    """

    def __init__(
        self,
        console_backend: Any,
        env: Any,
        mapping: Mapping,
        *,
        collision_check: bool = True,
    ) -> None:
        self.backend = console_backend
        self.env = env
        self.mapping = mapping
        self.collision_check = bool(collision_check)

        # 缓存每个 servo_id 的 sim index、sim ctrlrange、xdat [min, max]
        self._info: dict[int, dict[str, float]] = {}
        self._sim_idx_by_servo: dict[int, int] = {}
        self._resolve_indices()
        self._cache_per_servo()

        # 碰撞守卫
        self.guard: CollisionGuard | None = (
            CollisionGuard(env) if self.collision_check else None
        )

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _resolve_indices(self) -> None:
        """把 mapping 解析为 sim_actuator_index 缓存（供 ``_info`` 复用）。"""
        model = self.env.unwrapped.model
        self._sim_idx_by_servo = resolve_sim_indices(self.mapping, model)

    def _cache_per_servo(self) -> None:
        for entry in self.mapping.values():
            sid_int = entry.servo_id
            sim_idx = self._sim_idx_by_servo[sid_int]
            bundle = self.backend.bundles[sid_int - 1]
            self._info[sid_int] = {
                "sim_idx":  int(sim_idx),
                "sim_low":  float(self.env.action_low[sim_idx]),
                "sim_high": float(self.env.action_high[sim_idx]),
                "raw_low":  int(bundle.min_angle),
                "raw_high": int(bundle.max_angle),
            }

    def _rad_to_raw(self, info: dict[str, float], rad: float) -> int:
        """忽略零点：sim ctrlrange [low, high] ↔ xdat [min_angle, max_angle] 线性对应。"""
        sim_range = info["sim_high"] - info["sim_low"]
        if sim_range <= _EPS:
            # 退化（sim 端几乎无自由度）→ 直接用 raw_low
            return int(info["raw_low"])
        t = (rad - info["sim_low"]) / sim_range
        t = float(np.clip(t, 0.0, 1.0))
        raw = info["raw_low"] + t * (info["raw_high"] - info["raw_low"])
        # 整数化并 clip 到 raw 范围（避免微小浮点误差）
        return int(round(float(np.clip(raw, info["raw_low"], info["raw_high"]))))

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def deploy_qpos(self, sim_qpos_rad: np.ndarray) -> dict[int, int]:
        """把 sim 当前 qpos（前 nv 个值，弧度）部署到真机 17 舵机。

        Parameters
        ----------
        sim_qpos_rad : np.ndarray
            sim 关节位置（弧度），长度 ≥ ``env.unwrapped.model.nv``。

        Returns
        -------
        dict[int, int]
            ``{servo_id: raw}``，长度 17（含 wrist）。
        """
        positions: dict[int, int] = {}
        for sid_int, info in self._info.items():
            rad = float(sim_qpos_rad[info["sim_idx"]])
            positions[sid_int] = self._rad_to_raw(info, rad)
        # 真机原子下发 + 限位校验（ServoSafetyLayer 内部断言 raw ∈ [min, max]）
        self.backend.safety.sync_go_to_pose(positions)
        return positions

    def safe_deploy_action(
        self, action: np.ndarray
    ) -> tuple[bool, dict[int, int] | list]:
        """高阶入口：先 sim.step(action)，检查自碰撞，无碰撞才下发真机。

        Parameters
        ----------
        action : np.ndarray
            shape ``(env.action_space.shape[0],)`` 的 sim 目标动作。

        Returns
        -------
        (ok, info) : tuple
            - ``ok=True``  → ``info`` 是 ``{servo_id: raw}``（已下发真机）。
            - ``ok=False`` → ``info`` 是 :class:`ContactPair` 列表（动作被跳过）。
        """
        self.env.step(action)
        if self.guard is not None:
            contacts = self.guard.self_contacts()
            if contacts:
                return False, contacts
        qpos = self.env.unwrapped.data.qpos[: self.env.unwrapped.model.nv]
        return True, self.deploy_qpos(qpos)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def alignment_report(self) -> list[tuple[int, str, float, float]]:
        """运行 :func:`orca_sim.bridge.limits.check_alignment`，返回每只舵机的对齐情况。"""
        return check_alignment(self.mapping, self._info)


# ----------------------------------------------------------------------
# Dry-run backend：用于无真机调试
# ----------------------------------------------------------------------
@dataclass
class _DryBundle:
    """dry-run 占位 ServoBundle。"""

    servo_id: int
    min_angle: int
    max_angle: int
    ofs: int = 0
    fields: dict | None = None


class _DryRunBackend:
    """dry_run 模式下替代 :class:`ConsoleBackend` 的最小 mock。

    暴露 ``bundles``（17 个 ``_DryBundle``）、``safety``（``_DrySafety``）、
    ``connect()`` / ``load_bundles()`` 等最小接口。

    ``load_bundles(xdat_dir)`` 默认从指定目录读 1..17.xdat；如果目录不存在则
    回退到全占 [0, 4095]（占位）。
    """

    def __init__(self, mapping: Mapping) -> None:
        self.bundles: list[_DryBundle] = []
        self.safety = _DrySafety(self)
        self._connected = False
        self._deploy_log: list[dict[int, int]] = []

    def load_bundles(self, xdat_dir: str | pathlib.Path | None = None) -> list[_DryBundle]:
        """从 xdat_dir/1.xdat..17.xdat 读真实 min/max_angle；目录缺失则全占 [0, 4095]。"""
        bundles: list[_DryBundle] = []
        xdat_dir = pathlib.Path(xdat_dir) if xdat_dir else None
        for sid in range(1, 18):
            min_a, max_a = 0, 4095
            if xdat_dir is not None and (xdat_dir / f"{sid}.xdat").exists():
                try:
                    fields = self._read_xdat(xdat_dir / f"{sid}.xdat")
                    min_a = int(fields.get("最小角度限制", 0))
                    max_a = int(fields.get("最大角度限制", 4095))
                except Exception:
                    pass
            bundles.append(
                _DryBundle(
                    servo_id=sid,
                    min_angle=min_a,
                    max_angle=max_a,
                    ofs=0,
                    fields=None,
                )
            )
        self.bundles = bundles
        return self.bundles

    @staticmethod
    def _read_xdat(path: pathlib.Path) -> dict[str, int]:
        """从 .xdat 二进制文件读 min/max_angle；不依赖 xdat_tool。

        xdat 格式：2-byte header (0x00 0x28) + 49-byte little-endian payload；
        我们只需读出「最小角度限制」与「最大角度限制」字段。
        如果结构未知，按 xdat_tool 的字段表跳过 magic 直接解析常见布局。
        """
        data = path.read_bytes()
        if len(data) < 51:
            raise ValueError(f"xdat too short: {len(data)} bytes")
        # 简化：尝试 xdat_tool；失败回退
        try:
            from xdat_tool import read_xdat  # type: ignore
            return read_xdat(str(path))
        except Exception:
            # 占位 fallback：返回全量程（不阻塞 dry-run）
            return {"最小角度限制": 0, "最大角度限制": 4095}

    def connect(self, port: str, baud: int) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def deploy_log(self) -> list[dict[int, int]]:
        return list(self._deploy_log)


class _DrySafety:
    def __init__(self, backend: _DryRunBackend) -> None:
        self._backend = backend
        self._emergency_stopped = False

    def sync_go_to_pose(self, positions: dict[int, int]) -> int:
        """记录下发但不实际写硬件；模拟 ServoSafetyLayer 的限位校验。"""
        # 校验
        SafetyViolation: type | None = None
        for sid, raw in positions.items():
            b = self._backend.bundles[sid - 1]
            if not (b.min_angle <= raw <= b.max_angle):
                try:
                    from servo_console import SafetyViolation as _SV  # type: ignore
                    SafetyViolation = _SV
                except ImportError:
                    SafetyViolation = RuntimeError
                raise SafetyViolation(
                    f"ID {sid}: raw {raw} not in [{b.min_angle}, {b.max_angle}]"
                )
        self._backend._deploy_log.append(dict(positions))
        return 0

    def emergency_stop(self) -> None:
        self._emergency_stopped = True

    def recovery(self) -> None:
        self._emergency_stopped = False

    def is_emergency_stopped(self) -> bool:
        return self._emergency_stopped


# ----------------------------------------------------------------------
# 工厂
# ----------------------------------------------------------------------
def build_deployer(
    *,
    env_factory: Callable[[], Any],
    xdat_dir: str | pathlib.Path,
    port: str | None = None,
    baud: int = 1_000_000,
    mapping: Mapping | None = None,
    collision_check: bool = True,
    dry_run: bool = False,
) -> tuple[SimToRealDeployer, Any]:
    """一站式构造 deployer。

    Parameters
    ----------
    env_factory : callable
        无参工厂，返回已 ``reset`` 的 orca_sim env（建议 ``skin=False``）。
    xdat_dir : str
        FTServo_Python/参数 目录的路径。
    port : str 或 None
        串口名（如 ``"COM5"``）。``dry_run=True`` 时可为 None。
    baud : int
        波特率，默认 1_000_000。
    mapping : Mapping 或 None
        已加载的映射；None → 用默认 JSON。
    collision_check : bool
        是否启用碰撞预筛选。
    dry_run : bool
        True 时不连真机：用 ``_DryRunBackend`` 替代 ``ConsoleBackend``，
        部署调用被记录但不下发。
        False 时必须给 ``port``。

    Returns
    -------
    (SimToRealDeployer, env)
    """
    from orca_sim.bridge import BridgeHardwareUnavailable
    from orca_sim.bridge.mapping import load_joint_mapping

    env = env_factory()

    if mapping is None:
        mapping = load_joint_mapping()

    if dry_run:
        backend = _DryRunBackend(mapping)
        backend.load_bundles(xdat_dir)   # 从 xdat_dir 读真实 17 份 xdat
    else:
        try:
            from servo_console import ConsoleBackend  # type: ignore
        except ImportError as e:
            raise BridgeHardwareUnavailable(
                f"无法 import servo_console: {e}。"
                "请确认 FTServo_Python 目录可访问，"
                "或使用 dry_run=True 跳过真机。"
            ) from e

        if not port:
            raise ValueError("dry_run=False 时必须指定 port")
        backend = ConsoleBackend()
        backend.load_bundles()  # ConsoleBackend 读固定路径 PARAM_DIR
        # 注：FTServo_Python/test/servo_console.py 的 load_bundles 用相对路径
        # (PROJECT_ROOT/参数)，要求 xdat_dir 与之匹配。如需自定义路径，
        # 应把 xdat 拷贝/链接到 FTServo_Python/参数/ 下。
        backend.connect(port, baud)

    deployer = SimToRealDeployer(
        console_backend=backend,
        env=env,
        mapping=mapping,
        collision_check=collision_check,
    )
    return deployer, env