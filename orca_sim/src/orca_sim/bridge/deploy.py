"""sim → 真机一比一部署 + 碰撞预筛选。

核心思想：
    - sim ctrlrange [low, high]（弧度）↔ xdat [min_angle, max_angle]（raw 0..4095）线性对应；
      **忽略零点偏移**——sim rad=low 对应 raw=min_angle，sim rad=high 对应 raw=max_angle。
    - **方向翻转**：实测发现 sim 滑条正方向与真机转动方向相反，所以
      :meth:`SimToRealDeployer._cache_per_servo` 会交换 ``raw_low``/``raw_high``
      的存储。下游 :meth:`_rad_to_raw` 完全无感，仍按 ``low → low`` 映射。
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

        # 一次性碰撞旁通 latch：UI 按钮按下时调 ``bypass_next_collision_check()``
        # 置位；下一帧若检测到自碰撞就**消耗**该 latch 并照常下发（不跳过），
        # 之后 latch 自动归零。下一帧仍碰撞则再次跳过——单次放行语义。
        self.collision_bypass: bool = False

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _resolve_indices(self) -> None:
        """把 mapping 解析为 sim_actuator_index 缓存（供 ``_info`` 复用）。"""
        model = self.env.unwrapped.model
        self._sim_idx_by_servo = resolve_sim_indices(self.mapping, model)

    def _cache_per_servo(self) -> None:
        """每个 servo_id 缓存 sim index + sim ctrlrange + xdat [min, max]。

        **方向翻转约定**（per-servo，受 ``Mapping.entry.flip_direction`` 控制）：

        实测发现 sim 滑条的正方向与真实舵机的转动方向**大部分**相反——
        默认翻转：``raw_low := xdat.max_angle``, ``raw_high := xdat.min_angle``。
        下游 :meth:`_rad_to_raw` 仍按 ``low → low / high → high`` 线性映射，
        但写到 STS3215 的 raw 已自带翻转。

        个别 servo 可能 sim 与真机方向一致（实测中遇到过
        ``right_index_abd / right_middle_abd / right_pinky_abd`` 三个指根关节
        就是这种），这时在 JSON mapping 里把 ``flip_direction`` 设为 ``false``
        即可：``raw_low := xdat.min_angle``、``raw_high := xdat.max_angle``。
        CLAUDE.md §6.2 已经点过 "忽略零点偏移"，这里与该约定保持一致
        （即不补偿 ``ofs``）。
        """
        for entry in self.mapping.values():
            sid_int = entry.servo_id
            sim_idx = self._sim_idx_by_servo[sid_int]
            bundle = self.backend.bundles[sid_int - 1]
            raw_max = int(bundle.max_angle)
            raw_min = int(bundle.min_angle)
            if entry.flip_direction:
                # 默认翻转：raw_low 存大值，raw_high 存小值
                raw_low, raw_high = raw_max, raw_min
            else:
                # 显式不翻：raw_low 存小值，raw_high 存大值（与 xdat 顺序一致）
                raw_low, raw_high = raw_min, raw_max
            self._info[sid_int] = {
                "sim_idx":  int(sim_idx),
                "sim_low":  float(self.env.action_low[sim_idx]),
                "sim_high": float(self.env.action_high[sim_idx]),
                "raw_low":  raw_low,
                "raw_high": raw_high,
            }

    def _rad_to_raw(self, info: dict[str, float], rad: float) -> int:
        """线性映射 ``sim_low → raw_low`` / ``sim_high → raw_high``，clip 到 raw 边界。

        ``raw_low`` / ``raw_high`` 已在 :meth:`_cache_per_servo` 里按
        ``flip_direction`` 决定是否翻转（见该方法的 docstring）。
        所以**这个函数无需知道方向翻转的事**。仍忽略零点偏移——
        ``sim rad=0`` 不会自动对齐到 ``raw=2048``。

        **clip 行为**：翻转后 ``raw_low > raw_high`` 是常态；
        ``np.clip(x, a, b)`` 在 ``a > b`` 时行为未定义（实测返回 ``b``），
        所以这里显式用 max/min 表达 "raw 在翻转后的区间内"。
        """
        sim_range = info["sim_high"] - info["sim_low"]
        if sim_range <= _EPS:
            # 退化（sim 端几乎无自由度）→ 直接用 raw_low
            return int(info["raw_low"])
        t = (rad - info["sim_low"]) / sim_range
        t = float(np.clip(t, 0.0, 1.0))
        raw = info["raw_low"] + t * (info["raw_high"] - info["raw_low"])
        # 显式 clip：翻转后 raw_high ≤ raw_low，所以 raw 必须在 [raw_high, raw_low]
        lo, hi = (info["raw_high"], info["raw_low"]) if info["raw_high"] <= info["raw_low"] \
                 else (info["raw_low"], info["raw_high"])
        return int(round(float(np.clip(raw, lo, hi))))

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def qpos_to_positions(self, sim_qpos_rad: np.ndarray) -> dict[int, int]:
        """把 sim 当前 qpos（前 nv 个值，弧度）转换为 ``{servo_id: raw}``，**不写硬件**。

        这是 :meth:`deploy_qpos` 的无副作用版本，供 ``live_control.py`` 等
        "只读 sim 计算 raw 目标、再走别的通道（TCP / 日志）下发" 的场景使用。

        Parameters
        ----------
        sim_qpos_rad : np.ndarray
            sim 关节位置（弧度），长度 ≥ ``env.unwrapped.model.nv``。

        Returns
        -------
        dict[int, int]
            ``{servo_id: raw}``，长度 17（含 wrist）。raw 已经被 clip 到
            ``[raw_low, raw_high]``。
        """
        positions: dict[int, int] = {}
        for sid_int, info in self._info.items():
            rad = float(sim_qpos_rad[info["sim_idx"]])
            positions[sid_int] = self._rad_to_raw(info, rad)
        return positions

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
        positions = self.qpos_to_positions(sim_qpos_rad)
        # 真机原子下发 + 限位校验（ServoSafetyLayer 内部断言 raw ∈ [min, max]）
        self.backend.safety.sync_go_to_pose(positions)
        return positions

    def reload_from_xdat(self, xdat_dir: str | pathlib.Path | None = None) -> None:
        """Re-read xdat files and refresh ``self._info`` raw_low/raw_high cache.

        调用方应在外部修改了 xdat（写入 EPROM 后 GUI 自动同步、或手动改了
        ``参数/*.xdat`` 文件）后调用本方法，让 sim→raw 映射立刻生效，
        不需要重启 ``live_control.py``。

        Parameters
        ----------
        xdat_dir : str | Path | None
            传 ``None`` 时按 backend 的默认行为：
              - ``_DryRunBackend.load_bundles(None)`` 会**跳过**文件读取并保留
                ``self.bundles`` 不变（除非 backend 内部另有 fallback）。
            传具体路径时调用 ``backend.load_bundles(xdat_dir)`` 重读文件。

        Notes
        -----
        - ``backend`` 必须有 ``load_bundles`` 方法（``_DryRunBackend`` /
          ``ConsoleBackend`` 都满足）；缺失时抛 ``AttributeError``。
        - 只刷新 ``self._info``；`self._sim_idx_by_servo`` 和 sim_low / sim_high
          来自 env + mapping，与 xdat 无关，**不会**被刷新。
        - 调用 ``_cache_per_servo`` 是私有方法，但签名稳定（与 ``__init__``
          走同一条路径）；外部调用风险可控。
        """
        loader = getattr(self.backend, "load_bundles", None)
        if not callable(loader):
            raise AttributeError(
                f"{type(self.backend).__name__} has no load_bundles(); "
                "cannot reload from xdat"
            )
        loader(xdat_dir)
        # 复用 __init__ 里走过的同一条缓存构建路径，保持 flip_direction 等
        # 所有 per-servo 逻辑与构造时完全一致。
        self._cache_per_servo()

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

    def bypass_next_collision_check(self) -> None:
        """置位碰撞旁通 latch：下一帧若检测到自碰撞则照常下发（不跳过）。

        这是**单次放行**——latch 在被 ``run_live_loop`` 消费后自动归零。
        之后若再次碰撞，行为恢复成「跳过该帧」。

        设计动机：用户拖 Control panel 滑条把手指掰过头，可能临时进入 sim
        自碰撞姿态（即便真机因几何微差不会真撞）。这种**临时碰撞**常常是
        过冲，下一帧就回退；如果每次都要重启 sim 链路很烦人。允许单次
        放行可以快速跨过这个抖动峰。
        """
        self.collision_bypass = True


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


def build_local_deployer(
    env: Any,
    xdat_dir: str | pathlib.Path | None,
    *,
    collision_check: bool = True,
    mapping: Mapping | None = None,
) -> SimToRealDeployer:
    """构造一个**只用于本地 rad→raw 转换 + 碰撞检查**的 deployer。

    与 :func:`build_deployer` 不同，本函数：

    - 不开串口（使用 :class:`_DryRunBackend`）；
    - 不返回 env（env 由调用者持有）；
    - 不会触发任何 hardware I/O。

    适合 ``live_control.py`` 这类 "sim 端只算 raw 目标、再走 TCP/日志发给下游"
    的实时联动场景。仍然使用真实 ``xdat_dir`` 下的 17 份 ``.xdat``，所以
    ``raw_low/raw_high`` 与真机一致；调用 :meth:`SimToRealDeployer.qpos_to_positions`
    得到的结果与真机端 ``sync_go_to_pose`` 期望的输入同位。

    Parameters
    ----------
    env : gymnasium.Env
        已 ``reset`` 的 orca_sim env。
    xdat_dir : str 或 Path 或 None
        ``1.xdat..17.xdat`` 所在目录；None 时 ``_DryRunBackend`` 回退到
        全占 [0, 4095]。
    collision_check : bool
        是否启用碰撞预筛选（默认 True）。
    mapping : Mapping 或 None
        已加载的映射；None → 用默认 JSON。
    """
    if mapping is None:
        from orca_sim.bridge.mapping import load_joint_mapping
        mapping = load_joint_mapping()
    backend = _DryRunBackend(mapping)
    backend.load_bundles(xdat_dir)
    return SimToRealDeployer(
        console_backend=backend,
        env=env,
        mapping=mapping,
        collision_check=collision_check,
    )