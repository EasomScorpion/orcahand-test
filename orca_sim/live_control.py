"""live_control.py — MuJoCo viewer + 实时 sim→真机 17 舵机联动。

把 ``orca_sim`` 仿真里的右手（v1, ``skin=False``）实时同步到 STS3215 真机：

- **拖 MuJoCo Control panel 滑条** → ``data.ctrl`` → 物理 step → ``data.qpos`` →
  每帧把 ``{servo_id: raw}`` 推给真机。
- 默认是 **dry-run 模式**：不开串口，仅打印 17 路 raw 到终端。
- 加 ``--port COMx`` 表示"真机在附近"（**不打开串口**——pyserial 是独占的，
  串口已由 ``servo_console.py`` GUI 持有）；这种情况下会起一个本地 TCP socket，
  把 raw 目标发给 GUI 端的 ``RemotePoseReceiver`` 线程，由 GUI 走
  ``ServoSafetyLayer.sync_go_to_pose`` 写入硬件。

**与 ``scripts/sim_to_real.py`` 的区别**：本脚本是**实时每帧**跟随 viewer；
``sim_to_real.py`` 是 batch 跑 N 步后退出。

**与 ``view_v1.py`` 的区别**：本脚本做 viewer **并且**驱动真机；
``view_v1.py`` 只做 viewer，不碰硬件。

人工校准 EPROM 限位（``--calibrate-limits``）：GUI 按下「应用 sim 限位到 EPROM」
按钮时会反向发 ``{"type": "request_limits"}``，本脚本收到后回
``{"type": "apply_limits", "limits": {...}}``。GUI 走 ``ServoSafetyLayer.write_eprom_register``
写 EPROM 寄存器 9/11，不经过 sim 端串口（**live_control 永远不开串口**）。

用法：

    # 默认 dry-run（无真机），打开 viewer + 每帧打印 17 路 raw
    python live_control.py --no-render --max-fps 5

    # 拖滑条 + 干跑（无 TCP，纯打印）
    python live_control.py

    # 实时联动真机：先在另一终端跑 GUI（带 --remote-tcp-port），
    # 再用 --port COMx 启动本脚本
    python FTServo_Python/test/servo_console.py --remote-tcp-port 8765
    python live_control.py --port COM5 --tcp-port 8765

    # 人工校准 EPROM 限位：跑 viewer，把每根手指拖到机械极限，
    # 然后按 GUI 的「应用 sim 限位到 EPROM」按钮
    python live_control.py --port COM5 --calibrate-limits
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping as MappingT, Sequence

import mujoco
import numpy as np

# 让直接 `python live_control.py` 也能 import orca_sim
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orca_sim import OrcaHandRight  # noqa: E402
from orca_sim.bridge import build_local_deployer, load_joint_mapping  # noqa: E402
from orca_sim.bridge.deploy import SimToRealDeployer  # noqa: E402

logger = logging.getLogger("live_control")


# ============================================================================
# CLI
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    """构造 argparse。"""
    parser = argparse.ArgumentParser(
        prog="live_control",
        description=(
            "实时把 MuJoCo 仿真（v1 right hand, skin=False）映射到 STS3215 真机。"
            "默认 dry-run：仅打印 17 路 raw；加 --port COMx 启动 TCP 推送（仍不开串口）。"
        ),
    )
    parser.add_argument(
        "--port",
        default=None,
        help="硬件哨兵（如 COM5）。本脚本永远不调用串口；仅作为启动信息 + 启用 TCP server 的开关。",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=1_000_000,
        help="日志打印用；不影响本脚本（串口由 servo_console.py GUI 持有）。",
    )
    parser.add_argument(
        "--tcp-port",
        type=int,
        default=8765,
        help="本地 TCP server 端口（仅当 --port 设置时启用）。",
    )
    parser.add_argument(
        "--tcp-send-timeout",
        type=float,
        default=0.05,
        help=(
            "sim 端 TCP send 单帧最长时间（秒）。GUI 处理慢时 sim 主循环"
            " 立即 drop client 并继续推下一帧，而不是阻塞。"
            " 默认 0.05（50 ms），约 1.5 倍 --max-fps=30 的帧间隔。"
        ),
    )
    parser.add_argument(
        "--no-collision-check",
        action="store_true",
        help="关闭每帧 sim 自碰撞预筛选（默认开）。",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=100,
        help="日志打印用；speed/acc 不放入 TCP 协议，由 GUI 的 SafetyLimits 默认值决定。",
    )
    parser.add_argument(
        "--acc",
        type=int,
        default=10,
        help="日志打印用。",
    )
    parser.add_argument(
        "--xdat-dir",
        type=pathlib.Path,
        default=REPO_ROOT.parent / "FTServo_Python" / "参数",
        help="1.xdat..17.xdat 所在目录；用于 raw_low/raw_high。",
    )
    parser.add_argument(
        "--hand",
        choices=["right"],
        default="right",
        help="目前仅支持右手。",
    )
    parser.add_argument(
        "--version",
        default="v1",
        choices=["v1"],
        help="orca_sim 模型版本（目前仅 v1 右手映射被 servo_joint_mapping.json 覆盖）。",
    )
    parser.add_argument(
        "--env",
        choices=["right"],
        default="right",
        help="env 类型。",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="不打开 MuJoCo viewer（headless），用于自动化 / 无显示环境。",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=30.0,
        help="每秒最多帧数（实际帧率受 render 与 mj_step 影响）。",
    )
    parser.add_argument(
        "--calibrate-limits",
        action="store_true",
        help=(
            "人工校准 EPROM 限位模式：仍跑 viewer，但允许 GUI 反向拉取 "
            "「当前 17 路 sim 姿态 → raw 目标」作为 EPROM min/max 写入建议。"
            "需要同时带 --port 启动 TCP server；GUI 端按「应用 sim 限位到 EPROM」按钮触发。"
        ),
    )
    return parser


# ============================================================================
# 环境 / 部署器工厂
# ============================================================================
def make_env(args: argparse.Namespace) -> Any:
    """构造并 reset orca_sim 右手 env。"""
    render_mode = None if args.no_render else "human"
    env_cls = OrcaHandRight  # 计划内只支持右手
    env = env_cls(version=args.version, skin=False, render_mode=render_mode)
    env.reset()
    return env


def build_deployer_for_live(
    env: Any,
    xdat_dir: pathlib.Path,
    *,
    collision_check: bool,
) -> SimToRealDeployer:
    """构造本地 dry-run deployer（不开串口）。

    包装 :func:`bridge.build_local_deployer`，把 plan 里"避免在 live_control 中
    直接 import 私有 ``_DryRunBackend``"的约束固化下来。
    """
    return build_local_deployer(
        env=env,
        xdat_dir=xdat_dir,
        collision_check=collision_check,
        mapping=load_joint_mapping(),
    )


# ============================================================================
# 统计 / 主循环 / TCP server
# ============================================================================
@dataclass
class LoopStats:
    """主循环每帧计数。"""

    frames: int = 0
    sent: int = 0
    dry_run_frames: int = 0
    collisions_skipped: int = 0
    disconnected_frames: int = 0
    # GUI「远程 reload xdat」按钮触发 sim 端刷新 _info 缓存的次数
    reload_xdat_served: int = 0
    # TCP send 因 TimeoutError 抛出的次数（GUI 卡死导致 sim 快速 drop client）
    send_timeouts: int = 0


def run_live_loop(
    env: Any,
    deployer: SimToRealDeployer,
    *,
    tcp_server: "PoseTcpServer | None",
    fps: float,
    max_frames: int | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    step_fn: Callable[..., None] | None = None,
    render_fn: Callable[[], None] | None = None,
) -> LoopStats:
    """每帧：render → mj_step → 碰撞检查 → 转换 / 发送。

    ``render_fn`` / ``step_fn`` / ``sleep_fn`` 可被注入，便于测试；
    默认分别走 ``env.render`` / ``mujoco.mj_step`` / ``time.sleep``。

    ``max_frames`` 给定时，跑满 N 帧后退出循环（用于测试 / 批量运行）；
    生产模式下为 None，靠 ``KeyboardInterrupt`` 终止。

    每帧把 ``positions`` 同步写到 ``tcp_server.last_positions``（如果存在）；
    GUI 的「应用 sim 限位到 EPROM」按钮通过 ``request_limits`` 拉这个缓存。
    """
    if sleep_fn is None:
        sleep_fn = time.sleep
    if step_fn is None:
        step_fn = mujoco.mj_step
    if render_fn is None:
        render_fn = env.render
    stats = LoopStats()
    period = 1.0 / max(fps, 1e-3)
    while True:
        if max_frames is not None and stats.frames >= max_frames:
            return stats
        render_fn()  # 把 Control panel 滑条编辑同步进 data.ctrl
        step_fn(env.model, env.data, nstep=env.frame_skip)

        # 碰撞检查（在 mj_step 之后）
        contacts = deployer.guard.self_contacts() if deployer.guard else []
        if contacts:
            if deployer.collision_bypass:
                # 单次放行 latch：用户主动按下「解除碰撞限制」按钮，
                # 本帧跳过碰撞检查 + 照样下发真机，latch 自动清零。
                # 若下一帧仍碰撞，照旧被跳过。
                deployer.collision_bypass = False
                logger.info(
                    "Collision bypass latch consumed (frame=%d, contacts=%d)",
                    stats.frames, len(contacts),
                )
            else:
                stats.collisions_skipped += 1
                logger.warning(
                    "Skipping frame due to %d self-contact(s): %s",
                    len(contacts),
                    [(c.body1, c.body2, round(c.dist, 5)) for c in contacts[:3]],
                )
                stats.frames += 1
                sleep_fn(max(0.0, period))
                continue

        qpos = env.unwrapped.data.qpos[: env.unwrapped.model.nv]
        positions = deployer.qpos_to_positions(qpos)

        # 缓存最新 17 路 raw：GUI 「应用 sim 限位」按钮拉这个值
        if tcp_server is not None:
            tcp_server.update_last_positions(positions)

        if tcp_server is None:
            stats.dry_run_frames += 1
            logger.info(
                "dry-run frame=%d positions=%s",
                stats.frames,
                {sid: positions[sid] for sid in sorted(positions)},
            )
        else:
            if tcp_server.broadcast(positions):
                stats.sent += 1
            else:
                stats.disconnected_frames += 1
                # 区分 send 超时（GUI 卡）vs 其它断连（pipe reset 等）。
                # PoseTcpServer.broadcast 在 catch 块里写 last_send_exc；
                # 我们读这个来分类计数。
                if tcp_server.last_send_exc is TimeoutError:
                    stats.send_timeouts += 1

        stats.frames += 1
        sleep_fn(max(0.0, period))


# ----------------------------------------------------------------------------
# TCP server
# ----------------------------------------------------------------------------
def encode_pose(positions: MappingT[int, int]) -> bytes:
    """把 ``{servo_id: raw}`` 编码为 JSON 行（newline-delimited）。

    JSON object 的 key 必须是字符串，因此显式 ``str(int(sid))``。
    """
    payload = {
        "positions": {
            str(int(sid)): int(raw)
            for sid, raw in positions.items()
        },
    }
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


class PoseTcpServer:
    """本地 TCP server，把每帧 ``{servo_id: raw}`` 推送给 GUI 端的 receiver。

    仅绑定 ``127.0.0.1``，只接受 1 个客户端（典型用法：``servo_console.py`` 的
    ``RemotePoseReceiver``）。无认证；不应用于公网/多用户。

    - ``broadcast(positions)`` 在无 client 或 send 失败时返回 ``False``，不阻塞 sim。
    - daemon accept 线程，新 client 替换旧 client（先关旧 socket）。
    - **反向通道**：GUI 可发回 JSON 命令（newline-delimited），目前识别
      ``{"type": "bypass_collision"}``（按一次解除下一次碰撞限制）和
      ``{"type": "request_limits"}``（GUI「应用 sim 限位到 EPROM」按钮触发）。
      解析失败 / 未知 type 不会断线程。
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        *,
        on_command: Callable[[dict[str, Any]], None] | None = None,
        send_timeout: float = 0.05,
    ) -> None:
        if host != "127.0.0.1":
            raise ValueError(
                f"PoseTcpServer 仅允许 localhost 绑定；收到 host={host!r}"
            )
        if not (1 <= int(port) <= 65535):
            raise ValueError(f"port 必须在 [1, 65535]；收到 {port}")
        if send_timeout <= 0:
            raise ValueError(f"send_timeout 必须 > 0 秒；收到 {send_timeout}")
        self.host = host
        self.port = port
        self._listener: socket.socket | None = None
        self._client: socket.socket | None = None
        self._client_read_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._accept_thread: threading.Thread | None = None
        self.frames_sent = 0
        self.connection_count = 0
        self.commands_received = 0
        self.limits_requests = 0
        self._on_command = on_command
        # 最近一帧的 17 路 raw 缓存：GUI 「应用 sim 限位」按钮拉这个值
        self._last_positions: dict[int, int] = {}
        # apply_limits payload 构造器（注入便于测试）；默认走 self._last_positions
        self._limits_provider: Callable[[], dict[int, int]] | None = None
        # — TCP 写侧 timeout —
        # 实测 GUI 端 sync_go_to_pose 需 20-50 ms / 帧；写侧 timeout 50 ms 时
        # sim 主循环遇到卡 GUI 立即 drop，1 s 内重连，不阻塞 viewer。
        self._send_timeout = float(send_timeout)
        # 上一次 broadcast 抛出的异常类型（None / TimeoutError / BrokenPipeError / 等）；
        # 主循环用于分类计数（区分 send_timeout vs 其它断连）。
        self.last_send_exc: type | None = None

    def update_last_positions(self, positions: MappingT[int, int]) -> None:
        """主循环每帧调用，写入 ``_last_positions``（GUI 拉取时读这个）。"""
        with self._lock:
            self._last_positions = {int(k): int(v) for k, v in positions.items()}

    def get_last_positions(self) -> dict[int, int]:
        """读 ``_last_positions`` 拷贝（GUI 拉取 limits 时使用）。"""
        with self._lock:
            return dict(self._last_positions)

    def set_limits_provider(
        self, provider: Callable[[], dict[int, int]] | None
    ) -> None:
        """注入 limits 构造器（默认走 ``last_positions``，可被测试覆盖）。"""
        self._limits_provider = provider

    def _build_apply_limits_payload(self) -> dict[str, Any]:
        """构造 ``{"type": "apply_limits", "limits": {sid: {min, max}, ...}}``。

        默认把 ``last_positions`` 作为「最小」+「最大」都是同一个 raw 的建议；
        GUI 端可逐个 Yes/No 确认。注入 ``limits_provider`` 时可换成更复杂的
        「采样轨迹」的最小/最大提取。
        """
        provider = self._limits_provider or self.get_last_positions
        positions = provider()
        limits: dict[str, dict[str, int]] = {}
        for sid, raw in sorted(positions.items()):
            limits[str(int(sid))] = {
                "min": int(raw),
                "max": int(raw),
            }
        return {"type": "apply_limits", "limits": limits}

    def start(self) -> None:
        """启动 accept 线程；可重复调用（幂等）。"""
        if self._accept_thread is not None and self._accept_thread.is_alive():
            return
        self._stop.clear()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(1)
        sock.settimeout(0.5)
        self._listener = sock
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="pose-tcp-accept", daemon=True
        )
        self._accept_thread.start()
        logger.info("PoseTcpServer listening on %s:%d", self.host, self.port)

    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                client, _addr = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            client.settimeout(2.0)
            # 写侧 timeout 覆盖 Python 对称的 settimeout（最后一次 settimeout 对
            # 后续 send 生效）；读侧 2 s 仍由 _client_read_loop 内的 recv 继承。
            # 当 GUI 端 sync_go_to_pose 慢于 send_timeout，sim 主循环在
            # ~send_timeout 后立即 TimeoutError → drop client → 下一帧重连。
            client.settimeout(self._send_timeout)
            with self._lock:
                old = self._client
                self._client = client
                self.connection_count += 1
            if old is not None:
                try:
                    old.close()
                except OSError:
                    pass
            # 启动反向读线程：处理 GUI 发来的 bypass 命令
            self._client_read_thread = threading.Thread(
                target=self._client_read_loop,
                args=(client,),
                name="pose-tcp-read",
                daemon=True,
            )
            self._client_read_thread.start()
            logger.info("TCP client connected (total=%d)", self.connection_count)

    def _client_read_loop(self, client: socket.socket) -> None:
        """读 GUI 发来的命令，newline-delimited JSON 行。

        与本 socket 相关：连接断开（read 返回空 / OSError）即退出循环。
        """
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = client.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                # EOF
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    logger.warning("Drop malformed command line: %r", line[:80])
                    continue
                if not isinstance(msg, dict):
                    logger.warning("Drop non-dict command: %r", msg)
                    continue
                self.commands_received += 1
                msg_type = msg.get("type")
                if msg_type == "request_limits":
                    # GUI 主动拉取当前 17 路 raw 作为 EPROM min/max 建议
                    self.limits_requests += 1
                    payload = self._build_apply_limits_payload()
                    try:
                        client.sendall(
                            (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
                        )
                    except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                        logger.warning(
                            "Failed to send apply_limits to GUI (%s); dropping client", exc,
                        )
                        with self._lock:
                            if self._client is client:
                                self._client = None
                        try:
                            client.close()
                        except OSError:
                            pass
                        break
                    logger.info(
                        "Served apply_limits to GUI (request #%d, %d servos)",
                        self.limits_requests, len(payload.get("limits", {})),
                    )
                    continue
                if self._on_command is not None:
                    try:
                        self._on_command(msg)
                    except Exception:  # noqa: BLE001
                        logger.exception("on_command callback raised")
        # socket 已断开 → 清掉 _client 引用（如果还是自己）
        with self._lock:
            if self._client is client:
                self._client = None

    def broadcast(self, positions: MappingT[int, int]) -> bool:
        """把当前帧发到当前 client；无 client 或 send 失败时返回 False。

        写侧使用 ``self._send_timeout``（默认 50 ms）——在 ``_accept_loop`` 中
        已对 client socket 设过两次 settimeout，最后一次的 send_timeout 对
        send 生效。当 GUI 端处理慢于 send_timeout 时，``sendall`` 在
        ~send_timeout 后抛 :class:`TimeoutError`，本方法 catch 后 drop client
        并返回 False，让主循环在下一帧继续推进而不是被阻塞。
        """
        payload = encode_pose(positions)
        with self._lock:
            client = self._client
        if client is None:
            # 不重置 last_send_exc：让最近一次失败状态持续可见，
            # 直到下一次 broadcast 成功。
            return False
        try:
            client.sendall(payload)
            self.frames_sent += 1
            self.last_send_exc = None  # 成功发送后才清掉最近一次的异常记录
            return True
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError) as exc:
            self.last_send_exc = type(exc)
            logger.warning("TCP send failed (%s); dropping client", exc)
            with self._lock:
                if self._client is client:
                    self._client = None
            try:
                client.close()
            except OSError:
                pass
            return False

    def close(self) -> None:
        """关闭 listener + 当前 client；唤醒 accept 线程。"""
        self._stop.set()
        with self._lock:
            client = self._client
            self._client = None
        if client is not None:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                client.close()
            except OSError:
                pass
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)
            self._accept_thread = None
        if self._client_read_thread is not None:
            # daemon 线程；close 通常会让 socket 关闭，recv 失败后自行退出
            self._client_read_thread.join(timeout=1.0)
            self._client_read_thread = None


# ============================================================================
# 入口
# ============================================================================
def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_summary(stats: LoopStats, args: argparse.Namespace) -> None:
    """打印收尾统计。"""
    print("\n========== live_control summary ==========", file=sys.stderr)
    print(f"  total frames            : {stats.frames}", file=sys.stderr)
    print(f"  sent (TCP)              : {stats.sent}", file=sys.stderr)
    print(f"  dry-run frames          : {stats.dry_run_frames}", file=sys.stderr)
    print(f"  collisions skipped      : {stats.collisions_skipped}", file=sys.stderr)
    print(f"  disconnected frames     : {stats.disconnected_frames}", file=sys.stderr)
    if stats.send_timeouts > 0:
        # 只在有 send 超时时打印，避免无谓噪音
        print(f"  send_timeouts           : {stats.send_timeouts}", file=sys.stderr)
    print(f"  reload_xdat served      : {stats.reload_xdat_served}", file=sys.stderr)
    print(f"  port (sentinel)         : {args.port or '(dry-run)'}", file=sys.stderr)
    print(f"  tcp-port                : {args.tcp_port}", file=sys.stderr)
    if args.port:
        print(f"  tcp-send-timeout (s)    : {args.tcp_send_timeout}", file=sys.stderr)
    print(f"  calibrate-limits mode   : {args.calibrate_limits}", file=sys.stderr)
    print("==========================================", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    _configure_logging()
    args = build_parser().parse_args(argv)
    if args.max_fps <= 0:
        raise SystemExit(f"--max-fps 必须 > 0；收到 {args.max_fps}")
    if args.calibrate_limits and args.port is None:
        # calibrate-limits 需要 TCP server 转发给 GUI；强制要 --port
        raise SystemExit(
            "--calibrate-limits 必须同时带 --port COMx "
            "（GUI 端按「应用 sim 限位到 EPROM」按钮拉取限位建议）"
        )

    env: Any = None
    tcp_server: PoseTcpServer | None = None
    stats = LoopStats()
    try:
        env = make_env(args)
        deployer = build_deployer_for_live(
            env,
            xdat_dir=args.xdat_dir,
            collision_check=not args.no_collision_check,
        )

        if args.port is not None:
            def _on_cmd(msg: dict[str, Any]) -> None:
                """GUI 发来的反向命令分发。"""
                msg_type = msg.get("type")
                if msg_type == "bypass_collision":
                    deployer.bypass_next_collision_check()
                    logger.info("Received bypass_collision command from GUI")
                elif msg_type == "reload_xdat":
                    # GUI 改了 xdat 后按「🔄 重新加载 xdat 映射」按钮 →
                    # sim 端重新读取 xdat 并刷新 _info raw_high/raw_low。
                    # fire-and-forget；错误只打日志不影响主循环。
                    try:
                        deployer.reload_from_xdat(args.xdat_dir)
                        stats.reload_xdat_served += 1
                        logger.info(
                            "[tcp] reload_xdat → re-read %s "
                            "(total served: %d)",
                            args.xdat_dir, stats.reload_xdat_served,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[tcp] reload_xdat failed: %s", exc,
                        )
                else:
                    logger.warning("Unknown command type from GUI: %r", msg_type)
            tcp_server = PoseTcpServer(
                port=args.tcp_port,
                on_command=_on_cmd,
                send_timeout=args.tcp_send_timeout,
            )
            tcp_server.start()
            if args.calibrate_limits:
                logger.info(
                    "Calibrate-limits mode: GUI 可按「应用 sim 限位到 EPROM」按钮"
                    "拉取当前 17 路 raw → 写入 EPROM min/max。"
                )
            logger.info(
                "Hardware sentinel %s; TCP server on 127.0.0.1:%d "
                "(waiting for GUI receiver).",
                args.port,
                args.tcp_port,
            )
        else:
            logger.info(
                "Dry-run mode (no --port); positions will be logged, "
                "no TCP server, no serial I/O."
            )

        logger.info(
            "Config: hand=%s version=%s skin=False xdat_dir=%s "
            "collision_check=%s baud=%s speed=%s acc=%s max_fps=%s "
            "tcp_send_timeout=%s",
            args.hand, args.version, args.xdat_dir,
            not args.no_collision_check, args.baud, args.speed, args.acc,
            args.max_fps, args.tcp_send_timeout,
        )

        stats = run_live_loop(
            env,
            deployer,
            tcp_server=tcp_server,
            fps=args.max_fps,
        )
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    finally:
        if tcp_server is not None:
            tcp_server.close()
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                logger.exception("env.close() failed")
        _print_summary(stats, args)


if __name__ == "__main__":
    raise SystemExit(main())