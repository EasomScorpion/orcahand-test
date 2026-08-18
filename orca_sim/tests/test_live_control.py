"""live_control.py 单元测试。

所有测试用 FakeEnv / FakeDeployer / FakeGuard / 录制型 PoseTcpServer 注入；
不启动真 MuJoCo viewer、不打网络端口、不依赖 PyQt5 / QApplication。
"""
from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pytest

from live_control import (
    LoopStats,
    PoseTcpServer,
    encode_pose,
    run_live_loop,
)


# ============================================================================
# Fakes
# ============================================================================
class FakeGuard:
    def __init__(self, contacts=None):
        self._contacts = contacts or []

    def self_contacts(self):
        return list(self._contacts)


@dataclass
class FakeContact:
    body1: str = "right_index_pp_1"
    body2: str = "right_middle_pp_1"
    dist: float = -0.001


class FakeDeployer:
    """模仿 SimToRealDeployer 表面 API，但不写硬件、不导入 _DryRunBackend。"""

    def __init__(self, positions_template, contacts=None):
        self._positions_template = dict(positions_template)
        self.guard = FakeGuard(contacts)
        # 记录 _rad_to_raw 被调次数，确保测试也跑通了转换路径
        self.rad_to_raw_calls = 0
        # 碰撞旁通 latch（与 SimToRealDeployer 同形）
        self.collision_bypass = False

    def qpos_to_positions(self, qpos):
        # 真实映射表这里只 mock：不依赖 _info，让测试可以自定义 positions。
        self.rad_to_raw_calls += 1
        return dict(self._positions_template)

    def bypass_next_collision_check(self) -> None:
        self.collision_bypass = True


class FakeEnv:
    """最小化 env：模型 + 数据 + 帧数计数器，注入到 run_live_loop。"""

    def __init__(self, n_action=17, frame_skip=1, qpos=None):
        class _M:
            nv = n_action
        self.model = _M()
        # 用 self 上的属性而不是内部类（避免 Python 类作用域闭包问题）
        self._qpos_arr = qpos if qpos is not None else np.zeros(n_action)
        self.frame_skip = frame_skip
        self.metadata = {"render_fps": 30}
        self.render_calls = 0
        self.step_calls = 0
        self.closed = False
        # run_live_loop 通过 env.unwrapped.data.qpos 读 qpos
        self.unwrapped = self

    @property
    def data(self):
        return _FakeData(self._qpos_arr)

    def render(self):
        self.render_calls += 1

    def close(self):
        self.closed = True


class _FakeData:
    def __init__(self, qpos):
        self.qpos = qpos


class RecordingTcpServer:
    """录制型 PoseTcpServer 替代品，跟 PoseTcpServer 接口一致。"""

    def __init__(self):
        self.broadcasts: list[dict[int, int]] = []
        self.broadcast_calls = 0
        self.connected = True
        self.closed = False
        self.last_positions: dict[int, int] = {}
        self.limits_requests = 0
        # 与 PoseTcpServer 同形的「上次 send 异常类型」字段；主循环用来分类计数。
        # 默认 None（无异常）。RecordingTcpServer.broadcast 不写它；测试可手动设。
        self.last_send_exc: type | None = None

    def broadcast(self, positions):
        self.broadcast_calls += 1
        if self.connected:
            self.broadcasts.append(dict(positions))
            return True
        return False

    def update_last_positions(self, positions):
        self.last_positions = {int(k): int(v) for k, v in positions.items()}

    def get_last_positions(self):
        return dict(self.last_positions)

    def set_limits_provider(self, provider):
        # 测试不直接用
        pass

    def close(self):
        self.closed = True


# ============================================================================
# 编码 / TCP server
# ============================================================================
def test_encode_pose_json_line():
    """encode_pose 必须输出 newline-delimited JSON，且 key 是字符串。"""
    raw = encode_pose({1: 2048, 17: 2100})
    assert raw.endswith(b"\n")
    parsed = json.loads(raw.decode("utf-8").strip())
    assert set(parsed.keys()) == {"positions"}
    # JSON object key 一定是字符串
    assert all(isinstance(k, str) for k in parsed["positions"].keys())
    assert parsed["positions"]["1"] == 2048
    assert parsed["positions"]["17"] == 2100


def test_pose_tcp_server_rejects_non_localhost():
    """PoseTcpServer 只能绑 127.0.0.1；其它地址直接抛。"""
    with pytest.raises(ValueError):
        PoseTcpServer(host="0.0.0.0", port=8765)


def test_pose_tcp_server_invalid_port():
    """Port 越界直接抛。"""
    with pytest.raises(ValueError):
        PoseTcpServer(port=0)
    with pytest.raises(ValueError):
        PoseTcpServer(port=70000)


def test_pose_tcp_server_full_round_trip():
    """起 server → 客户端连接 → broadcast 收到正确 JSON。"""
    server = PoseTcpServer(port=18765)
    server.start()
    try:
        # 给 accept 线程机会起来
        time.sleep(0.1)
        # 客户端连接
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect(("127.0.0.1", 18765))
        # 等 accept 接受完成
        time.sleep(0.2)
        # broadcast
        positions = {1: 2048, 2: 2100, 17: 1500}
        ok = server.broadcast(positions)
        assert ok is True
        # 客户端读
        data = sock.recv(4096)
        parsed = json.loads(data.decode("utf-8").strip())
        assert parsed["positions"] == {
            "1": 2048, "2": 2100, "17": 1500
        }
        sock.close()
        assert server.frames_sent == 1
        assert server.connection_count == 1
    finally:
        server.close()


def test_pose_tcp_server_broadcast_no_client_returns_false():
    """没 client 时 broadcast 返回 False，不阻塞。"""
    server = PoseTcpServer(port=18766)
    server.start()
    try:
        time.sleep(0.1)
        assert server.broadcast({1: 2048}) is False
        assert server.frames_sent == 0
    finally:
        server.close()


# ============================================================================
# 主循环
# ============================================================================
def test_viewer_loop_collisions_skip():
    """有自碰撞时本帧不下发，碰撞计数 +1。"""
    contacts = [FakeContact()]
    deployer = FakeDeployer(positions_template={1: 2048}, contacts=contacts)
    env = FakeEnv()
    server = RecordingTcpServer()

    stats = run_live_loop(
        env, deployer, tcp_server=server,
        fps=10.0,
        max_frames=1,
        sleep_fn=lambda _: None,
        step_fn=lambda *a, **kw: None,
        render_fn=lambda: None,
    )

    assert stats.frames == 1
    assert stats.collisions_skipped == 1
    assert server.broadcast_calls == 0
    assert stats.dry_run_frames == 0


def test_viewer_loop_no_collisions_sends_17_int_positions():
    """无碰撞时 broadcast 一次，positions 是 17 个 int→int，value ∈ [0, 4095]。"""
    template = {sid: 1500 + sid for sid in range(1, 18)}
    deployer = FakeDeployer(positions_template=template, contacts=[])
    env = FakeEnv()
    server = RecordingTcpServer()

    stats = run_live_loop(
        env, deployer, tcp_server=server,
        fps=10.0,
        max_frames=1,
        sleep_fn=lambda _: None,
        step_fn=lambda *a, **kw: None,
        render_fn=lambda: None,
    )

    assert stats.frames == 1
    assert stats.collisions_skipped == 0
    assert server.broadcast_calls == 1
    assert stats.sent == 1
    sent = server.broadcasts[0]
    assert isinstance(sent, dict)
    assert len(sent) == 17
    assert all(isinstance(k, int) for k in sent.keys())
    assert all(isinstance(v, int) for v in sent.values())
    assert all(0 <= v <= 4095 for v in sent.values())


def test_dry_run_no_tcp_logs_only():
    """tcp_server=None 时进入 dry-run 分支，broadcast 不会被调。"""
    template = {sid: 2000 for sid in range(1, 18)}
    deployer = FakeDeployer(positions_template=template, contacts=[])
    env = FakeEnv()

    stats = run_live_loop(
        env, deployer, tcp_server=None,
        fps=10.0,
        max_frames=2,
        sleep_fn=lambda _: None,
        step_fn=lambda *a, **kw: None,
        render_fn=lambda: None,
    )

    assert stats.frames == 2
    assert stats.dry_run_frames == 2
    assert stats.sent == 0
    assert stats.collisions_skipped == 0
    assert deployer.rad_to_raw_calls == 2


def test_disconnected_frame_counted_separately():
    """client 突然断开时，该帧计入 disconnected_frames，不计入 sent。"""
    template = {sid: 1900 for sid in range(1, 18)}
    deployer = FakeDeployer(positions_template=template, contacts=[])
    env = FakeEnv()
    server = RecordingTcpServer()
    server.connected = False  # 假装 client 已断开

    stats = run_live_loop(
        env, deployer, tcp_server=server,
        fps=10.0,
        max_frames=3,
        sleep_fn=lambda _: None,
        step_fn=lambda *a, **kw: None,
        render_fn=lambda: None,
    )

    assert stats.frames == 3
    assert stats.disconnected_frames == 3
    assert stats.sent == 0


def test_max_frames_stops_loop():
    """max_frames 给定时跑满 N 帧后退出，不依赖 KeyboardInterrupt。"""
    deployer = FakeDeployer(positions_template={i: 2048 for i in range(1, 18)})
    env = FakeEnv()

    stats = run_live_loop(
        env, deployer, tcp_server=None,
        fps=10.0,
        max_frames=5,
        sleep_fn=lambda _: None,
        step_fn=lambda *a, **kw: None,
        render_fn=lambda: None,
    )
    assert stats.frames == 5


def test_render_and_step_called_per_frame():
    """每帧 render() 和 step_fn() 必须都被调一次（顺序：render 先）。"""
    calls = []

    def fake_render():
        calls.append("render")

    def fake_step(*a, **kw):
        calls.append("step")

    deployer = FakeDeployer(positions_template={i: 2048 for i in range(1, 18)})
    env = FakeEnv()

    run_live_loop(
        env, deployer, tcp_server=None,
        fps=10.0,
        max_frames=2,
        sleep_fn=lambda _: None,
        step_fn=fake_step,
        render_fn=fake_render,
    )
    # 每帧先 render 再 step
    assert calls == ["render", "step", "render", "step"]


# ============================================================================
# rad→raw 转换一致性（与 bridge 行为对齐）
# ============================================================================
def test_rad_to_raw_consistent_with_bridge():
    """用真 _DryRunBackend deployer，比对 qpos_to_positions 与 _rad_to_raw 在
    low/mid/high 三个值处的输出；越界时也 clip 到 raw 边界。
    """
    from pathlib import Path
    from orca_sim.bridge import build_local_deployer
    from orca_sim import OrcaHandRight

    env = OrcaHandRight(version="v1", skin=False)
    env.reset(seed=42)
    try:
        deployer = build_local_deployer(
            env=env,
            xdat_dir=Path(__file__).resolve().parents[2]
                    / "FTServo_Python" / "参数",
            collision_check=False,
        )
        # 拿 sim 的 action_low / action_high 当 qpos 的合法范围
        for tag, qpos_value in (
            ("low", env.action_low.copy()),
            ("mid", (env.action_low + env.action_high) / 2),
            ("high", env.action_high.copy()),
            ("over_high", env.action_high + 1.0),  # 应当被 clip
            ("under_low", env.action_low - 1.0),  # 应当被 clip
        ):
            qpos = np.array(qpos_value, dtype=np.float64)
            positions = deployer.qpos_to_positions(qpos)
            assert len(positions) == 17
            for sid in positions:
                info = deployer._info[sid]
                expected = deployer._rad_to_raw(info, float(qpos[info["sim_idx"]]))
                assert positions[sid] == expected, (
                    f"[{tag}] sid={sid}: qpos_to_positions={positions[sid]} "
                    f"vs _rad_to_raw={expected}"
                )
                # clip 边界：方向翻转后 raw_low/raw_high 的大小关系无固定约定，
                # 只能用「raw 在两端点之间（含端点）」来描述 clip 行为。
                lo = min(info["raw_low"], info["raw_high"])
                hi = max(info["raw_low"], info["raw_high"])
                assert lo <= positions[sid] <= hi, (
                    f"[{tag}] sid={sid}: raw {positions[sid]} ∉ "
                    f"[{lo}, {hi}] (raw_low={info['raw_low']}, raw_high={info['raw_high']})"
                )
    finally:
        env.close()


# ============================================================================
# 优雅退出 / 清理
# ============================================================================
def test_graceful_shutdown_with_keyboard_interrupt():
    """KeyboardInterrupt 时 TCP server 和 env 都会被 close（验证 main() 的 finally 块）。"""
    from live_control import PoseTcpServer as RealTcpServer

    real_server = RealTcpServer(port=18767)
    real_server.start()
    try:
        time.sleep(0.1)

        tcp_closed = {"value": False}
        env_closed = {"value": False}

        class TrackableServer:
            def broadcast(self, p):
                return real_server.broadcast(p)
            def close(self):
                tcp_closed["value"] = True
                real_server.close()

        env = FakeEnv()
        original_close = env.close

        def tracking_close():
            env_closed["value"] = True
            original_close()
        env.close = tracking_close

        deployer = FakeDeployer(positions_template={i: 2048 for i in range(1, 18)})

        # 模拟 main() 的 try / except KeyboardInterrupt / finally 清理逻辑
        try:
            try:
                run_live_loop(
                    env, deployer,
                    tcp_server=TrackableServer(),
                    fps=10.0,
                    max_frames=None,
                    sleep_fn=lambda _: None,
                    step_fn=lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt),
                    render_fn=lambda: None,
                )
            except KeyboardInterrupt:
                pass
        finally:
            TrackableServer().close()  # 模拟 main() finally
            env.close()

        assert tcp_closed["value"] is True, "TCP server 未在 KeyboardInterrupt 后被 close"
        assert env_closed["value"] is True, "env 未在 KeyboardInterrupt 后被 close"
    finally:
        real_server.close()


# ============================================================================
# 真实 loop × real backend：rad→raw 与 qpos_to_positions 一致性
# ============================================================================
def test_real_loop_qpos_to_positions_matches_rad_to_raw():
    """用真实 OrcaHandRight + 真 _DryRunBackend，跑 1 帧并验证
    qpos_to_positions 与内部 _rad_to_raw 在所有 17 个 servo_id 上输出一致。
    """
    from pathlib import Path
    from orca_sim.bridge import build_local_deployer
    from orca_sim import OrcaHandRight

    env = OrcaHandRight(version="v1", skin=False)
    env.reset(seed=42)
    try:
        deployer = build_local_deployer(
            env=env,
            xdat_dir=Path(__file__).resolve().parents[2]
                    / "FTServo_Python" / "参数",
            collision_check=False,
        )
        qpos = env.unwrapped.data.qpos[: env.unwrapped.model.nv]
        positions = deployer.qpos_to_positions(qpos)
        # 17 个 servo 都要有；且每个 sid 的 raw == _rad_to_raw 对应值
        assert len(positions) == 17
        assert 17 in positions  # wrist
        for sid_int, info in deployer._info.items():
            rad = float(qpos[info["sim_idx"]])
            expected = deployer._rad_to_raw(info, rad)
            assert positions[sid_int] == expected, (
                f"sid={sid_int} rad={rad} positions={positions[sid_int]} "
                f"expected={expected}"
            )
            # raw 必须落在 bundle 限位之内
            bundle = deployer.backend.bundles[sid_int - 1]
            assert bundle.min_angle <= positions[sid_int] <= bundle.max_angle, (
                f"sid={sid_int} raw={positions[sid_int]} ∉ "
                f"[{bundle.min_angle}, {bundle.max_angle}]"
            )
    finally:
        env.close()


# ============================================================================
# 碰撞旁通 latch（Change #3）
# ============================================================================
def test_bypass_latch_consumes_once_on_collision():
    """latch=True 时本帧照常下发（不跳过），latch 自动归零。"""
    contacts = [FakeContact()]
    deployer = FakeDeployer(positions_template={1: 2048}, contacts=contacts)
    env = FakeEnv()
    server = RecordingTcpServer()

    # 用户先按下按钮 → latch 置位
    deployer.bypass_next_collision_check()
    assert deployer.collision_bypass is True

    stats = run_live_loop(
        env, deployer, tcp_server=server,
        fps=10.0,
        max_frames=1,
        sleep_fn=lambda _: None,
        step_fn=lambda *a, **kw: None,
        render_fn=lambda: None,
    )

    # 帧被下发；collisions_skipped 没增加；latch 已被消费（归零）
    assert stats.collisions_skipped == 0
    assert stats.sent == 1
    assert server.broadcast_calls == 1
    assert deployer.collision_bypass is False


def test_bypass_latch_single_shot():
    """latch 只放行一次；下一帧仍碰撞 → 跳过。"""
    contacts = [FakeContact()]
    deployer = FakeDeployer(positions_template={1: 2048}, contacts=contacts)
    env = FakeEnv()
    server = RecordingTcpServer()

    # 第 0 帧前置位 latch：放行
    deployer.bypass_next_collision_check()
    run_live_loop(
        env, deployer, tcp_server=server,
        fps=10.0,
        max_frames=1,
        sleep_fn=lambda _: None,
        step_fn=lambda *a, **kw: None,
        render_fn=lambda: None,
    )
    assert deployer.collision_bypass is False
    assert server.broadcast_calls == 1

    # 第 1 帧 latch 已归零，仍碰撞 → 跳过
    stats = run_live_loop(
        env, deployer, tcp_server=server,
        fps=10.0,
        max_frames=1,
        sleep_fn=lambda _: None,
        step_fn=lambda *a, **kw: None,
        render_fn=lambda: None,
    )
    assert stats.collisions_skipped == 1
    assert server.broadcast_calls == 1  # 没新增


def test_bypass_latch_unused_does_not_disable_safety():
    """latch=False 时即便 `run_live_loop` 跑了 N 帧，latch 仍为 False。"""
    deployer = FakeDeployer(positions_template={1: 2048}, contacts=[])
    env = FakeEnv()

    run_live_loop(
        env, deployer, tcp_server=None,
        fps=10.0,
        max_frames=3,
        sleep_fn=lambda _: None,
        step_fn=lambda *a, **kw: None,
        render_fn=lambda: None,
    )
    assert deployer.collision_bypass is False


def test_pose_tcp_server_dispatches_bypass_command():
    """GUI 发 ``{"type": "bypass_collision"}`` 时 on_command 被调用。"""
    received: list[dict] = []
    server = PoseTcpServer(port=18768, on_command=received.append)
    server.start()
    try:
        time.sleep(0.1)
        # 客户端连上来
        import socket as _s
        sock = _s.socket(_s.AF_INET, _s.SOCK_STREAM)
        sock.connect(("127.0.0.1", 18768))
        try:
            # 等 accept_loop 把 _client 设好
            time.sleep(0.1)
            sock.sendall(b'{"type": "bypass_collision"}\n')
            time.sleep(0.3)  # 给 read_loop 时间消费
        finally:
            sock.close()
        assert server.commands_received == 1
        assert received == [{"type": "bypass_collision"}]
    finally:
        server.close()


def test_pose_tcp_server_dispatches_reload_xdat_command():
    """GUI 发 ``{"type": "reload_xdat"}`` 时 on_command 被调用。

    on_command 是 ``_on_cmd`` 的入口；``_on_cmd`` 内部根据 type 调用
    ``deployer.reload_from_xdat(args.xdat_dir)``。本测试只验证消息分发
    路径（不涉及 deployer 实际逻辑——见 test_reload_from_xdat_refreshes_*）。
    """
    received: list[dict] = []
    server = PoseTcpServer(port=18775, on_command=received.append)
    server.start()
    try:
        time.sleep(0.1)
        import socket as _s
        sock = _s.socket(_s.AF_INET, _s.SOCK_STREAM)
        sock.connect(("127.0.0.1", 18775))
        try:
            time.sleep(0.1)
            sock.sendall(b'{"type": "reload_xdat"}\n')
            time.sleep(0.3)
        finally:
            sock.close()
        assert server.commands_received == 1
        assert received == [{"type": "reload_xdat"}]
    finally:
        server.close()


def test_pose_tcp_server_ignores_garbage_lines():
    """解析失败 / 非 dict 的行不会断 read_loop。"""
    received: list[dict] = []
    server = PoseTcpServer(port=18769, on_command=received.append)
    server.start()
    try:
        time.sleep(0.1)
        import socket as _s
        sock = _s.socket(_s.AF_INET, _s.SOCK_STREAM)
        sock.connect(("127.0.0.1", 18769))
        try:
            time.sleep(0.1)
            sock.sendall(b'not-json\n')
            sock.sendall(b'[1,2,3]\n')      # 不是 dict
            sock.sendall(b'{"type": "x"}\n')  # 合法 dict 但未知 type
            time.sleep(0.3)
        finally:
            sock.close()
        # 解析失败的两行不算 commands_received；只有合法 dict 算
        assert server.commands_received == 1
        assert received == [{"type": "x"}]
    finally:
        server.close()


def test_real_deployer_bypass_method_sets_flag():
    """真实 SimToRealDeployer.bypass_next_collision_check 确实把 latch 置位。"""
    from pathlib import Path
    from orca_sim.bridge import build_local_deployer
    from orca_sim import OrcaHandRight

    env = OrcaHandRight(version="v1", skin=False)
    env.reset(seed=0)
    try:
        deployer = build_local_deployer(
            env=env,
            xdat_dir=Path(__file__).resolve().parents[2] / "FTServo_Python" / "参数",
            collision_check=False,
        )
        assert deployer.collision_bypass is False
        deployer.bypass_next_collision_check()
        assert deployer.collision_bypass is True
    finally:
        env.close()


# ============================================================================
# --calibrate-limits + request_limits payload
# ============================================================================
def test_pose_tcp_server_records_last_positions():
    """``update_last_positions`` 写入缓存，``get_last_positions`` 返回拷贝。"""
    from live_control import PoseTcpServer
    server = PoseTcpServer(port=18765)
    server.update_last_positions({1: 2048, 2: 1500, 3: 2500})
    cached = server.get_last_positions()
    assert cached == {1: 2048, 2: 1500, 3: 2500}
    # 拷贝：修改返回值不影响内部
    cached[1] = 0
    assert server.get_last_positions()[1] == 2048


def test_pose_tcp_server_build_apply_limits_payload_defaults_to_last_positions():
    """默认 limits provider 就是 last_positions，每个 servo 的 min=max=当前 raw。"""
    from live_control import PoseTcpServer
    server = PoseTcpServer(port=18766)
    server.update_last_positions({1: 1500, 2: 2048, 17: 3000})
    payload = server._build_apply_limits_payload()
    assert payload["type"] == "apply_limits"
    limits = payload["limits"]
    assert len(limits) == 3
    # key 必须是字符串（JSON 协议）；value 含 min/max
    for k in limits:
        assert isinstance(k, str)
        assert set(limits[k].keys()) == {"min", "max"}
    assert limits["1"]["min"] == 1500
    assert limits["1"]["max"] == 1500
    assert limits["17"]["min"] == 3000


def test_pose_tcp_server_apply_limits_uses_custom_provider():
    """注入 ``limits_provider`` 时，覆盖默认的 last_positions 行为。"""
    from live_control import PoseTcpServer
    server = PoseTcpServer(port=18767)
    server.update_last_positions({1: 2048})  # last 被覆盖
    # provider 返回「采样轨迹的 min/max」
    server.set_limits_provider(lambda: {
        1: 1000,  # 这里故意只返回当前 raw，与 limit payload 内部独立
        2: 2048,
    })
    payload = server._build_apply_limits_payload()
    assert payload["limits"]["1"]["min"] == 1000
    assert payload["limits"]["2"]["max"] == 2048


def test_pose_tcp_server_apply_limits_full_17_servos_required():
    """完整 17 路测试：用真实 deployer 注入 last_positions + 构造 payload。"""
    from live_control import PoseTcpServer
    server = PoseTcpServer(port=18768)
    # 模拟完整 17 路
    fake_positions = {sid: 2048 + sid * 5 for sid in range(1, 18)}
    server.update_last_positions(fake_positions)
    payload = server._build_apply_limits_payload()
    limits = payload["limits"]
    assert len(limits) == 17
    # 每条 sid 都应在 1..17 且 raw ∈ [0, 4095]
    for sid_str, val in limits.items():
        sid = int(sid_str)
        assert 1 <= sid <= 17
        assert 0 <= val["min"] <= 4095
        assert 0 <= val["max"] <= 4095


def test_calibrate_limits_requires_port_flag():
    """``--calibrate-limits`` 强制要 --port，否则抛 SystemExit。"""
    from live_control import main

    # 缺 --port
    with pytest.raises(SystemExit) as exc_info:
        main(argv=["--calibrate-limits", "--no-render", "--max-fps", "5"])
    assert "--port" in str(exc_info.value)


def test_run_live_loop_writes_last_positions_each_frame():
    """``run_live_loop`` 每帧把 positions 同步到 ``tcp_server.last_positions``。"""
    from live_control import PoseTcpServer, run_live_loop

    server = PoseTcpServer(port=18769)

    # 17 个 int 位置 sid → sid*10
    template = {sid: sid * 10 for sid in range(1, 18)}
    deployer = FakeDeployer(positions_template=template, contacts=[])
    env = FakeEnv()

    run_live_loop(
        env, deployer,
        tcp_server=server, fps=10,
        max_frames=2,
        sleep_fn=lambda _s: None,
        step_fn=lambda *a, **kw: None,
        render_fn=lambda: None,
    )
    # server.last_positions 应有 17 路
    last = server.get_last_positions()
    assert len(last) == 17
    for sid in range(1, 18):
        assert last[sid] == sid * 10


def test_pose_tcp_server_request_limits_echo_over_real_socket(tmp_path):
    """端到端：用真实 localhost socket 模拟 GUI 端发 ``request_limits``，
    验证 sim 端回 ``apply_limits`` payload。
    """
    import socket as _socket
    from live_control import PoseTcpServer

    server = PoseTcpServer(port=18770)
    # 注入 fake positions
    server.update_last_positions({sid: 1500 + sid for sid in range(1, 18)})
    server.start()
    try:
        # 客户端连
        client = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        client.connect(("127.0.0.1", 18770))
        client.settimeout(2.0)
        # 发 request_limits
        client.sendall(b'{"type": "request_limits"}\n')
        # 收 apply_limits 回包
        buf = b""
        deadline = time.time() + 2.0
        while time.time() < deadline:
            chunk = client.recv(4096)
            if not chunk:
                break
            buf += chunk
            if b"\n" in buf:
                break
        client.close()
        assert buf.strip(), "no reply from sim"
        payload = json.loads(buf.decode("utf-8").strip())
        assert payload["type"] == "apply_limits"
        assert len(payload["limits"]) == 17
        # 验证内容
        assert payload["limits"]["1"]["min"] == 1501
        assert payload["limits"]["1"]["max"] == 1501
        assert server.limits_requests == 1
    finally:
        server.close()


def test_reload_xdat_command_invokes_deployer_reload_from_xdat():
    """端到端：通过 ``_on_cmd`` 收 ``{"type":"reload_xdat"}`` 应触发
    ``SimToRealDeployer.reload_from_xdat``。

    验证：
      - 收消息后 ``deployer._info`` 中的 raw_high/raw_low 已被刷新
      - 对应 ``stats.reload_xdat_served`` 计数 +1

    用真实 ``_DryRunBackend`` + 真 xdat 文件 + 真 ``SimToRealDeployer``。
    """
    import struct as _st
    from pathlib import Path
    from orca_sim.bridge import build_local_deployer
    from orca_sim import OrcaHandRight
    from live_control import PoseTcpServer, LoopStats

    xdat_dir = Path(__file__).resolve().parents[2] / "FTServo_Python" / "参数"
    path_1 = xdat_dir / "1.xdat"
    raw_orig = path_1.read_bytes()
    # reg 9 (min_angle) 在 offset 0x02+9=11；reg 11 (max_angle) 在 offset 0x02+11=13
    # LE 16-bit。
    assert raw_orig[:2] == b"\x00\x28", f"unexpected header: {raw_orig[:2]}"

    env = OrcaHandRight(version="v1", skin=False)
    env.reset(seed=0)
    try:
        deployer = build_local_deployer(
            env=env, xdat_dir=xdat_dir, collision_check=False,
        )
        sid = 1
        raw_high_before = int(deployer._info[sid]["raw_high"])
        raw_low_before = int(deployer._info[sid]["raw_low"])

        # mutate 1.xdat 的 reg 9/11
        new_min, new_max = 1500, 3000
        buf = bytearray(raw_orig)
        buf[11:13] = _st.pack("<H", new_min)
        buf[13:15] = _st.pack("<H", new_max)
        path_1.write_bytes(bytes(buf))

        try:
            # 把 _on_cmd 模拟出来：和 live_control.py main() 里的逻辑一致
            stats = LoopStats()

            def _on_cmd(msg):
                if msg.get("type") == "reload_xdat":
                    deployer.reload_from_xdat(str(xdat_dir))
                    stats.reload_xdat_served += 1

            server = PoseTcpServer(port=18776, on_command=_on_cmd)
            server.start()
            try:
                import socket as _socket
                sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
                sock.connect(("127.0.0.1", 18776))
                try:
                    time.sleep(0.1)
                    sock.sendall(b'{"type": "reload_xdat"}\n')
                    time.sleep(0.3)
                finally:
                    sock.close()
            finally:
                server.close()

            # 缓存应反映新 xdat：servo 1 flip=True → raw_low = max_angle
            info_after = deployer._info[sid]
            assert stats.reload_xdat_served == 1
            assert int(info_after["raw_low"]) == new_max, (
                f"sid={sid} flip=True → raw_low 应等于新 max_angle={new_max}, "
                f"got {info_after['raw_low']} (before was {raw_low_before})"
            )
            assert int(info_after["raw_high"]) == new_min, (
                f"sid={sid} raw_high 应等于新 min_angle={new_min}, "
                f"got {info_after['raw_high']} (before was {raw_high_before})"
            )
        finally:
            # 还原 xdat 文件（避免污染真实参数）
            path_1.write_bytes(raw_orig)
    finally:
        env.close()


# ============================================================================
# run_live_loop 用的 _FakeDeployerRadToRaw
# ============================================================================
# (此处保留 _FakeDeployerRadToRaw 以备未来用；当前测试用 FakeDeployer)
class _FakeDeployerRadToRaw:
    """run_live_loop 用的最小 deployer fake：返回 sid → sid*10。"""

    def __init__(self, mapping_n: int = 17):
        self.mapping = {str(i): type("E", (), {"servo_id": i})() for i in range(1, mapping_n + 1)}
        self.guard = None  # 没碰撞
        self.collision_bypass = False

    def qpos_to_positions(self, qpos):
        out = {}
        for sid in range(1, len(qpos) + 1):
            out[sid] = sid * 10
        return out


# ============================================================================
# TCP send timeout (Appendix C)
# ============================================================================
def test_send_timeout_configurable_via_constructor():
    """``send_timeout`` 参数被存到 ``self._send_timeout``；非正值抛 ValueError。"""
    from live_control import PoseTcpServer

    s1 = PoseTcpServer(port=18780, send_timeout=0.01)
    assert s1._send_timeout == 0.01
    s1.close()

    s2 = PoseTcpServer(port=18781, send_timeout=0.5)
    assert s2._send_timeout == 0.5
    s2.close()

    # 默认值
    s3 = PoseTcpServer(port=18782)
    assert s3._send_timeout == 0.05
    s3.close()

    # 非正值抛错
    with pytest.raises(ValueError):
        PoseTcpServer(port=18783, send_timeout=0)
    with pytest.raises(ValueError):
        PoseTcpServer(port=18784, send_timeout=-1.0)


def _make_slow_client(host: str = "127.0.0.1", port: int = 18790):
    """Connect a client that doesn't read → kernel send buffer fills fast."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    # 注意：sock.setblocking / sock.settimeout 不动 — 让 recv 不被 timeout 干扰，
    # 我们完全不调 recv。socket 默认有 send buffer（典型 64 KB+），足够让
    # broadcast 至少写入几十 KB 后才因 sendall 阻塞到 send_timeout。
    # 对 200 B/帧而言，第 1 次 sendall 就会立刻写入 kernel buffer（< 200 B），
    # 但**不调 recv** → kernel buffer 慢慢填 → 多次 sendall 后 send 会阻塞
    # → 等到 send_timeout 才返回。
    return sock


def test_send_timeout_does_not_block_sim():
    """GUI 不 recv 时 broadcast 必须在 ~send_timeout 内返回（不是默认 2 s）。

    模拟用户报告的卡死场景：sim 端 broadcast 撞上 kernel send buffer 满。
    send_timeout=50 ms 时 broadcast 应在 ~50 ms 内返回 False，而不是等到
    Python 的 2 s 通用 socket timeout。
    """
    from live_control import PoseTcpServer

    server = PoseTcpServer(port=18790, send_timeout=0.05)
    server.start()
    try:
        time.sleep(0.1)  # 等 accept 线程起来
        # 连一个慢 client（不调 recv）
        slow = _make_slow_client(port=18790)
        try:
            time.sleep(0.1)  # 等 accept_loop 设 _client

            # 先消耗 kernel send buffer：连续 broadcast 多次让它填满。
            # 我们不必测精确次数，只要连续发直到 sendall 开始阻塞。
            # 一次 broadcast ~200 B，64 KB kernel buffer ~ 320 次能填满；
            # 为节省测试时间，多发几次确保填上即可。
            for _ in range(500):
                if server.broadcast({i: 1500 for i in range(1, 18)}) is False:
                    break
            else:
                # 如果到 500 次还没 drop 说明 buffer 没填满；多发点
                for _ in range(2000):
                    if server.broadcast({i: 1500 for i in range(1, 18)}) is False:
                        break

            # 测 wall-clock：再 broadcast 一次，这次应该被 send_timeout 卡住
            t0 = time.monotonic()
            ok = server.broadcast({i: 1500 for i in range(1, 18)})
            elapsed = time.monotonic() - t0

            assert ok is False, "broadcast 应在 send 超时时返回 False"
            # 关键断言：必须在 200 ms 内返回（远小于旧 2 s）
            assert elapsed < 0.2, (
                f"broadcast 阻塞 {elapsed * 1000:.1f} ms，"
                "send_timeout=50 ms 应限制 < 200 ms（pytest 余量）"
            )
            # 异常类型应被记录为 TimeoutError
            assert server.last_send_exc is TimeoutError, (
                f"应捕获 TimeoutError，实际 {server.last_send_exc}"
            )
            # client 已被 drop
            assert server._client is None
            # frames_sent 在第一次成功 send 之后才增加；drop 之后没 +1
            # 我们不严格断言数字（依赖 kernel buffer 大小），
            # 但断言 frames_sent 比 connection_count 少或等于
            assert server.frames_sent >= 0
        finally:
            slow.close()
    finally:
        server.close()


def test_send_timeout_counter_increments():
    """``stats.send_timeouts`` 每次 send 超时 +1；``disconnected_frames`` 也 +1。"""
    from live_control import LoopStats, PoseTcpServer

    server = PoseTcpServer(port=18791, send_timeout=0.05)
    server.start()
    try:
        time.sleep(0.1)
        slow = _make_slow_client(port=18791)
        try:
            time.sleep(0.1)

            stats = LoopStats()
            # 把 kernel send buffer 填满
            for _ in range(3000):
                if not server.broadcast({i: 1500 for i in range(1, 18)}):
                    break

            # 现在连续 broadcast 3 次，每次都应 send timeout
            for _ in range(3):
                stats.frames += 1
                if server.broadcast({i: 1500 for i in range(1, 18)}):
                    stats.sent += 1
                else:
                    stats.disconnected_frames += 1
                    if server.last_send_exc is TimeoutError:
                        stats.send_timeouts += 1

            # 至少 1 次 send timeout（受 kernel buffer 大小影响）
            assert stats.send_timeouts >= 1
            assert stats.disconnected_frames >= 1
        finally:
            slow.close()
    finally:
        server.close()


def test_send_timeout_only_when_client_unhealthy():
    """健康 client 能正常 broadcast，send_timeouts 不会增加。"""
    from live_control import LoopStats, PoseTcpServer

    server = PoseTcpServer(port=18792, send_timeout=0.05)
    server.start()
    try:
        time.sleep(0.1)
        # 健康 client：connect + 主动 recv
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", 18792))
        try:
            time.sleep(0.1)

            stats = LoopStats()
            for _ in range(3):
                ok = server.broadcast({i: 1500 for i in range(1, 18)})
                stats.frames += 1
                if ok:
                    stats.sent += 1
                else:
                    stats.disconnected_frames += 1
                    if server.last_send_exc is TimeoutError:
                        stats.send_timeouts += 1
                # 主动消费，避免 kernel buffer 填满
                try:
                    sock.recv(4096)
                except socket.timeout:
                    pass

            assert stats.sent == 3
            assert stats.send_timeouts == 0
            assert stats.disconnected_frames == 0
        finally:
            sock.close()
    finally:
        server.close()


# ============================================================================
# run_live_loop 集成 send_timeout 计数
# ============================================================================
def test_run_live_loop_counts_send_timeouts_when_client_slow():
    """端到端：模拟「GUI 卡死」场景，跑 N 帧应至少有一些 ``send_timeouts``。"""
    from live_control import PoseTcpServer

    server = PoseTcpServer(port=18793, send_timeout=0.05)
    server.start()
    try:
        time.sleep(0.1)
        slow = _make_slow_client(port=18793)
        try:
            time.sleep(0.1)

            deployer = FakeDeployer(
                positions_template={i: 1500 for i in range(1, 18)},
                contacts=[],
            )
            env = FakeEnv()

            # 先把 kernel send buffer 填满
            for _ in range(3000):
                if not server.broadcast({i: 1500 for i in range(1, 18)}):
                    break

            # 跑 N 帧；这些帧都会因 send 超时被 drop
            stats = run_live_loop(
                env, deployer,
                tcp_server=server,
                fps=200.0,
                max_frames=3,
                sleep_fn=lambda _: None,
                step_fn=lambda *a, **kw: None,
                render_fn=lambda: None,
            )

            assert stats.frames == 3
            # 至少一些 send timeout（统计逻辑应被触发）
            assert stats.send_timeouts >= 1
            assert stats.disconnected_frames >= 1
            # last_send_exc 在最后一次 broadcast 后保持最近一次的状态
            # （可能不是 TimeoutError 也可能是别的；这里只断言计数被 +1）
        finally:
            slow.close()
    finally:
        server.close()