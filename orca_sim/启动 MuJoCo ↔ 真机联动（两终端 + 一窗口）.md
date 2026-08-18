# 启动 MuJoCo ↔ 真机联动（两终端 + 一窗口）

> 本文档说明如何把 orca_sim 的 MuJoCo viewer 与 FTServo_Python 真机（STS3215 × 17）实时联动。
> **pyserial 串口独占**，所以需要：1 个 GUI 终端（持有串口）+ 1 个 sim 终端（驱动 sim，通过本地 TCP 推 raw 目标给 GUI）+ 1 个 MuJoCo viewer 窗口。

---

## 0. 激活 venv（每次新终端都要做）

**PowerShell**：

```powershell
cd C:\Users\28422\Desktop\internship\orcahand-test-main\orca_sim
.\orca\Scripts\Activate.ps1
```

**Git Bash / WSL / Linux / macOS**：

```bash
cd orca_sim
source orca/Scripts/activate          # Windows bash
source orca/bin/activate              # Linux / macOS
```

看到命令行提示符前面出现 `(orca)` 就说明激活成功。如果报「禁止运行脚本」：

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\orca\Scripts\Activate.ps1
```

> 退出：`deactivate`

---

## 1. 终端 1：起 GUI（持有串口 + 起远端接收器）

```bash
cd ../FTServo_Python/test
python servo_console.py --remote-tcp-port 8765
```

GUI 启动后：

1. 端口框输入 **`COM5`**（或你的实际串口号）
2. 点 **「连接」**
3. 控制 tab 应出现 **「远端控制: 已连接 — 127.0.0.1:8765」**（绿色）
4. 控制 tab 同时出现橙色 **「解除碰撞限制 (latch)」** 按钮（待用）

> 不连真机只跑 sim？这步可以跳过；终端 2 直接 `--no-render` 即可。

---

## 2. 终端 2：起 sim driver

```bash
cd C:\Users\28422\Desktop\internship\orcahand-test-main\orca_sim
source orca/Scripts/activate        # Git Bash
# 或 .\orca\Scripts\Activate.ps1     # PowerShell

python live_control.py --port COM5 --max-fps 30
```

- 会弹出 MuJoCo viewer，左侧是 Control panel
- 终端 1 的 GUI **不会**被脚本动到 —— `live_control.py` 自己**永远不开串口**，只通过 `127.0.0.1:8765` 把 raw 目标推给 GUI 端的 `RemotePoseReceiver` 线程
- **拖 Control panel 滑条 → sim 物理 step → 17 路 raw → 真机 17 个舵机跟随**

---

## 3. 日常使用

| 操作 | 现象 |
| --- | --- |
| 拖 Control panel 滑条 | 17 路 raw 即时发到 GUI → 真机跟随；终端 2 日志每帧打 `dry-run frame=N positions={...}` 或 `sent` 计数 +1 |
| 拖过头触发 sim 自碰撞 | 该帧被跳过，**真机保持不动**；终端 2 打 `Skipping frame due to N self-contact(s)`，summary 字段 `collisions_skipped` 累加 |
| 临时想让真机硬过去 | 点 GUI 上的 **「解除碰撞限制 (latch)」** 按钮 → 下一帧照常下发 → latch 自动归零；再撞会被再次跳过（**单次放行**） |
| 退出 | 在终端 2 按 `Ctrl+C` → viewer 关闭 → 打印 summary（`sent / collisions_skipped / disconnected_frames` 等） |

### 解除碰撞限制 (latch) 详解

- **目的**：拖滑条时偶发的临时过冲（手指掰过头），sim 检测到自碰撞就跳过，但下一帧就回退 —— 这种**抖动峰**如果每次都重启链路很烦人
- **行为**：按一次后，**下一帧**若检测到自碰撞则照样下发（不跳过）；latch 自动归零
- **不持久**：下一帧仍碰撞 → 再次被跳过（这是设计，**不是 bug**）
- **不要**用它当"永久禁用碰撞检查"用 —— 碰撞是真机可能撞的信号

---

## 4. 不连真机的几种 dry-run 模式

| 用途 | 命令 |
| --- | --- |
| 无 viewer、无 TCP、只打 raw log（headless） | `python live_control.py --no-render --max-fps 5` |
| 弹 viewer + 拖滑条 + 只打 log（无 TCP、无真机） | `python live_control.py --max-fps 30` |
| 弹 viewer + 起 TCP + 等 GUI（默认） | `python live_control.py --port COM5 --max-fps 30` |

---

## 5. 完整命令速查

| 你想做什么 | 命令 |
| --- | --- |
| GUI 占串口 + 起接收器 | `cd FTServo_Python/test && python servo_console.py --remote-tcp-port 8765` |
| sim 联动真机（弹 viewer） | `cd orca_sim && python live_control.py --port COM5 --max-fps 30` |
| sim 人工校准 EPROM 限位（弹 viewer + GUI 「应用 sim 限位」按钮可用） | `cd orca_sim && python live_control.py --port COM5 --calibrate-limits --max-fps 30` |
| sim dry-run（headless） | `cd orca_sim && python live_control.py --no-render --max-fps 5` |
| sim dry-run + viewer | `cd orca_sim && python live_control.py --max-fps 30` |
| 跑测试（验证 109 用例） | `python -m pytest tests/ tests/bridge/ tests/test_live_control.py -q --ignore=tests/retarget` |

---

## 6. 排错速查

| 现象 | 原因 / 处理 |
| --- | --- |
| GUI 报 `serial.serialutil.SerialException` | 串口被别的进程占用（pyserial 独占）。关掉其它终端 / Arduino IDE / 串口助手 |
| 终端 2 报 `OSError: [Errno 98] Address already in use` | 上次 live_control 没关干净，`--tcp-port` 换一个（如 `--tcp-port 8766`）并同步 GUI 的 `--remote-tcp-port 8766` |
| 拖滑条真机不动，但终端 2 日志显示 `sent` 在涨 | 终端 1 GUI 没真连串口。检查 GUI 状态栏是不是「未连接」 |
| 拖滑条真机不动，终端 2 日志 `collisions_skipped` 一直涨 | 自碰撞预筛把帧都跳了。点 GUI 「解除碰撞限制 (latch)」手动放行一次 |
| 滑条正方向与真机转动方向相反 | 已处理：bridge 在 `SimToRealDeployer._cache_per_servo` 按 per-servo `flip_direction` 字段决定是否翻转（默认翻；JSON 已对 `right_index_abd / right_middle_abd / right_pinky_abd` 三个关节显式 `false`）。详见 `orca_sim/README.md` §「方向翻转」+ `orca_sim/CLAUDE.md §13.1` |
| 想改某个关节的翻转方向 | 编辑 `orca_sim/src/orca_sim/bridge/data/servo_joint_mapping.json`：给该 servo 加 `"flip_direction": false`（不翻）或 `"flip_direction": true`（翻）；改完无需重启 sim 之外的任何东西，重新跑 `live_control.py` 即可 |
| 滑条在边缘真机没反应 / 位置偏差 | 用 [`scripts/calibrate_sim_to_real.py`](scripts/calibrate_sim_to_real.py)：先 `inspect` 看哪几只舵机有死区/范围不匹配，再用 `sample` + 真机采样、`fit` 自动拟合；详见 `orca_sim/CLAUDE.md §13.6` |
| 想把 sim ctrlrange 当作舵机机械极限写入 EPROM | 见下方「§人工校准 EPROM 限位（calibrate-limits）」 |
| `--debug-mapping` / `--debug-interval` 报错 `unrecognized arguments` | 这两个 flag 已删除（per 用户反馈"实时打印没啥用"）。改用 `--max-fps 30` + 看终端日志即可 |
| GUI 起动但没看到「解除碰撞限制」按钮 | 没用 `--remote-tcp-port` 启动；重新 `python servo_console.py --remote-tcp-port 8765` |

---

## 6.5 人工校准 EPROM 限位（`--calibrate-limits`）

如果你想把当前 sim Control panel 拖到的姿态当作舵机的**机械极限**写进 EPROM（寄存器 9 / 11），用这个工作流：

### 工作流

```bash
# 终端 1（不变）
cd FTServo_Python/test
python servo_console.py --remote-tcp-port 8765
# GUI 连 COM5 → 控制 tab 出现「📥 应用 sim 限位到 EPROM (17 路)」紫色按钮

# 终端 2（新增 --calibrate-limits）
cd orca_sim
python live_control.py --port COM5 --calibrate-limits --max-fps 30
# 会弹 viewer + 起 TCP server，并在日志提示 calibrate-limits 模式
```

1. 终端 2 弹 MuJoCo Control panel → 把每根手指拖到**真机物理极限**位置（看着真机反应）。
2. 满意后，按 GUI 控制 tab 上的 **「📥 应用 sim 限位到 EPROM (17 路)」** 按钮。
3. GUI 向 sim 发 `{"type": "request_limits"}` → sim 回 `{"type": "apply_limits", "limits": {...}}`（17 路 raw 当前值同时作为 min 和 max 的建议）。
4. GUI 弹窗列出 17 路建议值 + 「⚠ EPROM 写入不可逆」警告 → 你点 Yes → 走 `ServoSafetyLayer.write_eprom_register(sid, 9, 2, min)` + `write_eprom_register(sid, 11, 2, max)` 自动 unLock/lock。
5. 写完后 GUI 读回 EPROM 验证 → 内存中的 `ServoBundle.min_angle / max_angle` 同步更新 → UI SpinBox 刷新。

### 协议

```json
# GUI → sim
{"type": "request_limits"}\n

# sim → GUI 回包
{"type": "apply_limits", "limits": {"1": {"min": 1500, "max": 1500}, "2": {...}, ..., "17": {...}}}\n
```

JSON key 是字符串；每条含 `min / max ∈ [0..4095]`；恰好 17 路；GUI 校验失败会弹 `QMessageBox.warning`，不写。

### 注意事项

- **live_control.py 永远不开串口**：所有 EPROM 写入都在 GUI 端持有串口的状态下完成（pyserial 独占），sim 端只通过 TCP 转发建议。
- **每按一次按钮只写一次**：没有 latch，单次操作。如果想再写一遍，按一次按钮即可。
- **不要把 min == max 作为最终限位**：`apply_limits` 默认把「当前 raw」同时作为 min 和 max。建议工作流：先把手指拖到**最小机械极限**点 → 记下值 → 再拖到**最大机械极限**点 → 写入。**当前实现只是把最后一帧的 raw 作为 min=max 建议**，需要更精细的两端采样可以扩展 `PoseTcpServer._build_apply_limits_payload`（注入 `set_limits_provider`）。
- **claude memory**: 你反馈「raw = 0..4095 编码位置值」，已在 docstring 注释（[live_control.py:23](live_control.py#L23)）里说清楚 raw ∈ [0..4095] 的 12-bit 编码语义。

---

---

## 7. 关键文件

| 路径 | 作用 |
| --- | --- |
| `orca_sim/live_control.py` | 顶层 sim driver，弹 viewer + 起 TCP server + 每帧推 raw + 处理 `request_limits` 回 `apply_limits` |
| `orca_sim/src/orca_sim/bridge/deploy.py` | `SimToRealDeployer`：方向翻转 + rad→raw 转换 + 碰撞预筛选 + bypass latch |
| `orca_sim/src/orca_sim/bridge/data/servo_joint_mapping.json` | 17 servo ↔ sim actuator 权威映射 |
| `FTServo_Python/test/servo_console.py` | GUI：持有串口 + `RemotePoseReceiver` + 「解除碰撞限制」按钮 + 「📥 应用 sim 限位到 EPROM」按钮 |
| `orca_sim/scripts/calibrate_sim_to_real.py` | 手工校准工具：`inspect` 静态对齐表 / `sample` 交互采样 / `fit` 自动拟合 |

---

详细协议 / 风险点见 [`orca_sim/README.md`](README.md)「实时 sim→真机 控制」章节，以及顶层 [`CLAUDE.md`](../CLAUDE.md) §3 联动入口。
