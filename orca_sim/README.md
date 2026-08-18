<p align="center">
  <img src="https://huggingface.co/datasets/fracapuano/blogs/resolve/main/orca_sim.png" alt="orca_sim header" width="600"/>
</p>


`orca_sim` provides simulation environments for the ORCA hand.
You can start building your ORCA hand today at [orcahand.com](https://www.orcahand.com/).

## Install

We recommend using python 3.11 inside of a virtual environment.
You can create a virtual environment using `uv` ([how to install uv](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
uv venv orca --python 3.11
source orca/bin/activate
uv pip install orca_sim
```

Alternatively, you can use `conda` ([how to install conda](https://www.anaconda.com/docs/getting-started/miniconda/install)):
```bash
conda create -n orca python=3.11 -y
conda activate orca
python -m pip install orca_sim
```
As we are continuously iterating on `orca_sim`, you can fetch the latest `main` building this package from source, so to be in the loop with the latest developments.

```bash
git clone https://github.com/orcahand/orca_sim
cd orca_sim && uv pip install -e .
```

> [!WARNING] 
We are still iterating (a lot!) on this package. If you need stability, consider sticking to the Pypi package (`pip install orca_sim`).

## Getting started

`orca_sim` follows the [Gymnasium](https://gymnasium.farama.org/) API, and uses [Mujoco](https://mujoco.readthedocs.io/en/stable/overview.html) for physics simulation and rendering.
You can instantiate an environment with one hand (or both) via:

```python
from orca_sim import OrcaHandRight  # or OrcaHandLeft, OrcaHandCombined

env = OrcaHandRight()
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```
The 'extended' version of different hands contain additional bodies (incl. inertial properties) such as the camera mount, the U2D2 board and fans.

### Hands versioning

By default, any environment defaults to the latest fully-supported embodiment files.

```python
from orca_sim import OrcaHandRight, OrcaHandRightExtended

env = OrcaHandRight()  # latest version of the standard right hand
extended_env = OrcaHandRightExtended()  # latest version of the extended right hand
```

`orca_sim` stored different hand versions under versioned `scenes/` and `models/` directories. You can still pin an older version explicitly when needed:

```python
from orca_sim import OrcaHandCombinedExtended

env = OrcaHandCombinedExtended(version="v1")  # loads the v1 hand
```

See our [`random_policy.py`](random_policy.py) example to see how to instantiate and interface an ORCA hand.

## Sample task: in-hand cube orientation

`orca_sim` now also ships a task-level example that augments the right hand with a free-floating cube whose one target face is colored red:

```python
from orca_sim import OrcaHandRightCubeOrientation

env = OrcaHandRightCubeOrientation(version="v2", render_mode="human")
obs, info = env.reset(seed=0)
```

By default, the task resets to a palm-up open-hand pose with the cube resting on the palm and the red face pointing downward. You can also randomize the initial cube orientation while keeping it unsolved:

```python
env = OrcaHandRightCubeOrientation(
    version="v2",
    initial_red_face="random",
    cube_pos_xy_jitter=0.01,
)
obs, info = env.reset(seed=0)
```

If you want to keep the environment completely deterministic while you are still building it, you can use the nominal reset directly and add randomization later:

```python
env = OrcaHandRightCubeOrientation(version="v2")

nominal = env.nominal_reset_options()
obs, info = env.reset(options=nominal)

randomized = env.sample_randomized_reset_options(
    seed=0,
    initial_red_face="random",
    cube_pos_xy_jitter=0.01,
)
obs, info = env.reset(options=randomized)
```

The implementation is intentionally split so it doubles as a porting template:

- The task scene lives in [`src/orca_sim/scenes/v2/scene_right_cube_orientation.xml`](src/orca_sim/scenes/v2/scene_right_cube_orientation.xml) and composes the existing hand MJCF with a single task cube.
- The nominal palm-up hand pose and in-palm cube spawn are now authored into the task-specific scene/model files, so opening the XML directly in MuJoCo shows the intended setup.
- The task logic lives in [`src/orca_sim/task_envs.py`](src/orca_sim/task_envs.py), including reset-time cube randomization and optional hand-pose overrides for custom MJCF layouts.

## 实时 sim→真机 控制 (`live_control.py`)

`live_control.py` 会打开 MuJoCo viewer 加载 v1 右手，每帧把仿真的 `qpos` 转成 raw 舵机目标，再推给真实的 STS3215 × 17 机械手。它定位是**校准 / 对账 / sanity check 工具**，不是长期跑策略用的 driver。

脚本**自己不开串口** —— pyserial 是独占的，串口由兄弟仓库里的 `servo_console.py` GUI 持有。`live_control.py` 只通过本地 TCP socket（`127.0.0.1:<tcp-port>`）把目标发过去，GUI 端的 `RemotePoseReceiver` 线程收到后再走 `ServoSafetyLayer.sync_go_to_pose` 写硬件。

> **方向翻转（per-servo）**：用户实测发现 sim 滑条的正方向与真实舵机的转动方向**大部分**相反，但也有少量关节方向一致。bridge 在 [`SimToRealDeployer._cache_per_servo`](src/orca_sim/bridge/deploy.py) 里按 [`ServoEntry.flip_direction`](src/orca_sim/bridge/mapping.py)（默认 `true`）决定是否交换 `raw_low`/`raw_high` 存储语义 —— `true` 时 `sim_low → xdat_max_angle`、`sim_high → xdat_min_angle`，`false` 时与 xdat 顺序一致。从 `qpos_to_positions` 出来的 raw 已经按各 servo 的标记决定是否翻转，调用方（包括 `live_control.py`）无需感知。当前 JSON 把 `right_index_abd / right_middle_abd / right_pinky_abd`（servo 3/4/5）三个指根关节设为 `flip_direction: false`。

> **碰撞旁通 latch**：每帧 sim 自碰撞预筛选会把碰撞帧**跳过**。如果只是临时性过冲（比如拖滑条把手指掰过头），可以在 GUI 上按「**解除碰撞限制 (latch)**」按钮，下一次碰撞帧会**照样下发**到真机，之后 latch 自动归零；再撞会再次被跳过。详细协议见下文。

> **人工校准 EPROM 限位（`--calibrate-limits`）**：拖 sim Control panel 滑条到真机机械极限，按 GUI「**📥 应用 sim 限位到 EPROM (17 路)**」按钮，工具把当前 17 路 sim 姿态 → raw → 走 `ServoSafetyLayer.write_eprom_register(sid, 9, 2, min)` + `write_eprom_register(sid, 11, 2, max)` 写 EPROM 寄存器 9/11（自动 unLock/lock）。详见 `orca_sim/启动 MuJoCo ↔ 真机联动（两终端 + 一窗口）.md` §6.5。

### 前置条件

脚本需要 `[dev,bridge]` extras（如果你也要跑 GUI 才需要 PyQt5）：

```bash
cd orca_sim
source orca/bin/activate          # Linux / macOS
.\orca\Scripts\activate           # Windows bash
.\orca\Scripts\Activate.ps1       # Windows PowerShell
uv pip install -e ".[dev,bridge]"
uv pip install PyQt5              # 只有跑 servo_console.py（GUI）时才需要
```

### 快速参考

| 命令 | 作用 |
| --- | --- |
| `python live_control.py --no-render --max-fps 5` | 无头干跑：加载 sim，每帧打印 17 路 raw；不弹 viewer、不开 TCP、不动串口。 |
| `python live_control.py --max-fps 30` | 弹 MuJoCo viewer 干跑；拖 Control panel 滑条让手指动，看日志里 raw 数值变化。 |
| `python live_control.py --port COM5 --max-fps 30` | 弹 viewer + 起 `127.0.0.1:8765` 的 TCP server 等 GUI 来连。**仍然不开串口。** |

### CLI 旗标

| 旗标 | 默认值 | 行为 |
| --- | ---: | --- |
| `--port` | `None` | 硬件哨兵（例如 `COM5`）。设置后才起 TCP server。**脚本自己永远不开串口。** |
| `--baud` | `1_000_000` | 仅打日志用；串口由 GUI 持有。 |
| `--tcp-port` | `8765` | 设置 `--port` 时使用的本地 TCP server 端口。 |
| `--no-collision-check` | 关 | 跳过每帧 `CollisionGuard.self_contacts()` 预筛。 |
| `--speed`, `--acc` | `100`, `10` | 仅打日志用；不放入 TCP 协议。GUI 端走 `SafetyLimits.RESET_SPEED`/`RESET_ACC` 默认值。 |
| `--xdat-dir` | `../FTServo_Python/参数` | 用来推 raw 边界的 xdat 目录。 |
| `--hand`, `--version`, `--env` | `right`, `v1`, `right` | 目前只接好了右手 v1 映射。 |
| `--no-render` | 关 | 无头模式：不弹 MuJoCo viewer，`render_mode=None`。 |
| `--max-fps` | `30.0` | 循环帧率上限。 |

### 端到端工作流

1. **先在另一个终端起 GUI**，让它持有串口：

   ```bash
   cd ../FTServo_Python/test
   python servo_console.py --remote-tcp-port 8765
   ```

   在 GUI 里选 `COM5`（或你的实际端口），点「连接」，确认出现「远端控制」指示灯。

2. **再起 sim driver**：

   ```bash
   python live_control.py --port COM5 --max-fps 30
   ```

3. **逐个拖 MuJoCo Control panel 里的滑条**，观察日志里 raw 数值变化 —— 与 `servo_joint_mapping.json` 的 `sim_actuator` 字段对照，确认物理上是哪几根手指在动。

4. 如果对不上，改 [`src/orca_sim/bridge/data/servo_joint_mapping.json`](src/orca_sim/bridge/data/servo_joint_mapping.json) 的 `sim_actuator` 字段，然后重启脚本即可。

### 碰撞旁通 latch（GUI 按钮）

GUI 启动时只要带 `--remote-tcp-port`，控制 tab 就会出现一个橙色按钮：

> **解除碰撞限制 (latch)** — 按一次后，下一帧若 sim 检测到自碰撞则照常下发（不跳过）；下一帧仍碰撞 → 再次跳过（**单次放行**）。

底层协议：GUI 在同一 TCP 连接上向 `live_control.py` 发一帧 newline-delimited JSON：

```json
{"type": "bypass_collision"}
```

`PoseTcpServer._client_read_loop` 线程收到后调用 `on_command` 回调 → `main()` 里把它分发给 `SimToRealDeployer.bypass_next_collision_check()`，置位 `collision_bypass` latch。下一帧若发生碰撞，`run_live_loop` 消费 latch、归零、再下发。

### 协议细节（备查）

**SIM → GUI（positions）** —— 一帧一行 JSON，以 `\n` 结尾：

  ```json
  {"positions":{"1":2048,"2":1744,"3":2141,"4":1971,"5":2067,"6":1765,"7":1703,"8":1945,"9":1662,"10":1575,"11":1608,"12":1997,"13":1993,"15":1684,"16":1460,"17":1822}}
  ```

  注意 wire 上 JSON object 的 key 是字符串；GUI 收到后会再 `int()` 一次，然后才传给 `sync_go_to_pose`。

**GUI → SIM（commands）** —— 同 socket 反方向，一行一命令：

  ```json
  {"type": "bypass_collision"}
  ```

  解析失败 / 未知 `type` / 非 dict 的行被忽略，不会断 read_loop。

- 只绑 `127.0.0.1`；**不要**改到 `0.0.0.0`。
- listen backlog = 1（只允许一个 GUI client）；新连接会替换旧连接。
- receiver 会忽略解析失败的 JSON、越界值、空 positions —— 不退出线程。

### 测试

```bash
python -m pytest tests/test_live_control.py -v
```

覆盖范围：JSON wire 格式、TCP server（host/port 校验、完整 round-trip、无 client 时 broadcast、反向命令分发 + 垃圾行容错）、主循环（碰撞跳过、dry-run、断开帧、`max_frames` 停机、render/step 顺序、**碰撞旁通 latch 单次消费**）、`KeyboardInterrupt` 优雅退出、与真实 `OrcaHandRight` env 的 rad→raw 转换一致性。

## 手工校准 sim ↔ 真机 (`scripts/calibrate_sim_to_real.py`)

如果 MuJoCo 滑条拖到边缘时真机"没反应"，或者 sim 手指位置与真机偏差明显，问题通常出在 sim ↔ xdat 范围没对齐。本工具做两件事：

1. **静态检查**：列出每只舵机的 sim range / xdat range / 中点 / 不对称 / 死区，标出哪几只值得人工核。
2. **交互采样 + 自动拟合**：拖 sim 到 N 个采样点（默认 5 个），脚本让你输入"我看到真机的 raw 是多少"，自动算线性 fit 并建议 `flip_direction` 的正确值。

### 三步走

```bash
# 1) 静态检查（不需要真机）
python scripts/calibrate_sim_to_real.py inspect

# 2) 交互采样（要 GUI + 真机）
#    终端 1：python FTServo_Python/test/servo_console.py --remote-tcp-port 8765
#    终端 2：弹 viewer，按提示输入每个采样点的真实 raw
python scripts/calibrate_sim_to_real.py sample --output calibration.csv

# 3) 自动拟合 + 给出 flip_direction 建议
python scripts/calibrate_sim_to_real.py fit --csv calibration.csv
```

详细输出见 [`orca_sim/scripts/calibrate_sim_to_real.py`](scripts/calibrate_sim_to_real.py) 的 docstring 与 `orca_sim/CLAUDE.md §13.6`。

