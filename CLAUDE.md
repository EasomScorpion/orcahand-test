# CLAUDE.md — FT 项目导航（hardware ↔ simulator 联动）

> `FT/` 是三个相互独立的子项目共存的工作区，用于让真实 STS3215 × 17 舵机手与 orca_sim 仿真互通。
> **本文档是导航页**：每个子项目都有自己的详细 CLAUDE.md，请进子目录看。

---

## 1. 项目定位

本目录解决两件事：

1. **sim 知道真机的物理限位 / 碰撞**：orca_sim v1 右手 skin=False 模式下的 MuJoCo 物理（关节 + STL 碰撞几何）能作为「动作可执行性」过滤器——sim 自碰撞的动作，真机也会撞。
2. **sim 训练结果 → 真机一比一执行**：sim 关节角（弧度）→ 17 个 STS3215 raw 目标（0..4095）的确定性映射 + `ServoSafetyLayer.sync_go_to_pose` 原子下发。

**范围**：
- orca_sim **v1** 右手 + `skin=False`（骨头对骨头碰撞，与真机一致）
- 真机 **STS3215 × 17**（右手 1..16 + wrist=17）
- FTServo_Python/ 侧**不修改**任何文件
- orca_core/ 当前不在本项目范围内

---

## 2. 目录结构

```
FT/
├── CLAUDE.md                        # 本文档（导航页）
├── FTServo_Python/                  # 真机端：PyQt5 控制台 + Feetech SDK + 17 份 xdat 参数
│   ├── CLAUDE.md                    # 子项目自己的 CLAUDE.md（如果存在）
│   ├── test/servo_console.py        # 17 舵机控制台（ConsoleBackend / ServoSafetyLayer）
│   ├── scservo_sdk/                 # Feetech 协议 SDK
│   ├── xdat_tool.py                 # EPROM 读写工具
│   ├── 参数/1.xdat .. 17.xdat       # 17 份舵机 EPROM 快照（含最小/最大角度限制）
│   └── 其它 demos
├── orca_sim/                        # 仿真端：Gymnasium + MuJoCo
│   ├── CLAUDE.md                    # 详细的子项目说明
│   ├── pyproject.toml
│   ├── src/orca_sim/
│   │   ├── envs.py                  # BaseOrcaHandEnv + 6 个手型子类
│   │   ├── task_envs.py             # OrcaHandRightCubeOrientation
│   │   └── bridge/                  # ← 联动层入口（新增）
│   │       ├── __init__.py
│   │       ├── mapping.py           # 17 舵机映射查询
│   │       ├── collision.py         # sim 自碰撞守卫
│   │       ├── deploy.py            # sim → 真机部署器
│   │       ├── limits.py            # sim/xdat 对齐 advisory
│   │       └── data/servo_joint_mapping.json
│   ├── scripts/sim_to_real.py       # ← CLI 入口（新增）
│   ├── tests/
│   │   ├── test_envs.py / test_versions.py / test_registry.py
│   │   └── bridge/                  # ← 联动测试（新增）
│   └── 其它 demo 脚本
└── orca_core/                       # 独立子项目（本次未涉及）
```

---

## 3. 联动入口

**代码入口**：`orca_sim/src/orca_sim/bridge/`

```python
from orca_sim.bridge import (
    load_joint_mapping,
    CollisionGuard,
    SimToRealDeployer,
    build_deployer,
    check_alignment,
)
```

**CLI 入口**：`orca_sim/scripts/sim_to_real.py`

**一键运行**（dry-run，无需真机）：
```bash
cd orca_sim
source orca/Scripts/activate
python scripts/sim_to_real.py --dry-run --random-steps 30
```

**连真机运行**（需要 USB-UART 与 FTServo_Python 权限）：
```bash
python scripts/sim_to_real.py --port COM5 --random-steps 30
```

---

## 4. 17 舵机 ↔ sim actuator 映射速查表

来源：`orca_sim/src/orca_sim/bridge/data/servo_joint_mapping.json`（用户提供的真机权威映射）。

| servo_id | 中文关节名 | orca_sim v1 actuator | joint_role |
|----------|-----------|----------------------|------------|
| 1 | 食指指尖 | `right_index_pip` | pip |
| 2 | 食指指中 | `right_index_mcp` | mcp |
| 3 | 食指指根 | `right_index_abd` | abd |
| 4 | 中指指根 | `right_middle_abd` | abd |
| 5 | 小拇指指根 | `right_pinky_abd` | abd |
| 6 | 小拇指指尖 | `right_pinky_pip` | pip |
| 7 | 小拇指指中 | `right_pinky_mcp` | mcp |
| 8 | 中指指尖 | `right_middle_pip` | pip |
| 9 | 中指指中 | `right_middle_mcp` | mcp |
| 10 | 无名指指中 | `right_ring_mcp` | mcp |
| 11 | 无名指指尖 | `right_ring_pip` | pip |
| 12 | 无名指指根 | `right_ring_abd` | abd |
| 13 | 大拇指大指根 | `right_thumb_abd` | abd |
| 14 | 大拇指小指根 | `right_thumb_mcp` | mcp |
| 15 | 大拇指指尖 | `right_thumb_dip` | pip |
| 16 | 大拇指指中 | `right_thumb_pip` | pip |
| 17 | 腕关节（wrist） | `right_wrist` | wrist |

**v1 MJCF 声明顺序**（`models/v1/right.mjcf:294-315`，从 0 起）：wrist → pinky (3) → ring (3) → middle (3) → index (3) → thumb (4)。

---

## 5. 常用命令

### 5.0 激活虚拟环境（先做这一步）

orca_sim 的依赖装在 `orca_sim/orca/` 这个 uv venv 里。**每次新开终端都要先激活**才能 `import mujoco` 等。包是常驻磁盘的，重启电脑不会丢；只有改了 `pyproject.toml` 才需要重跑 `uv pip install -e ".[dev,bridge]"`。

**Windows PowerShell（默认）：**

```powershell
cd .\orca_sim
.\orca\Scripts\Activate.ps1
```

看到命令行提示符前面出现 `(orca)` 就说明激活成功：

```
(orca) PS C:\Users\28422\Desktop\internship\FT\orca_sim>
```

> 如果报「禁止运行脚本」：
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
> .\orca\Scripts\Activate.ps1
> ```

**Windows Git Bash / WSL / Linux / macOS：**

```bash
cd orca_sim
source orca/Scripts/activate        # Windows bash
source orca/bin/activate           # Linux / macOS
```

**退出 venv：**

```powershell
deactivate
```

---

> 安装一次后常用命令如下（venv 激活见 §5.0）。

```bash
# === 安装（orca_sim）===
cd orca_sim
uv venv orca --python 3.11          # 已存在 orca/ venv
source orca/Scripts/activate         # Windows bash；Linux/macOS 用 source orca/bin/activate
uv pip install -e ".[dev,bridge]"    # dev=pytest；bridge=pyserial

# === 运行 sim ===
python view_v1.py --env right --mode static          # v1 static 查看器
python random_policy.py --env right --version v1 --steps 100

# === 联动：sim → 真机（dry-run，无需真机）===
python scripts/sim_to_real.py --dry-run --random-steps 20 --collision-check
python scripts/sim_to_real.py --dry-run --target-json pose.json

# === 联动：sim → 真机（需要 USB-UART）===
python scripts/sim_to_real.py --port COM5 --baud 1000000 --random-steps 30
python scripts/sim_to_real.py --port COM5 --target-json pose.json

# === 联动：policy rollout ===
python scripts/sim_to_real.py --port COM5 --policy-path my_policy.npy --sync-every 5

# === 测试 ===
python -m pytest tests/ tests/bridge/ -q
```

---

## 6. 已知风险

1. **几何失配**：sim STL 与打印件尺寸/外形不完全一致时，sim 通过的动作真机可能还是会撞。部署前需确认 STL 与打印件几何一致。
2. **忽略零点的语义**：bridge 直接把 sim ctrlrange [low, high] ↔ xdat [min_angle, max_angle] 线性对应，**不补偿任何零点偏移**。若 sim ctrlrange 不对称（v1 `thumb_abd` 是 `[-1.08211, 0]`），真机舵机也会偏向 raw 范围的一侧——这是物理事实，不修复。
3. **`<exclude>` 段**：v1 MJCF 已对相邻手指段 exclude，bridge 检测到的接触是「真正的」自碰撞（不是误判）。
4. **FTServo 串口独占**：orca_sim bridge 与 `servo_console.py` GUI 不能同时跑（pyserial 独占打开）。
5. **MAX_TORQUE = 100 vs xdat = 50**：`ServoSafetyLayer.SafetyLimits.MAX_TORQUE=100` 比真机出厂的 50 更大；不影响本计划（不写 EPROM）。若要拉低力矩，需在 `servo_console` GUI 跑「应用安全底线」按钮对齐 EPROM。
6. **sim 步进时序差**：bridge 用「sim step → 部署一次」串行；真机内部 PID 走自己的速度曲线。bridge 默认 speed=100 / acc=10；CLI 可加 `--speed 500 --acc 30` 调快。
7. **`load_bundles` 路径**：FTServo_Python/test/servo_console.py 的 `ConsoleBackend.load_bundles()` 用固定路径 `参数/`，要求 xdat 位于 FTServo_Python/参数/ 下；不要挪动位置。
8. **orca_core/ 不在范围**：本次项目不集成 orca_core（其 `MotorClient` / `MockDynamixelClient` 是另一条链路，与本 bridge 并行存在，不冲突）。

---

## 7. 测试入口

```bash
cd orca_sim
source orca/Scripts/activate
python -m pytest tests/ tests/bridge/ -q
```

**原 orca_sim 测试**（tests/）共 35 个，已在 v0.1.0 通过。
**bridge 测试**（tests/bridge/）新增约 25 个：

- `test_mapping.py` — JSON 加载、servo_id ↔ sim_actuator 双向查表、缺失/越界报错
- `test_deploy.py` — 弧度→raw 转换、ServoSafetyLayer 校验路径、单调性、忽略零点、clip 行为
- `test_collision.py` — CollisionGuard 自碰撞过滤、threshold 边界、过滤背景 geom
- `test_limits.py` — sim ctrlrange 与 xdat 范围对齐检查（advisory）

全部测试使用 fake/mock backend，**不需要真机**。

---

## 8. 已验证结论（本机环境）

- Python 3.11.5（orca_sim venv）。
- orca_sim 原 35 个测试全部通过。
- bridge 子包 ~25 个测试全部通过（dry-run 模式）。
- `python scripts/sim_to_real.py --dry-run --random-steps 30` 正常运行，每步打印碰撞检测 + raw 部署结果。
- 真机端验证需要 USB-UART + STS3215 × 17 实物；本机暂无此硬件，跳过。

---

## 9. 一句话总结

`FT/` = 真实手控制程序 (`FTServo_Python/`) + 仿真环境 (`orca_sim/`) + 联动桥 (`orca_sim/src/orca_sim/bridge/`)。

**核心思路**：用 sim 自带的 STL 碰撞检测过滤「动作可执行性」；sim ctrlrange ↔ xdat raw 线性映射（忽略零点）；通过 `ServoSafetyLayer.sync_go_to_pose` 原子下发 17 个 STS3215 舵机。