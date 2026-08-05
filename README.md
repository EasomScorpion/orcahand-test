# FT — Orca 手硬件 ↔ 仿真联动

> 真实 STS3215 × 17 舵机手 ↔ orca_sim 仿真互通的桥梁

本仓库是 `FT/` 工作区的根目录，目标是让仿真训练结果**一比一**地部署到真机。

## 项目结构

```
.
├── FTServo_Python/      # 真机端：PyQt5 控制台 + Feetech SDK + 17 份舵机参数 xdat
├── orca_sim/            # 仿真端：Gymnasium + MuJoCo（右手 v1，skin=False 骨头对骨头碰撞）
├── orca_core/           # 独立控制核心（不在本联动范围内）
├── orca_teleop/         # 远程操作：手部追踪（MediaPipe / Vision Pro 等）→ 关节目标
├── bridge/              # 联动桥（orca_sim/src/orca_sim/bridge/）
├── 参数/                 # 17 份舵机出厂参数快照
├── CLAUDE.md            # 项目导航（详细文档看这里）
└── README.md            # 本文件
```

## 17 舵机 ↔ sim actuator 映射

来源：`orca_sim/src/orca_sim/bridge/data/servo_joint_mapping.json`（真机权威映射）。

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

## 快速开始

### 1. 激活仿真端虚拟环境

```bash
cd orca_sim
uv venv orca --python 3.11       # 首次：创建 venv（已存在可跳过）
source orca/Scripts/activate     # Windows bash；Linux/macOS 用 source orca/bin/activate
uv pip install -e ".[dev,bridge]"   # 首次或改了 pyproject.toml 时
```

### 2. 跑仿真（不需要真机）

```bash
cd orca_sim
python view_v1.py --env right --mode static         # v1 静态查看器
python random_policy.py --env right --version v1 --steps 100
```

### 3. 跑联动（dry-run，不需要真机）

```bash
cd orca_sim
python scripts/sim_to_real.py --dry-run --random-steps 30 --collision-check
```

### 4. 跑联动（需要 USB-UART + 真机）

```bash
python scripts/sim_to_real.py --port COM5 --baud 1000000 --random-steps 30
```

### 5. 跑测试

```bash
cd orca_sim
source orca/Scripts/activate
python -m pytest tests/ tests/bridge/ -q
```

> 📖 详细文档看 [CLAUDE.md](./CLAUDE.md)。

## 核心思路

1. **sim 自带 STL 碰撞检测**作为「动作可执行性」过滤器：sim 自碰撞的动作真机也会撞。
2. **sim ctrlrange ↔ xdat raw 线性映射**（忽略零点偏移）——确定性、可复现。
3. **原子下发**：通过 `ServoSafetyLayer.sync_go_to_pose` 一次性同步 17 个 STS3215 舵机。

## 已知风险

详见 [CLAUDE.md §6](./CLAUDE.md)。摘要：
- **几何失配**：sim STL 与打印件不完全一致时，sim 通过的动作真机可能还是会撞
- **FTServo 串口独占**：bridge 与 `servo_console.py` GUI 不能同时跑
- **orca_core/ 不在范围**：其 `MotorClient` / `MockDynamixelClient` 是另一条链路

## 分支约定

- `main` — 稳定版本（朋友上传的版本）
- `dev-ftservo` — 当前开发分支（联动物理层 + 仿真）

## License

TBD
