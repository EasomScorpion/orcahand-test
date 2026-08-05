# CLAUDE.md — orca_sim 项目要点

> 面向 Claude / 后续助手的项目速查手册。完整说明请见 [README.md](README.md) 与 `项目说明.md`。

## 1. 项目定位

`orca_sim` 是 ORCA Hand 的仿真环境集合，基于 **Gymnasium** API + **MuJoCo** 物理引擎与渲染。
参考：[orcahand.com](https://www.orcahand.com/)。
当前示例任务：**手中立方体方向调整**（in-hand cube reorientation，右手机 + 自由立方体，一个红色面为 target）。

## 2. 环境（Python / 工具链）

- Python：`>= 3.10`（推荐 **3.11**；本仓库已使用 `uv` 创建 `orca` 虚拟环境，Python 3.11.5）。
- 关键依赖（`pyproject.toml`）：
  - `gymnasium>=0.29`
  - `mujoco>=3.1`
  - `numpy>=1.26`
  - dev: `pytest>=8`
- 包内源码位于 `src/orca_sim/`，包数据包括：
  - `scenes/*/*.xml`、`models/*/*.mjcf`
  - STL 资源（视觉 / 碰撞）：`assets/mjcf/{left,right}/{visual,collision}/*.stl`
- 当前已存在的 venv 路径：`orca/`（uv 创建，已激活可直接 `source orca/Scripts/activate` 使用）。

## 3. 仓库目录结构

```
orca_sim/
├── README.md                    # 安装、入门、版本说明
├── 项目说明.md                   # 中文项目说明
├── pyproject.toml               # 依赖与打包配置
├── random_policy.py             # 通用 random policy 入口（多手 / 多版本）
├── view_v1.py                   # v1 快捷查看脚本（static/demo/random 三种模式）
├── src/orca_sim/
│   ├── __init__.py              # 对外暴露所有 env 类 + register_envs
│   ├── envs.py                  # BaseOrcaHandEnv + 6 个普通 env 类
│   ├── task_envs.py             # OrcaHandRightCubeOrientation（任务示例）
│   ├── registry.py              # gym.register 所有版本化 env_id
│   ├── versions.py              # 版本解析（list_versions/resolve_scene_path 等）
│   ├── scenes/{v1,v2}/*.xml     # 场景 XML（包含 scene.xml 公共背景 + MJCF include）
│   └── models/{v1,v2}/*.mjcf    # 手部 MJCF 模型；STL 资源在 assets/mjcf/ 下
└── tests/                       # pytest，conftest 把 src/ 加入 sys.path
    ├── conftest.py
    ├── test_envs.py
    ├── test_versions.py
    └── test_registry.py
```

## 4. 双手变体与版本

环境类（位于 [src/orca_sim/envs.py](src/orca_sim/envs.py)）：

- `OrcaHandLeft`、`OrcaHandRight`
- `OrcaHandLeftExtended`、`OrcaHandRightExtended`（带 U2D2 板、风扇、相机支架等带惯性的附加体）
- `OrcaHandCombined`、`OrcaHandCombinedExtended`（双手组合）

任务 env：`OrcaHandRightCubeOrientation`（[src/orca_sim/task_envs.py](src/orca_sim/task_envs.py:14)）。

版本（`versions.py`）：
- `LATEST_VERSION = "v2"`；通过 `scenes/<v?>/scene_left.xml` 自动发现 `v1`、`v2`。
- 用法：`env = OrcaHandRight(version="v1")`（默认 `version=None` 会用最新）。
- `register_envs()` 在 gym 注册中心以 `OrcaHandXxx-vN` 形式注册。

观测 / 动作尺寸（来自 `tests/test_envs.py` / `tests/test_registry.py`）：
- 单手：`obs=(34,)`、`action=(17,)`。
- 双手：`obs=(68,)`、`action=(34,)`。
- 立方体方向任务：`obs=(51,)`、`action=(17,)`，`info` 含 `red_face_up_alignment` 等。

## 5. 安装与运行

```bash
# 用 uv 创建 venv（推荐）
uv venv orca --python 3.11
source orca/Scripts/activate          # Windows (bash)；Linux/macOS 用 source orca/bin/activate
uv pip install -e .                   # 本仓库源码安装

# 跑测试
python -m pytest tests/ -q            # 当前通过 31 个测试
```

> macOS 上需要 `mjpython` 运行 human render 模式；`render_mode="rgb_array"` 在无显示环境下仍可工作。

## 5.1 启动 viewer 的常用命令

> **先激活 venv**（两个 CLAUDE.md 都强调这条）：
> - **PowerShell**（Windows 默认）：`.\orca\Scripts\Activate.ps1`（看到 `(orca)` 提示符即激活）
> - **Git Bash / WSL / Linux / macOS**：`source orca/Scripts/activate`（Windows bash）或 `source orca/bin/activate`（Linux/macOS）
> - 详细说明与「禁止运行脚本」绕开方法见 [FT/CLAUDE.md §5.0](../CLAUDE.md#50-激活虚拟环境先做这一步)

```bash
source orca/Scripts/activate          # Windows (bash)；Linux/macOS 用 source orca/bin/activate

# === view_v1.py：v1 专用的快速查看器（static/demo/random 三种模式） ===
# static：手动拖 Control 滑条控制每根手指
python view_v1.py --env right --mode static
# demo：自动循环 张开 ↔ 握拳
python view_v1.py --env right --mode demo
# random：随机动作
python view_v1.py --env right --mode random --steps 200

# 切换手型 / 版本
python view_v1.py --env left              --mode static
python view_v1.py --env combined          --mode static
python view_v1.py --env right_extended    --mode static   # 含 U2D2 / 风扇 / 相机支架
python view_v1.py --env right_cube_orientation --version v2 --mode static  # 立方体方向任务

# 无皮肤 v1
python view_v1.py --env right --mode static --no-skin

# 无皮肤 + 立方体任务
python view_v1.py --env right_cube_orientation --version v1 --mode static --no-skin

# 代码侧
from orca_sim import OrcaHandRight
env = OrcaHandRight(version="v1", skin=False)

# === random_policy.py：通用 random policy 入口（任意 env_id / 任意版本） ===
python random_policy.py --env combined_extended --version v2 --render-mode human --steps 1000
# --env 可选: left / left_extended / right / right_extended / right_cube_orientation / combined / combined_extended
# --version: v1 / v2（不填默认最新，即 v2）
# --render-mode: human（交互视图） / rgb_array（离屏）
# --steps: 0 = 一直跑到 Ctrl+C

# === 退出 viewer ===
# 在 viewer 内点 File → Quit；或回到终端按 Ctrl+C。
```

> **为什么 static 模式拖滑条手指会动**：`view_v1.py:_static_loop` 在 `env.render()` 之后调用 `mujoco.mj_step(env.model, env.data, nstep=env.frame_skip)`，把 Control 面板的滑条值真正推进物理。详见 [view_v1.py:74-86](view_v1.py#L74-L86)。

## 6. 快速使用

```python
from orca_sim import OrcaHandRight, OrcaHandRightCubeOrientation

# 仅右手（v1）
env = OrcaHandRight(version="v1", render_mode="rgb_array")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()

# 立方体方向任务（v2）
env = OrcaHandRightCubeOrientation(version="v2")
obs, info = env.reset(seed=0)

# 自定义 reset 选项
nominal    = env.nominal_reset_options()
randomized = env.sample_randomized_reset_options(seed=0, initial_red_face="random",
                                                 cube_pos_xy_jitter=0.01)
env.reset(options=randomized)
```

辅助脚本：
- `python random_policy.py --env combined --render-mode human --version v1` — 随机策略 + 交互视图。
- `python view_v1.py --env right --mode demo` — 开合循环；`mode` ∈ `static | demo | random`。

`view_v1.py` 关键参数（[view_v1.py](view_v1.py)）：
- `FLEX_IDX = [3,4,7,10,13,16]`（thumb_pip/dip + 四指 pip）
- `MCP_IDX  = [6,9,12,15]`（四指 mcp）
- 默认 `--env right --version v1 --mode static`。

## 7. 架构与代码要点

### 7.1 BaseOrcaHandEnv（Gymnasium `Env[np.ndarray, np.ndarray]`）

[src/orca_sim/envs.py:14](src/orca_sim/envs.py#L14) 关键约定：
- `metadata = {"render_modes": ["human","rgb_array"], "render_fps": 30}`
- `frame_skip = 5`，仿真 `mujoco.mj_step(model, data, nstep=frame_skip)`。
- 动作空间从 `model.actuator_ctrlrange` 推导并被 `np.clip` 在 step 内裁剪到 `[action_low, action_high]`。
- `reset(options={"qpos","qvel"})`：可以直接覆盖完整 qpos/qvel；其它 env 的 reset 也会做 `mj_forward`。
- `render_mode="human"`：懒加载 `mujoco.viewer.launch_passive`，启动时调用 `mujoco.mjv_defaultFreeCamera` 把自由相机的视角回到 scene 默认。
- 默认相机名：`"closeup"`。
- 终止 / 截断 / 信息 getter 可被子类覆盖（`_get_terminated / _get_truncated / _get_info / _get_reward / _get_obs`）。

### 7.2 OrcaHandRightCubeOrientation（任务示例）

[src/orca_sim/task_envs.py:14](src/orca_sim/task_envs.py#L14) 关键约定：
- 假设每个 actuator 驱动一个 `mjJNT_HINGE` 关节（`_resolve_actuator_qpos_indices` 会做断言）。
- 默认初始红面朝下（`initial_red_face="down"`）；可设为 `"random"`，且会在采样时过滤 `red_face_up_alignment >= 0.95`（即不会一开局就已成功）。
- `cube_pos_xy_jitter` 标量或 `(2,)`，用于在 xy 上随机化立方体位置。
- Reward：`0.5 * (alignment + 1)` + `0.10 * lift_bonus` − `drop_penalty`（红面与世界 +z 对齐得分；立方体 z 抬高得分；掉落扣分）。
- 终止条件：`red_face_up_alignment >= cos(15°)`（默认 15° 容差）或立方体掉落（z<drop_height，默认 0.05）。
- 截断条件：`max_episode_steps = 200`（可改）。
- 观测在 base 观测后拼接 `red_face_world_normal`（3）和 `red_face_up_alignment`（1）。
- Reset 允许覆盖 `qpos`、`qvel`、`hand_pose_by_joint`、`hand_qpos`、`cube_pos`、`cube_quat`、`cube_qvel`、`settle_steps` 等。
- `mj_resetData` 后写入 hand/cube qpos，再调用 `_compose_ctrl_from_qpos()` 保证 ctrl 与 qpos 一致。
- 提供 `nominal_reset_options()` 与 `sample_randomized_reset_options(seed=...)` 作为可重现的 reset 工厂。

## 8. 资源打包

`[tool.setuptools.package-data]` 中声明了所有 XML/MJCF/STL 资源路径。**任何新文件如果未被声明，运行 `pip install -e .` 安装的副本就会缺失这些数据。** 修改资源目录结构时，记得同步更新 `pyproject.toml` 中的 `package-data`。

## 9. 常见操作

- 切换到不同手版本：
  ```python
  env = OrcaHandLeftExtended(version="v2")
  ```
- 通过 gym 注册中心（受 `versions.py` 控制）：
  ```python
  from orca_sim import register_envs
  register_envs()
  import gymnasium as gym
  env = gym.make("OrcaHandRight-v1")
  ```
- 写入自定义 qpos 重置：
  ```python
  env.reset(options={"qpos": np.zeros(env.model.nq)})
  ```

## 10. 测试覆盖（`tests/`，共 31 用例，均已通过）

- [tests/test_envs.py](tests/test_envs.py)：reset/step smoke、qpos 注入、动作限幅、非法 render_mode、形状校验。
- [tests/test_versions.py](tests/test_versions.py)：版本发现、未知版本报错、v1/v2 全部场景 XML 直接加载。
- [tests/test_registry.py](tests/test_registry.py)：两次调用 `register_envs()` 幂等；每个 `OrcaXxx-vN` env_id 都能 make 并 reset。

调试建议：
- `tests/` 在缺包时不会自动加入 `src/`，但 `conftest.py` 已经把 `src` 注入 `sys.path`，运行无需 `pip install -e .`。
- 资源问题（找不到 `scene_*.xml` 或 STL）优先检查 `pyproject.toml` 的 `package-data`。

## 11. 已验证结论（本机环境）

- 平台：Windows 11，`uv 0.11.7` 创建的 `orca/` 已激活。
- `python --version` 报告 **3.12.12**；venv 内 Python 报告 **3.11.5**。在 venv 中 `gymnasium`、`mujoco`、`numpy`、`orca_sim` 可正常导入。
- `python -m pytest tests/`：**35 passed**。
- v1 下 7 个环境类均可 reset + step，形状如下：
  - `OrcaHandRight-v1` / `OrcaHandLeft-v1` / Extended 系列 → `obs=(34,)`、`act=(17,)`
  - `OrcaHandCombined-v1` → `obs=(68,)`、`act=(34,)`
  - `OrcaHandRightCubeOrientation-v1` → `obs=(51,)`、`act=(17,)`
- `python view_v1.py --env right --mode random --steps 3` 正常运行（默认 `render_mode="human"`，使用 rgb_array 需修改该脚本或加交互式显示）。

> 注：CI/headless 环境建议运行 `view_v1.py` 时改 `--render-mode rgb_array`（脚本目前没有该参数），或改 `random_policy.py` 的 `--render-mode rgb_array`，并对静态交互 viewer 改成 `mjpython`（macOS）。

## 12. 无皮肤（skin=False）模式

当你**还没打印 silicone 指腹 / 掌心皮肤**时，可以用 `skin=False` 在仿真里把皮肤关掉，让模型与实际硬件一致。

### 12.1 用法

```python
from orca_sim import OrcaHandRight, OrcaHandRightCubeOrientation

env = OrcaHandRight(version="v1", skin=False)        # 无皮肤 v1
env = OrcaHandRightCubeOrientation(version="v1", skin=False)  # 立方体任务也无皮肤
```

命令行：

```bash
python view_v1.py --env right --mode static --no-skin
python view_v1.py --env right_cube_orientation --version v1 --mode static --no-skin
```

### 12.2 做了什么 / 没做什么

实现位置：[src/orca_sim/envs.py](src/orca_sim/envs.py) 的 `_disable_skin_geoms(model)`。

| 维度 | skin=True（默认） | skin=False |
| --- | --- | --- |
| MJCF 文件 / STL 资源 | 不变 | 不变（无侵入） |
| `ngeom` / `nmesh` 数量 | 63 / 62（v1 right） | 同样 63 / 62，**皮肤 geom 还在** |
| 皮肤 geom 的 `geom_rgba[:, 3]` | `1.0`（白色） | `0.0`（不可见） |
| 皮肤 geom 的 `contype` / `conaffinity` | 部分 `1`（v1 还参与碰撞） | `0`（彻底不参与碰撞） |
| 物理仿真 | 与设计一致 | 物体不会"撞"到指腹上 |
| env 的 `_disabled_skin_geom_count` | 0 | 22（v1 right 实际数） |

`skin=False` **不会**真的从 `.mjcf` 删除 mesh / geom 行（避免处理 `<include>` 相对路径的复杂性），而是在 `MjModel` 加载后立刻把它们"关掉"。

### 12.3 v1 "只打印骨头" 用的 STL 清单

如果要 3D 打印**没有皮肤**的版本（v1），只需要下表中"骨头"那列；所有 `*_skin.stl` 不必打印：

| 部位 | 骨头 STL（必打） | 皮肤 STL（`skin=False` 模式可省） |
| --- | --- | --- |
| 底座 | `right_visual_tower_main.stl`、`_hull.stl`、`_text.stl` | — |
| 掌心 | `right_visual_palm.stl` | `right_visual_palm_skin.stl` |
| 拇指 MP / PP / IP / DP | `right_visual_thumb_mp.stl`、`_pp.stl`、`_ip.stl`、`_dp.stl` | `right_visual_thumb_ip_skin.stl`、`_dp_skin.stl` |
| 食指 PP / IP | `right_visual_index_pp.stl`、`_ip.stl` | `right_visual_index_pp_skin.stl`、`_ip_skin.stl` |
| 中指 PP / IP | `right_visual_middle_pp.stl`、`_ip.stl` | `right_visual_middle_pp_skin.stl`、`_ip_skin.stl` |
| 无名指 PP / IP | `right_visual_ring_pp.stl`、`_ip.stl` | `right_visual_ring_pp_skin.stl`、`_ip_skin.stl` |
| 小指 PP / IP | `right_visual_pinky_pp.stl`、`_ip.stl` | `right_visual_pinky_pp_skin.stl`、`_ip_skin.stl` |

> 左手 / 双手 / Extended 版的 STL 在 `src/orca_sim/models/v1/assets/mjcf/left/` 与 `models/v1/assets/mjcf/right/` 同样按 `*_skin.stl` 后缀区分。v2 命名是 `*_Skin.stl`（大写 S），逻辑一样。

### 12.4 如果你后续**真的**打印了皮肤

- 仿真侧只需把 `skin=True`（或不传、留默认）即可恢复。
- 不需要改任何代码、也不需要清缓存。

