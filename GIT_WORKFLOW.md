# Git 日常使用流程（FT 项目）

> 本项目使用一个仓库管全部子项目（`FTServo_Python/`、`orca_sim/`、`orca_core/`、`orca_teleop/`），
> 远程分支约定：`main`（朋友） / `dev-ftservo`（我们）。
> 本机默认分支：`main`，上游追踪 `origin/dev-ftservo`。

## 1. 核心心智模型

### Push 不会覆盖历史

```
本地                                  远程 (origin/dev-ftservo)
─────                                  ────────────────────────
d6c57c7 ─┐
          ├─ 179fa00 ─┐
                     ├─ 2cc8a1e ── push ──→ d6c57c7 ─ 179fa00 ─ 2cc8a1e ─ (新commit)
                                                    ↑          ↑         ↑
                                              这些永远在    这些还在   你刚push
```

- ✅ 第 1 版 push 上去后，**哪怕你后来改了 100 版、删了 100 版，第 1 版永远在**
- ✅ 每次 commit 有**唯一 hash**（如 `2cc8a1e`），随时 `git checkout <hash>` 回去
- ⚠️ 唯一会"覆盖历史"的是 `git push --force`（**别轻易用**）

### 工作区 / 暂存区 / 仓库 三态

```
   工作区 (working tree)          暂存区 (staging)              仓库 (.git)
   ─────────────────────         ──────────────────           ─────────────
   文件实际内容                   下次要提交的内容              所有历史 commit
        │                              │                          │
   你直接编辑                  git add <file>               git commit
        │                              │                          │
        └───── git checkout ───────────┴────── git reset ─────────┘
              (撤销工作区)            (取消暂存)              (撤销最近 commit)
```

## 2. 日常五步

### Step 1：看改了什么

```bash
git status                    # 列出改了哪些文件
git diff                      # 看具体改了什么（绿 + 新增，红 - 删除）
git diff path/to/file.py      # 只看某一个文件
```

### Step 2：暂存要提交的文件

```bash
# 方式 A：手动挑（推荐）
git add orca_sim/src/orca_sim/bridge/deploy.py
git add FTServo_Python/test/servo_console.py

# 方式 B：全加
git add .

# 方式 C：交互式（适合不熟的改动）
git add -i
```

### Step 3：写 commit message 并提交

```bash
git commit -m "bridge: sim→真机部署器加上碰撞 guard"
```

#### Commit message 规范（按本项目结构）

```
<作用域>: <一句话总结>

[可选：详细说明]
```

| 作用域 | 含义 |
|--------|------|
| `bridge` | `orca_sim/src/orca_sim/bridge/` 下的联动代码 |
| `orca_sim` | 仿真环境本身 |
| `orca_teleop` | 远程操作（手部追踪→关节） |
| `orca_core` | 核心控制 |
| `FTServo` | 真机端舵机控制 |
| `docs` | 仅文档（CLAUDE.md / README 等） |
| `chore` | 杂项（.gitignore、依赖等） |
| `test` | 测试 |

**好例子：**
- `bridge: deploy.py 修复 thumb_abd 的 [-1.08211, 0] 不对称范围映射`
- `orca_sim: envs.py 给右手加 wrist 关节上限校验`
- `FTServo: servo_console.py 同步 speed/acc 默认值到 bridge`
- `docs: 更新 CLAUDE.md §6 已知风险`
- `servo-3: xdat 更新（调整最大角度限位）`

### Step 4：推送到远程

```bash
# 第一次推送（已配 upstream 之后可直接 git push）
git push origin main:dev-ftservo

# 或者用简化命令（已配 upstream 后）
git push
```

> 💡 `main:dev-ftservo` 的含义：**把本地的 `main` 推送到远程的 `dev-ftservo` 分支**。
> 这样不会动朋友的 `main` 分支。

**如果想让 `git push` 默认推到 `dev-ftservo`（推荐）：**

```bash
# 一次性设置
git push --set-upstream origin main:dev-ftservo

# 之后直接
git push
```

### Step 5：查看历史

```bash
git log --oneline              # 一行一条
git log --oneline --graph      # 图形化（含分支）
git log --oneline -20          # 最近 20 条
git log --stat                 # 看每次改了哪些文件
git show <commit-id>           # 看某次的具体改动（如 git show 2cc8a1e）
```

## 3. 后悔药（按危险程度排序）

| 想做什么 | 命令 | 危险度 |
|---------|------|--------|
| 撤销文件的本地改动 | `git checkout -- <file>` | ⚠️ 改动永久丢失 |
| 取消暂存（保留改动） | `git reset HEAD <file>` | ✅ 安全 |
| 改最近 commit 的 message | `git commit --amend -m "新 message"` | ⚠️ 改了 hash |
| 改到一半要切走做别的 | `git stash` / `git stash pop` | ✅ 安全 |
| 回到某次 commit 的状态（不删历史） | `git checkout <commit-id>` | ✅ 安全 |
| 撤销某次 commit（生成反向 commit） | `git revert <commit-id>` | ✅ 安全，会多一个 commit |
| **删掉最近 commit 重新来** | `git reset --hard HEAD~1` | 🔴 改动永久丢失 |
| **覆盖远程历史** | `git push --force` | 🔴 灾难性，**永远别用** |

## 4. 本项目特有的注意事项

### 4.1 `FTServo_Python/参数/*.xdat` 是关键文件

这 17 份是舵机出厂参数快照（每份含最小/最大角度限制、波特率、ID 等）。
**改一份就 commit 一次**（带 `servo-N: xdat 更新` 信息），方便回溯"某次故障前舵机参数是啥"。

```bash
git add FTServo_Python/参数/3.xdat
git commit -m "servo-3: 调整最大角度限位 2700→3000"
```

### 4.2 `orca_sim/src/orca_sim/bridge/data/servo_joint_mapping.json` 是联动核心

改了它要**单独 commit** 并明确说改了哪个 servo 的映射：

```bash
git add orca_sim/src/orca_sim/bridge/data/servo_joint_mapping.json
git commit -m "bridge: mapping 修复 servo-15 (thumb_dip) 的 joint_role 标记"
```

### 4.3 Windows 上的 CRLF 警告

提交时偶尔会看到：

```
warning: in the working copy of 'xxx', LF will be replaced by CRLF the next time Git touches it
```

**正常现象**，是 Windows 自动转换，**忽略即可**，不污染提交。

### 4.4 串口独占

`orca_sim/scripts/sim_to_real.py`（bridge）和 `FTServo_Python/test/servo_console.py`（GUI）**不能同时跑**——pyserial 独占打开。两个都开必然有一个失败。

## 5. 远程协作

### 当前远程结构

```
origin (https://github.com/EasomScorpion/orcahand-test.git)
├── main         ← 朋友上传的版本（不要动）
└── dev-ftservo  ← 你的开发分支（你 push 上来都进这里）
```

### 拉朋友最新代码

```bash
# 先看朋友 main 上有没有新东西
git fetch origin main

# 拉到本地新分支 review，不动你当前的 main
git checkout -b friend-main origin/main

# 看完了切回来
git checkout main
```

### 合并朋友的改动到你的 dev-ftservo

```bash
git fetch origin main
git merge origin/main            # 或 git rebase origin/main
git push                        # 推回去
```

如果合并有冲突，git 会告诉你哪些文件冲突，进去手动改 → `git add` → `git commit` → `git push`。

## 6. Tag：给版本打标记

> 用来标记某个 commit 是「一个值得留念的版本」——比如 `v1.0`、`v0.2`。
> Tag 指向某个 commit 永久不动，像 commit 一样有完整历史。

### 6.1 两种 tag

| 类型 | 命令 | 说明 |
|------|------|------|
| **轻量 tag** | `git tag v1.0` | 只是一个指向某次 commit 的指针 |
| **带注释 tag**（**推荐**） | `git tag -a v1.0 -m "..."` | 像 commit 一样有作者、日期、说明 |

**永远用带注释的 tag**——以后 `git show v1.0` 能看到完整的发布说明，GitHub 也会把它渲染成 release 页面。

### 6.2 完整流程：给当前 HEAD 打 1.0 tag

```bash
# 1. 确认要在哪个 commit 上打 tag
git log --oneline -5
# 比如看到 dcb3489 是最新

# 2. 打 tag（带说明）
git tag -a v1.0 -m "v1.0: 完成 sim→真机联动基础版

包含：
- 17 舵机 ↔ sim actuator 映射
- CollisionGuard 自碰撞过滤
- SimToRealDeployer 部署器
- ServoSafetyLayer 原子下发
- 35 + 25 个测试全过
"

# 3. 推送 tag（⚠️ tag 默认不随 git push 上传，必须显式 push）
git push origin v1.0

# 4. 验证
git tag -l                      # 列出所有本地 tag
git ls-remote --tags origin     # 列出所有远程 tag
git show v1.0                   # 看 tag 详情（commit + 说明 + 作者）
```

之后去 https://github.com/EasomScorpion/orcahand-test/releases 会自动看到 "v1.0" release（GitHub 把带注释的 tag 渲染成 release）。

### 6.3 给过去的某个 commit 打 tag

```bash
# 给 d6c57c7（init 那个）打 tag
git tag -a v0.1 d6c57c7 -m "v0.1: 初始导入"
git push origin v0.1
```

### 6.4 Tag 命名规范（建议）

| 场景 | 命名 | 例子 |
|------|------|------|
| 完整发布版 | `v<主>.<次>.<补丁>` | `v1.0.0`、`v1.2.3` |
| 未发布的里程碑 | `v<主>.<次>` | `v0.1`、`v0.2` |
| 临时快照 | `snapshot-<日期>` | `snapshot-2026-08-05` |
| 预发布 | `v<主>.<次>.<补丁>-<预发标签>` | `v1.0.0-rc1`、`v1.0.0-beta2` |

> 💡 **不要轻易打 `v1.0`**——它通常意味着"对外发布的稳定版"。开发中的里程碑用 `v0.x` 即可。

### 6.5 切到某个 tag 看代码

```bash
# 看 v1.0 那个版本的代码（会进入 "detached HEAD" 状态）
git checkout v1.0

# 看完切回主分支
git checkout main
```

> ⚠️ 在 detached HEAD 状态下**别 commit 新东西**——那些 commit 不属于任何分支，容易丢。要临时改东西的话，先建个分支：`git checkout -b fix-from-v1.0 v1.0`。

### 6.6 删 tag

```bash
git tag -d v1.0                  # 删本地 tag
git push origin --delete v1.0    # 删远程 tag
```

## 7. 速查表

```bash
# === 改完代码 ===
git status
git diff
git add <file>
git commit -m "scope: 描述"
git push                          # 已配 upstream 后

# === 后悔 ===
git checkout -- <file>            # 撤销文件改动
git reset HEAD <file>             # 取消暂存
git stash / git stash pop         # 临时收起 / 恢复
git revert <hash>                 # 撤销某次 commit（不删历史）
git reset --hard HEAD~1           # 删最近 commit（🔴 危险）

# === 看历史 ===
git log --oneline
git log --oneline --graph
git show <hash>
git log --stat

# === 远程 ===
git remote -v                     # 看远程地址
git fetch origin                  # 拉所有远程更新（不合并）
git pull                          # 拉 + 合并
git push                          # 推

# === Tag ===
git tag                           # 列出本地 tag
git tag -l "v1.*"                 # 按模式过滤
git tag -a v1.0 -m "说明"         # 给当前 HEAD 打带注释 tag
git tag -a v1.0 <commit> -m "..." # 给指定 commit 打 tag
git show v1.0                     # 看 tag 详情
git push origin v1.0              # 推单个 tag
git push origin --tags            # 推所有 tag
git checkout v1.0                  # 切到 tag 的代码（detached HEAD）
git tag -d v1.0                   # 删本地 tag
git push origin --delete v1.0     # 删远程 tag
```
