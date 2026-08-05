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

## 6. 速查表

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
```
