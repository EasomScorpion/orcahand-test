import sys
sys.path.insert(0, "src")
import numpy as np
import mujoco
from orca_sim.envs import OrcaHandRight

env = OrcaHandRight(version="v1", skin=False, render_mode="rgb_array")
env.reset(seed=42)
m = env.unwrapped.model
d = env.unwrapped.data

# 测 index 的 ip body
# 1) 全 0 状态
d.ctrl[:] = 0
for _ in range(200):
    mujoco.mj_step(m, d, nstep=1)
mujoco.mj_forward(m, d)
print("全 0 状态：")
print(f"  qpos[7] (index_pip) = {d.qpos[7]:.4f}")
print(f"  qpos[6] (index_mcp) = {d.qpos[6]:.4f}")
print(f"  qpos[5] (index_abd) = {d.qpos[5]:.4f}")
print(f"  xpos[right_index_ip] = {d.xpos[m.body('right_index_ip').id]}")

# 2) 把 pip=hi
d.ctrl[:] = 0
d.ctrl[7] = m.actuator_ctrlrange[7, 1]  # pip hi = +1.885
print(f"\n设 ctrl[7] = {m.actuator_ctrlrange[7, 1]:.4f} (pip hi)")
for i in range(10):
    mujoco.mj_step(m, d, nstep=1)
print(f"10 步后 qpos[7] = {d.qpos[7]:.4f}")
print(f"  xpos[right_index_ip] = {d.xpos[m.body('right_index_ip').id]}")

# 3) 同样设 mcp=hi
d.ctrl[:] = 0
d.ctrl[6] = m.actuator_ctrlrange[6, 1]  # mcp hi = +1.658
print(f"\n设 ctrl[6] = {m.actuator_ctrlrange[6, 1]:.4f} (mcp hi)")
for i in range(10):
    mujoco.mj_step(m, d, nstep=1)
print(f"10 步后 qpos[6] = {d.qpos[6]:.4f}")
print(f"  xpos[right_index_ip] = {d.xpos[m.body('right_index_ip').id]}")

# 4) 设 mcp+pip 都 hi
d.ctrl[:] = 0
d.ctrl[6] = m.actuator_ctrlrange[6, 1]
d.ctrl[7] = m.actuator_ctrlrange[7, 1]
print(f"\n设 mcp+pip 都 hi")
for i in range(10):
    mujoco.mj_step(m, d, nstep=1)
print(f"  qpos[6]={d.qpos[6]:.4f} qpos[7]={d.qpos[7]:.4f}")
print(f"  xpos[right_index_ip] = {d.xpos[m.body('right_index_ip').id]}")
print(f"  xpos[right_index_pp] = {d.xpos[m.body('right_index_pp').id]}")
print(f"  xpos[right_index_mp] = {d.xpos[m.body('right_index_mp').id]}")