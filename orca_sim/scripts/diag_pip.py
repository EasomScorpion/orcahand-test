import sys
sys.path.insert(0, "src")
import numpy as np
import mujoco
from orca_sim.envs import OrcaHandRight

env = OrcaHandRight(version="v1", skin=False, render_mode="rgb_array")
env.reset(seed=42)
m = env.unwrapped.model
d = env.unwrapped.data

# 测试：mcp=hi 已经弯曲到位后，再设 pip=hi，ip 应该进一步移动
# rest
d.ctrl[:] = 0
for _ in range(500):
    mujoco.mj_step(m, d, nstep=1)
ip_rest = d.xpos[m.body("right_index_ip").id, :3].copy()

# 只 mcp=hi
d.ctrl[:] = 0
d.ctrl[6] = m.actuator_ctrlrange[6, 1]
for _ in range(500):
    mujoco.mj_step(m, d, nstep=1)
ip_mcp = d.xpos[m.body("right_index_ip").id, :3].copy()
print(f"mcp=hi: ip = {ip_mcp*1000}mm")
print(f"  Δ vs rest = {(ip_mcp - ip_rest)*1000}mm  |Δ|={np.linalg.norm(ip_mcp-ip_rest)*1000:.2f}mm")

# 在 mcp=hi 基础上再加 pip=hi
d.ctrl[7] = m.actuator_ctrlrange[7, 1]
for _ in range(500):
    mujoco.mj_step(m, d, nstep=1)
ip_mcp_pip = d.xpos[m.body("right_index_ip").id, :3].copy()
print(f"\nmcp+pip=hi: ip = {ip_mcp_pip*1000}mm")
print(f"  Δ vs mcp-only = {(ip_mcp_pip - ip_mcp)*1000}mm  |Δ|={np.linalg.norm(ip_mcp_pip-ip_mcp)*1000:.2f}mm")
print(f"  Δ vs rest = {(ip_mcp_pip - ip_rest)*1000}mm  |Δ|={np.linalg.norm(ip_mcp_pip-ip_rest)*1000:.2f}mm")

# 只 pip=hi（mcp 不动）
d.ctrl[:] = 0
d.ctrl[7] = m.actuator_ctrlrange[7, 1]
for _ in range(500):
    mujoco.mj_step(m, d, nstep=1)
ip_pip_only = d.xpos[m.body("right_index_ip").id, :3].copy()
print(f"\nonly pip=hi: ip = {ip_pip_only*1000}mm")
print(f"  Δ vs rest = {(ip_pip_only - ip_rest)*1000}mm  |Δ|={np.linalg.norm(ip_pip_only-ip_rest)*1000:.2f}mm")