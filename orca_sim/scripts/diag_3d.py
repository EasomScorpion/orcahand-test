import sys
sys.path.insert(0, "src")
import numpy as np
import mujoco
from orca_sim.envs import OrcaHandRight

env = OrcaHandRight(version="v1", skin=False, render_mode="rgb_array")
env.reset(seed=42)
m = env.unwrapped.model
d = env.unwrapped.data

print("=" * 70)
print("逐个驱动 mcp/pip，看 ip body 位置变化（rest vs hi vs lo）")
print("=" * 70)

# 对 index：测 right_index_ip body 的位置变化
# rest 状态
d.ctrl[:] = 0
for _ in range(200):
    mujoco.mj_step(m, d, nstep=1)
rest = d.xpos[m.body("right_index_ip").id, :3].copy()
print(f"\nright_index_ip rest: {rest}")

# 只动 mcp = hi
d.ctrl[:] = 0
d.ctrl[6] = m.actuator_ctrlrange[6, 1]  # mcp hi
for _ in range(200):
    mujoco.mj_step(m, d, nstep=1)
p_mcp_hi = d.xpos[m.body("right_index_ip").id, :3].copy()
print(f"right_index_ip mcp=hi: {p_mcp_hi}  delta={p_mcp_hi - rest}")

# 只动 pip = hi
d.ctrl[:] = 0
d.ctrl[7] = m.actuator_ctrlrange[7, 1]  # pip hi
for _ in range(200):
    mujoco.mj_step(m, d, nstep=1)
p_pip_hi = d.xpos[m.body("right_index_ip").id, :3].copy()
print(f"right_index_ip pip=hi: {p_pip_hi}  delta={p_pip_hi - rest}")

# 同时 mcp+pip hi
d.ctrl[:] = 0
d.ctrl[6] = m.actuator_ctrlrange[6, 1]
d.ctrl[7] = m.actuator_ctrlrange[7, 1]
for _ in range(200):
    mujoco.mj_step(m, d, nstep=1)
p_both_hi = d.xpos[m.body("right_index_ip").id, :3].copy()
print(f"right_index_ip mcp+pip=hi: {p_both_hi}  delta={p_both_hi - rest}")

# lo
d.ctrl[:] = 0
d.ctrl[6] = m.actuator_ctrlrange[6, 0]
d.ctrl[7] = m.actuator_ctrlrange[7, 0]
for _ in range(200):
    mujoco.mj_step(m, d, nstep=1)
p_both_lo = d.xpos[m.body("right_index_ip").id, :3].copy()
print(f"right_index_ip mcp+pip=lo: {p_both_lo}  delta={p_both_lo - rest}")

print("\n=== 总结 ===")
print(f"hi 时 ip 距离 MCP 更远: {(p_both_hi - rest)}")
print(f"lo 时 ip 距离 MCP 更远: {(p_both_lo - rest)}")