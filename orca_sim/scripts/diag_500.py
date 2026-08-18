import sys
sys.path.insert(0, "src")
import numpy as np
import mujoco
from orca_sim.envs import OrcaHandRight

env = OrcaHandRight(version="v1", skin=False, render_mode="rgb_array")
env.reset(seed=42)
m = env.unwrapped.model
d = env.unwrapped.data

# 关键测试：分别设每个 actuator=hi，看 ip body 位置变化（充分收敛）
def get_xpos():
    return {
        "mp": d.xpos[m.body("right_index_mp").id, :3].copy(),
        "pp": d.xpos[m.body("right_index_pp").id, :3].copy(),
        "ip": d.xpos[m.body("right_index_ip").id, :3].copy(),
    }

# rest
d.ctrl[:] = 0
for _ in range(500):
    mujoco.mj_step(m, d, nstep=1)
rest = get_xpos()
print(f"rest: mp={rest['mp']} pp={rest['pp']} ip={rest['ip']}")

for label, aidx, joint_name in [
    ("abd hi", 5, "abd"),
    ("mcp hi", 6, "mcp"),
    ("pip hi", 7, "pip"),
]:
    d.ctrl[:] = 0
    d.ctrl[aidx] = m.actuator_ctrlrange[aidx, 1]
    for _ in range(500):
        mujoco.mj_step(m, d, nstep=1)
    cur = get_xpos()
    d_mp = (cur["mp"] - rest["mp"]) * 1000
    d_pp = (cur["pp"] - rest["pp"]) * 1000
    d_ip = (cur["ip"] - rest["ip"]) * 1000
    print(f"\n{label} ({joint_name}):")
    print(f"  Δmp = {d_mp}  |Δmp|={np.linalg.norm(d_mp):6.2f}mm")
    print(f"  Δpp = {d_pp}  |Δpp|={np.linalg.norm(d_pp):6.2f}mm")
    print(f"  Δip = {d_ip}  |Δip|={np.linalg.norm(d_ip):6.2f}mm")
    print(f"  qpos[{aidx}]={d.qpos[m.jnt_qposadr[int(m.actuator_trnid[aidx, 0])]]:.3f}")