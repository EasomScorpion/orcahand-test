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
print("每根手指：mcp=hi vs pip=hi 的指尖 body 位移对比")
print("=" * 70)

# 4 根非拇指手指：(actuator_mcp, actuator_pip, tip_body)
fingers = [
    ("index", 6, 7, "right_index_ip"),
    ("middle", 9, 10, "right_middle_ip"),
    ("ring", 12, 13, "right_ring_ip"),
    ("pinky", 15, 16, "right_pinky_ip"),
]

for name, amcp, apip, tip_b in fingers:
    saved = d.ctrl.copy()
    # rest
    d.ctrl[:] = 0
    for _ in range(200):
        mujoco.mj_step(m, d, nstep=1)
    rest = d.xpos[m.body(tip_b).id, :3].copy()
    # mcp hi
    d.ctrl[:] = 0
    d.ctrl[amcp] = m.actuator_ctrlrange[amcp, 1]
    for _ in range(200):
        mujoco.mj_step(m, d, nstep=1)
    delta_mcp = d.xpos[m.body(tip_b).id, :3] - rest
    # pip hi
    d.ctrl[:] = 0
    d.ctrl[apip] = m.actuator_ctrlrange[apip, 1]
    for _ in range(200):
        mujoco.mj_step(m, d, nstep=1)
    delta_pip = d.xpos[m.body(tip_b).id, :3] - rest
    # combined
    d.ctrl[:] = 0
    d.ctrl[amcp] = m.actuator_ctrlrange[amcp, 1]
    d.ctrl[apip] = m.actuator_ctrlrange[apip, 1]
    for _ in range(200):
        mujoco.mj_step(m, d, nstep=1)
    delta_both = d.xpos[m.body(tip_b).id, :3] - rest

    mcp_norm = np.linalg.norm(delta_mcp)
    pip_norm = np.linalg.norm(delta_pip)
    both_norm = np.linalg.norm(delta_both)
    ratio = pip_norm / (mcp_norm + 1e-6)
    print(f"{name:7s}: |mcp delta|={mcp_norm*1000:5.1f}mm |pip delta|={pip_norm*1000:5.1f}mm |both|={both_norm*1000:5.1f}mm  pip/mcp={ratio:.2f}")
    d.ctrl[:] = saved