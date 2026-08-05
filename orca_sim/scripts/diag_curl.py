import sys
sys.path.insert(0, 'src')
import numpy as np
import mujoco
from orca_sim.envs import OrcaHandRight

env = OrcaHandRight(version="v1", skin=False, render_mode="rgb_array")
env.reset(seed=42)
m = env.unwrapped.model
d = env.unwrapped.data
mujoco.mj_forward(m, d)

mcp_tip = {
    "thumb":  ("right_thumb_mp", "right_thumb_dp"),
    "index":  ("right_index_mp", "right_index_ip"),
    "middle": ("right_middle_mp","right_middle_ip"),
    "ring":   ("right_ring_mp",  "right_ring_ip"),
    "pinky":  ("right_pinky_mp", "right_pinky_ip"),
}

print("=== qpos0 (rest) fingertip <-> MCP distance ===")
for finger, (mcp_b, tip_b) in mcp_tip.items():
    p1 = d.xpos[m.body(mcp_b).id, :3]
    p2 = d.xpos[m.body(tip_b).id, :3]
    print(f"  {finger:7s}: dist={np.linalg.norm(p2-p1)*1000:5.1f}mm")

saved = d.qpos.copy()
flex = []
for i in range(m.nu):
    name = m.actuator(i).name
    if any(x in name for x in ["mcp_actuator", "pip_actuator", "dip_actuator"]):
        if "thumb_abd" not in name:
            flex.append(i)

print("\n=== flexion = ctrl hi (max) ===")
for a in flex:
    hi = m.actuator_ctrlrange[a, 1]
    d.ctrl[a] = hi
    qa = int(m.jnt_qposadr[int(m.actuator_trnid[a, 0])])
    d.qpos[qa] = hi
mujoco.mj_forward(m, d)
for finger, (mcp_b, tip_b) in mcp_tip.items():
    p1 = d.xpos[m.body(mcp_b).id, :3]
    p2 = d.xpos[m.body(tip_b).id, :3]
    print(f"  {finger:7s}: dist={np.linalg.norm(p2-p1)*1000:5.1f}mm")

print("\n=== flexion = ctrl lo (min) ===")
for a in flex:
    lo = m.actuator_ctrlrange[a, 0]
    d.ctrl[a] = lo
    qa = int(m.jnt_qposadr[int(m.actuator_trnid[a, 0])])
    d.qpos[qa] = lo
mujoco.mj_forward(m, d)
for finger, (mcp_b, tip_b) in mcp_tip.items():
    p1 = d.xpos[m.body(mcp_b).id, :3]
    p2 = d.xpos[m.body(tip_b).id, :3]
    print(f"  {finger:7s}: dist={np.linalg.norm(p2-p1)*1000:5.1f}mm")

print("\n=== summary ===")
print("Rest  dist > lo dist > hi dist  →  hi 是弯（指尖靠近 MCP）")
print("Rest  dist < lo dist < hi dist  →  hi 是伸（指尖远离 MCP）")

d.qpos[:] = saved
mujoco.mj_forward(m, d)