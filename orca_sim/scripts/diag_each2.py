"""正确诊断：每个 actuator 动时，看它直接控制的 body 相对 parent 的位移。"""
import sys
sys.path.insert(0, "src")
import numpy as np
import mujoco
from orca_sim.envs import OrcaHandRight

env = OrcaHandRight(version="v1", skin=False, render_mode="rgb_array")
env.reset(seed=42)
m = env.unwrapped.model
d = env.unwrapped.data

# 每个 actuator → 它直接控制的 body（从 MJCF 推出）
#   index:  actuator 5=abd → body right_index_mp
#           actuator 6=mcp → body right_index_pp
#           actuator 7=pip → body right_index_ip
#   thumb:  actuator 1=mcp → body right_thumb_mp
#           actuator 2=abd → body right_thumb_pp
#           actuator 3=pip → body right_thumb_ip
#           actuator 4=dip → body right_thumb_dp

# 每根手指的 actuator 列表 (aidx, 直接控制的 body)
test_index = [
    (5, "right_index_mp"),    # abd → mp
    (6, "right_index_pp"),    # mcp → pp
    (7, "right_index_ip"),    # pip → ip
]
test_thumb = [
    (1, "right_thumb_mp"),    # mcp → mp
    (2, "right_thumb_pp"),    # abd → pp
    (3, "right_thumb_ip"),    # pip → ip
    (4, "right_thumb_dp"),    # dip → dp
]

# rest 状态
d.ctrl[:] = 0
for _ in range(200):
    mujoco.mj_step(m, d, nstep=1)
mujoco.mj_forward(m, d)

def test_actuator(aidx, body_name, label):
    """设 ctrl=hi，收敛，看 body 相对 rest 的位移。"""
    saved = d.ctrl.copy()
    d.ctrl[:] = 0
    d.ctrl[aidx] = m.actuator_ctrlrange[aidx, 1]
    for _ in range(200):
        mujoco.mj_step(m, d, nstep=1)
    mujoco.mj_forward(m, d)

    # body 自己的位置变化
    bid = m.body(body_name).id
    # parent body
    parent_id = int(m.body_parentid[bid])
    parent_pos = d.xpos[parent_id, :3].copy() if parent_id > 0 else np.zeros(3)

    rest_body = d.xpos[bid, :3].copy()
    rest_parent = d.xpos[parent_id, :3].copy() if parent_id > 0 else np.zeros(3)
    rel_rest = rest_body - rest_parent
    rel_now = d.xpos[bid, :3] - parent_pos
    delta = rel_now - rel_rest

    print(f"  {label:50s}  ctrl hi |Δ|={np.linalg.norm(delta)*1000:6.2f}mm  dx={delta[0]*1000:+6.2f} dy={delta[1]*1000:+6.2f} dz={delta[2]*1000:+6.2f}")
    d.ctrl[:] = saved
    for _ in range(100):
        mujoco.mj_step(m, d, nstep=1)
    mujoco.mj_forward(m, d)

print("=" * 80)
print("INDEX finger：每个 actuator 动时，它直接控制的 body 相对 parent 的位移")
print("=" * 80)
for aidx, body in test_index:
    test_actuator(aidx, body, f"actuator {aidx} ({m.actuator(aidx).name}) → body {body}")

print()
print("=" * 80)
print("THUMB finger：同上")
print("=" * 80)
for aidx, body in test_thumb:
    test_actuator(aidx, body, f"actuator {aidx} ({m.actuator(aidx).name}) → body {body}")