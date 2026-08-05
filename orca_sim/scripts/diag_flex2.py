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
print("逐个用 d.ctrl 驱动每个 flexion 关节到 hi/lo")
print("=" * 70)

test_plan = [
    (1, "right_thumb_mcp", "right_thumb_dp", "right_thumb_mp"),
    (3, "right_thumb_pip", "right_thumb_dp", "right_thumb_mp"),
    (4, "right_thumb_dip", "right_thumb_dp", "right_thumb_mp"),
    (6, "right_index_mcp", "right_index_ip", "right_index_mp"),
    (7, "right_index_pip", "right_index_ip", "right_index_mp"),
    (9, "right_middle_mcp","right_middle_ip","right_middle_mp"),
    (10,"right_middle_pip","right_middle_ip","right_middle_mp"),
    (12,"right_ring_mcp", "right_ring_ip", "right_ring_mp"),
    (13,"right_ring_pip", "right_ring_ip", "right_ring_mp"),
    (15,"right_pinky_mcp", "right_pinky_ip", "right_pinky_mp"),
    (16,"right_pinky_pip", "right_pinky_ip", "right_pinky_mp"),
]

print(f"\n{'actuator':30s}  {'rest':>8s}  {'hi':>8s}  {'lo':>8s}  趋势")
print("-" * 70)
saved_ctrl = d.ctrl.copy()
for aidx, jname, tip_b, base_b in test_plan:
    # 先把所有 ctrl 设为 0
    d.ctrl[:] = 0
    # rest
    mujoco.mj_forward(m, d)
    dist_rest = np.linalg.norm(d.xpos[m.body(tip_b).id, :3] - d.xpos[m.body(base_b).id, :3]) * 1000
    # hi：跑 200 步让 servo 收敛
    d.ctrl[:] = 0
    d.ctrl[aidx] = m.actuator_ctrlrange[aidx, 1]
    for _ in range(200):
        mujoco.mj_step(m, d, nstep=1)
    dist_hi = np.linalg.norm(d.xpos[m.body(tip_b).id, :3] - d.xpos[m.body(base_b).id, :3]) * 1000
    # lo
    d.ctrl[:] = 0
    d.ctrl[aidx] = m.actuator_ctrlrange[aidx, 0]
    for _ in range(200):
        mujoco.mj_step(m, d, nstep=1)
    dist_lo = np.linalg.norm(d.xpos[m.body(tip_b).id, :3] - d.xpos[m.body(base_b).id, :3]) * 1000
    trend = "hi=弯" if dist_hi < dist_lo else "hi=伸"
    print(f"{m.actuator(aidx).name:30s}  {dist_rest:7.1f}mm  {dist_hi:7.1f}mm  {dist_lo:7.1f}mm  {trend}")

d.ctrl[:] = saved_ctrl