import sys
sys.path.insert(0, "src")
import numpy as np
import mujoco
from orca_sim.envs import OrcaHandRight

env = OrcaHandRight(version="v1", skin=False, render_mode="rgb_array")
env.reset(seed=42)
m = env.unwrapped.model
d = env.unwrapped.data
mujoco.mj_forward(m, d)

print("=" * 70)
print("逐个驱动每个 flexion 关节到 hi/lo，看指尖到 base 的距离")
print("=" * 70)

# 4 个非拇指手指：abd + mcp + pip
# thumb：mcp + abd + pip + dip
# 我们关注 mcp/pip/dip 这三个 flexion 关节
# 已知它们的 joint 名

test_plan = [
    # (actuator_idx, joint_name, tip_body, mcp_body)
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

print(f"\n{'actuator':30s}  {'rest':>8s}  {'hi':>8s}  {'lo':>8s}  hi_vs_lo")
print("-" * 70)
saved = d.qpos.copy()
for aidx, jname, tip_b, base_b in test_plan:
    # rest 距离
    mujoco.mj_forward(m, d)
    dist_rest = np.linalg.norm(d.xpos[m.body(tip_b).id, :3] - d.xpos[m.body(base_b).id, :3]) * 1000
    # hi
    hi = m.actuator_ctrlrange[aidx, 1]
    qa = int(m.jnt_qposadr[int(m.actuator_trnid[aidx, 0])])
    d.qpos[qa] = hi
    mujoco.mj_forward(m, d)
    dist_hi = np.linalg.norm(d.xpos[m.body(tip_b).id, :3] - d.xpos[m.body(base_b).id, :3]) * 1000
    # lo
    lo = m.actuator_ctrlrange[aidx, 0]
    d.qpos[qa] = lo
    mujoco.mj_forward(m, d)
    dist_lo = np.linalg.norm(d.xpos[m.body(tip_b).id, :3] - d.xpos[m.body(base_b).id, :3]) * 1000
    # 恢复
    d.qpos[:] = saved
    trend = "hi=弯" if dist_hi < dist_lo else "hi=伸"
    print(f"{m.actuator(aidx).name:30s}  {dist_rest:7.1f}mm  {dist_hi:7.1f}mm  {dist_lo:7.1f}mm  {trend}")

d.qpos[:] = saved
mujoco.mj_forward(m, d)