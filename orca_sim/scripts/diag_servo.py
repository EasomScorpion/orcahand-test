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
print("设 ctrl 后 qpos 实际值是多少（看 servo 是否生效）")
print("=" * 70)

# 看每个 flexion 关节：set ctrl hi/lo 后，qpos 跟到哪？
test = [
    (1, "right_thumb_mcp", 1),
    (3, "right_thumb_pip", 3),
    (4, "right_thumb_dip", 4),
    (6, "right_index_mcp", 6),
    (7, "right_index_pip", 7),
    (9, "right_middle_mcp", 9),
    (10, "right_middle_pip", 10),
    (12, "right_ring_mcp", 12),
    (13, "right_ring_pip", 13),
    (15, "right_pinky_mcp", 15),
    (16, "right_pinky_pip", 16),
]

for aidx, aname, qidx in test:
    saved = d.ctrl.copy()
    # hi
    d.ctrl[:] = 0
    d.ctrl[aidx] = m.actuator_ctrlrange[aidx, 1]
    for _ in range(300):
        mujoco.mj_step(m, d, nstep=1)
    qpos_hi = d.qpos[qidx]
    # lo
    d.ctrl[:] = 0
    d.ctrl[aidx] = m.actuator_ctrlrange[aidx, 0]
    for _ in range(300):
        mujoco.mj_step(m, d, nstep=1)
    qpos_lo = d.qpos[qidx]
    target_hi = m.actuator_ctrlrange[aidx, 1]
    target_lo = m.actuator_ctrlrange[aidx, 0]
    print(f"{aname:25s} target_hi={target_hi:+.3f} reached={qpos_hi:+.3f} | target_lo={target_lo:+.3f} reached={qpos_lo:+.3f}")
    d.ctrl[:] = saved
    # 收敛
    d.ctrl[:] = 0
    for _ in range(100):
        mujoco.mj_step(m, d, nstep=1)