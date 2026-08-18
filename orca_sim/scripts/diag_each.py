"""彻底诊断：每个 actuator 单独动到 hi 时，整条手指链上每个 body 的位移。

不预设 mcp/pip/dip 语义——只看谁动了哪里。
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import mujoco
from orca_sim.envs import OrcaHandRight

env = OrcaHandRight(version="v1", skin=False, render_mode="rgb_array")
env.reset(seed=42)
m = env.unwrapped.model
d = env.unwrapped.data

# index 手指的所有 body（按从 root 到 tip）
INDEX_BODIES = ["right_index_mp", "right_index_pp", "right_index_ip"]
THUMB_BODIES = ["right_thumb_mp", "right_thumb_pp", "right_thumb_ip", "right_thumb_dp"]

def measure_bodies(bodies, rest_xpos):
    """对每个 body 算 hi 时相对 rest 的位移"""
    deltas = {}
    for b in bodies:
        idx = m.body(b).id
        cur = d.xpos[idx, :3]
        deltas[b] = (cur - rest_xpos[b]) * 1000  # mm
    return deltas

# rest 状态
d.ctrl[:] = 0
for _ in range(200):
    mujoco.mj_step(m, d, nstep=1)
rest_index = {b: d.xpos[m.body(b).id, :3].copy() for b in INDEX_BODIES}
rest_thumb = {b: d.xpos[m.body(b).id, :3].copy() for b in THUMB_BODIES}

# 每个 index actuator 单独 hi
print("=" * 70)
print("INDEX：每个 actuator 单独 hi 时，链上每个 body 的位移")
print("=" * 70)
for aidx in [5, 6, 7]:  # abd, mcp, pip
    d.ctrl[:] = 0
    d.ctrl[aidx] = m.actuator_ctrlrange[aidx, 1]
    for _ in range(200):
        mujoco.mj_step(m, d, nstep=1)
    deltas = measure_bodies(INDEX_BODIES, rest_index)
    print(f"\nactuator {aidx} ({m.actuator(aidx).name}) = hi:")
    for b, delta in deltas.items():
        norm = np.linalg.norm(delta)
        print(f"  {b:25s}: dx={delta[0]:+6.1f} dy={delta[1]:+6.1f} dz={delta[2]:+6.1f}  |Δ|={norm:5.1f}mm")

# thumb
print("\n" + "=" * 70)
print("THUMB：每个 actuator 单独 hi 时，链上每个 body 的位移")
print("=" * 70)
for aidx in [1, 2, 3, 4]:  # mcp, abd, pip, dip
    d.ctrl[:] = 0
    d.ctrl[aidx] = m.actuator_ctrlrange[aidx, 1]
    for _ in range(200):
        mujoco.mj_step(m, d, nstep=1)
    deltas = measure_bodies(THUMB_BODIES, rest_thumb)
    print(f"\nactuator {aidx} ({m.actuator(aidx).name}) = hi:")
    for b, delta in deltas.items():
        norm = np.linalg.norm(delta)
        print(f"  {b:25s}: dx={delta[0]:+6.1f} dy={delta[1]:+6.1f} dz={delta[2]:+6.1f}  |Δ|={norm:5.1f}mm")