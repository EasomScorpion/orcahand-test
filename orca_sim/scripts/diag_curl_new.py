import sys, time
sys.path.insert(0, "src")
import numpy as np
import mujoco
from orca_sim.envs import OrcaHandRight
from orca_sim.retarget import CurlSolver, CurlSolverConfig

env = OrcaHandRight(version="v1", skin=False, render_mode="rgb_array")
env.reset(seed=42)
m = env.unwrapped.model
d = env.unwrapped.data

solver = CurlSolver(m, d, CurlSolverConfig())
t0 = time.time()
# 模拟张开手
landmarks = np.zeros((21, 3), dtype=np.float64)
landmarks[0] = [0, 0, 0]
landmarks[9] = [0, 0, 0.085]  # MIDDLE_MCP
landmarks[8] = [0, -0.05, 0.18]    # INDEX_TIP（远离 MCP = 张开）
landmarks[12] = [0, 0, 0.20]
landmarks[16] = [0.02, 0, 0.18]
landmarks[20] = [0.04, 0.02, 0.15]
landmarks[4] = [-0.04, 0.04, 0.05]
landmarks[2] = [-0.04, 0.02, 0.04]  # THUMB_MCP

# 第一次会触发 lookup 构建
qpos_open = solver.solve(landmarks)
print(f"build + solve 张开: {time.time()-t0:.2f}s")
print(f"qpos 张开: {np.round(qpos_open, 3)}")

# 握拳：所有 TIP 靠近 MCP
landmarks2 = landmarks.copy()
landmarks2[8] = [0, -0.02, 0.10]
landmarks2[12] = [0, 0, 0.08]
landmarks2[16] = [0.01, 0, 0.08]
landmarks2[20] = [0.02, 0.01, 0.06]
landmarks2[4] = [-0.02, 0.02, 0.02]

t0 = time.time()
qpos_fist = solver.solve(landmarks2)
print(f"solve 握拳: {(time.time()-t0)*1000:.1f}ms")
print(f"qpos 握拳: {np.round(qpos_fist, 3)}")

print()
print("=== 对比：哪些关节变化了 ===")
diff = qpos_fist - qpos_open
for i in range(17):
    if abs(diff[i]) > 0.01:
        print(f"  actuator {i:2d}  {m.actuator(i).name:30s}  open={qpos_open[i]:+.3f}  fist={qpos_fist[i]:+.3f}  delta={diff[i]:+.3f}")

# 实际看 mcp actuator 写入 ctrl 后手是否真的动了
print()
print("=== 验证：把 qpos 设为 ctrl，看 mcp=hi/lo 时手指是否真的弯曲 ===")
# 把所有 mcp actuator (6,9,12,15) 设为 hi
d.ctrl[:] = 0
for aidx in [6, 9, 12, 15]:
    d.ctrl[aidx] = m.actuator_ctrlrange[aidx, 1]
for _ in range(200):
    mujoco.mj_step(m, d, nstep=1)
print("mcp=hi 后 5 指尖位置：")
for b in ["right_thumb_dp", "right_index_ip", "right_middle_ip", "right_ring_ip", "right_pinky_ip"]:
    p = d.xpos[m.body(b).id, :3]
    print(f"  {b}: y={p[1]*1000:+.1f}mm z={p[2]*1000:+.1f}mm")

env.close()