import sys
sys.path.insert(0, "src")
import numpy as np
import mujoco
from orca_sim.envs import OrcaHandRight

env = OrcaHandRight(version="v1", skin=False, render_mode="rgb_array")
env.reset(seed=42)
m = env.unwrapped.model
d = env.unwrapped.data

# 看 actuator 0 的完整属性
print("Actuator 0:", m.actuator(0).name)
print("  ctrlrange:", m.actuator_ctrlrange[0])
print("  gear:", m.actuator_gainprm[0])
print("  bias:", m.actuator_biasprm[0])
print("  trntype:", m.actuator_trntype[0])
print("  trnid:", m.actuator_trnid[0])

# mj_default 关节阻尼/弹簧
jid = int(m.actuator_trnid[0, 0])
print(f"\nJoint 0 ({m.joint(jid).name}):")
print(f"  range: {m.jnt_range[jid]}")
print(f"  stiffness (spring): {m.jnt_stiffness[jid]}")
print(f"  damping: {m.jnt_damping[jid]}")
print(f"  armature: {m.jnt_armature[jid]}")

# 测试：直接 mj_forward 不 step，看 ctrl 直接设置的影响
# position actuator 是直接 ctrl → 关节位置
saved = d.qpos.copy()
print(f"\nrest qpos[0]={d.qpos[0]:.4f}")
d.ctrl[0] = m.actuator_ctrlrange[0, 1]  # hi
print(f"set ctrl[0]=hi={m.actuator_ctrlrange[0, 1]:.4f}")
mujoco.mj_forward(m, d)
print(f"after mj_forward, qpos[0]={d.qpos[0]:.4f}")

d.qpos[:] = saved
mujoco.mj_forward(m, d)