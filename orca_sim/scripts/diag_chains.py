import sys
sys.path.insert(0, "src")
import mujoco
from orca_sim.envs import OrcaHandRight

env = OrcaHandRight(version="v1", skin=False, render_mode="rgb_array")
env.reset(seed=42)
m = env.unwrapped.model
d = env.unwrapped.data

print("=" * 70)
print("每根手指的关节链：body 名 + joint 名 + actuator 名 + qpos 索引")
print("=" * 70)

for finger_prefix in ["thumb", "index", "middle", "ring", "pinky"]:
    print(f"\n--- {finger_prefix} ---")
    # 找所有以 finger_prefix 开头的 body
    bodies = []
    for i in range(m.nbody):
        nm = m.body(i).name or ""
        if nm.startswith(f"right_{finger_prefix}"):
            bodies.append((i, nm))
    for bid, bname in bodies:
        # 这个 body 的 parent joint
        parent = int(m.body_parentid[bid])
        if parent == 0:
            continue
        # 找连接这个 body 的 joint
        for jid in range(m.njnt):
            if int(m.jnt_bodyid[jid]) == bid:
                jname = m.joint(jid).name
                qa = int(m.jnt_qposadr[jid])
                lo, hi = m.jnt_range[jid]
                # 找驱动这个 joint 的 actuator
                acts = []
                for aid in range(m.nu):
                    if int(m.actuator_trnid[aid, 0]) == jid:
                        acts.append(m.actuator(aid).name)
                print(f"  {bname:25s} <- joint: {jname:25s} qpos_idx={qa:2d} range=[{lo:+.3f},{hi:+.3f}]  actuators: {acts}")
                break