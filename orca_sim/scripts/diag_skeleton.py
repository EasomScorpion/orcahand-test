"""diag_skeleton.py —— 打印 SimSkeleton 抽出的右手 17 body 骨架全貌。

用法：
    python scripts/diag_skeleton.py
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main():
    from orca_sim.envs import OrcaHandRight
    from orca_sim.retarget import SimSkeleton

    env = OrcaHandRight(version="v1", skin=False, render_mode="rgb_array")
    env.reset(seed=42)
    skel = SimSkeleton.from_model(env.unwrapped.model)

    print("=" * 90)
    print(f"SimSkeleton: {len(skel.bones)} body / 17 actuator")
    print(f"palm body: {skel.palm_body_name}")
    print(f"5 fingertip bodies (顺序 thumb→pinky): {skel.fingertip_body_names}")
    print("=" * 90)
    print(f"{'body':22s} {'parent':22s} {'joint':22s} {'axis':15s} {'range':15s} {'ref':7s}")
    print("-" * 90)

    for name in sorted(skel.bones.keys()):
        b = skel.get_bone(name)
        # 找 parent body 名
        from_orca = env.unwrapped.model
        parent_name = (
            env.unwrapped.model.body(b.parent_body_id).name
            if b.parent_body_id > 0 else "world"
        )
        ax = b.joint_axis_local
        ax_str = "no joint" if np.all(ax == 0) else f"({ax[0]:+.2f},{ax[1]:+.2f},{ax[2]:+.2f})"
        rng = b.joint_range if b.joint_id >= 0 else ("-", "-")
        rng_str = (
            f"[{rng[0]:+.3f},{rng[1]:+.3f}]" if isinstance(rng, tuple) else "-"
        )
        ref = f"{b.joint_ref:+.3f}" if b.joint_id >= 0 else "-"
        print(
            f"{name:22s} {parent_name:22s} "
            f"{(env.unwrapped.model.joint(b.joint_id).name if b.joint_id >= 0 else '-'):22s} "
            f"{ax_str:15s} {rng_str:15s} {ref:7s}"
        )

    print()
    print(f"指尖链长 (mp->tip):")
    for finger, length in skel.fingertip_local_lengths().items():
        print(f"  {finger}: {length*1000:5.1f}mm")

    print()
    print(f"qpos<->body 映射 (qpos idx -> body_id):")
    for qa, bid in sorted(skel.qpos_to_body.items()):
        nm = env.unwrapped.model.body(bid).name
        print(f"  qpos[{qa:2d}] -> {nm}")

    env.close()


if __name__ == "__main__":
    import numpy as np
    main()
