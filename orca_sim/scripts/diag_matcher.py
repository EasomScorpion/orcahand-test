"""diag_matcher.py —— 对 calibration_data.json 跑 BoneMatcher round-trip。

用法：
    python scripts/diag_matcher.py
    python scripts/diag_matcher.py --pose fist
"""

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pose", default=None,
                   help="只看某个手势（默认全跑）")
    p.add_argument("--no-ik", action="store_true",
                   help="不用 LM IK 微调（只看启发式初值）")
    args = p.parse_args()

    import mujoco
    import numpy as np
    from orca_sim.envs import OrcaHandRight
    from orca_sim.retarget import SimSkeleton, BoneMatcher, BoneMatcherConfig

    env = OrcaHandRight(version="v1", skin=False, render_mode="rgb_array")
    env.reset(seed=42)
    m, d = env.unwrapped.model, env.unwrapped.data
    skel = SimSkeleton.from_model(m)
    matcher = BoneMatcher(
        skel,
        BoneMatcherConfig(max_iterations=20, lm_damping=1e-3, use_heuristic_init=True),
    )

    cal = json.load(open("calibration_data.json", encoding="utf-8"))

    print("=" * 90)
    print(f"{'pose':12s} {'qpos':40s} {'solve_ms':>9s}")
    print("-" * 90)

    poses = cal["poses"]
    if args.pose:
        poses = {args.pose: poses[args.pose]}

    for name, info in poses.items():
        lm = np.array(info["landmarks"])
        matcher.reset()
        t0 = time.time()
        if args.no_ik:
            qpos = matcher._heuristic_init(lm, m)
        else:
            qpos = matcher.solve(lm, data=d)
        dt = (time.time() - t0) * 1000
        qstr = "[" + ", ".join(f"{x:+.2f}" for x in qpos[:8]) + ", ...]"
        print(f"{name:12s} {qstr:40s} {dt:7.2f}")

    env.close()


if __name__ == "__main__":
    main()
