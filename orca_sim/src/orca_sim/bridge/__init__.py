"""orca_sim ↔ FTServo_Python 联动子包。

仅作用于 orca_sim 端（v1 右手 + skin=False，17 个 STS3215）。FTServo_Python/ 侧不修改任何文件，
通过 sys.path 复用 :mod:`servo_console` 与 :mod:`scservo_sdk` 的现有 API。

对外接口：
    - :func:`load_joint_mapping`  - 加载 17 舵机 ↔ sim actuator 的权威映射。
    - :class:`CollisionGuard`    - sim 自碰撞守卫（data.contact 过滤）。
    - :class:`SimToRealDeployer` - 弧度→raw→sync_go_to_pose 的确定性映射 + 碰撞预筛选。
    - :func:`build_deployer`     - 一站式工厂：构造 env、连接/装载 bundles、返回 Deployer。

sys.path 注入策略：自动把 ``../FTServo_Python`` 与 ``../FTServo_Python/test`` 加入
``sys.path``（仅在路径存在时）。导入失败抛 :class:`BridgeHardwareUnavailable`。
"""

from __future__ import annotations

import os
import pathlib
import sys

# ----------------------------------------------------------------------
# sys.path 注入：让 FTServo_Python 可被发现
# ----------------------------------------------------------------------
# bridge 包位于 orca_sim/src/orca_sim/bridge/，
# 路径向上 4 级才是 FT/ 大目录。
_HERE = pathlib.Path(__file__).resolve()
_FT_ROOT = _HERE.parents[4] / "FTServo_Python"
if _FT_ROOT.exists():
    for sub in ("", "test"):
        p = str(_FT_ROOT if not sub else _FT_ROOT / sub)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


# ----------------------------------------------------------------------
# 公共异常
# ----------------------------------------------------------------------
class BridgeHardwareUnavailable(RuntimeError):
    """FTServo_Python 端缺失（pyserial / scservo_sdk / servo_console 不可导入）。"""


# ----------------------------------------------------------------------
# 对外导出
# ----------------------------------------------------------------------
from orca_sim.bridge.mapping import (  # noqa: E402
    Mapping,
    load_joint_mapping,
)
from orca_sim.bridge.collision import CollisionGuard  # noqa: E402
from orca_sim.bridge.deploy import SimToRealDeployer, build_deployer  # noqa: E402
from orca_sim.bridge.limits import check_alignment  # noqa: E402

__all__ = [
    "BridgeHardwareUnavailable",
    "Mapping",
    "load_joint_mapping",
    "CollisionGuard",
    "SimToRealDeployer",
    "build_deployer",
    "check_alignment",
]