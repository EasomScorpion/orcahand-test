# FTServo_Python 教程（STS3215 × 17 个舵机）

> 本教程专为本项目定制：你使用的是 **17 个 STS3215 舵机**（SMS/STS 系列，磁编码、12 位、0~4095），因此只保留 [sms_sts/](sms_sts/) 目录下的相关示例和 API。已删除与 SCSCL、HLS 系列无关的所有内容。

---

## 目录

- [1. 你的硬件与目录对应关系](#1-你的硬件与目录对应关系)
- [2. 准备工作](#2-准备工作)
- [3. 共通模板：所有示例都在套用这套初始化代码](#3-共通模板所有示例都在套用这套初始化代码)
- [4. 第一个程序 ping.py — 探测舵机（先用它确认 17 个舵机都在线）](#4-第一个程序-pingpy--探测舵机先用它确认-17-个舵机都在线)
- [5. read.py — 周期回读位置和速度](#5-readpy--周期回读位置和速度)
- [6. write.py — 单舵机位置控制](#6-writepy--单舵机位置控制)
- [7. read_write.py — 写入并等待到位](#7-read_writepy--写入并等待到位)
- [8. reg_write.py — 异步写 + RegAction 严格同步启动](#8-reg_writepy--异步写--regaction-严格同步启动)
- [9. sync_write.py — 一帧广播写多个舵机](#9-sync_writepy--一帧广播写多个舵机)
- [10. sync_read.py — 一帧广播读多个舵机（17 个舵机必备）](#10-sync_readpy--一帧广播读多个舵机17-个舵机必备)
- [11. wheel.py — 轮式（恒速）模式](#11-wheelpy--轮式恒速模式)
- [12. 单位换算公式](#12-单位换算公式)
- [13. 常见错误码解读](#13-常见错误码解读)
- [14. 故障排查清单](#14-故障排查清单)

---

## 1. 你的硬件与目录对应关系

| 你的硬件 | SDK 适配类 | 示例目录 |
|---|---|---|
| **17 个 STS3215** | `sms_sts` | [`sms_sts/`](sms_sts/) |

库源码固定在 [`scservo_sdk/`](scservo_sdk/)（`port_handler.py` / `protocol_packet_handler.py` / `sms_sts.py` 等）。**SCSCL 与 HLS 的示例和模块与本项目无关**，本教程不再涉及。

STS3215 关键参数（用于贯穿全文）

| 项 | 取值 |
|---|---|
| 位置分辨率 | 12 位（0 ~ 4095） |
| 字节序 | 小端 |
| 加速度 | 支持 |
| 同步读 SYNC READ | ✅ 支持 |
| 同步写 SYNC WRITE | ✅ 支持 |
| 轮式模式 | 通过 `WheelMode()` 切 |
| 默认波特率 | 1 Mbps |
| 默认 ID | 1（出厂默认） |

> 17 个同型号舵机出厂默认 ID 都是 1，必须**逐个上电、单独改 ID**，否则总线会冲突。

---

## 2. 准备工作

### 2.1 硬件连接
- 17 个 STS3215 串联在一根半双工总线上
- 总线接到 USB 转串口模块（FT232 / CP2102 / CH340 都行）
- **总线供电单独接**（7.4V 锂电池或稳压电源），USB 只提供逻辑电平
- 半双工需要方向控制：1kΩ+1kΩ 电阻分压，或专用半双工收发器（如 MAX485 / SP485 / 飞特自家 URT 板）

### 2.2 安装依赖
```bash
pip install pyserial
```

### 2.3 确认串口号

| 系统 | 查看方式 |
|---|---|
| Linux | `ls /dev/ttyUSB*` 或 `/dev/ttyACM*` |
| Windows | 设备管理器 → 端口 (COM 和 LPT) |

把示例代码里的 `/dev/ttyUSB0` 改成你的串口号再运行。

### 2.4 运行目录

所有示例都用 `sys.path.append("..")`，**必须在仓库根目录的下一级运行**：

```bash
cd FTServo_Python/sms_sts
python3 ping.py
```

或者先 `pip install ftservo-python-sdk` 再从任何目录运行（不需要 `sys.path.append("..")`）。

---

## 3. 共通模板：所有示例都在套用这套初始化代码

下面这段几乎出现在**每一个示例**里：

```python
import sys
sys.path.append("..")                    # 让 Python 找到上层目录的 scservo_sdk 包
from scservo_sdk import *                 # 把 SDK 所有符号平铺导入

# ① 创建串口句柄（Windows 改成 "COM3" 等）
portHandler = PortHandler('/dev/ttyUSB0')

# ② 选择适配层（STS3215 固定用 sms_sts）
packetHandler = sms_sts(portHandler)

# ③ 打开串口
if not portHandler.openPort():
    print("Failed to open the port"); quit()

# ④ 设置波特率（必须与舵机一致，1Mbps 是出厂默认）
if not portHandler.setBaudRate(1000000):
    print("Failed to change the baudrate"); quit()

# ⑤ ... 业务代码 ...

# ⑥ 关闭串口
portHandler.closePort()
```

理解这六步，就能读懂所有示例。

---

## 4. 第一个程序 ping.py — 探测舵机（先用它确认 17 个舵机都在线）

> **作用**：检测指定 ID 的舵机是否在总线、通信是否正常，并读出舵机型号号。STS3215 的型号号 = **1540**。

[ping.py](sms_sts/ping.py) 完整内容：

```python
sys.path.append("..")
from scservo_sdk import *

portHandler = PortHandler('/dev/ttyUSB0')
packetHandler = sms_sts(portHandler)

if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port"); quit()

if portHandler.setBaudRate(1000000):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate"); quit()

scs_model_number, scs_comm_result, scs_error = packetHandler.ping(1)
if scs_comm_result != COMM_SUCCESS:
    print(packetHandler.getTxRxResult(scs_comm_result))
else:
    print("[ID:%03d] ping Succeeded. SCServo model number : %d" % (1, scs_model_number))
if scs_error != 0:
    print(packetHandler.getRxPacketError(scs_error))

portHandler.closePort()
```

**返回值**

| 变量 | 含义 |
|---|---|
| `scs_model_number` | 舵机型号（STS3215 = **1540**，可据此核对） |
| `scs_comm_result` | 通信结果码（`COMM_SUCCESS` = 0 为成功） |
| `scs_error` | 舵机状态错误码（电压/角度/过热等） |

**典型成功输出**

```
Succeeded to open the port
Succeeded to change the baudrate
[ID:001] ping Succeeded. SCServo model number : 1540
```

### 4.1 验证 17 个舵机都在线

这是 17 舵机项目**第一步必须跑通**的脚本：

```python
import sys
sys.path.append("..")
from scservo_sdk import *

portHandler = PortHandler('/dev/ttyUSB0')
packetHandler = sms_sts(portHandler)

if not portHandler.openPort(): quit()
if not portHandler.setBaudRate(1000000): quit()

print("Scanning IDs 1..17 ...")
for scs_id in range(1, 18):  # 1~17
    model, comm, err = packetHandler.ping(scs_id)
    if comm == COMM_SUCCESS and err == 0:
        print("  ID:%02d  OK    model=%d" % (scs_id, model))
    else:
        print("  ID:%02d  FAIL  comm=%s err=%s" %
              (scs_id, packetHandler.getTxRxResult(comm),
               packetHandler.getRxPacketError(err)))

portHandler.closePort()
```

**使用建议**

- 17 个舵机出厂默认 ID 全是 1，必须**逐个上电**用 SDK 改成 1~17 后再串联。
- 改 ID 前要 `unLockEprom()`，改完 `LockEprom()`。
- 通信失败但线缆/电源正常时，多半是波特率不匹配。

---

## 5. read.py — 周期回读位置和速度

> **作用**：以 1Hz 周期读取舵机的当前位置和当前速度，纯粹观测，不下发任何指令。

[read.py](sms_sts/read.py) 完整内容：

```python
sys.path.append("..")
from scservo_sdk import *

portHandler = PortHandler('/dev/ttyUSB0')
packetHandler = sms_sts(portHandler)

if not portHandler.openPort(): quit()
if not portHandler.setBaudRate(1000000): quit()

while 1:
    scs_present_position, scs_present_speed, scs_comm_result, scs_error = \
        packetHandler.ReadPosSpeed(1)
    if scs_comm_result != COMM_SUCCESS:
        print(packetHandler.getTxRxResult(scs_comm_result))
    else:
        print("[ID:%03d] PresPos:%d PresSpd:%d" % (1, scs_present_position, scs_present_speed))
    if scs_error != 0:
        print(packetHandler.getRxPacketError(scs_error))
    time.sleep(1)
```

**API 详解**

| 调用 | 返回 |
|---|---|
| `ReadPos(id)` | `(position, comm_result, error)` |
| `ReadSpeed(id)` | `(speed, comm_result, error)` |
| `ReadPosSpeed(id)` | `(position, speed, comm_result, error)` ← 一次读 4 字节，**更快** |
| `ReadMoving(id)` | `(moving(0/1), comm_result, error)` |

**使用建议**

- `ReadPosSpeed` 比 `ReadPos + ReadSpeed` 快接近一倍（一次往返 vs 两次往返）。
- 17 舵机场景下不要这样单循环轮询，会非常慢；改用 [`sync_read.py`](#10-sync_readpy--一帧广播读多个舵机17-个舵机必备) 一次读全部。

---

## 6. write.py — 单舵机位置控制

> **作用**：让单个舵机在位置 0 和位置 4095 之间来回摆动。

[write.py](sms_sts/write.py) 完整内容：

```python
sys.path.append("..")
from scservo_sdk import *

portHandler = PortHandler('/dev/ttyUSB0')
packetHandler = sms_sts(portHandler)

if not portHandler.openPort(): quit()
if not portHandler.setBaudRate(1000000): quit()

while 1:
    # 转到 4095：速度 V=60×0.732=43.92rpm，加速度 A=50×8.7deg/s²
    scs_comm_result, scs_error = packetHandler.WritePosEx(1, 4095, 60, 50)
    if scs_comm_result != COMM_SUCCESS:
        print(packetHandler.getTxRxResult(scs_comm_result))
    elif scs_error != 0:
        print(packetHandler.getRxPacketError(scs_error))

    time.sleep(((4095-0)/(60*50) + (60*50)/(50*100) + 0.05))

    # 回到 0
    scs_comm_result, scs_error = packetHandler.WritePosEx(1, 0, 60, 50)
    if scs_comm_result != COMM_SUCCESS:
        print(packetHandler.getTxRxResult(scs_comm_result))
    elif scs_error != 0:
        print(packetHandler.getRxPacketError(scs_error))

    time.sleep(((4095-0)/(60*50) + (60*50)/(50*100) + 0.05))

portHandler.closePort()
```

**`WritePosEx(id, position, speed, acc)`** 是位置模式最常用的接口，写一次立即执行。

---

## 7. read_write.py — 写入并等待到位

> **作用**：下发目标位置后，**轮询 ReadMoving 直到到位**才发下一条指令 —— 比 `sleep` 固定时长更稳健。

[read_write.py](sms_sts/read_write.py)：

```python
def read(SCS_ID):
    while 1:
        scs_present_position, scs_present_speed, scs_comm_result, scs_error = \
            packetHandler.ReadPosSpeed(SCS_ID)
        if scs_comm_result != COMM_SUCCESS:
            print(packetHandler.getTxRxResult(scs_comm_result))
        else:
            print("[ID:%03d] PresPos:%d PresSpd:%d" % (SCS_ID, scs_present_position, scs_present_speed))
        if scs_error != 0:
            print(packetHandler.getRxPacketError(scs_error))

        moving, scs_comm_result, scs_error = packetHandler.ReadMoving(SCS_ID)
        if scs_comm_result != COMM_SUCCESS:
            print(packetHandler.getTxRxResult(scs_comm_result))

        if moving == 0:
            break
    return

# ... 初始化略 ...

while 1:
    packetHandler.WritePosEx(1, 4095, 60, 50)
    read(1)   # 等到位
    packetHandler.WritePosEx(1, 0,   60, 50)
    read(1)
```

**注意源码里 `read(SCS_ID)` 内部实际写死了 ID 1**（`ReadPosSpeed(1)` / `ReadMoving(1)`），是 bug，多舵机场景应改为 `(SCS_ID)`。

**使用建议**

- 轨迹段之间需要"等到位"的场景，用这种 ReadMoving 轮询方式更可靠。
- 严格实时场景可以把 `time.sleep` 去掉，纯靠 while 1 轮询。

---

## 8. reg_write.py — 异步写 + RegAction 严格同步启动

> **作用**：把指令先缓存到每个舵机的寄存器，**等所有舵机都缓存完毕后再广播 ACTION 触发**，确保 17 个舵机**严格同时启动**，避免因串口通信延迟造成的相位差。

[reg_write.py](sms_sts/reg_write.py)：

```python
sys.path.append("..")
from scservo_sdk import *

portHandler = PortHandler('/dev/ttyUSB0')
packetHandler = sms_sts(portHandler)

if not portHandler.openPort(): quit()
if not portHandler.setBaudRate(1000000): quit()

while 1:
    # 把 ID 1~17 都下发到位置 4095（这一步只写寄存器，不执行！）
    for scs_id in range(1, 18):
        scs_comm_result, scs_error = packetHandler.RegWritePosEx(scs_id, 4095, 60, 50)
        if scs_comm_result != COMM_SUCCESS:
            print(packetHandler.getTxRxResult(scs_comm_result))
        if scs_error != 0:
            print(packetHandler.getRxPacketError(scs_error))
    packetHandler.RegAction()

    time.sleep(((4095-0)/(60*50) + (60*50)/(50*100) + 0.05))

    # 同样方式下发回 0
    for scs_id in range(1, 18):
        scs_comm_result, scs_error = packetHandler.RegWritePosEx(scs_id, 0, 60, 50)
        if scs_comm_result != COMM_SUCCESS:
            print(packetHandler.getTxRxResult(scs_comm_result))
        if scs_error != 0:
            print(packetHandler.getRxPacketError(scs_error))
    packetHandler.RegAction()

    time.sleep(((4095-0)/(60*50) + (60*50)/(50*100) + 0.05))
```

**和 write.py 的关键区别**

| | write.py | reg_write.py |
|---|---|---|
| 启动时刻 | 每个舵机收到指令就启动 | 全部指令到达后，ACTION 一齐触发 |
| 多舵机同步 | ❌ 受通信延迟影响 | ✅ 严格同步 |
| 适用场景 | 单舵机调试 | 17 舵机仿生动作、机械臂协调 |

---

## 9. sync_write.py — 一帧广播写多个舵机

> **作用**：把 17 个舵机的位置/速度/加速度打包成**一帧**发到总线，比逐个下发快得多，并且**一帧下发的时间一致性比 RegWrite 还高**。

[sync_write.py](sms_sts/sync_write.py)：

```python
sys.path.append("..")
from scservo_sdk import *

portHandler = PortHandler('/dev/ttyUSB0')
packetHandler = sms_sts(portHandler)

if not portHandler.openPort(): quit()
if not portHandler.setBaudRate(1000000): quit()

while 1:
    # 把 ID 1~17 的目标加入同步写队列
    for scs_id in range(1, 18):
        scs_addparam_result = packetHandler.SyncWritePosEx(scs_id, 4095, 60, 50)
        if scs_addparam_result != True:
            print("[ID:%03d] groupSyncWrite addparam failed" % scs_id)

    # 一帧广播写入所有舵机
    scs_comm_result = packetHandler.groupSyncWrite.txPacket()
    if scs_comm_result != COMM_SUCCESS:
        print(packetHandler.getTxRxResult(scs_comm_result))

    # 必须 clear，否则下次会重复发旧数据
    packetHandler.groupSyncWrite.clearParam()

    time.sleep(((4095-0)/(60*50) + (60*50)/(50*100) + 0.05))

    for scs_id in range(1, 18):
        scs_addparam_result = packetHandler.SyncWritePosEx(scs_id, 0, 60, 50)
        if scs_addparam_result != True:
            print("[ID:%03d] groupSyncWrite addparam failed" % scs_id)

    scs_comm_result = packetHandler.groupSyncWrite.txPacket()
    if scs_comm_result != COMM_SUCCESS:
        print(packetHandler.getTxRxResult(scs_comm_result))

    packetHandler.groupSyncWrite.clearParam()

    time.sleep(((4095-0)/(60*50) + (60*50)/(50*100) + 0.05))
```

**API**

```python
packetHandler.SyncWritePosEx(id, position, speed, acc)
# 内部写 ACC~GOAL_SPEED 共 7 字节（小端）

packetHandler.groupSyncWrite.txPacket()    # 一帧广播
packetHandler.groupSyncWrite.clearParam()  # 必须清空
```

**和 reg_write 的区别**

| | reg_write | sync_write |
|---|---|---|
| 通信帧数 | 1 帧 N 次（每舵机一发） + 1 帧 ACTION | 1 帧 N（一个广播包） |
| 时间一致性 | 高（取决于 ACTION 时刻） | 极高（总线一帧下发，硬件同步启动） |

---

## 10. sync_read.py — 一帧广播读多个舵机（17 个舵机必备）

> **作用**：一帧发起读请求，17 个舵机各自回包，由 SDK 拆分到 `data_dict`，是**多舵机实时状态回读**的关键（50Hz+ 也能扛）。

[sync_read.py](sms_sts/sync_read.py)：

```python
sys.path.append("..")
from scservo_sdk import *

portHandler = PortHandler('/dev/ttyUSB0')
packetHandler = sms_sts(portHandler)

if not portHandler.openPort(): quit()
if not portHandler.setBaudRate(1000000): quit()

# 从 PRESENT_POSITION_L (地址 56) 开始读 4 字节（位置 2B + 速度 2B）
groupSyncRead = GroupSyncRead(packetHandler, SMS_STS_PRESENT_POSITION_L, 4)

while 1:
    # ① 加入 ID 1~17
    for scs_id in range(1, 18):
        scs_addparam_result = groupSyncRead.addParam(scs_id)
        if scs_addparam_result != True:
            print("[ID:%03d] groupSyncRead addparam failed" % scs_id)

    # ② 一帧发请求 + 收所有回包
    scs_comm_result = groupSyncRead.txRxPacket()
    if scs_comm_result != COMM_SUCCESS:
        print(packetHandler.getTxRxResult(scs_comm_result))

    # ③ 逐个读
    for scs_id in range(1, 18):
        scs_data_result, scs_error = groupSyncRead.isAvailable(scs_id, SMS_STS_PRESENT_POSITION_L, 4)
        if scs_data_result == True:
            pos = groupSyncRead.getData(scs_id, SMS_STS_PRESENT_POSITION_L, 2)
            spd = groupSyncRead.getData(scs_id, SMS_STS_PRESENT_SPEED_L, 2)
            print("[ID:%03d] PresPos:%d PresSpd:%d" % (scs_id, pos, packetHandler.scs_tohost(spd, 15)))
        else:
            print("[ID:%03d] groupSyncRead getdata failed" % scs_id)
            continue
        if scs_error != 0:
            print(packetHandler.getRxPacketError(scs_error))

    # ④ 一定要 clearParam
    groupSyncRead.clearParam()
    time.sleep(1)
```

**为什么速度要 `scs_tohost(...)`？** 因为 STS3215 用 15 位"符号位+绝对值"表示负速度，库读出来是 16 位无符号整数，必须用 `scs_tohost(value, 15)` 还原成有符号数。

**17 舵机实测性能参考**

- 1Mbps 下一次 sync_read 约 3~5ms（含 17 个回包）
- 去掉 `time.sleep(1)`，循环即可跑到 50~100Hz

**SMS/STS 系列寄存器速查**

| 常量 | 地址 | 含义 |
|---|---|---|
| `SMS_STS_PRESENT_POSITION_L/H` | 56, 57 | 当前 位置（2 字节） |
| `SMS_STS_PRESENT_SPEED_L/H` | 58, 59 | 当前 速度（2 字节，符号位编码） |
| `SMS_STS_PRESENT_LOAD_L/H` | 60, 61 | 当前 负载 |
| `SMS_STS_PRESENT_VOLTAGE` | 62 | 当前 电压（V×10） |
| `SMS_STS_PRESENT_TEMPERATURE` | 63 | 当前 温度（°C） |
| `SMS_STS_MOVING` | 66 | 运动标志 |
| `SMS_STS_PRESENT_CURRENT_L/H` | 69, 70 | 当前 电流 |

> **小技巧**：要做电流/温度监控时，把 `GroupSyncRead(packetHandler, addr, 4)` 中的 `addr` 换成对应寄存器，按 4 字节或 2 字节读即可。

---

## 11. wheel.py — 轮式（恒速）模式

> **作用**：让 STS3215 像直流电机一样**连续旋转**，可指定速度和方向。

[wheel.py](sms_sts/wheel.py)：

```python
sys.path.append("..")
from scservo_sdk import *

portHandler = PortHandler('/dev/ttyUSB0')
packetHandler = sms_sts(portHandler)

if not portHandler.openPort(): quit()
if not portHandler.setBaudRate(1000000): quit()

# 一次性切模式（写入 EPROM 后掉电保持）
scs_comm_result, scs_error = packetHandler.WheelMode(1)
if scs_comm_result != COMM_SUCCESS:
    print(packetHandler.getTxRxResult(scs_comm_result))
elif scs_error != 0:
    print(packetHandler.getRxPacketError(scs_error))

while 1:
    # 正转
    scs_comm_result, scs_error = packetHandler.WriteSpec(1, 60, 50)   # speed=60, acc=50
    time.sleep(5)

    # 停止（速度降到 0）
    scs_comm_result, scs_error = packetHandler.WriteSpec(1, 0, 50)
    time.sleep(2)

    # 反转（速度为负）
    scs_comm_result, scs_error = packetHandler.WriteSpec(1, -50, 50)
    time.sleep(5)

    # 停止
    scs_comm_result, scs_error = packetHandler.WriteSpec(1, 0, 50)
    time.sleep(2)
```

**`WriteSpec(id, speed, acc)`** 三参数：
- `speed` **支持负数**表示反转
- `acc=0` 表示无加速度限制（瞬时变速）

**注意**：STS3215 在 `WheelMode()` 下**位置限位失效**，要回到位置控制模式需调用 `WritePosEx` 或重新给 `MODE` 寄存器写 0。

---

## 12. 单位换算公式

### 12.1 速度 V
```
实际转速 (rpm) = V × 0.732
V 范围：0 ~ 4095（注意 speed 用 15 位符号位编码，负数最高位置 1）
```

### 12.2 加速度 A
```
实际加速度 (deg/s²) = A × 8.7
A 范围：0 ~ 254
```

### 12.3 位置 P
```
实际角度 (deg) = P × (360 / 4096) ≈ P × 0.0879
P 范围：0 ~ 4095
```

### 12.4 等待时间估算

代码中常出现这样的 sleep：
```python
time.sleep(((4095-0)/(60*50) + (60*50)/(50*100) + 0.05))
```

拆解含义：
- `(P1-P0)/(V*50)` = **匀速段时间**（V×50 是 "步/s"，再换算到秒级）
- `(V*50)/(A*100)` = **加减速段时间**（A×100 是"步/s²"，加上起步到匀速的时间）
- `+ 0.05` = 安全余量 50ms

> 这是飞特官方示例里的经验公式，物理量纲不一定严谨，但作为 sleep 时长是够用的。如果要求严格到点停下，请用 [`read_write.py` 模式](#7-read_writepy--写入并等待到位)的 ReadMoving 轮询。

---

## 13. 常见错误码解读

### 13.1 通信返回码（`scs_comm_result`）

由 `packetHandler.getTxRxResult(code)` 翻译：

| 码 | 含义 | 常见原因 |
|---|---|---|
| `COMM_SUCCESS(0)` | 成功 | — |
| `COMM_PORT_BUSY(-1)` | 端口忙 | 上一个操作未完成；多线程冲突 |
| `COMM_TX_FAIL(-2)` | 发送失败 | 写串口字节数对不上，端口已断开 |
| `COMM_RX_FAIL(-3)` | 接收失败 | 同上 |
| `COMM_TX_ERROR(-4)` | 包长度超 250 | 罕见，几乎只发生在拼错包时 |
| `COMM_RX_WAITING(-5)` | 仍在接收 | 多见于调试；正常流程不应出现 |
| `COMM_RX_TIMEOUT(-6)` | 接收超时 | 舵机没回包 → ID 错 / 波特率错 / 断电 / 半双工线序错 |
| `COMM_RX_CORRUPT(-7)` | 数据损坏 / 校验和错 | 总线噪声太大，加磁环；或波特率不匹配 |
| `COMM_NOT_AVAILABLE(-9)` | 不支持 | 调用了不适用于该系列的 API |

### 13.2 舵机状态错误（`scs_error`）

由 `packetHandler.getRxPacketError(error)` 翻译（位标志）：

| 位 | 含义 | 处理建议 |
|---|---|---|
| `ERRBIT_VOLTAGE(1)` | 输入电压异常 | 检查供电（6~12V） |
| `ERRBIT_ANGLE(2)` | 角度传感器异常 | 重新上电或 `reOfsCal` |
| `ERRBIT_OVERHEAT(4)` | 过热 | 降负载或加散热 |
| `ERRBIT_OVERELE(8)` | 过流 | 检查机械是否卡死 |
| `ERRBIT_OVERLOAD(32)` | 过载 | 减负载或提高 `MAX_TORQUE` |

---

## 14. 故障排查清单

按顺序逐项排查可以解决 95% 的问题：

1. **跑 [§4 的"扫描 ID 1~17"脚本](#41-验证-17-个舵机都在线)**
   - 17 个舵机必须全部 OK 才算总线就绪
2. **物理连接**
   - USB 转串口是否被系统识别（Linux: `dmesg | grep ttyUSB`）
   - 总线供电是否充足（7.4V/2A 起，17 个舵机建议 5A+）
   - 半双工接线是否正确（TX/RX 是否走 1kΩ+1kΩ 电阻或专用方向控制芯片）
3. **波特率**
   - 默认 1Mbps，新买的舵机或用上位机改过波特率的话，要和舵机一致
4. **ID 冲突**
   - 17 个 STS3215 的 ID 必须 1~17 唯一；多个相同 ID 会同时回包，校验和失败
5. **运动卡死**
   - 加大 `MAX_TORQUE` / `MIN_STARTUP_FORCE`，或降低机械负载
6. **回读不到数据**
   - `sync_read` 时确保 `addParam` 后 `txRxPacket` 成功，且每个 ID 都 `isAvailable` 验证
7. **多舵机相位差**
   - 用 [`sync_write.py`](#9-sync_writepy--一帧广播写多个舵机) 代替 [`reg_write.py`](#8-reg_writepy--异步写--regaction-严格同步启动)，进一步降低延迟
8. **通信噪声大**
   - 总线加磁环/双绞线，或把波特率从 1Mbps 降到 500K
9. **修改 ID/波特率不生效**
   - 改 EPROM 前必须 `unLockEprom()`，改完 `LockEprom()`
   - 改波特率后必须同步改代码里的 `setBaudRate`
10. **17 舵机总线响应慢**
    - 确认 `LATENCY_TIMER = 50` 是否过大，可改到 5~16 ms

---

## 附录：示例代码索引

| 程序 | 文件 |
|---|---|
| ping.py | [sms_sts/ping.py](sms_sts/ping.py) |
| read.py | [sms_sts/read.py](sms_sts/read.py) |
| write.py | [sms_sts/write.py](sms_sts/write.py) |
| reg_write.py | [sms_sts/reg_write.py](sms_sts/reg_write.py) |
| read_write.py | [sms_sts/read_write.py](sms_sts/read_write.py) |
| sync_read.py | [sms_sts/sync_read.py](sms_sts/sync_read.py) |
| sync_write.py | [sms_sts/sync_write.py](sms_sts/sync_write.py) |
| wheel.py | [sms_sts/wheel.py](sms_sts/wheel.py) |

库内部实现请见 [scservo_sdk/README.md](scservo_sdk/README.md)。