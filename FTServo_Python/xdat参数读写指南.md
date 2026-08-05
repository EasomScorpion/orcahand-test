# STS3215 `.xdat` 参数读写指南

本文档配套仓库根目录下的 `xdat_tool.py` 使用。所有操作均使用同一个工具完成。

---

## 1. 文件格式

`.xdat` 是 Feetech 调试助手（FD）导出的 **STS3215 EPROM 区 dump**，**每个文件 51 字节**。

| 偏移 | 长度 | 含义 |
|------|------|------|
| `0x00` – `0x01` | 2 | 文件头（固定 `00 28`，调试软件标记，非校验码） |
| `0x02` – `0x32` | 49 | EPROM 寄存器数据，从 STS3215 地址 0 开始，**多字节字段小端序** |

> 没有 CRC / checksum。Feetech 没有按字节校验，因此解析时只比对长度与魔术头即可。

### 寄存器表（地址 → 字段名）

```text
 0  固件主版本       (u8)   只读
 1  固件次版本       (u8)   只读
 2  预留            (u8)
 3  舵机主版本       (u8)   只读
 4  舵机次版本       (u8)   只读
 5  ID             (u8)
 6  波特率           (u8)   0=1M, 1=0.5M, 2=250K, 3=128K, 4=115200 ...
 7  返回延时         (u8)
 8  应答状态级别      (u8)   0=不响应, 1=只读, 2=全部响应
 9  最小角度限制      (u16, LE)   0-4095
11  最大角度限制      (u16, LE)   0-4095
13  最高温度上限      (u8)   ℃
14  最高输入电压      (u8)   ×0.1 V
15  最低输入电压      (u8)   ×0.1 V
16  最大扭矩         (u16, LE)   0-1000
18  相位            (u8)
19  卸载条件         (u8)   位掩码
20  LED 报警条件     (u8)
21  P 系数          (u8)
22  D 系数          (u8)
23  I 系数          (u8)
24  最小启动力       (u16, LE)
26  顺时针死区       (u8)
27  逆时针死区       (u8)
28  保护电流         (u16, LE)   mA
30  角度分辨率       (u8)   1=360°, 0=300°
31  位置偏移         (u16, LE)   0-4095 (机械零位补偿)
33  运行模式         (u8)   0=位置, 1=速度, 2=PWM
34  保护扭矩         (u8)   %
35  保护时间         (u8)   ×10 ms
36  过载扭矩         (u8)   %
37  速度闭环 P       (u8)
38  过流保护时间      (u8)
39  速度闭环 I       (u8)
```

角度换算公式：  
**角度° = 原始值 × 360 / 4095**（分辨率=1 时）  
**角度° = 原始值 × 300 / 4095**（分辨率=0 时，老固件）

---

## 2. 工具安装

工具零依赖，使用 **Python 3.7+**：

```bash
# 不需要 pip install
python xdat_tool.py --help
```

---

## 3. 读取参数

### 3.1 读取单个文件

```bash
python xdat_tool.py read 参数/3.xdat
```

输出示例：

```
=== 参数\3.xdat ===
    [ 0] 固件主版本              = 3       (默认 3)
    [ 1] 固件次版本              = 10      (默认 10)
    [ 5] ID                 = 3       (默认 1)
    [ 9] 最小角度限制             = 1800    [158.24°]  (默认 0)
   [11] 最大角度限制             = 2452    [215.56°]  (默认 4095)
   ...
```

带 `*` 的字段表示**实测值 ≠ 默认参考值**，需重点关注。

### 3.2 读取整个文件夹

```bash
python xdat_tool.py read 参数/
```

会按 ID 顺序依次打印所有 17 个 `.xdat`。

### 3.3 当作 Python 模块用

```python
from xdat_tool import read_xdat

d = read_xdat('参数/3.xdat')
print(d['ID'], d['最小角度限制'], d['位置偏移'])
```

`read_xdat()` 返回一个 `dict`，key 是中文字段名，value 是整数。

---

## 4. 修改参数

### 4.1 命令行 set（带自动备份）

```bash
python xdat_tool.py set 参数/3.xdat 最大角度限制 2600
# → 参数/3.xdat: 最大角度限制 2452 → 2600
# → 原文件已备份到 参数/3.xdat.bak
```

- `字段名` 支持**模糊匹配**（如 `角度` 会命中 `最大角度限制`）
- 新值是 **10 进制整数**
- 第一次写时会自动把原文件改名 `<原文件>.bak`
- 范围越界会被拒绝（例如把 `ID` 改成 256）

### 4.2 还原某字段为默认

```python
from xdat_tool import read_xdat, default_fields, write_xdat

d = read_xdat('参数/3.xdat')
d['P 系数'] = default_fields()['P 系数']
write_xdat('参数/3.xdat', d)
```

或者直接整体还原：

```bash
python xdat_tool.py restore 参数/3.xdat
```

### 4.3 批量修改并打包新文件

```python
import os
from xdat_tool import read_xdat, write_xdat

src_dir = '参数'
dst_dir = '参数_new'
os.makedirs(dst_dir, exist_ok=True)

for fn in sorted(os.listdir(src_dir)):
    if not fn.endswith('.xdat'): continue
    d = read_xdat(os.path.join(src_dir, fn))
    # 例: 把所有 ID=3 改成 ID=30 (假设物理上是另一个总线)
    if d['ID'] == 3:
        d['ID'] = 30
    write_xdat(os.path.join(dst_dir, fn), d)
```

---

## 5. 导出 CSV 汇总

```bash
python xdat_tool.py csv 参数 -o 参数汇总.csv
```

生成 UTF-8-BOM 编码的 CSV，**Excel / WPS 直接双击打开不乱码**。

CSV 列：

| 列 | 含义 |
|---|---|
| 文件, ID | 元信息 |
| 最小角度(°), 最大角度(°), 中位角度(°), 角度幅度(°) | 由原始 4096 编码换算得到 |
| 位置偏移(°) | 中位零位补偿角度 |
| 固件主/次版本, 舵机主/次版本 | 版本信息 |
| 其余字段名 | 原始整数值 |

---

## 6. 进阶：把参数**真正烧进舵机**

`.xdat` 只是离线的配置快照。如果想刷到真实舵机，使用仓库自带的 [scservo_sdk](scservo_sdk/)。

```python
from scservo_sdk import PortHandler, sms_sts

port = PortHandler('COM5')      # Windows; Linux 是 /dev/ttyUSB0
sms   = sms_sts(port)
port.openPort()
sms.setBaudRate(1000000)        # 跟 .xdat 里的波特率保持一致

# 把 1.xdat 烧给 ID=1 的舵机
from xdat_tool import read_xdat
cfg = read_xdat('参数/1.xdat')

target_id = cfg['ID']
sms.WriteByte(target_id, 21, cfg['P 系数'])           # 地址 21 = P_COEF
sms.WriteWord(target_id, 31, cfg['位置偏移'])        # 地址 31 = OFS_L

# 改完后,把舵机的 EPROM 再 dump 一次,对比是否一致
sms.SaveEEPROM(target_id)
```

> ⚠ **操作前请断电保护**：修改 EPROM 时舵机可能短暂失联，建议一次只改一台、每个字段之间 `time.sleep(0.05)`。

---

## 7. 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| `文件长度 0 ≠ 预期 51` | 文件被截断或编码转换过 | 重新从调试助手导出 |
| 解析后角度值像乱码 | 文件头不是 `00 28` | 有些版本调试助手导出用别的头，手工 import 后改头部再 dump |
| 烧不进舵机 | 波特率不匹配 | 先确认 `.xdat` 第 6 字节（地址 6 = 波特率），与 `setBaudRate()` 一致 |
| 修改 ID 后找不到舵机 | 总线 ID 冲突 | 改 ID 前确认该地址没被占用 |
| 角度值跟实际差 90° | 0/1 边沿方向 | 把 `位置偏移` 加 / 减 1024 后重测 |

---

## 8. 命令总览

```text
python xdat_tool.py read <文件/目录>          # 打印参数表
python xdat_tool.py csv <目录> [-o 输出.csv]   # 批量导出
python xdat_tool.py set <文件> <字段> <新值>    # 修改一项
python xdat_tool.py restore <文件>            # 全部还原
```

工具源码 [xdat_tool.py](xdat_tool.py)，导入时直接 `from xdat_tool import read_xdat, write_xdat, default_fields`。
