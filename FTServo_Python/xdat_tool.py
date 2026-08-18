#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STS3215 .xdat 参数读写工具
===========================
Feetech 调试助手导出的 .xdat 文件解析与回写工具。

文件格式 (51 字节):
    [0x00-0x01]  文件头 (固定 0x00 0x28, 调试软件标记)
    [0x02-0x32]  49 字节寄存器数据 (STS3215 EPROM, 地址 0-48)
                 多字节字段使用小端序

用法:
    python xdat_tool.py read <文件/目录>
    python xdat_tool.py csv <参数目录> [-o 参数汇总.csv]
    python xdat_tool.py set <文件> <字段> <新值>
    python xdat_tool.py restore <文件>     # 还原所有字段为默认值
"""

import sys
import csv
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# 寄存器表 (地址, 名称, 默认参考值, 长度字节, 区域)
# 数据来源: scservo_sdk/sms_sts.py + STS3215 飞特官方寄存器表
# ---------------------------------------------------------------------------
REGISTERS = [
    (0,  "固件主版本",        3,    1, "EPROM"),
    (1,  "固件次版本",        10,   1, "EPROM"),
    (2,  "预留",              0,    1, "EPROM"),
    (3,  "舵机主版本",        9,    1, "EPROM"),
    (4,  "舵机次版本",        3,    1, "EPROM"),
    (5,  "ID",                1,    1, "EPROM"),
    (6,  "波特率",            0,    1, "EPROM"),
    (7,  "返回延时",          0,    1, "EPROM"),
    (8,  "应答状态级别",      1,    1, "EPROM"),
    (9,  "最小角度限制",      0,    2, "EPROM"),
    (11, "最大角度限制",      4095, 2, "EPROM"),
    (13, "最高温度上限(℃)",   70,   1, "EPROM"),
    (14, "最高电压(0.1V)",    80,   1, "EPROM"),
    (15, "最低电压(0.1V)",    40,   1, "EPROM"),
    (16, "最大扭矩",          50,   2, "EPROM"),
    (18, "相位",              12,   1, "EPROM"),
    (19, "卸载条件",          44,   1, "EPROM"),
    (20, "LED报警条件",       47,   1, "EPROM"),
    (21, "P系数",             32,   1, "EPROM"),
    (22, "D系数",             32,   1, "EPROM"),
    (23, "I系数",             0,    1, "EPROM"),
    (24, "最小启动力",        16,   2, "EPROM"),
    (26, "顺时针死区",        1,    1, "EPROM"),
    (27, "逆时针死区",        1,    1, "EPROM"),
    (28, "保护电流(mA)",      500,  2, "EPROM"),
    (30, "角度分辨率",        1,    1, "EPROM"),
    (31, "位置偏移",          977,  2, "EPROM"),
    (33, "运行模式",          0,    1, "EPROM"),
    (34, "保护扭矩",          10,   1, "EPROM"),
    (35, "保护时间",          10,   1, "EPROM"),
    (36, "过载扭矩",          10,   1, "EPROM"),
    (37, "速度闭环P",         10,   1, "EPROM"),
    (38, "过流保护时间",      200,  1, "EPROM"),
    (39, "速度闭环I",         200,  1, "EPROM"),
]

HEADER_MAGIC = bytes([0x00, 0x28])
PAYLOAD_SIZE = 49
FILE_SIZE = 2 + PAYLOAD_SIZE  # 51 字节

# 角度相关的字段 (原始 0-4095 编码 ↔ 0-360 度)
ANGLE_FIELDS = {"最小角度限制", "最大角度限制", "位置偏移"}


def check_payload_size():
    """校验寄存器表是否能装进 49 字节"""
    last_addr = max(a + s for a, _, _, s, _ in REGISTERS)
    assert last_addr <= PAYLOAD_SIZE, f"寄存器表越界: {last_addr} > {PAYLOAD_SIZE}"


# ---------------------------------------------------------------------------
# 核心: 读取与写入
# ---------------------------------------------------------------------------
def read_xdat(path):
    """读取单个 .xdat, 返回 {字段名: 整数值}"""
    with open(path, 'rb') as f:
        raw = f.read()
    if len(raw) != FILE_SIZE:
        raise ValueError(f"{path}: 文件长度 {len(raw)} ≠ 预期 {FILE_SIZE}")
    if raw[:2] != HEADER_MAGIC:
        print(f"⚠  {path}: 文件头为 {raw[:2].hex()} (非标准 {HEADER_MAGIC.hex()})",
              file=sys.stderr)
    payload = raw[2:]
    out = {}
    for addr, name, default, size, region in REGISTERS:
        if size == 1:
            v = payload[addr]
        else:                                              # 2 字节小端
            v = payload[addr] + payload[addr + 1] * 256
        out[name] = v
    return out


def write_xdat(path, fields):
    """根据字段字典写回 .xdat (保留 2 字节文件头)"""
    buf = bytearray(HEADER_MAGIC + bytes(PAYLOAD_SIZE))
    for addr, name, default, size, region in REGISTERS:
        if name == "预留":
            continue                                          # 跳过保留字段
        v = int(fields.get(name, default))
        if v < 0 or v >= (1 << (8 * size)):
            raise ValueError(f"{name}={v} 超出 {size} 字节范围")
        if size == 1:
            buf[2 + addr] = v & 0xFF
        else:
            buf[2 + addr]     = v & 0xFF
            buf[2 + addr + 1] = (v >> 8) & 0xFF
    Path(path).write_bytes(bytes(buf))


def default_fields():
    """返回一份全新默认值的字段字典"""
    return {name: default for _, name, default, _, _ in REGISTERS}


# ---------------------------------------------------------------------------
# 派生: CSV 输出
# ---------------------------------------------------------------------------
def parse_folder(folder):
    """解析整个文件夹, 返回 [{__file__, ID, ...字段...}, ...]"""
    files = sorted(Path(folder).glob('*.xdat'),
                   key=lambda p: int(p.stem))
    results = []
    for p in files:
        d = read_xdat(p)
        d['__file__'] = p.name
        results.append(d)
    return results


def export_csv(results, out_path):
    """导出为 UTF-8-BOM 编码 CSV (Excel 直接打开不乱码)"""
    # 字段顺序: 文件名 / ID / 角度派生 / EPROM 字段
    base_fields = [name for _, name, _, _, _ in REGISTERS
                   if name not in ("ID", "预留")]

    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        # 表头
        w.writerow(['文件', 'ID', '最小角度(°)', '最大角度(°)', '中位角度(°)',
                    '角度幅度(°)', '位置偏移(°)'] + base_fields)
        for r in results:
            mn = r['最小角度限制']
            mx = r['最大角度限制']
            mid = round((mn + mx) / 2 * 360 / 4095, 2)
            span = round((mx - mn) * 360 / 4095, 2)
            ofs_deg = round(r['位置偏移'] * 360 / 4095, 2)
            w.writerow([
                r['__file__'], r['ID'],
                round(mn * 360 / 4095, 2),
                round(mx * 360 / 4095, 2),
                mid, span, ofs_deg,
                *[r[k] for k in base_fields],
            ])
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_read(args):
    targets = []
    p = Path(args.path)
    if p.is_dir():
        targets = sorted(p.glob('*.xdat'))
    elif p.is_file():
        targets = [p]
    else:
        sys.exit(f"路径不存在: {args.path}")

    for t in targets:
        d = read_xdat(t)
        print(f"\n=== {t} ===")
        for addr, name, default, size, region in REGISTERS:
            v = d[name]
            star = "*" if v != default and name != "预留" else " "
            extra = f"  [{round(v * 360 / 4095, 2)}°]" if name in ANGLE_FIELDS else ""
            extra += f"  [{v/10:.1f}V]" if "(0.1V)" in name else ""
            print(f"  {star} [{addr:>2}] {name:<18} = {v:<6}{extra}  (默认 {default})")


def cmd_csv(args):
    results = parse_folder(args.folder)
    if not results:
        sys.exit(f"在 {args.folder} 中找不到 .xdat")
    out = export_csv(results, args.out)
    print(f"已导出 {len(results)} 行 → {out}")


def cmd_set(args):
    d = read_xdat(args.path)
    if args.field not in d:
        # 模糊匹配
        cands = [n for n in d if args.field in n]
        if len(cands) == 1:
            args.field = cands[0]
        else:
            sys.exit(f"未知字段 {args.field!r}, 可选: {list(d.keys())}")
    old = d[args.field]
    d[args.field] = args.value
    write_xdat(args.path, d)
    print(f"  {args.path}: {args.field} {old} → {args.value}")
    # 备份
    bak = args.path + '.bak'
    if not Path(bak).exists():
        Path(args.path).rename(bak)
        write_xdat(args.path, d)
        print(f"  原文件已备份到 {bak}")


def cmd_restore(args):
    write_xdat(args.path, default_fields())
    print(f"  {args.path} 已还原为出厂默认")


def main():
    check_payload_size()

    parser = argparse.ArgumentParser(
        description='STS3215 .xdat 参数读写工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('read', help='读取并打印参数')
    p.add_argument('path', help='.xdat 文件或目录')

    p = sub.add_parser('csv', help='批量导出 CSV')
    p.add_argument('folder', help='参数 目录')
    p.add_argument('-o', '--out', default='参数汇总.csv')

    p = sub.add_parser('set', help='修改字段')
    p.add_argument('path', help='.xdat 文件')
    p.add_argument('field', help='字段名 (模糊匹配)')
    p.add_argument('value', type=int, help='新值 (10 进制)')

    p = sub.add_parser('restore', help='还原为出厂默认')
    p.add_argument('path')

    args = parser.parse_args()
    {'read': cmd_read, 'csv': cmd_csv,
     'set': cmd_set, 'restore': cmd_restore}[args.cmd](args)


if __name__ == '__main__':
    main()
