#!/usr/bin/env python
"""
Decode Feetech .xdat servo preset files.

Usage:
    python decode_servo_preset.py [preset_file_or_folder]

Examples:
    python decode_servo_preset.py 参数/1.xdat
    python decode_servo_preset.py 参数
"""

import sys
import os
import glob

# Register names for STS3215 (from official Feetech memory table)
REG_NAMES = {
    0: "Firmware main version",
    1: "Firmware sub version",
    2: "Reserved",
    3: "Model main version",
    4: "Model sub version",
    5: "ID",
    6: "Baud rate",
    7: "Return delay",
    8: "Response status level",
    9: "Min angle limit (L)",
    10: "Min angle limit (H)",
    11: "Max angle limit (L)",
    12: "Max angle limit (H)",
    13: "Max temperature limit",
    14: "Max input voltage",
    15: "Min input voltage",
    16: "Max torque (L)",
    17: "Max torque (H)",
    18: "Phase",
    19: "Unload conditions",
    20: "LED alarm conditions",
    21: "P coefficient",
    22: "D coefficient",
    23: "I coefficient",
    24: "Min starting force (L)",
    25: "Min starting force (H)",
    26: "CW dead zone",
    27: "CCW dead zone",
    28: "Protect current (L)",
    29: "Protect current (H)",
    30: "Angle resolution",
    31: "Position correction (L)",
    32: "Position correction (H)",
    33: "Operating mode",
    34: "Protect torque",
    35: "Protect time",
    36: "Overload torque",
    37: "Overcurrent protect time",
    38: "Reserved",
    39: "Reserved",
    40: "Torque switch (SRAM)",
    41: "Acceleration (SRAM)",
    42: "Goal position (L) (SRAM)",
    43: "Goal position (H) (SRAM)",
    44: "Goal time (L) (SRAM)",
    45: "Goal time (H) (SRAM)",
    46: "Goal speed (L) (SRAM)",
    47: "Goal speed (H) (SRAM)",
    48: "Torque limit (L) (SRAM)",
}

READ_ONLY_REGS = {0, 1, 2, 3, 4, 18}
TWO_BYTE_REGS = {9, 11, 16, 24, 28, 31, 42, 44, 46}


def le(data, idx):
    """Little-endian 16-bit value from data[idx], data[idx+1]."""
    return data[idx] | (data[idx + 1] << 8)


def format_reg(reg, data, idx):
    """Return a human-readable string for a register."""
    name = REG_NAMES.get(reg, "Unknown")
    raw = data[idx]
    if reg in TWO_BYTE_REGS:
        val = le(data, idx)
        extra = ""
        if reg == 9:
            extra = " -> min angle = %d steps" % val
        elif reg == 11:
            extra = " -> max angle = %d steps" % val
        elif reg == 16:
            extra = " -> %.1f%% max torque" % (val / 10.0)
        elif reg == 24:
            extra = " -> %.1f%% min starting force" % (val / 10.0)
        elif reg == 28:
            extra = " -> %.1f mA protect current" % (val * 6.26)
        elif reg == 31:
            extra = " -> position correction = %d steps" % val
        elif reg == 42:
            extra = " -> goal position = %d steps" % val
        elif reg == 44:
            extra = " -> goal time = %d ms" % (val * 10)
        elif reg == 46:
            extra = " -> goal speed = %d" % val
        return "  reg%2d %-32s = %5d (0x%04x)%s" % (reg, name, val, val, extra)
    else:
        extra = ""
        if reg == 5:
            extra = " -> servo ID %d" % raw
        elif reg == 6:
            bauds = {0: 1000000, 1: 500000, 2: 250000, 3: 128000, 4: 115200,
                     5: 76800, 6: 57600, 7: 38400}
            extra = " -> %d bps" % bauds.get(raw, raw)
        elif reg == 7:
            extra = " -> %d us" % (raw * 2)
        elif reg == 13:
            extra = " -> %d C" % raw
        elif reg in (14, 15):
            extra = " -> %.1f V" % (raw * 0.1)
        elif reg == 19:
            bits = []
            if raw & 0x01: bits.append("voltage")
            if raw & 0x02: bits.append("sensor")
            if raw & 0x04: bits.append("temp")
            if raw & 0x08: bits.append("current")
            if raw & 0x10: bits.append("angle")
            if raw & 0x20: bits.append("overload")
            extra = " -> " + (", ".join(bits) if bits else "none")
        elif reg in (34, 36):
            extra = " -> %d%%" % raw
        elif reg in (35, 37):
            extra = " -> %d ms" % (raw * 10)
        elif reg == 48:
            extra = " -> %.1f%% torque limit" % (raw / 10.0)
        return "  reg%2d %-32s = %5d (0x%02x)%s" % (reg, name, raw, raw, extra)


def decode_file(path):
    """Print decoded contents of one .xdat file."""
    with open(path, "rb") as f:
        data = f.read()

    if len(data) != 51:
        print("WARNING: %s is %d bytes, expected 51" % (path, len(data)))

    print("=" * 70)
    print("File: %s" % path)
    print("Header: 0x%02x 0x%02x" % (data[0], data[1]))
    print("Register data: bytes 2..50  ->  registers 0..48")
    print("-" * 70)

    i = 2
    while i < len(data):
        reg = i - 2
        if reg in READ_ONLY_REGS:
            line = format_reg(reg, data, i)
            print(line + "  [READ-ONLY]")
        else:
            print(format_reg(reg, data, i))
        i += 2 if reg in TWO_BYTE_REGS else 1


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "参数")

    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "*.xdat")),
                       key=lambda p: int(os.path.basename(p).split(".")[0]))
        if not files:
            print("No .xdat files found in %s" % target)
            return
        for f in files:
            decode_file(f)
    elif os.path.isfile(target):
        decode_file(target)
    else:
        print("Not found: %s" % target)


if __name__ == "__main__":
    main()
