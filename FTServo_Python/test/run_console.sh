#!/bin/bash
# ============================================================
# 舵机控制台 (STS3215 × 17) 启动脚本 — macOS / Linux
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$ROOT_DIR/../.venv/bin/python"

echo "========================================"
echo "  舵机控制台 (STS3215 × 17) 启动"
echo "========================================"
echo

# --- Step 1: find Python ---
echo "[1/4] 查找 Python 环境..."
if [ -x "$VENV_PYTHON" ]; then
    PYTHON="$VENV_PYTHON"
    echo "  使用 .venv: $VENV_PYTHON"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
    echo "  使用系统 python3: $(which python3)"
else
    echo "  错误: 未找到 Python 3.7+"
    exit 1
fi

# --- Step 2: check PyQt ---
echo "[2/4] 检查 PyQt..."
if ! "$PYTHON" -c "import PyQt5" &> /dev/null; then
    if "$PYTHON" -c "import PyQt6" &> /dev/null; then
        echo "  检测到 PyQt6（将自动适配）"
    else
        echo "  缺少 PyQt5，正在安装..."
        "$PYTHON" -m pip install PyQt5
    fi
fi

# --- Step 3: check pyserial ---
echo "[3/4] 检查 pyserial..."
if ! "$PYTHON" -c "import serial" &> /dev/null; then
    echo "  缺少 pyserial，正在安装..."
    "$PYTHON" -m pip install pyserial
fi

# --- Step 4: launch ---
echo "[4/4] 启动控制台..."
echo
cd "$SCRIPT_DIR"
exec "$PYTHON" servo_console.py
