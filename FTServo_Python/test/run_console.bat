@echo off
REM Servo Console Launcher (STS3215 × 17)
REM ======================================

echo [1/3] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.7+
    pause
    exit /b 1
)

echo [2/3] Checking dependencies (PyQt5 + pyserial)...
python -c "import PyQt5, serial" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Missing dependencies, installing...
    pip install PyQt5 pyserial
    if errorlevel 1 (
        echo [ERROR] Dependency install failed.
        pause
        exit /b 1
    )
)

echo [3/3] Starting Servo Console...
echo.
python servo_console.py

if errorlevel 1 (
    echo.
    echo [ERROR] Program exited with errors. Check messages above.
    pause
)