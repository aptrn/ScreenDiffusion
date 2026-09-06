@echo off
setlocal
title ScreenDiffusion Setup (uv)
cd /d "%~dp0"

echo ========================================================
echo         ScreenDiffusion Environment Setup
echo ========================================================
echo.

echo [1/3] Checking for uv...
where uv >nul 2>&1
if %errorlevel% equ 0 goto :have_uv

echo [INFO] uv not found. Installing it now (one-time, ~30 seconds)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if %errorlevel% neq 0 goto :uv_error

:: The installer adds uv to PATH for NEW shells; add it to THIS one too.
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>&1
if %errorlevel% neq 0 goto :uv_error

:have_uv
for /f "delims=" %%v in ('uv --version') do echo [OK] %%v

echo.
echo [2/3] Preparing Python 3.11...
echo       (uv downloads its own Python - no manual install needed)
uv python install 3.11
if %errorlevel% neq 0 goto :setup_error

echo.
echo [3/3] Creating the isolated environment and installing dependencies...
echo       This downloads several GB of CUDA/PyTorch wheels - please be patient.
echo.
uv sync
if %errorlevel% neq 0 goto :setup_error

echo.
echo ========================================================
echo    Setup complete! The environment is locked and stable.
echo.
echo    Everything lives in the .venv folder in this directory.
echo    Nothing was installed into your system Python.
echo.
echo    Next: run run.bat to launch ScreenDiffusion.
echo ========================================================
pause
exit /b 0

:: ==========================================
:: ERROR HANDLERS
:: ==========================================

:uv_error
echo.
echo [ERROR] Could not install uv automatically.
echo         Install it manually from https://docs.astral.sh/uv/ and re-run setup.bat.
pause
exit /b 1

:setup_error
echo.
echo [ERROR] Setup failed. See the messages above for details.
echo         A common cause is an interrupted download - just re-run setup.bat,
echo         uv will resume from where it left off.
pause
exit /b 1
