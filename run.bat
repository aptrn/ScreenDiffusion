@echo off
setlocal
title Launching ScreenDiffusion
cd /d "%~dp0"

set "PATH=%USERPROFILE%\.local\bin;%PATH%"

where uv >nul 2>&1
if %errorlevel% neq 0 goto :no_uv
if not exist ".venv\" goto :no_venv

echo Starting ScreenDiffusion...

:: --frozen: use the locked versions as-is, never re-resolve on launch.
uv run --frozen python main_gpu_addon.py

:: Keep the window open if the application crashes
pause
exit /b 0

:no_uv
echo [ERROR] uv is not installed. Please run setup.bat first.
pause
exit /b 1

:no_venv
echo [ERROR] No environment found (.venv is missing). Please run setup.bat first.
pause
exit /b 1
