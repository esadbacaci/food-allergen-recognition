@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python scripts\status.py
pause
