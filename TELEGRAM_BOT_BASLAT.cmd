@echo off
rem Telegram ilerleme botunu baslatir (bu pencere acik kaldigi surece calisir).
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python scripts\telegram_bot.py
pause
