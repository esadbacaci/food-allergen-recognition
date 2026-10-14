@echo off
rem Egitim zincirini baslatir veya kaldigi yerden devam ettirir. Calisiyorsa ikinci kez baslatmaz.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
start "Egitim zinciri" /min "C:\Program Files\Gitinash.exe" scripts/start_chain.sh
echo Baslatma istegi gonderildi. Ilerleme icin durum.cmd dosyasini calistirin.
pause
