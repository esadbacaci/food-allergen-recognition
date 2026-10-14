#!/bin/bash
# Starts scripts/run_all.sh unless a chain is already running (two chains writing the same
# run folders would corrupt results). Used by EGITIMI_DEVAM_ETTIR.cmd.
cd "$(dirname "$0")/.."
LOCK=../03_deneyler/.run_all.lock
running=$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'scripts.run\.py' }).Count")
if [ "${running//[$'\r\n ']/}" != "0" ]; then
  echo "Egitim zaten calisiyor (scripts/run.py aktif). Ikinci kez baslatilmadi."; exit 1
fi
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK")
  if tasklist //FI "PID eq $old" 2>/dev/null | grep -qi "bash"; then
    echo "Egitim zinciri zaten calisiyor (PID $old). Ikinci kez baslatilmadi."; exit 1
  fi
fi
cat /proc/$$/winpid > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
echo "=== $(date '+%Y-%m-%d %H:%M:%S') zincir (yeniden) baslatildi" >> ../03_deneyler/run_all.log
bash scripts/run_all.sh >> ../03_deneyler/run_all.log 2>> ../03_deneyler/run_all_stderr.log
