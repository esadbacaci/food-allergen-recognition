# -*- coding: utf-8 -*-
"""Telegram progress bot for the experiment chain (standard library only, read-only).

Setup (once):
  1. In Telegram, talk to @BotFather -> /newbot -> copy the token.
  2. Put it into 02_kod/telegram_config.json (git-ignored):  {"token": "123:ABC..."}
  3. Start: TELEGRAM_BOT_BASLAT.cmd, then send /start to your bot. The first chat that
     sends /start is saved as the only allowed chat; other chats are ignored.

Commands: /durum (progress + current run, epoch by epoch), /son (last finished runs), /yardim
Automatic messages: a run finished (validation accuracy), errors in run_all_stderr.log,
no new epoch for 40 minutes while the chain should be running, chain finished.
"""
import http.client
import json
import sys
import time
import traceback
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodallergen import paths  # noqa: E402
import scripts.status as status  # noqa: E402

CONFIG = paths.REPO / "telegram_config.json"
STALL_MINUTES = 40
HOT_C = 85
LOG = paths.RUNS / "run_all.log"
ERR = paths.RUNS / "run_all_stderr.log"


def load_config():
    if not CONFIG.exists():
        CONFIG.write_text(json.dumps({"token": "BURAYA_BOTFATHER_TOKENI", "chat_id": None}, indent=2))
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if not cfg.get("token") or "BURAYA" in cfg["token"]:
        sys.exit(f"Telegram token eksik: {CONFIG} dosyasına BotFather'dan aldığınız token'ı yazın.")
    return cfg


def api(cfg, method, **params):
    url = f"https://api.telegram.org/bot{cfg['token']}/{method}"
    data = urllib.parse.urlencode(params).encode()
    with urllib.request.urlopen(url, data=data, timeout=40) as r:
        return json.loads(r.read().decode())


def send(cfg, text):
    if cfg.get("chat_id"):
        for i in range(0, len(text), 3900):  # Telegram limit 4096 characters
            api(cfg, "sendMessage", chat_id=cfg["chat_id"], text=text[i:i + 3900])


def gpu_temp():
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        t, p, u = [x.strip() for x in out.split(",")[:3]]
        return int(float(t)), f"GPU {t}°C, {float(p):.0f} W, kullanım %{u}"
    except Exception:
        return None, "GPU bilgisi alınamadı"


def training_alive():
    """True if a 'scripts/run.py' training process exists."""
    import subprocess
    cmd = ("@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
           "Where-Object { $_.CommandLine -match 'scripts.run\\.py' }).Count")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True,
                             text=True, timeout=60).stdout.strip()
        return int(out or 0) > 0
    except Exception:
        return True  # unknown: do not claim that training stopped


def progress_text():
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        status.main()
    return buf.getvalue() + "\n" + gpu_temp()[1]


def finished_runs():
    return {str(rd.relative_to(paths.RUNS)): rd for _, rd, *_ in status.planned() if (rd / "DONE").exists()}


def last_epoch_time():
    t = 0.0
    for _, rd, *_ in status.planned():
        log = rd / "train.log"
        if log.exists():
            t = max(t, log.stat().st_mtime)
    return t


def main():
    cfg = load_config()
    offset = None
    known = set(finished_runs())
    err_size = ERR.stat().st_size if ERR.exists() else 0
    stall_warned = False
    done_announced = "ALL_EXPERIMENTS_DONE" in (LOG.read_text(encoding="utf-8", errors="ignore") if LOG.exists() else "")
    net_fail = 0
    hot_warned = 0.0
    print("Bot çalışıyor. Telegram'da botunuza /start yazın. Kapatmak için bu pencereyi kapatın.", flush=True)
    while True:
        try:
            # ---- commands
            params = {"timeout": 25}
            if offset is not None:
                params["offset"] = offset
            for upd in api(cfg, "getUpdates", **params).get("result", []):
                offset = upd["update_id"] + 1
                msg = upd.get("message") or {}
                chat = (msg.get("chat") or {}).get("id")
                text = (msg.get("text") or "").strip().lower()
                if chat is None:
                    continue
                if cfg.get("chat_id") is None and text.startswith("/start"):
                    cfg["chat_id"] = chat
                    CONFIG.write_text(json.dumps(cfg, indent=2))
                if chat != cfg.get("chat_id"):
                    continue  # only the registered chat may use the bot
                if text.startswith(("/durum", "/start")):
                    send(cfg, progress_text())
                elif text.startswith("/son"):
                    send(cfg, status.detail(last_finished=15))
                else:
                    send(cfg, "Komutlar: /durum  /son")
            # ---- automatic messages
            now_done = finished_runs()
            for name in sorted(set(now_done) - known):
                m = json.loads((now_done[name] / "metrics.json").read_text())
                test = f", test %{m['test']['acc']*100:.2f}" if "test" in m else ""
                send(cfg, f"✅ Bitti: {name}\ndoğrulama %{m['val']['acc']*100:.2f} (en iyi epoch {m['best_epoch']}){test}")
            known = set(now_done)
            if ERR.exists() and ERR.stat().st_size > err_size:
                new = ERR.read_text(encoding="utf-8", errors="ignore")[err_size:]
                err_size = ERR.stat().st_size
                bad = [ln for ln in new.splitlines() if any(k in ln for k in ("Traceback", "Error", "FAILED"))
                       and "Warning" not in ln]
                if bad:
                    send(cfg, "⚠️ Hata kaydı:\n" + "\n".join(bad[-8:]))
            log_txt = LOG.read_text(encoding="utf-8", errors="ignore") if LOG.exists() else ""
            if "ALL_EXPERIMENTS_DONE" in log_txt and not done_announced:
                send(cfg, "🎉 Tüm deneyler bitti.\n\n" + progress_text())
                done_announced = True
            idle = (time.time() - last_epoch_time()) / 60
            if not done_announced and idle > STALL_MINUTES and not stall_warned:
                if training_alive():
                    send(cfg, f"🐢 Eğitim ÇALIŞIYOR ama yavaşladı: son epoch {idle:.0f} dakika önce bitti. "
                              f"Sizin bir şey yapmanıza gerek yok; otomatik kontrol nedenini inceleyecek.")
                else:
                    send(cfg, f"⛔ Eğitim süreci DURMUŞ (son epoch {idle:.0f} dakika önce). "
                              f"02_kod\\EGITIMI_DEVAM_ETTIR.cmd dosyasına çift tıklayın; kaldığı yerden devam eder.")
                stall_warned = True
            if idle < 5:
                stall_warned = False
            temp, info = gpu_temp()
            if temp is not None and temp >= HOT_C and time.time() - hot_warned > 1800:
                send(cfg, f"🌡️ Ekran kartı sıcak: {info}. Kasa havalandırmasını kontrol edin; kart kendini "
                          f"korumak için yavaşlar, zarar görmez.")
                hot_warned = time.time()
            net_fail = 0
        except (OSError, http.client.HTTPException) as e:  # network drop, reset, timeout
            net_fail += 1
            wait = min(300, 10 * 2 ** min(net_fail, 5))
            print(f"{time.strftime('%H:%M:%S')} Telegram bağlantısı kesildi ({type(e).__name__}); "
                  f"{wait} sn sonra yeniden denenecek. Eğitim etkilenmez.", flush=True)
            time.sleep(wait)
        except Exception:
            traceback.print_exc()
            time.sleep(30)


if __name__ == "__main__":
    main()
