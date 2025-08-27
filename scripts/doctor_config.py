# scripts/doctor_config.py
import os, sys, json

# adaugă rădăcina repo-ului în PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config

print("CONFIG PATH:", getattr(config, "__file__", "?"))
keys = ["TTS_MODE","TTS_VOICE","TTS_FORMAT","TTS_RATE","TTS_VOLUME","CHAT_MODEL"]
print(json.dumps({k: getattr(config, k, "<MISSING>")} for k in keys))
