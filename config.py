# config.py
import os
from dotenv import load_dotenv

load_dotenv()  # citește .env din rădăcină

# ✔️ doar modelele mici pentru chat
ALLOWED_CHAT_MODELS = {"gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano"}

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
if CHAT_MODEL not in ALLOWED_CHAT_MODELS:
    raise ValueError(
        f"CHAT_MODEL={CHAT_MODEL} nu este permis. Alege unul din: {sorted(ALLOWED_CHAT_MODELS)}."
    )

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY lipsește din .env sau din variabilele de mediu")

# Embeddings ieftine + Chroma persist
EMBED_MODEL = "text-embedding-3-small"
PERSIST_DIR = "chroma_store"

# Debug opțional
DEBUG = os.getenv("DEBUG", "0") == "1"

# --- 🎤 TTS defaults (folosite de UI) ---
# Mod: 'openai' (tts-1), 'offline' (pyttsx3), 'off'
TTS_MODE = os.getenv("TTS_MODE", "openai").lower()
if TTS_MODE not in {"openai", "offline", "off"}:
    raise ValueError("TTS_MODE trebuie să fie: openai | offline | off")

# Voce & format pentru tts-1
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")
TTS_FORMAT = os.getenv("TTS_FORMAT", "mp3").lower()  # mp3 | wav
if TTS_FORMAT not in {"mp3", "wav"}:
    raise ValueError("TTS_FORMAT trebuie să fie 'mp3' sau 'wav'")

def _as_int(env_name: str, default: int) -> int:
    try:
        return int(os.getenv(env_name, str(default)))
    except Exception:
        return default

def _as_float(env_name: str, default: float) -> float:
    try:
        return float(os.getenv(env_name, str(default)))
    except Exception:
        return default

TTS_RATE = _as_int("TTS_RATE", 170)
TTS_VOLUME = _as_float("TTS_VOLUME", 0.8)
