# stt/transcribe.py
import os
from pathlib import Path
from typing import Tuple

from pydub import AudioSegment
from openai import OpenAI

SUPPORTED_AUDIO_TYPES = ("mp3", "wav", "m4a", "ogg")

def audio_duration_seconds(path: str | Path) -> float:
    seg = AudioSegment.from_file(str(path))
    return len(seg) / 1000.0

def transcribe_offline(path: str | Path, model_size: str | None = None) -> str:
    """
    0$ – transcriere locală cu faster-whisper.
    Prima rulare va descărca modelul (ex: 'tiny' ~ 75MB).
    """
    from faster_whisper import WhisperModel

    size = model_size or os.getenv("FWHISPER_MODEL", "tiny")
    device = os.getenv("FWHISPER_DEVICE", "cpu")
    compute_type = "int8" if device == "cpu" else "float16"

    model = WhisperModel(size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(str(path), temperature=0.0)
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip()

def transcribe_openai(path: str | Path, client: OpenAI, model: str = "gpt-4o-mini-transcribe") -> str:
    """
    STT cu model mic (plătit). Alternativ poți folosi 'whisper-1'.
    """
    with open(path, "rb") as f:
        resp = client.audio.transcriptions.create(model=model, file=f)
    return resp.text.strip()

def transcribe_file(path: Path, engine: str, client: OpenAI | None = None, model: str = "gpt-4o-mini-transcribe") -> Tuple[str, float]:
    dur = audio_duration_seconds(path)
    if engine == "offline":
        text = transcribe_offline(path)
    else:
        if client is None:
            raise ValueError("OpenAI client required for engine='openai'.")
        text = transcribe_openai(path, client, model=model)
    return text, dur
