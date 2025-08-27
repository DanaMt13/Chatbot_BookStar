# stt/transcribe.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, List

import os
import io
import math

# pydub pentru conversia robustă a audio (necesită ffmpeg în PATH – în Docker e instalat)
from pydub import AudioSegment
from pydub.utils import which

# OpenAI client e injectat din UI (doar pentru online STT)
# from openai import OpenAI  # NU crea client aici, îl primești ca argument

# --------- Setări generale ----------
SUPPORTED_AUDIO_TYPES = {"mp3", "wav", "m4a", "ogg", "flac", "webm"}

# Asigură binarele ffmpeg/ffprobe pentru pydub
AudioSegment.converter = which("ffmpeg") or "ffmpeg"
AudioSegment.ffprobe   = which("ffprobe") or "ffprobe"


def audio_duration_seconds(p: Path) -> float:
    """Durata estimată a fișierului audio (secunde)."""
    seg = AudioSegment.from_file(p)
    return len(seg) / 1000.0


def _convert_to_wav16k_mono(src_path: Path) -> Path:
    """
    Convertor sigur -> WAV, 16kHz, mono, 16-bit PCM (format optim pentru STT).
    Returnează calea către fișierul convertit.
    """
    seg = AudioSegment.from_file(src_path)
    seg = seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # 16-bit
    out_path = src_path.with_suffix(".stt16.wav")
    seg.export(out_path, format="wav")
    return out_path


# ---------------- OFFLINE (faster-whisper) ----------------
def _offline_whisper_transcribe(
    wav16_mono_path: Path,
    language: str = "auto",
    model_size: str = None,
) -> Tuple[str, float, dict]:
    """
    Transcriere offline cu faster-whisper.
    Returnează: (text, durată_secunde, info_dict)
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError("Lipsete faster-whisper. Instalează: pip install faster-whisper")

    if model_size is None:
        model_size = os.getenv("FWHISPER_MODEL", "base")  # tiny/base/small

    # Compute type: int8 pentru CPU e ok; dacă ai AVX, e rapid; small = mai precis
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # Heuristici bune pentru vorbire casual:
    # - vad_filter: reduce părți cu zgomot
    # - beam_size: 5 → acuratețe mai bună
    # - language: "ro" forțează limba; None = autodetect
    lang = None if (language is None or language.lower() == "auto") else language.lower()

    initial_prompt = (
        "Context: discuție despre cărți, recomandări de lectură, autori, personaje, genuri. "
        "Termeni precum: carte, roman, distopie, fantasy, magie, Tolkien, Orwell, Huxley, "
        "detectiv, crimă, sf, prietenie, aventură, dragoste, război."
    )

    segments, info = model.transcribe(
        str(wav16_mono_path),
        language=lang,
        task="transcribe",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
        best_of=5,
        temperature=[0.0, 0.2, 0.4],
        condition_on_previous_text=True,
        initial_prompt=initial_prompt,
    )

    text_parts: List[str] = []
    avg_logprob = 0.0
    seg_count = 0
    for seg in segments:
        t = (seg.text or "").strip()
        if t:
            text_parts.append(t)
        # avg_logprob nu e mereu prezent, dar îl colectăm când există
        if hasattr(seg, "avg_logprob") and seg.avg_logprob is not None:
            avg_logprob += seg.avg_logprob
            seg_count += 1

    text = " ".join(text_parts).strip()
    if seg_count > 0:
        avg_logprob /= max(1, seg_count)
    else:
        avg_logprob = -math.inf

    dur = audio_duration_seconds(wav16_mono_path)
    info_dict = {
        "detected_language": getattr(info, "language", None),
        "avg_logprob": avg_logprob,
        "duration": dur,
    }
    return text, dur, info_dict


# ---------------- ONLINE (OpenAI) ----------------
def _online_openai_transcribe(
    wav16_mono_path: Path,
    client,
    model: str = "gpt-4o-mini-transcribe",
    language: str = "auto",
) -> Tuple[str, float, dict]:
    """
    Online STT (OpenAI). Suportă gpt-4o-mini-transcribe și fallback la whisper-1.
    """
    lang_arg = None if (language is None or language.lower() == "auto") else language.lower()
    prompt_hint = (
        "Context: recomandări de cărți, autori, personaje, genuri, teme ca distopie, fantasy, magie, crimă."
    )
    with open(wav16_mono_path, "rb") as f:
        try:
            resp = client.audio.transcriptions.create(
                model=model,
                file=f,
                language=lang_arg,           # respectă limba când e setată
                prompt=prompt_hint
            )
            text = getattr(resp, "text", "") or ""
        except Exception:
            # fallback robust
            f.seek(0)
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=lang_arg,
                prompt=prompt_hint
            )
            text = getattr(resp, "text", "") or ""

    dur = audio_duration_seconds(wav16_mono_path)
    return text.strip(), dur, {"engine": "openai", "model": model, "language": lang_arg}


# ---------------- API principală ----------------
def transcribe_file(
    audio_path: Path,
    engine: str = "offline",               # "offline" | "openai"
    client=None,
    model: str = "gpt-4o-mini-transcribe", # doar pentru online
    language: str = "auto",                # "auto" sau cod ISO ("ro", "en", ...)
) -> Tuple[str, float]:
    """
    Transcrie fișierul audio cu engine-ul selectat. Întoarce (text, durata_secunde).
    Face conversia la WAV 16k mono înainte de transcriere.
    Dacă offline produce text „gibberish”, face fallback la online (dacă există client).
    """
    audio_path = Path(audio_path)
    wav_path = _convert_to_wav16k_mono(audio_path)

    if engine.lower().startswith("off"):
        text, dur, info = _offline_whisper_transcribe(
            wav_path,
            language=language,
            model_size=os.getenv("FWHISPER_MODEL", "base")
        )

        # Fallback: dacă iese gol sau pare „gibberish” și avem client → încearcă online
        bad = (not text) or (len(text.split()) <= 2)
        if bad and client is not None:
            text2, dur2, _ = _online_openai_transcribe(wav_path, client, model=model, language=language)
            return (text2 or text or ""), (dur2 if dur2 else dur)

        return text, dur

    else:
        # Online direct
        if client is None:
            raise RuntimeError("Ai ales engine='openai' dar nu ai furnizat client OpenAI.")
        text, dur, _ = _online_openai_transcribe(wav_path, client, model=model, language=language)
        return text, dur
