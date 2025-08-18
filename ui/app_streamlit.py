# ui/app_streamlit.py
# -*- coding: utf-8 -*-
# Smart Librarian – Streamlit UI (RAG + Tool + TTS + STT + AI Images low-cost)

import os, sys, time, re
from pathlib import Path

# --- permite importuri din rădăcina proiectului când rulăm din /ui ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from openai import OpenAI

# AI image generator (low-cost cu cache, gpt-image-1)
from img.ai_gen import generate_ai_cover_bytes, generate_ai_scene_bytes

# config robust (NU folosi "from config import ...")
import config as CFG

# ---- Config & defaults ------------------------------------------------------
PERSIST_DIR     = CFG.PERSIST_DIR
OPENAI_API_KEY  = CFG.OPENAI_API_KEY

TTS_MODE_DEFAULT   = getattr(CFG, "TTS_MODE", "off")   # openai | offline | off (implicit off)
TTS_VOICE_DEFAULT  = getattr(CFG, "TTS_VOICE", "alloy")
TTS_FORMAT_DEFAULT = getattr(CFG, "TTS_FORMAT", "mp3") # mp3 | wav
TTS_RATE_DEFAULT   = int(getattr(CFG, "TTS_RATE", 170))
TTS_VOL_DEFAULT    = float(getattr(CFG, "TTS_VOLUME", 0.8))

# AUDIO_DIR: env > config > default (Docker /tmp/audio, local data/tmp_audio)
AUDIO_DIR = os.getenv(
    "AUDIO_DIR",
    getattr(CFG, "AUDIO_DIR", "/tmp/audio" if os.path.exists("/.dockerenv") else "data/tmp_audio")
)

# ---- RAG & Chat -------------------------------------------------------------
from rag.embed_store import load_summaries, init_vector_store
from rag.retriever import debug_candidates
from chatbot import chat

# ---- STT (upload + offline/online) -----------------------------------------
from stt.transcribe import (
    transcribe_file,
    SUPPORTED_AUDIO_TYPES,
    audio_duration_seconds,
)

# ---- OpenAI client (TTS / Images / STT online) -----------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# ---- UI meta ---------------------------------------------------------------
st.set_page_config(page_title="📚 Smart Librarian", page_icon="📚", layout="centered")
st.title("📚 Smart Librarian — RAG + Tool Calling (mini chat models only)")
st.caption(
    f"Vector store: **ChromaDB** (`{PERSIST_DIR}`) • Embeddings: **text-embedding-3-small** • "
    f"Chat model: **gpt-4o-mini / gpt-4.1-mini / gpt-4.1-nano** (impuse în config)"
)

# ---- utils -----------------------------------------------------------------
def guess_title_from_answer(answer: str, known_titles: list[str] | None = None) -> str | None:
    """
    Încearcă să extragă titlul din răspunsul modelului.
    1) pattern-uri frecvente (cu **bold**, ghilimele românești „...” sau standard "...")
    2) 'Rezumat detaliat: „Titlu”'
    3) dacă avem known_titles, căutăm apariții case-insensitive
    """
    if not answer:
        return None

    patterns = [
        r"Îți recomand[^:\n]*:\s*\*\*(.+?)\*\*",
        r"Îți recomand[^„\"]*[„\"]\s*([^„”\"\n]+?)\s*[”\"]",
        r"Rezumat detaliat:\s*[„\"]\s*([^„”\"\n]+?)\s*[”\"]",
        r"[„\"]\s*([^„”\"\n]+?)\s*[”\"]\s*\(.*?rezumat|.*?autor",  # titlu urmat de paranteze
    ]
    import re
    for pat in patterns:
        m = re.search(pat, answer, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    if known_titles:
        low = answer.lower()
        # preferăm cel mai lung match (evită 'It' vs 'It Ends With Us')
        candidates = sorted(known_titles, key=lambda t: -len(t))
        for t in candidates:
            if t.lower() in low:
                return t

    return None

@st.cache_resource(show_spinner=False)
def get_known_titles() -> list[str]:
    try:
        books = load_summaries()
        return [b.get("title", "") for b in books if b.get("title")]
    except Exception:
        return []


def fmt_seconds(s: float) -> str:
    s = int(round(s))
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}"

def measure_and_enforce_limit(audio_path: Path, fmt: str, max_seconds: int) -> tuple[Path, float, bool]:
    """Măsoară durata (pydub). Taie la max_seconds dacă e nevoie."""
    try:
        from pydub import AudioSegment
    except Exception:
        return audio_path, -1.0, False
    seg = AudioSegment.from_file(audio_path)
    dur = len(seg) / 1000.0
    if max_seconds and dur > max_seconds:
        seg = seg[: max_seconds * 1000]
        cut_path = audio_path.with_name(f"{audio_path.stem}_cut{audio_path.suffix}")
        seg.export(cut_path, format=fmt.lower())
        return cut_path, max_seconds, True
    return audio_path, dur, False

def cleanup_old_audio(folder: str, max_age_seconds: int = 3600):
    try:
        p = Path(folder)
        if not p.exists():
            return
        now = time.time()
        for f in p.glob("*"):
            try:
                if f.is_file() and (now - f.stat().st_mtime) > max_age_seconds:
                    f.unlink()
            except Exception:
                pass
    except Exception:
        pass

cleanup_old_audio(AUDIO_DIR, max_age_seconds=3600)

# ---- TTS helpers ------------------------------------------------------------
def tts_openai_tts1(text: str, voice: str = "alloy", fmt: str = "mp3", max_seconds: int = 150) -> tuple[Path, float, bool]:
    """TTS OpenAI (tts-1). Prefer streaming; fallback la .create()."""
    out_dir = Path(AUDIO_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tts1_{int(time.time())}.{fmt}"

    try:
        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=voice,
            response_format=fmt,
            input=text,
        ) as resp:
            resp.stream_to_file(out_path)
    except Exception:
        try:
            resp = client.audio.speech.create(model="tts-1", voice=voice, input=text, format=fmt)
        except TypeError as e:
            if "format" in str(e):
                resp = client.audio.speech.create(model="tts-1", voice=voice, input=text, response_format=fmt)
            else:
                raise
        if hasattr(resp, "stream_to_file"):
            resp.stream_to_file(out_path)
        else:
            data = getattr(resp, "read", lambda: None)() or getattr(resp, "content", None)
            if not data:
                raise RuntimeError("TTS: răspuns neașteptat, fără bytes.")
            out_path.write_bytes(data)

    return measure_and_enforce_limit(out_path, fmt, max_seconds)

def tts_offline_pyttsx3(text: str, rate: int = 170, vol: float = 0.8, max_seconds: int = 150) -> tuple[Path | None, float, bool]:
    """TTS offline (pyttsx3) -> .wav"""
    try:
        import pyttsx3
    except ImportError:
        st.error("Lipsește `pyttsx3`. Instalează: `pip install pyttsx3`.")
        return None, -1.0, False
    out_dir = Path(AUDIO_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pyttsx3_{int(time.time())}.wav"
    eng = pyttsx3.init()
    eng.setProperty("rate", int(rate))
    eng.setProperty("volume", float(vol))
    eng.save_to_file(text, str(out_path))
    eng.runAndWait()
    return measure_and_enforce_limit(out_path, "wav", max_seconds)

def render_audio(path: Path, duration_sec: float, was_truncated: bool):
    if not path or not Path(path).exists():
        return
    mime = "audio/mpeg" if path.suffix.lower() == ".mp3" else "audio/wav"
    info = " (tăiat la limită)" if was_truncated else ""
    if duration_sec >= 0:
        st.caption(f"⏱️ Durată audio: {fmt_seconds(duration_sec)}{info}")
    st.audio(str(path), format=mime)
    with open(path, "rb") as f:
        st.download_button("⬇️ Descarcă audio", data=f, file_name=path.name, mime=mime, key="download_audio_btn")

# ---- bootstrap RAG (o singură dată) ----------------------------------------
@st.cache_resource(show_spinner=True)
def bootstrap_collection():
    summaries = load_summaries()
    collection = init_vector_store(summaries, persist_path=PERSIST_DIR)
    return collection, len(summaries)

collection, n_books = bootstrap_collection()

# ---- Session state ----------------------------------------------------------
for key, default in [
    ("last_answer", ""),
    ("last_audio_path", ""),
    ("last_audio_dur", -1.0),
    ("last_audio_trunc", False),
    ("query_last", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Opțiuni")
    show_debug = st.checkbox("Arată Top-K (debug RAG)", value=False, key="show_debug_chk")
    k = st.slider("Top-K pentru debug", 1, 8, 5, key="topk_slider")

    st.markdown("---")
    st.subheader("🔈 Text-to-Speech (manual)")
    mode_to_idx = {"openai": 0, "offline": 1, "off": 2}
    idx_default = mode_to_idx.get(TTS_MODE_DEFAULT, 2)
    tts_choice = st.radio(
        "Motor TTS",
        ["OpenAI tts-1 (plătit)", "Offline pyttsx3 (gratuit)", "Oprit"],
        index=idx_default,
        key="tts_choice_radio",
    )
    voice = st.text_input("Voce (tts-1)", value=TTS_VOICE_DEFAULT, help="ex: alloy", key="voice_input")
    audio_fmt = st.selectbox("Format audio (tts-1)", ["mp3", "wav"], index=0 if TTS_FORMAT_DEFAULT == "mp3" else 1, key="audiofmt_sel")
    tts_rate = st.slider("Viteză (pyttsx3)", 100, 250, TTS_RATE_DEFAULT, key="tts_rate_slider")
    tts_volume = st.slider("Volum (pyttsx3)", 0.1, 1.0, float(TTS_VOL_DEFAULT), key="tts_vol_slider")
    max_secs = st.slider("Limită durată audio (sec)", 10, 300, 150, help="Implicit 150s = 2m30s", key="maxsecs_slider")

    st.markdown("---")
    st.subheader("🎙️ Speech-to-Text (manual)")
    stt_choice = st.radio("Motor STT", ["Offline (0$)", "OpenAI gpt-4o-mini-transcribe (plătit)"], index=0, key="stt_choice_radio")

# ---- Întrebare: Enter sau Buton --------------------------------------------
st.subheader("Pune o întrebare")

def run_recommendation(q: str):
    if not q.strip():
        st.warning("Introdu o întrebare.")
        return
    try:
        if st.session_state.get("show_debug_chk"):
            st.write("### 🔎 Top-K din Chroma (debug)")
            for idx, (title, dist) in enumerate(debug_candidates(q, collection, top_k=st.session_state.get("topk_slider", 5)), start=1):
                st.write(f"{idx}. **{title}** · dist: `{dist:.4f}`")
        with st.spinner("Gândesc o recomandare..."):
            answer = chat(q, collection)
        st.session_state["last_answer"] = answer
        st.session_state["last_audio_path"] = ""
        st.session_state["last_audio_dur"] = -1.0
        st.session_state["last_audio_trunc"] = False
        st.session_state["query_last"] = q
    except Exception as e:
        st.error(f"Eroare: {e}")

# 1) Formular (Enter trimite) – FĂRĂ `key` la form_submit_button
with st.form("ask_form", clear_on_submit=False):
    query = st.text_input(
        "Exemple: „Vreau o carte despre libertate și magie”, „Ce recomanzi pentru cineva care iubește distopiile?”, „Ce este The Hobbit?”",
        value=st.session_state.get("query_last") or "Vreau o carte despre libertate și magie",
        key="query_text",
    )
    submitted = st.form_submit_button("💬 Cere o recomandare (Enter)")

if submitted:
    run_recommendation(st.session_state.get("query_text", ""))

# 2) Buton clasic (cu key unic) – rămâne cu key ca să nu ai ID duplicat
if st.button("💬 Cere o recomandare", key="ask_btn_click"):
    run_recommendation(st.session_state.get("query_text", ""))

# ---- STT: încarcă fișier și întreabă ---------------------------------------
st.markdown("### 🎙️ Transcrie un fișier audio (≤ 2:30)")
audio_file = st.file_uploader("Încarcă .mp3 / .wav / .m4a / .ogg", type=list(SUPPORTED_AUDIO_TYPES), key="uploader_audio")

def _save_uploaded(tmp_dir: Path, file) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    p = tmp_dir / f"upload_{int(time.time())}_{file.name}"
    with open(p, "wb") as f:
        f.write(file.read())
    return p

if st.button("📝 Transcrie & întreabă", key="stt_btn", disabled=audio_file is None):
    if audio_file is None:
        st.warning("Încarcă un fișier audio.")
    else:
        tmp_dir = Path(AUDIO_DIR) / "uploads"
        path = _save_uploaded(tmp_dir, audio_file)
        dur = audio_duration_seconds(path)
        if dur > st.session_state.get("maxsecs_slider", 150):
            st.error(f"Fișier prea lung: ~{int(dur)}s. Limita este {st.session_state.get('maxsecs_slider', 150)}s (2m30s).")
        else:
            engine = "offline" if st.session_state.get("stt_choice_radio","Offline").startswith("Offline") else "openai"
            try:
                txt, real_dur = transcribe_file(path, engine, client=client, model="gpt-4o-mini-transcribe")
                st.success(f"Transcriere (~{int(real_dur)}s) realizată.")
                with st.expander("Text transcris"):
                    st.write(txt)
                with st.spinner("Gândesc o recomandare..."):
                    answer = chat(txt, collection)
                st.session_state["last_answer"] = answer
                st.session_state["last_audio_path"] = ""
                st.session_state["last_audio_dur"] = -1.0
                st.session_state["last_audio_trunc"] = False
                st.session_state["query_last"] = txt
            except Exception as e:
                st.error(f"Eroare STT: {e}")

# ---- Afișare răspuns + TTS + Imagini ---------------------------------------
if st.session_state["last_answer"]:
    st.markdown("### 💡 Răspuns")
    st.markdown(st.session_state["last_answer"])

    # TTS manual
    if st.button("🔈 Citește răspunsul", key="tts_btn"):
        try:
            prev = st.session_state.get("last_audio_path")
            if prev:
                try:
                    pprev = Path(prev)
                    if pprev.exists():
                        pprev.unlink()
                except Exception:
                    pass
            if st.session_state.get("tts_choice_radio") == "OpenAI tts-1 (plătit)":
                p, d, t = tts_openai_tts1(
                    st.session_state["last_answer"],
                    voice=st.session_state.get("voice_input", TTS_VOICE_DEFAULT),
                    fmt=st.session_state.get("audiofmt_sel", "mp3"),
                    max_seconds=st.session_state.get("maxsecs_slider", 150),
                )
            elif st.session_state.get("tts_choice_radio") == "Offline pyttsx3 (gratuit)":
                p, d, t = tts_offline_pyttsx3(
                    st.session_state["last_answer"],
                    rate=st.session_state.get("tts_rate_slider", TTS_RATE_DEFAULT),
                    vol=st.session_state.get("tts_vol_slider", TTS_VOL_DEFAULT),
                    max_seconds=st.session_state.get("maxsecs_slider", 150),
                )
            else:
                p, d, t = None, -1.0, False
            if p:
                st.session_state["last_audio_path"] = str(p)
                st.session_state["last_audio_dur"] = d
                st.session_state["last_audio_trunc"] = t
        except Exception as e:
            st.error(f"Nu am reușit să generez audio: {e}")

    if st.session_state["last_audio_path"]:
        render_audio(
            Path(st.session_state["last_audio_path"]),
            st.session_state["last_audio_dur"],
            st.session_state["last_audio_trunc"],
        )

# === Imagini AI pentru recomandare (1 buton, fără salvare pe disc) ==========
st.markdown("### 🖼️ Imagine AI pentru recomandare")

# deduce titlul din răspuns + fallback pe titlurile cunoscute din YAML
known_titles = get_known_titles()
title_auto = guess_title_from_answer(st.session_state.get("last_answer", ""), known_titles=known_titles)

colL, colR = st.columns([1, 1])
with colL:
    img_kind = st.selectbox("Tip imagine", ["Copertă", "Scenă"], key="img_kind_sel")
with colR:
    size = st.selectbox(
        "Dimensiune",
        ["512x512", "1024x1024", "1024x1536", "1536x1024"],
        index=0,
        key="img_size_sel"
    )

quality = st.selectbox("Calitate (cost)", ["low", "medium", "high"], index=0, key="img_quality_sel")
st.caption("🪙 Recomandat: low @ 512×512 (dacă e acceptat) sau 1024×1024 ≈ $0.011")

themes_text = st.text_input("Teme principale (opțional)", value="", key="img_themes_input")
scene_text = st.text_area(
    "Scenă / descriere scurtă (doar pentru 'Scenă')",
    value=st.session_state.get("query_last", ""),
    key="img_scene_textarea",
) if st.session_state.get("img_kind_sel") == "Scenă" else ""

disabled = not bool(title_auto)
if st.button("🖼️ Generează imagine", key="img_generate_btn", disabled=disabled):
    if not title_auto:
        st.error("❗ Nu am găsit automat titlul cărții în răspuns. Cere o recomandare mai întâi.")
    else:
        try:
            if st.session_state["img_kind_sel"] == "Copertă":
                data, filename = generate_ai_cover_bytes(
                    title_auto,
                    st.session_state.get("img_themes_input", ""),
                    client=client,
                    size=st.session_state.get("img_size_sel", "1024x1024"),
                    quality=st.session_state.get("img_quality_sel", "low"),
                )
            else:
                data, filename = generate_ai_scene_bytes(
                    title_auto,
                    st.session_state.get("img_scene_textarea", ""),
                    st.session_state.get("img_themes_input", ""),
                    client=client,
                    size=st.session_state.get("img_size_sel", "1024x1024"),
                    quality=st.session_state.get("img_quality_sel", "low"),
                )

            st.image(data, caption=f"{st.session_state['img_kind_sel']} pentru: {title_auto}")
            st.download_button(
                "⬇️ Descarcă PNG",
                data=data,
                file_name=filename,
                mime="image/png",
                key="img_download_btn",
            )
            st.success(f"Generată cu succes pentru „{title_auto}”.")
        except Exception as e:
            st.error(f"Eroare la generarea imaginii AI: {e}")
else:
    if not title_auto:
        st.info("ℹ️ Butonul este dezactivat până există un răspuns cu un titlu detectat.")

st.markdown("---")
st.caption("Smart Librarian — RAG + Tool • ChromaDB + OpenAI (mini chat models only) • Streamlit")
