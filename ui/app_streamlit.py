# ui/app_streamlit.py
# -*- coding: utf-8 -*-
# Smart Librarian — Streamlit UI (RAG + Tool + TTS + STT + AI Images) cu dovezi RAG

import os, sys, time, re
from pathlib import Path

# --- permite importuri din rădăcina proiectului când rulăm din /ui ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from openai import OpenAI

# Microfon: preferă audio-recorder-streamlit; fallback pe st_mic_recorder
try:
    from audio_recorder_streamlit import audio_recorder  # type: ignore
    HAS_AUDIO_REC = True
except Exception:
    HAS_AUDIO_REC = False

try:
    from st_mic_recorder import mic_recorder  # type: ignore
    HAS_MIC_REC = True
except Exception:
    HAS_MIC_REC = False

# AI Image generator (gpt-image-1)
from img.ai_gen import generate_ai_cover_bytes, generate_ai_scene_bytes

# config robust
import config as CFG

# ---- Config & defaults ------------------------------------------------------
PERSIST_DIR     = CFG.PERSIST_DIR
OPENAI_API_KEY  = CFG.OPENAI_API_KEY

TTS_MODE_DEFAULT   = getattr(CFG, "TTS_MODE", "off")   # openai | offline | off
TTS_VOICE_DEFAULT  = getattr(CFG, "TTS_VOICE", "alloy")
TTS_FORMAT_DEFAULT = getattr(CFG, "TTS_FORMAT", "mp3") # mp3 | wav
TTS_RATE_DEFAULT   = int(getattr(CFG, "TTS_RATE", 170))
TTS_VOL_DEFAULT    = float(getattr(CFG, "TTS_VOLUME", 0.8))

AUDIO_DIR = os.getenv(
    "AUDIO_DIR",
    getattr(CFG, "AUDIO_DIR", "/tmp/audio" if os.path.exists("/.dockerenv") else "data/tmp_audio")
)

# ---- RAG & Chat -------------------------------------------------------------
from rag.embed_store import load_summaries, init_vector_store
from rag.retriever import debug_candidates, semantic_search
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

# minimal CSS pt. look curat
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }
.stButton button { border-radius: 12px; }
.stAlert, .stCodeBlock, .stDataFrame { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("📚 Smart Librarian — RAG + Tool Calling")
st.caption(
    f"Vector store: **ChromaDB** (`{PERSIST_DIR}`) • Embeddings: **text-embedding-3-small** • "
    f"Chat model: **{getattr(CFG, 'CHAT_MODEL', 'gpt-4o-mini')}**"
)

# ---- utils -----------------------------------------------------------------
def guess_title_from_answer(answer: str, known_titles: list[str] | None = None) -> str | None:
    """
    Extrage titlul recomandat din răspunsul asistentului.
    Prioritate:
    1) linia cu „Îți recomand …”
    2) primul element din „Top potriviri (RAG)”
    3) orice titlu 'quoted'
    4) fallback: cel mai devreme match din known_titles
    """
    if not answer:
        return None
    txt = answer.strip()

    # 1) „Îți recomand …”
    patterns = [
        r"Îți recomand[^:\n]*:\s*\*\*(.+?)\*\*",
        r"Îți recomand[^\n]*?[„\"]\s*([^„”\"\n]+?)\s*[”\"]",
        r"Recomandare[^:\n]*:\s*\*\*(.+?)\*\*",
        r"Recomand[^\n]*?[„\"]\s*([^„”\"\n]+?)\s*[”\"]",
    ]
    for pat in patterns:
        m = re.search(pat, txt, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # 2) Top potriviri: primul item
    m = re.search(r"Top\s+potriviri.*?\n\s*1\.\s*([^\n·]+)", txt, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(" .")

    # 3) quoted fallback
    m = re.search(r"[„\"]\s*([^„”\"\n]+?)\s*[”\"]", txt)
    if m:
        return m.group(1).strip()

    # 4) earliest match din known_titles
    if known_titles:
        low = txt.lower()
        matches = []
        for t in known_titles:
            t_str = (t if isinstance(t, str) else str(t)).strip()
            if not t_str:
                continue
            idx = low.find(t_str.lower())
            if idx != -1:
                matches.append((idx, t_str))
        if matches:
            matches.sort(key=lambda x: x[0])
            return matches[0][1]

    return None


def _time_debug_candidates(q: str, k: int):
    t0 = time.perf_counter()
    tops = list(debug_candidates(q, collection, top_k=k))
    ms = (time.perf_counter() - t0) * 1000.0
    return tops, ms


@st.cache_resource(show_spinner=False)
def get_known_titles() -> list[str]:
    try:
        books = load_summaries()
        titles: list[str] = []
        for b in books:
            t = b.get("title")
            if t is None:
                continue
            if not isinstance(t, str):
                t = str(t)
            t = t.strip()
            if t:
                titles.append(t)
        titles = list(dict.fromkeys(titles))
        return titles
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
        from pydub.utils import which
        AudioSegment.converter = which("ffmpeg") or "ffmpeg"
        AudioSegment.ffprobe   = which("ffprobe") or "ffprobe"
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
            resp = client.audio.speech.create(model="tts-1", voice=voice, input=text, response_format=fmt)
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
    ("last_title_auto", None),
    ("known_titles_cache", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ✅ inițializare unică pentru câmpul de input
if "query_text" not in st.session_state:
    st.session_state["query_text"] = "Vreau o carte despre libertate și magie"

# ---- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Opțiuni")
    st.checkbox("Arată Top-K (debug RAG)", value=False, key="show_debug_chk")
    st.slider("Top-K pentru debug", 1, 8, 5, key="topk_slider")

    st.markdown("---")
    st.subheader("🔈 Text-to-Speech (manual)")
    mode_to_idx = {"openai": 0, "offline": 1, "off": 2}
    st.radio(
        "Motor TTS",
        ["OpenAI tts-1 (plătit)", "Offline pyttsx3 (gratuit)", "Oprit"],
        index=mode_to_idx.get(TTS_MODE_DEFAULT, 2),
        key="tts_choice_radio",
    )
    st.text_input("Voce (tts-1)", value=TTS_VOICE_DEFAULT, key="voice_input")
    st.selectbox("Format audio (tts-1)", ["mp3", "wav"], index=0 if TTS_FORMAT_DEFAULT == "mp3" else 1, key="audiofmt_sel")
    st.slider("Viteză (pyttsx3)", 100, 250, TTS_RATE_DEFAULT, key="tts_rate_slider")
    st.slider("Volum (pyttsx3)", 0.1, 1.0, float(TTS_VOL_DEFAULT), key="tts_vol_slider")
    st.slider("Limită durată audio (sec)", 10, 300, 150, key="maxsecs_slider")

    st.markdown("---")
    st.subheader("🎙️ Speech-to-Text (manual)")
    st.radio("Motor STT", ["Offline (0$)", "OpenAI gpt-4o-mini-transcribe (plătit)"], index=0, key="stt_choice_radio")
    stt_lang = st.selectbox(
    "Limba audio",
    ["auto", "ro", "en", "fr", "de", "es", "it", "pt"],
    index=1,  # ro by default
    key="stt_lang_sel",
    help="Forțează limba pentru recunoaștere. 'auto' poate greși pe înregistrări scurte/noisy."
    )

    fwhisper_size = st.selectbox(
        "Model STT offline (faster-whisper)",
        ["tiny", "base", "small"], index=1, key="fwhisper_size_sel",
        help="tiny = cel mai rapid; base = recomandat; small = mai precis dar mai lent"
    )
    os.environ["FWHISPER_MODEL"] = fwhisper_size

# ---- Întrebare: Enter/Buton ------------------------------------------------
st.subheader("Pune o întrebare")

def run_recommendation(q: str):
    if not q or not q.strip():
        st.warning("Introdu o întrebare.")
        return

    try:
        # Debug RAG (opțional)
        if st.session_state.get("show_debug_chk"):
            tops, rag_ms = _time_debug_candidates(q, st.session_state.get("topk_slider", 5))
            with st.expander(f"🔎 RAG (Top-{len(tops)}) • {rag_ms:.0f} ms", expanded=False):
                st.caption(f"🔧 Caut semantic ca: `{q}`")
                try:
                    import pandas as pd  # type: ignore
                    df = pd.DataFrame(
                        [{"#": i+1, "Titlu": t, "Distanță": round(d, 4)} for i, (t, d) in enumerate(tops)]
                    )
                    st.dataframe(df, hide_index=True, use_container_width=True)
                except Exception:
                    for i, (t, d) in enumerate(tops, start=1):
                        st.write(f"{i}. **{t}** · dist: `{d:.4f}`")

                # === Badge de încredere bazat pe d1 și gap față de locul 2 ==========
                if tops:
                    try:
                        d1 = float(tops[0][1])
                    except Exception:
                        d1 = float("inf")

                    if len(tops) >= 2:
                        try:
                            d2 = float(tops[1][1])
                        except Exception:
                            d2 = d1
                        delta = d2 - d1
                    else:
                        delta = float("inf")

                    # Heuristici simple pentru "confidence"
                    if (d1 < 0.95 and delta >= 0.15):
                        confidence = "High"
                    elif (d1 < 1.10 and delta >= 0.08):
                        confidence = "Medium"
                    else:
                        confidence = "Low"

                    # Cosine distance -> similarity ≈ 1 - distance (doar informativ)
                    sim_est = 0.0 if (d1 == float("inf")) else max(0.0, 1.0 - d1)
                    st.caption(
                        f"📈 Încredere RAG: **{confidence}** "
                        f"(d1={d1:.4f}, Δ={('∞' if delta==float('inf') else f'{delta:.4f}')}, sim≈{sim_est:.3f})"
                    )

            # Snippete semantice pentru Top-K
            ev = semantic_search(q, collection, top_k=st.session_state.get("topk_slider", 5))
            with st.expander("🧭 Dovezi RAG (snippete pentru Top-K)", expanded=False):
                if not ev:
                    st.caption("Nu am putut extrage snippete (colecție goală sau eroare).")
                else:
                    for i, e in enumerate(ev, 1):
                        st.markdown(f"**{i}. {e['title']}** · dist: `{e['distance']:.4f}`\n\n> {e['snippet']}")

        # Răspunsul de recomandare
        with st.spinner("Gândesc o recomandare..."):
            answer = chat(q, collection)

        # Persistă răspunsul + reset audio
        st.session_state["last_answer"] = answer
        st.session_state["last_audio_path"] = ""
        st.session_state["last_audio_dur"] = -1.0
        st.session_state["last_audio_trunc"] = False
        st.session_state["query_last"] = q

        # Titluri cunoscute – cache în sesiune
        known = st.session_state.get("known_titles_cache")
        if not isinstance(known, list) or not known:
            raw = get_known_titles()
            known = [t.strip() for t in raw if isinstance(t, str) and t and t.strip()]
            st.session_state["known_titles_cache"] = known

        # Extrage titlul și îl „îngheață” pentru secțiunea de imagini
        title = guess_title_from_answer(answer, known_titles=known)
        st.session_state["last_title_auto"] = title

    except Exception as e:
        st.error(f"Eroare: {e}")


# ✅ Form (Enter trimite) — fără value=, păstrăm doar key în session_state
with st.form("ask_form", clear_on_submit=False):
    st.text_input(
        "Exemple: „Vreau o carte despre libertate și magie”, „Ce recomanzi pentru cineva care iubește distopiile?”, „Ce este The Hobbit?”",
        key="query_text",
        placeholder="Scrie aici…",
    )
    submitted = st.form_submit_button("💬 Cere o recomandare (Enter)")

if submitted:
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
            st.error(f"Fișier prea lung: ~{int(dur)}s. Limita este {st.session_state.get('maxsecs_slider', 150)}s.")
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

# === 🎤 Întrebare prin microfon =============================================
st.markdown("### 🎤 Întrebare prin microfon (0$ offline)")

if not (HAS_AUDIO_REC or HAS_MIC_REC):
    st.info("Pentru microfon: adaugă în requirements.txt `audio-recorder-streamlit==0.0.8` sau `streamlit-mic-recorder==0.0.8` și reconstruiește Docker.")
else:
    wav_bytes = None
    if HAS_AUDIO_REC:
        wav_bytes = audio_recorder(
            pause_threshold=2.0, sample_rate=44100,
            text="🎙️ Start / Stop", recording_color="#e8f0fe",
            neutral_color="#f0f2f6", icon_name="microphone", icon_size="2x",
        )
    elif HAS_MIC_REC:
        rec = mic_recorder(start_prompt="🎙️ Înregistrează", stop_prompt="⏹️ Oprește & transcrie", format="wav", key="mic_recorder_widget")
        if isinstance(rec, (bytes, bytearray)):
            wav_bytes = rec
        elif isinstance(rec, dict) and "bytes" in rec:
            wav_bytes = rec["bytes"]

    if wav_bytes:
        tmp_dir = Path(AUDIO_DIR) / "mic"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        wav_path = tmp_dir / f"mic_{int(time.time())}.wav"
        wav_path.write_bytes(wav_bytes)

        try:
            dur = audio_duration_seconds(wav_path)
        except Exception:
            dur = -1

        max_secs = st.session_state.get("maxsecs_slider", 150)
        if dur != -1 and dur > max_secs:
            st.error(f"Înregistrarea are ~{int(dur)}s, limita este {max_secs}s.")
        else:
            engine = "offline" if st.session_state.get("stt_choice_radio", "Offline").startswith("Offline") else "openai"
            try:
                txt, real_dur = transcribe_file(wav_path, engine, client=client, model="gpt-4o-mini-transcribe", language=st.session_state.get("stt_lang_sel", "ro"))
                st.success(f"Transcriere (~{int(real_dur)}s) realizată din microfon.")
                with st.expander("Text transcris (microfon)"):
                    st.write(txt)
                with st.spinner("Gândesc o recomandare..."):
                    answer = chat(txt, collection)
                st.session_state["last_answer"] = answer
                st.session_state["query_last"] = txt
                st.session_state["last_audio_path"] = ""
                st.session_state["last_audio_dur"] = -1.0
                st.session_state["last_audio_trunc"] = False
            except Exception as e:
                st.error(f"Eroare STT (microfon): {e}")

# ---- Afișare răspuns + TTS + Imagini ---------------------------------------
if st.session_state["last_answer"]:
    st.markdown("### 💡 Răspuns")
    last_q = st.session_state.get("query_last", "").strip()
    if last_q:
        st.chat_message("user").write(last_q)
    st.chat_message("assistant").markdown(st.session_state["last_answer"])

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

# === Imagini AI pentru recomandare (nu salvează pe disc) ====================
st.markdown("### 🖼️ Imagine AI pentru recomandare")

known_titles = get_known_titles()
title_auto = st.session_state.get("last_title_auto")
if not title_auto:
    raw_known = st.session_state.get("known_titles_cache")
    if not isinstance(raw_known, list) or not raw_known:
        raw_known = get_known_titles()
        raw_known = [t.strip() for t in raw_known if isinstance(t, str) and t and t.strip()]
        st.session_state["known_titles_cache"] = raw_known
    title_auto = guess_title_from_answer(st.session_state.get("last_answer", ""), known_titles=raw_known)
    st.session_state["last_title_auto"] = title_auto

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
            with st.status("🎨 Generez imaginea...", expanded=True) as s:
                s.update(label="📤 Trimit prompt către model...", state="running")
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
                s.update(label=" Post-procesare rezultat...", state="running")
                st.image(data, caption=f"{st.session_state['img_kind_sel']} pentru: {title_auto}")
                st.download_button(
                    "⬇️ Descarcă PNG", data=data, file_name=filename, mime="image/png", key="img_download_btn",
                )
                s.update(label="Done! Imagine generată.", state="complete")
                st.toast("Imagine generată", icon="🎉")
        except Exception as e:
            st.error(f"Eroare la generarea imaginii AI: {e}")
else:
    if not title_auto:
        st.info("ℹ️ Butonul este dezactivat până există un răspuns cu un titlu detectat.")

st.markdown("---")
st.caption("Smart Librarian — RAG + Tool • ChromaDB + OpenAI (mini chat models only) • Streamlit")
