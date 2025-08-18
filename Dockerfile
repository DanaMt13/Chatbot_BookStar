# Dockerfile
FROM python:3.11-slim

# Evită fișiere .pyc și bufferizare stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- System deps (audio + STT/TTS) ---
# ffmpeg        -> necesar pydub și procesare audio
# espeak-ng     -> backend TTS pentru pyttsx3 pe Linux
# libasound2    -> ALSA (audio)
# libgomp1      -> necesar pentru faster-whisper/ctranslate2 pe CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg espeak-ng libasound2 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Dependențe Python ---
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# --- Codul aplicației ---
COPY . .

# --- User non-root ---
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

# --- Directoare runtime scriibile în container ---
ENV AUDIO_DIR=/tmp/audio \
    COVERS_DIR=/tmp/covers
RUN mkdir -p "${AUDIO_DIR}" "${COVERS_DIR}"

# (opțional) fă ușor de montat în host: vector store-ul Chroma
VOLUME ["/app/chroma_store"]

EXPOSE 8501

# Streamlit UI
CMD ["streamlit", "run", "ui/app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
