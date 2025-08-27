# api/main.py
from __future__ import annotations
import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config as CFG
from rag.embed_store import load_summaries, init_vector_store
from rag.retriever import debug_candidates, semantic_search
from tools.summary_tool import get_summary_by_title
from chatbot import chat  # folosește RAG-first strict + tool
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

# --- bootstrap RAG o singură dată ---
PERSIST_DIR = CFG.PERSIST_DIR
summaries = load_summaries()
collection = init_vector_store(summaries, persist_path=PERSIST_DIR)

# --- FastAPI ---
app = FastAPI(title="Smart Librarian API", version="1.0")

# CORS pentru frontend (vite rulează pe 5173 de obicei)
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    os.getenv("FRONTEND_ORIGIN", "")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---- Schemas ----
class RecommendReq(BaseModel):
    query: str
    top_k: int = 5

class TopItem(BaseModel):
    title: str
    distance: float

class EvidenceItem(BaseModel):
    title: str
    distance: float
    snippet: str

class RecommendResp(BaseModel):
    title: Optional[str] = None               # titlul recomandat (din RAG)
    answer_markdown: str                      # textul final (MD) generat de chat+tool
    topk: List[TopItem]                       # top-K din RAG (distanțe)
    evidence: List[EvidenceItem]              # snippete (fragment) pentru primele K
    confidence: str                           # High / Medium / Low (euristic)
    d1: float                                 # distanța primului rezultat
    gap: float                                # d2 - d1 (∞ dacă nu există d2)

# ---- Heuristici simple de încredere ----
MAX_GOOD_DISTANCE = 1.00
def _confidence_from_pairs(pairs: list[tuple[str, float]]) -> tuple[str, float, float]:
    if not pairs:
        return "Low", float("inf"), 0.0
    d1 = float(pairs[0][1])
    d2 = float(pairs[1][1]) if len(pairs) > 1 else float("inf")
    gap = d2 - d1
    if (d1 < MAX_GOOD_DISTANCE and gap >= 0.12):
        conf = "High"
    elif (d1 < 1.10 and gap >= 0.08):
        conf = "Medium"
    else:
        conf = "Low"
    return conf, d1, gap

# ---- Routes ----
@app.post("/recommend", response_model=RecommendResp)
def recommend(req: RecommendReq):
    q = (req.query or "").strip()
    k = max(1, min(req.top_k, 8))

    # 1) Top-K (titluri + distanțe)
    pairs = list(debug_candidates(q, collection, top_k=k))
    topk = [TopItem(title=t, distance=float(d)) for (t, d) in pairs]

    # 2) Dovezi/snippete pentru top-K
    ev_raw = semantic_search(q, collection, top_k=k) or []
    evidence = [EvidenceItem(title=e["title"], distance=float(e["distance"]), snippet=e["snippet"]) for e in ev_raw]

    # 3) Recomandarea finală (RAG-first + tool) – text markdown
    answer_md = chat(q, collection)

    # 4) Încredere RAG
    confidence, d1, gap = _confidence_from_pairs(pairs)

    # 5) Titlul ales (primul din Top-K; LLM-ul e doar pentru formulare)
    title = pairs[0][0] if pairs else None

    return RecommendResp(
        title=title,
        answer_markdown=answer_md,
        topk=topk,
        evidence=evidence,
        confidence=confidence,
        d1=float(d1),
        gap=float(gap if gap != float("inf") else 1e9),
    )

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
