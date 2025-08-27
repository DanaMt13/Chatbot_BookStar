# rag/retriever.py
from __future__ import annotations
from typing import List, Tuple, Dict
import re, unicodedata

# ---------- helpers ----------
def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")

def _norm_query(s: str) -> str:
    s = _nfkc(s).strip()
    s = re.sub(r"\s+", " ", s).lower()
    return s

def _best_snippet(doc: str, query: str, max_len: int = 220) -> str:
    if not doc:
        return ""
    q = _norm_query(query)
    words = [w for w in re.findall(r"[a-zA-Zăâîșțéèüöß0-9]+", q) if len(w) > 2]
    txt = doc.replace("\n", " ").strip()
    idxs = [txt.lower().find(w) for w in words]
    idxs = [i for i in idxs if i != -1]
    start = max(0, min(idxs) - 40) if idxs else 0
    snippet = txt[start:start+max_len].strip()
    return snippet

# ---------- public API ----------
def debug_candidates(query: str, collection, top_k: int = 5) -> List[Tuple[str, float]]:
    q = _norm_query(query)
    if not q:
        return []
    res = collection.query(
        query_texts=[q],
        n_results=top_k,
        include=["distances", "metadatas"],
    )
    titles = (res.get("metadatas") or [[]])[0]
    dists  = (res.get("distances") or [[]])[0]
    out = []
    for meta, dist in zip(titles, dists):
        t = (meta or {}).get("title", "")
        out.append((t, float(dist)))
    return out

def semantic_search(query: str, collection, top_k: int = 5) -> List[Dict]:
    q = _norm_query(query)
    if not q:
        return []
    res = collection.query(
        query_texts=[q],
        n_results=top_k,
        include=["distances", "metadatas", "documents"],
    )
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    docs  = (res.get("documents") or [[]])[0]
    items = []
    for meta, dist, doc in zip(metas, dists, docs):
        title = (meta or {}).get("title", "")
        items.append({
            "title": title,
            "distance": float(dist),
            "snippet": _best_snippet(doc or "", q),
        })
    return items

def auto_search_books(query: str, collection, top_k: int = 5) -> dict:
    items = semantic_search(query, collection, top_k=top_k)
    pairs = [(it["title"], it["distance"]) for it in items]
    best_title   = items[0]["title"] if items else None
    best_dist    = items[0]["distance"] if items else None
    best_snippet = items[0]["snippet"] if items else ""
    return {
        "best_title": best_title,
        "best_distance": best_dist,
        "best_snippet": best_snippet,
        "candidates": pairs,
    }
