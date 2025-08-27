# rag/embed_store.py
from __future__ import annotations
from pathlib import Path
import re, unicodedata, json, yaml
import chromadb
from chromadb.utils import embedding_functions
from config import PERSIST_DIR, OPENAI_API_KEY, EMBED_MODEL  # asigură-te că există în config

BOOKS_YAML = Path("data/book_summaries.yaml")

def _norm_title(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s or "").strip().lower().split())

def _slug(s: str) -> str:
    s = _norm_title(s)
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")

def load_summaries() -> list[dict]:
    if not BOOKS_YAML.exists():
        return []
    data = yaml.safe_load(BOOKS_YAML.read_text(encoding="utf-8")) or []
    out: list[dict] = []
    for it in (data if isinstance(data, list) else []):
        title = str(it.get("title", "")).strip()
        summary = str(it.get("summary", "")).strip()
        full = str(it.get("full_summary", "") or "").strip()
        themes = it.get("themes") or []
        if isinstance(themes, str):
            themes = [t.strip() for t in themes.split(",") if t.strip()]
        if title and (summary or full):
            out.append({
                "title": title,
                "summary": summary,
                "full_summary": full,
                "themes": themes,
            })
    return out

def _fingerprint(rows: list[dict]) -> str:
    # hash stabil al conținutului semantic
    norm = []
    for r in rows:
        norm.append({
            "t": r.get("title", ""),
            "s": r.get("summary", ""),
            "f": r.get("full_summary", ""),
            "th": r.get("themes", []),
        })
    norm.sort(key=lambda x: (x["t"] or "").lower())
    raw = json.dumps(norm, ensure_ascii=False, sort_keys=True)
    import hashlib
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def init_vector_store(summaries: list[dict], persist_path: str = PERSIST_DIR):
    fp_new = _fingerprint(summaries)
    fp_file = Path(persist_path) / "books.sha1"
    fp_old = fp_file.read_text(encoding="utf-8").strip() if fp_file.exists() else None

    client = chromadb.PersistentClient(path=persist_path)
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBED_MODEL,
    )

    # (Re)creează colecția dacă e nevoie
    try:
        coll = client.get_or_create_collection(
        name="books",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},  # <— adaugă asta
    )
    except Exception:
        coll = None

    need_rebuild = (coll is None) or (fp_old != fp_new)

    if need_rebuild:
        try:
            client.delete_collection("books")
        except Exception:
            pass
        coll = client.get_or_create_collection(name="books", embedding_function=ef)

        ids, docs, metas = [], [], []
        for r in summaries:
            parts = []
            if r.get("summary"):       parts.append(r["summary"])
            if r.get("full_summary"):  parts.append(r["full_summary"])
            if r.get("themes"):        parts.append("Themes: " + ", ".join(r["themes"]))
            text = "\n\n".join(parts).strip()
            if not text:
                continue
            ids.append(f"id:{_slug(r['title'])}")
            docs.append(text)
            # ⚠️ metadate DOAR primitive
            metas.append({
                "title": r["title"],
                "themes_txt": ", ".join(r.get("themes", [])) if r.get("themes") else ""
            })

        if ids:
            coll.add(ids=ids, documents=docs, metadatas=metas)
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        fp_file.write_text(fp_new, encoding="utf-8")

    else:
        if coll is None:
            coll = client.get_or_create_collection(name="books", embedding_function=ef)

    return coll
