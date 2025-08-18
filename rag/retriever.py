# rag/retriever.py
from typing import Optional, List, Tuple, Union, Dict

# ---------------- Helpers ----------------
def _pairs_from_result(res) -> List[Tuple[str, float]]:
    """
    Transformă răspunsul Chroma în perechi (title, distance).
    Distanța mai mică = mai similar.
    """
    metas = (res or {}).get("metadatas") or [[]]
    dists = (res or {}).get("distances") or [[]]
    if not metas or not metas[0]:
        return []
    out: List[Tuple[str, float]] = []
    for m, d in zip(metas[0], dists[0] if dists else []):
        t = (m or {}).get("title")
        if t is not None:
            try:
                out.append((t, float(d)))
            except Exception:
                out.append((t, 0.0))
    return out

# ---------------- API clasic ----------------
def search_books(
    query: str,
    collection,
    top_k: int = 1,
    return_scores: bool = False,
) -> Optional[Union[str, List[Tuple[str, float]]]]:
    """
    Caută semantic în ChromaDB.
    - top_k=1 & return_scores=False => întoarce titlul (str) sau None.
    - altfel => întoarce lista [(title, distance), ...].
    """
    if not query or not isinstance(query, str):
        return None if not return_scores else []

    if top_k < 1:
        top_k = 1

    res = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )
    pairs = _pairs_from_result(res)

    if not pairs:
        return None if (top_k == 1 and not return_scores) else []

    if top_k == 1 and not return_scores:
        return pairs[0][0]

    return pairs

# --------------- top_k dinamic ---------------
def auto_search_books(
    query: str,
    collection,
    k_probe: int = 5,
    dist_confident: float = 0.12,
    gap_ratio_confident: float = 0.15,
    dist_weak: float = 0.35,
    k_max: int = 8,
) -> Dict[str, Union[str, int, List[Tuple[str, float]]]]:
    """
    Alege automat top_k după „încredere” sugerată de distanțe:
      - dacă top-1 e foarte bun sau există gap mare față de #2 -> k_selected = 1
      - altfel extinde căutarea (k_probe sau k_max)
    """
    if not query or not isinstance(query, str):
        return {"best_title": None, "k_selected": 0, "pairs": []}

    res = collection.query(
        query_texts=[query],
        n_results=max(1, k_probe),
        include=["metadatas", "documents", "distances"],
    )
    pairs = _pairs_from_result(res)
    if not pairs:
        return {"best_title": None, "k_selected": 0, "pairs": []}

    pairs.sort(key=lambda x: x[1])
    top1 = pairs[0]
    top2 = pairs[1] if len(pairs) > 1 else None

    if top1[1] <= dist_confident:
        return {"best_title": top1[0], "k_selected": 1, "pairs": pairs}

    if top2:
        gap_ratio = (top2[1] - top1[1]) / max(top1[1], 1e-6)
        if gap_ratio >= gap_ratio_confident:
            return {"best_title": top1[0], "k_selected": 1, "pairs": pairs}

    k_selected = k_probe if top1[1] < dist_weak else k_max
    if k_selected > len(pairs):
        res2 = collection.query(
            query_texts=[query],
            n_results=k_selected,
            include=["metadatas", "documents", "distances"],
        )
        pairs = _pairs_from_result(res2)
        pairs.sort(key=lambda x: x[1])

    best = pairs[0][0] if pairs else None
    return {"best_title": best, "k_selected": min(k_selected, len(pairs)), "pairs": pairs}

# --------------- debug transparență ---------------
def debug_candidates(query: str, collection, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Întoarce Top-K ca listă [(title, distance), ...] sortată crescător după distanță.
    """
    if top_k < 1:
        top_k = 1

    res = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )
    pairs = _pairs_from_result(res)
    pairs.sort(key=lambda x: x[1])
    return pairs
