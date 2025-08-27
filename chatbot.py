# chatbot.py — RAG strict: titlul final = top-1 din vector store

from typing import Optional, List, Tuple
import json
from openai import OpenAI

from config import CHAT_MODEL, OPENAI_API_KEY, MODERATION_ENABLED
from rag.retriever import auto_search_books, semantic_search   # <-- avem și snippete
from tools.summary_tool import TOOL_SPEC, get_summary_by_title
from safety.moderation import moderate_text, explain_categories

client = OpenAI(api_key=OPENAI_API_KEY)

FALLBACK_BAD_WORDS = {
    "nigger","nigga","hitler","nazist","nazi","jidan","țigan","tigan",
    "hate","ură","fuck","shit","idiot","prost","imbecil","handicapat"
}

# praguri doar pentru mesaje de “încredere”, nu influențează alegerea (care e strict top-1)
MAX_SHOW_ITEMS = 5

def _fallback_blocklist(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in FALLBACK_BAD_WORDS)

def _blocked_message(reason: str) -> str:
    return (
        "Aș vrea să păstrăm conversația respectuoasă și sigură. "
        f"Mesajul tău pare să încalce regulile de conținut ({reason}).\n\n"
        "Te rog reformulează fără termeni de ură, violență, hărțuire sau conținut sexual explicit, "
        "iar eu te ajut imediat. =)"
    )

def _format_topk_section(pairs: List[Tuple[str, float]], k_selected: Optional[int] = None) -> str:
    if not pairs:
        return ""
    if k_selected is None:
        k_selected = min(len(pairs), MAX_SHOW_ITEMS)
    lines = []
    for i, (title, dist) in enumerate(pairs[:k_selected], 1):
        try:
            d = float(dist)
            sim = 1.0 - d
        except Exception:
            d = dist
            sim = 0.0
        lines.append(f"{i}. **{title}** · dist: `{d:.4f}` · sim: `{sim:.4f}`")
    return "\n\n---\n🔎 Top potriviri (RAG)\n" + "\n".join(lines)

def _extract_pairs(auto_obj) -> List[Tuple[str, float]]:
    # acceptă forme variate din retriever
    if not isinstance(auto_obj, dict):
        return []
    pairs = auto_obj.get("candidates") or []
    out = []
    for it in pairs:
        if isinstance(it, (list, tuple)) and len(it) >= 2:
            out.append((str(it[0]), float(it[1])))
        elif isinstance(it, dict) and "title" in it and "distance" in it:
            out.append((str(it["title"]), float(it["distance"])))
    return out

def chat(user_query: str, collection) -> str:
    # 0) Moderation
    if MODERATION_ENABLED:
        mod = moderate_text(user_query)
        if mod.get("flagged"):
            reason = explain_categories(mod.get("categories", {})) or "conținut interzis"
            return _blocked_message(reason)
        if mod.get("error") and _fallback_blocklist(user_query):
            return _blocked_message("conținut interzis (fallback local)")
    else:
        if _fallback_blocklist(user_query):
            return _blocked_message("conținut interzis")

    # 1) RAG: obține Top-K și alege STRICT top-1
    auto = auto_search_books(user_query, collection) or {}
    pairs = _extract_pairs(auto)
    if not pairs:
        # fără potriviri: răspuns bland
        return "Nu am găsit o potrivire relevantă în biblioteca curentă. Încearcă să formulezi altfel interesul (ex.: teme, gen, ton)."

    # top-1 (titlu + distanță)
    best_title, best_dist = pairs[0]
    topk_section = _format_topk_section(pairs, k_selected=min(len(pairs), MAX_SHOW_ITEMS))

    # snippete pentru titlul ales (arătăm de ce îl propunem)
    evidence_snip = ""
    try:
        ev = semantic_search(user_query, collection, top_k=MAX_SHOW_ITEMS)
        for e in ev:
            if (e.get("title") or "").strip().lower() == best_title.strip().lower():
                evidence_snip = e.get("snippet", "")
                break
    except Exception:
        pass

    # 2) Forțăm tool calling pentru titlul ales (LLM NU mai poate alege altceva)
    system_msg = {
        "role": "system",
        "content": (
            "Ești Smart Librarian. Titlul final a fost ales de sistem pe baza RAG și este FIXAT.\n"
            "TREBUIE să apelezi funcția get_summary_by_title cu EXACT acest titlu, apoi vei genera răspunsul final.\n"
            "Nu inventa titluri și nu schimba titlul decis."
        )
    }
    chosen_msg = {"role": "system", "content": f"CHOSEN_TITLE={best_title}"}
    user_msg   = {"role": "user", "content": (user_query or "").strip()}

    # forțăm tool-ul explicit
    first = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[system_msg, chosen_msg, user_msg],
        tools=TOOL_SPEC,
        tool_choice={"type": "function", "function": {"name": "get_summary_by_title"}},
    )
    assistant_msg = first.choices[0].message

    # 3) Executăm tool-ul (cu titlul fixat)
    summary_text = get_summary_by_title(best_title)

    # 4) Al doilea apel — cere modelului să redacteze folosind rezumatul primit
    messages = [
        system_msg,
        chosen_msg,
        user_msg,
        {
            "role": "assistant",
            "tool_calls": assistant_msg.tool_calls,
            "content": assistant_msg.content or ""
        },
        {
            "role": "tool",
            "tool_call_id": assistant_msg.tool_calls[0].id if assistant_msg.tool_calls else "tool_call_id",
            "name": "get_summary_by_title",
            "content": summary_text
        },
        {
            "role": "system",
            "content": (
                "Formatează răspunsul astfel:\n"
                f"1) Recomandă **{best_title}** în 4–6 fraze (conversațional).\n"
                "2) Apoi o secțiune „📖 Rezumat detaliat:” cu exact textul primit de la tool.\n"
                "3) O secțiune „🔎 Dovezi RAG (căutare semantică):” cu 1–2 fraze, folosind snippetul oferit mai jos.\n"
                "Nu adăuga altă carte."
            )
        },
        {"role": "system", "content": f"RAG_SNIPPET={evidence_snip or ''}"}
    ]
    final = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    text = final.choices[0].message.content or ""

    # 5) Atașăm Top-K folosit
    return f"{text}{topk_section}"
