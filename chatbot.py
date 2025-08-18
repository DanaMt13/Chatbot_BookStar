# chatbot.py
# -*- coding: utf-8 -*-
import json
from typing import Optional
from openai import OpenAI

from config import CHAT_MODEL, OPENAI_API_KEY
from rag.retriever import search_books
from rag.embed_store import load_summaries, init_vector_store
from tools.summary_tool import TOOL_SPEC, get_summary_by_title

client = OpenAI(api_key=OPENAI_API_KEY)

def is_offensive(text: str) -> bool:
    blocked = {"idiot", "prost", "ură", "urăsc", "urât", "dispreț", "fuck", "shit"}
    low = text.lower()
    return any(w in low for w in blocked)

def build_messages(user_query: str, candidate_title: Optional[str]):
    system_msg = {
        "role": "system",
        "content": (
            "Ești Smart Librarian, un asistent de recomandări de cărți. "
            "Folosește contextul RAG pentru a alege un titlu relevant. "
            "După ce alegi TITLUL FINAL, apelează funcția (tool) get_summary_by_title "
            "cu exact acel titlu. Apoi îmbină răspunsul într-un mesaj final, concis "
            "(4–6 fraze) + o secțiune '📖 Rezumat detaliat' cu textul primit din tool. "
            "Nu inventa titluri inexistente."
        ),
    }
    context_msg = {"role": "system", "content": f"RAG_CANDIDATE_TITLE={candidate_title or 'None'}"}
    user_msg = {"role": "user", "content": user_query.strip()}
    return [system_msg, context_msg, user_msg]

def _extract_tool_title(tool_call) -> Optional[str]:
    try:
        args = json.loads(tool_call.function.arguments or "{}")
        return args.get("title")
    except Exception:
        return None

def chat(user_query: str, collection) -> str:
    if is_offensive(user_query):
        return ("Aș vrea să păstrăm conversația respectuoasă 😊. Reformulează te rog întrebarea "
                "fără termeni ofensatori și te ajut imediat.")

    # 1) RAG -> candidat
    candidate_title = search_books(user_query, collection, top_k=1)

    # 2) Primul apel — încurajăm tool-calling
    messages = build_messages(user_query, candidate_title)
    first = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=TOOL_SPEC,
        tool_choice="auto",
        temperature=0.4,
    )
    assistant_msg = first.choices[0].message

    # adaugă assistant + eventualele tool_calls în istoricul pentru al doilea apel
    messages.append({
        "role": "assistant",
        "content": assistant_msg.content or "",
        "tool_calls": assistant_msg.tool_calls
    })

    # 3) Executăm tool-ul dacă a fost cerut
    did_tool_call = False
    final_title_from_tool = None
    if assistant_msg.tool_calls:
        for tool_call in assistant_msg.tool_calls:
            if tool_call.function.name == "get_summary_by_title":
                did_tool_call = True
                title_arg = _extract_tool_title(tool_call) or candidate_title or ""
                final_title_from_tool = title_arg
                summary_text = get_summary_by_title(title_arg)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "get_summary_by_title",
                    "content": summary_text
                })

    # 4) Al doilea apel — compunere răspuns final
    if did_tool_call:
        final = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.4,
        )
        content = final.choices[0].message.content
        # asigură-te că există „📖 Rezumat detaliat” (în caz că modelul n-a formatat)
        if "📖 Rezumat detaliat" not in (content or ""):
            # ultimul mesaj tool conține rezumatul
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            tool_text = tool_msgs[-1]["content"] if tool_msgs else ""
            book_title = final_title_from_tool or candidate_title or "cartea recomandată"
            content = (
                f"Îți recomand: **{book_title}**.\n\n"
                f"{content or ''}\n\n"
                f"📖 Rezumat detaliat:\n{tool_text}"
            )
        return content

    # 5) Fallback — dacă modelul nu a chemat tool-ul, îl chemăm noi
    safe_title = candidate_title or "o carte potrivită intereselor tale"
    summary_text = get_summary_by_title(candidate_title or "")
    return (
        f"Îți recomand: **{safe_title}**.\n\n"
        f"📖 Rezumat detaliat:\n{summary_text}"
    )

# Mic test manual
if __name__ == "__main__":
    summaries = load_summaries()
    col = init_vector_store(summaries)
    for q in [
        "Ce recomanzi pentru cineva care iubește distopiile?",
        "Ce este The Hobbit?",
        "Vreau o carte despre libertate și magie",
    ]:
        print("\n=== Q:", q)
        print(chat(q, col))
