# safety/moderation.py
from openai import OpenAI
import os

_MODEL = os.getenv("MODERATION_MODEL", "omni-moderation-latest")
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

def moderate_text(text: str) -> dict:
    """Returnează dict cu 'flagged', 'categories', 'scores' + 'error' dacă a eșuat."""
    try:
        resp = _client.moderations.create(model=_MODEL, input=text or "")
        r = resp.results[0]
        return {
            "flagged": bool(r.flagged),
            "categories": dict(r.categories),
            "scores": dict(r.category_scores),
            "error": None,
        }
    except Exception as e:
        return {"flagged": False, "categories": {}, "scores": {}, "error": str(e)}

def explain_categories(categories: dict) -> str:
    if not categories:
        return ""
    labels = [k.replace("_", " ") for k, v in categories.items() if v]
    return ", ".join(labels)
