# img/ai_gen.py
import base64, hashlib
from pathlib import Path
from typing import Tuple
from openai import OpenAI

try:
    import requests  # pentru fallback pe URL
except Exception:
    requests = None


def _slug(s: str) -> str:
    s = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s.strip()).strip("_")
    return s or "img"

def build_cover_prompt(title: str, themes: str = "") -> str:
    themes = (themes or "").strip()
    themes_part = f" Themes: {themes}." if themes else ""
    return (
        f"Design an illustrated book cover concept for the novel '{title}'."
        f"{themes_part} Visualize the core motifs and mood without placing any text on the image. "
        "Cinematic lighting, cohesive color palette, painterly, clean composition."
    )

def build_scene_prompt(title: str, scene: str = "", themes: str = "") -> str:
    scene = (scene or "").strip() or "depict the core conflict and atmosphere"
    themes = (themes or "").strip()
    themes_part = f"(themes: {themes})" if themes else ""
    return (
        f"Key scene illustration for '{title}'. {scene} {themes_part}. "
        "Wide shot, cinematic mood, no text on the image."
    )

# ---- apel OpenAI cu fallback pe toate variantele posibile -------------------
def _images_generate_any(client: OpenAI, prompt: str, size: str, quality: str | None):
    """
    Încearcă pe rând:
      - response_format="b64_json" (+ quality dacă e acceptat)
      - fără response_format (unele SDK-uri o refuză)
      - fără quality (dacă dă eroare)
    """
    kwargs = dict(model="gpt-image-1", prompt=prompt, size=size)

    # încearcă cu response_format=b64_json
    try:
        if quality:
            kwargs["quality"] = quality
        return client.images.generate(response_format="b64_json", **kwargs)
    except Exception as e1:
        # încearcă fără response_format
        try:
            return client.images.generate(**kwargs)
        except Exception:
            # încearcă fără quality
            try:
                kwargs.pop("quality", None)
                return client.images.generate(**kwargs)
            except Exception as e3:
                raise RuntimeError(f"OpenAI Images API a eșuat: {e3}") from e1

def _resp_first_item(resp):
    # resp.data[0] la majoritatea SDK-urilor
    if hasattr(resp, "data"):
        data = resp.data
        if isinstance(data, (list, tuple)) and data:
            return data[0]
    # fallback pt. eventuale dict-uri
    if isinstance(resp, dict):
        data = resp.get("data") or []
        if data:
            return data[0]
    return None

def _bytes_from_item(item) -> bytes | None:
    # b64_json (preferat)
    b64 = getattr(item, "b64_json", None)
    if not b64 and isinstance(item, dict):
        b64 = item.get("b64_json")
    if b64:
        return base64.b64decode(b64)

    # url (fallback – necesită requests)
    url = getattr(item, "url", None)
    if not url and isinstance(item, dict):
        url = item.get("url")
    if url:
        if requests is None:
            raise RuntimeError("API a returnat URL, dar 'requests' nu este instalat.")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.content

    return None

def _choose_size(size: str) -> str:
    # GPT Image 1 suportă în special 1024^2 și 1024×1536/1536×1024.
    allowed = {"512x512", "1024x1024", "1024x1536", "1536x1024"}
    return size if size in allowed else "1024x1024"

def _choose_quality(q: str | None) -> str | None:
    # tabelul tău indică low/medium/high; dacă SDK-ul nu-l acceptă, scoatem param.
    if not q:
        return "low"
    q = q.lower()
    return q if q in {"low", "medium", "high"} else "low"

def _filename(kind: str, title: str, size: str) -> str:
    slug = _slug(title)
    return f"{slug}_{kind}_{size}.png"

def _generate_image_bytes(prompt: str, client: OpenAI, size: str, quality: str | None) -> bytes:
    size = _choose_size(size)            # acceptă 512x512, 1024x1024, 1024x1536, 1536x1024
    quality = _choose_quality(quality)   # low/medium/high (dacă nu e acceptat, scoatem param)
    try:
        resp = _images_generate_any(client, prompt=prompt, size=size, quality=quality)
    except Exception as e_first:
        # fallback: dacă 512x512 nu e suportat de endpointul curent → încearcă 1024x1024
        if size == "512x512":
            resp = _images_generate_any(client, prompt=prompt, size="1024x1024", quality=quality)
            size = "1024x1024"
        else:
            raise

    item = _resp_first_item(resp)
    if not item:
        raise RuntimeError("API nu a returnat niciun element de imagine.")
    data = _bytes_from_item(item)
    if not data:
        raise RuntimeError("API nu a returnat nici b64_json, nici url pentru imagine.")
    return data


# ------------------ API public: funcții care întorc BYTES + nume fișier -----
def generate_ai_cover_bytes(
    title: str,
    themes: str,
    client: OpenAI,
    size: str = "1024x1024",
    quality: str | None = "low",
) -> Tuple[bytes, str]:
    prompt = build_cover_prompt(title, themes)
    data = _generate_image_bytes(prompt, client=client, size=size, quality=quality)
    return data, _filename("cover", title, _choose_size(size))

def generate_ai_scene_bytes(
    title: str,
    scene: str,
    themes: str,
    client: OpenAI,
    size: str = "1024x1024",
    quality: str | None = "low",
) -> Tuple[bytes, str]:
    prompt = build_scene_prompt(title, scene, themes)
    data = _generate_image_bytes(prompt, client=client, size=size, quality=quality)
    return data, _filename("scene", title, _choose_size(size))
