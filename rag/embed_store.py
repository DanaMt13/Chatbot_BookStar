# rag/embed_store.py
import os
from dotenv import load_dotenv

# 🔽 IMPORTANT: importă yaml!
import yaml

from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# dacă vrei să centralizezi totul, folosim config
from config import EMBED_MODEL, PERSIST_DIR, OPENAI_API_KEY

load_dotenv()  # .env deja este folosit în config

def load_summaries(file_path: str = "data/book_summaries.yaml"):
    """
    Încarcă lista de cărți și rezumate scurte din YAML.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if not isinstance(data, list) or not data:
                raise ValueError("Fișierul YAML trebuie să conțină o listă cu elemente.")
            return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Nu găsesc {file_path}. Verifică path-ul.")
    except Exception as e:
        raise RuntimeError(f"Eroare la citirea YAML: {e}")

def init_vector_store(summaries, persist_path: str = PERSIST_DIR):
    """
    Initializează/angajează ChromaDB și adaugă embedding-uri OpenAI (text-embedding-3-small).
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY lipsește din .env")

    client = PersistentClient(path=persist_path)

    embedding_fn = OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBED_MODEL
    )

    collection = client.get_or_create_collection(
        name="books",
        embedding_function=embedding_fn
    )

    # Adăugăm/actualizăm elementele; dacă există ID duplicat, îl ignorăm
    for book in summaries:
        title = book.get("title")
        summary = book.get("summary")
        if not title or not summary:
            continue
        try:
            collection.add(
                ids=[title],
                documents=[summary],
                metadatas=[{"title": title}]
            )
        except Exception:
            # probabil există deja; poți ignora sau trata altfel
            pass

    return collection
