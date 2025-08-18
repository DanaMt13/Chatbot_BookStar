# main.py
from rag.embed_store import load_summaries, init_vector_store
from chatbot import chat

def main():
    # 1) Încarcă rezumatele scurte
    summaries = load_summaries()
    print(f"✅ Am încărcat {len(summaries)} rezumate scurte din YAML.")

    # 2) Inițializează ChromaDB + embeddings
    collection = init_vector_store(summaries)
    print("✅ Vector store inițializat.")

    # 3) CLI simplu
    print("\n📚 Smart Librarian — scrie o întrebare (ex: 'Vreau o carte despre magie și război').")
    print("   Scrie 'exit' pentru a ieși.\n")

    while True:
        q = input("Tu: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("La revedere! 👋")
            break
        try:
            answer = chat(q, collection)
            print("\n" + answer + "\n")
        except Exception as e:
            print(f"⚠️ Eroare: {e}\n")

if __name__ == "__main__":
    main()
