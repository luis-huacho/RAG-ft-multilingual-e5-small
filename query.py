"""
query.py - Consulta el RAG: retrieval + generacion opcional.

Uso:
    python query.py "pregunta" --index base
    python query.py "pregunta" --index ft_L2 --generation
"""

import argparse
import os
import pickle
import time
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BASE_MODEL = "intfloat/multilingual-e5-small"
LLM_MODEL = "gpt-4o-mini-2024-07-18"

SYSTEM_PROMPT = (
    "Eres un asistente que responde preguntas basandose unicamente en el contexto proporcionado. "
    "Si la respuesta no se encuentra en el contexto, indica que no tienes suficiente informacion."
)


def resolve_paths(index_mode):
    """Resuelve rutas de modelo y indice FAISS segun el modo."""
    if index_mode == "base":
        return BASE_MODEL, "output/faiss_base/"
    else:
        n = index_mode.split("L")[1]
        return f"output/ft_model_L{n}/", f"output/faiss_{index_mode}/"


def main():
    parser = argparse.ArgumentParser(description="Consulta RAG")
    parser.add_argument("query", help="Texto de la consulta")
    parser.add_argument("--index", default="base", help="base o ft_L{n}")
    parser.add_argument("--generation", action="store_true", help="Generar respuesta con LLM")
    parser.add_argument("--top-k", type=int, default=5, help="Chunks a recuperar")
    args = parser.parse_args()

    model_path, index_dir = resolve_paths(args.index)

    # Device info
    if DEVICE.startswith("cuda"):
        print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {DEVICE}")

    # Cargar chunks
    with open("output/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    chunks_by_id = {c["chunk_id"]: c for c in chunks}

    # Cargar indice FAISS y mapping
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "chunk_ids.pkl"), "rb") as f:
        chunk_ids = pickle.load(f)

    # Cargar modelo de embeddings
    model = SentenceTransformer(model_path, device=DEVICE)

    # --- Retrieval ---
    t_start = time.perf_counter()
    query_embedding = model.encode(
        [f"query: {args.query}"],
        normalize_embeddings=True,
    )
    query_embedding = np.array(query_embedding, dtype=np.float32)
    scores, indices = index.search(query_embedding, args.top_k)
    t_retrieval = (time.perf_counter() - t_start) * 1000

    # Recopilar resultados
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        cid = chunk_ids[idx]
        chunk = chunks_by_id[cid]
        results.append({"rank": rank + 1, "score": float(score), "chunk": chunk})

    # Mostrar resultados de retrieval
    print(f'\nQuery: "{args.query}"')
    print(f"Index: {args.index}")
    print(f"Top {args.top_k} resultados:")
    print("-" * 60)
    for r in results:
        c = r["chunk"]
        preview = c["text"][:120].replace("\n", " ")
        print(f"[{r['rank']}] score={r['score']:.4f} | {c['chunk_id']} | {c['source_file']}")
        print(f"    \"{preview}...\"")
        print()

    # --- Generacion (opcional) ---
    if args.generation:
        context = "\n---\n".join(r["chunk"]["text"] for r in results)
        user_message = f"Contexto:\n---\n{context}\n---\n\nPregunta: {args.query}\n\nRespuesta:"

        client = OpenAI()
        t_start = time.perf_counter()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )
        t_generation = (time.perf_counter() - t_start) * 1000
        answer = response.choices[0].message.content

        print("Respuesta generada (gpt-4o-mini):")
        print("=" * 60)
        print(answer)
        print()
        print(f"Latency: retrieval={t_retrieval:.0f}ms | generation={t_generation:.0f}ms | total={t_retrieval + t_generation:.0f}ms")
    else:
        print(f"Latency: retrieval={t_retrieval:.0f}ms")


if __name__ == "__main__":
    main()
