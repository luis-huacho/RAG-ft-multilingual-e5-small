"""
faiss_index.py - Genera embeddings y construye indice FAISS persistente.

Uso:
    python faiss_index.py base          # Con modelo base e5-small
    python faiss_index.py ft_L2         # Con modelo fine-tuned L2

Indice: IndexFlatIP (inner product sobre vectores normalizados = cosine similarity)
Dimension: 384
"""

import sys
import os
import pickle
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BASE_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384
BATCH_SIZE = 64


def print_device_info():
    if DEVICE.startswith("cuda"):
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"Device: {DEVICE} ({name}, {mem:.0f}GB VRAM)")
    else:
        print(f"Device: {DEVICE}")


def resolve_paths(mode):
    """Resuelve rutas de modelo y output segun el modo."""
    if mode == "base":
        return BASE_MODEL, "output/faiss_base/"
    elif mode.startswith("ft_L"):
        n = mode.split("L")[1]
        return f"output/ft_model_L{n}/", f"output/faiss_{mode}/"
    else:
        print(f"Modo no valido: {mode}")
        print("Uso: python faiss_index.py <base|ft_L1|ft_L2|...>")
        sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print("Uso: python faiss_index.py <base|ft_L1|ft_L2|...>")
        sys.exit(1)

    mode = sys.argv[1]
    model_path, output_dir = resolve_paths(mode)

    print_device_info()
    print(f"Modo: {mode}")
    os.makedirs(output_dir, exist_ok=True)

    # Cargar chunks del corpus RAG
    print("Cargando chunks desde output/chunks.pkl")
    with open("output/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print(f"  {len(chunks):,} chunks cargados")

    # Cargar modelo de embeddings
    print(f"Cargando modelo: {model_path}")
    model = SentenceTransformer(model_path, device=DEVICE)

    # Preparar textos con prefijo "passage: " (requerido por e5)
    texts = [f"passage: {chunk['text']}" for chunk in chunks]

    # Generar embeddings
    print(f"Generando embeddings (batch_size={BATCH_SIZE})...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    # Construir indice FAISS
    print(f"Construyendo indice FAISS (IndexFlatIP, dim={EMBEDDING_DIM})")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    print(f"  Vectores en el indice: {index.ntotal:,}")

    # Guardar indice
    faiss.write_index(index, os.path.join(output_dir, "index.faiss"))

    # Guardar mapping posicion -> chunk_id
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    with open(os.path.join(output_dir, "chunk_ids.pkl"), "wb") as f:
        pickle.dump(chunk_ids, f)

    print(f"Guardado en {output_dir}")


if __name__ == "__main__":
    main()
