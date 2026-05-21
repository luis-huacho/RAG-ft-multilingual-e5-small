"""
ft_data.py - Genera pares de entrenamiento (query, positive) para fine-tuning.

Flujo:
    1. Genera un testset RAGAS desde el corpus FT (NO el corpus RAG)
    2. Extrae pares (question, reference_context) del testset
    3. Guarda como datos de entrenamiento

Uso:
    python ft_data.py
    python ft_data.py --size 500 --simple 0.5 --reasoning 0.3 --multi_context 0.2
"""

import argparse
import os
import pickle
import json
import torch
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

load_dotenv()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LLM_MODEL = "gpt-4o-mini-2024-07-18"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


def find_best_chunk(context_text, chunks, threshold=0.3):
    """
    Busca el chunk con mayor overlap de palabras con el contexto de referencia.
    Primero intenta substring match (mas preciso), luego word overlap.
    """
    # Intento 1: substring match
    for chunk in chunks:
        if context_text in chunk["text"] or chunk["text"] in context_text:
            return chunk["text"]

    # Intento 2: word overlap
    ctx_words = set(context_text.lower().split())
    if not ctx_words:
        return None

    best_match = None
    best_score = 0

    for chunk in chunks:
        chunk_words = set(chunk["text"].lower().split())
        overlap = len(ctx_words & chunk_words) / len(ctx_words)
        if overlap > best_score:
            best_score = overlap
            best_match = chunk["text"]

    if best_score >= threshold:
        return best_match
    return None


def main():
    parser = argparse.ArgumentParser(description="Generar datos de entrenamiento para FT")
    parser.add_argument("--size", type=int, default=500, help="Preguntas a generar con RAGAS")
    parser.add_argument("--simple", type=float, default=0.5, help="Proporcion simple")
    parser.add_argument("--reasoning", type=float, default=0.3, help="Proporcion reasoning")
    parser.add_argument("--multi_context", type=float, default=0.2, help="Proporcion multi_context")
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    # Cargar chunks del corpus FT
    print("Cargando chunks FT desde output/chunks_ft.pkl")
    with open("output/chunks_ft.pkl", "rb") as f:
        chunks = pickle.load(f)
    print(f"  {len(chunks):,} chunks cargados")

    # Crear documentos LangChain desde los chunks FT
    documents = [
        Document(
            page_content=c["text"],
            metadata={"chunk_id": c["chunk_id"], "source": c["source_file"]},
        )
        for c in chunks
    ]

    # Configurar RAGAS
    print(f"Configurando RAGAS con {LLM_MODEL}...")
    llm = ChatOpenAI(model=LLM_MODEL)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
    )

    generator = TestsetGenerator.from_langchain(llm, llm, embeddings)

    distributions = {
        simple: args.simple,
        reasoning: args.reasoning,
        multi_context: args.multi_context,
    }

    print(f"Generando testset RAGAS desde corpus FT (size={args.size})...")
    print(f"  Distribucion: simple={args.simple:.0%}, reasoning={args.reasoning:.0%}, multi_context={args.multi_context:.0%}")

    testset = generator.generate_with_langchain_docs(
        documents,
        test_size=args.size,
        distributions=distributions,
    )

    df = testset.to_pandas()

    # Guardar testset intermedio (auditoria)
    ft_testset = []
    for _, row in df.iterrows():
        ft_testset.append({
            "question": row["question"],
            "ground_truth": row.get("ground_truth", ""),
            "reference_contexts": row.get("contexts", []),
            "question_type": row.get("evolution_type", "unknown"),
        })

    with open("output/ft_testset.json", "w", encoding="utf-8") as f:
        json.dump(ft_testset, f, ensure_ascii=False, indent=2)
    print(f"Testset intermedio guardado en output/ft_testset.json")

    # Extraer pares de entrenamiento
    print("Extrayendo pares de entrenamiento...")
    train_pairs = []
    skipped = 0

    for entry in ft_testset:
        question = entry["question"]
        contexts = entry["reference_contexts"]

        if not contexts:
            skipped += 1
            continue

        for ctx in contexts:
            match = find_best_chunk(ctx, chunks)
            if match:
                train_pairs.append({
                    "query": question,
                    "positive": match,
                })
            else:
                skipped += 1

    # Guardar datos de entrenamiento
    with open("output/ft_train_data.json", "w", encoding="utf-8") as f:
        json.dump(train_pairs, f, ensure_ascii=False, indent=2)

    print(f"  Pares generados: {len(train_pairs):,}")
    print(f"  Descartados (sin match): {skipped}")
    print(f"Guardado en output/ft_train_data.json")


if __name__ == "__main__":
    main()
