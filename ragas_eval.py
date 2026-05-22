"""
ragas_eval.py - Genera testset y evalua pipeline RAG.

Subcomandos:
    python ragas_eval.py testset --size 50
    python ragas_eval.py evaluate base
    python ragas_eval.py evaluate ft_L2

Metricas:
    - LLM-Judge (RAGAS): Faithfulness, Answer Relevancy, Context Precision, Context Recall
    - IR: NDCG@5, Recall@5, Precision@5
    - Latency: retrieval, generation, total
"""

import argparse
import os
import pickle
import json
import math
import time
import random
from datetime import datetime
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
from tqdm import tqdm

load_dotenv()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BASE_MODEL = "intfloat/multilingual-e5-small"
LLM_MODEL = "gpt-4o-mini-2024-07-18"
TOP_K = 5

SYSTEM_PROMPT = (
    "Eres un asistente que responde preguntas basandose unicamente en el contexto proporcionado. "
    "Si la respuesta no se encuentra en el contexto, indica que no tienes suficiente informacion."
)


# --- Metricas IR ---

def is_relevant(retrieved_text, reference_contexts, threshold=0.3):
    """
    Un chunk es relevante si tiene overlap de palabras >= threshold
    con alguno de los contextos de referencia.
    """
    # Intento rapido: substring match
    for ref in reference_contexts:
        if ref in retrieved_text or retrieved_text in ref:
            return True

    # Fallback: word overlap
    retrieved_words = set(retrieved_text.lower().split())
    for ref in reference_contexts:
        ref_words = set(ref.lower().split())
        if not ref_words:
            continue
        overlap = len(retrieved_words & ref_words) / len(ref_words)
        if overlap >= threshold:
            return True
    return False


def ndcg_at_k(relevances, k=5):
    """Normalized Discounted Cumulative Gain."""
    rels = relevances[:k]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def compute_ir_metrics(relevances, n_relevant_total, k=5):
    """Calcula NDCG@k, Recall@k y Precision@k."""
    rels = relevances[:k]
    n_relevant_retrieved = sum(rels)
    return {
        "ndcg_at_5": round(ndcg_at_k(rels, k), 4),
        "recall_at_5": round(n_relevant_retrieved / max(n_relevant_total, 1), 4),
        "precision_at_5": round(n_relevant_retrieved / k, 4),
    }


# --- Subcomando: testset ---

def cmd_testset(args):
    """Genera testset sintetico desde el corpus RAG usando RAGAS."""
    os.makedirs("output", exist_ok=True)

    # Cargar chunks RAG
    print("Cargando chunks RAG desde output/chunks.pkl")
    with open("output/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print(f"  {len(chunks):,} chunks")

    # Samplear subconjunto para RAGAS (construye grafo de conocimiento con 1 llamada LLM por doc)
    if args.max_docs and len(chunks) > args.max_docs:
        random.seed(42)
        chunks = random.sample(chunks, args.max_docs)
        print(f"  Sampleados {args.max_docs:,} chunks para generacion de testset")

    # Crear documentos LangChain
    documents = [
        Document(
            page_content=c["text"],
            metadata={"chunk_id": c["chunk_id"], "source": c["source_file"]},
        )
        for c in chunks
    ]

    # Configurar generador RAGAS
    print(f"Configurando RAGAS con {LLM_MODEL}...")
    llm = ChatOpenAI(model=LLM_MODEL)
    embeddings = HuggingFaceEmbeddings(
        model_name=BASE_MODEL,
        model_kwargs={"device": DEVICE},
    )
    generator = TestsetGenerator.from_langchain(llm, llm, embeddings)

    distributions = {
        simple: args.simple,
        reasoning: args.reasoning,
        multi_context: args.multi_context,
    }

    print(f"Generando testset (size={args.size})")
    print(f"  Distribucion: simple={args.simple:.0%}, reasoning={args.reasoning:.0%}, multi_context={args.multi_context:.0%}")

    testset = generator.generate_with_langchain_docs(
        documents,
        test_size=args.size,
        distributions=distributions,
    )

    df = testset.to_pandas()

    # Convertir a formato JSON
    testset_data = []
    for _, row in df.iterrows():
        testset_data.append({
            "question": row["question"],
            "ground_truth": row.get("ground_truth", ""),
            "reference_contexts": row.get("contexts", []),
            "question_type": row.get("evolution_type", "unknown"),
        })

    with open("output/testset.json", "w", encoding="utf-8") as f:
        json.dump(testset_data, f, ensure_ascii=False, indent=2)

    print(f"Guardado en output/testset.json ({len(testset_data)} preguntas)")


# --- Subcomando: evaluate ---

def cmd_evaluate(args):
    """Evalua un pipeline RAG (base o ft_L{n})."""
    mode = args.mode

    # Resolver rutas segun modo
    if mode == "base":
        model_path = BASE_MODEL
        index_dir = "output/faiss_base/"
        output_path = "output/eval_base.json"
    else:
        n = mode.split("L")[1]
        model_path = f"output/ft_model_L{n}/"
        index_dir = f"output/faiss_{mode}/"
        output_path = f"output/eval_{mode}.json"

    if DEVICE.startswith("cuda"):
        print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print(f"Evaluando pipeline: {mode}")

    # Cargar testset
    with open("output/testset.json", "r", encoding="utf-8") as f:
        testset = json.load(f)
    print(f"Cargando testset: {len(testset)} preguntas")

    # Cargar chunks, indice FAISS y modelo de embeddings
    with open("output/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    chunks_by_id = {c["chunk_id"]: c for c in chunks}

    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "chunk_ids.pkl"), "rb") as f:
        chunk_ids = pickle.load(f)

    print(f"Cargando modelo: {model_path}")
    embed_model = SentenceTransformer(model_path, device=DEVICE)

    client = OpenAI()

    # --- Ejecutar retrieval + generacion por cada pregunta ---
    raw_results = []
    print("Ejecutando retrieval + generacion...")

    for entry in tqdm(testset):
        question = entry["question"]
        ref_contexts = entry.get("reference_contexts", [])

        # Retrieval
        t0 = time.perf_counter()
        q_emb = embed_model.encode(
            [f"query: {question}"],
            normalize_embeddings=True,
        )
        q_emb = np.array(q_emb, dtype=np.float32)
        scores, indices_arr = index.search(q_emb, TOP_K)
        t_retrieval = (time.perf_counter() - t0) * 1000

        # Evaluar relevancia de chunks recuperados
        retrieved = []
        relevances = []
        retrieved_texts = []

        for rank, (score, idx) in enumerate(zip(scores[0], indices_arr[0])):
            cid = chunk_ids[idx]
            chunk = chunks_by_id[cid]
            rel = is_relevant(chunk["text"], ref_contexts)
            relevances.append(1 if rel else 0)
            retrieved_texts.append(chunk["text"])
            retrieved.append({
                "chunk_id": cid,
                "source_file": chunk["source_file"],
                "score": float(score),
                "rank": rank + 1,
                "is_relevant": rel,
                "text_preview": chunk["text"][:150],
            })

        # Generacion con LLM
        context_text = "\n---\n".join(retrieved_texts)
        user_msg = f"Contexto:\n---\n{context_text}\n---\n\nPregunta: {question}\n\nRespuesta:"

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )
        t_generation = (time.perf_counter() - t0) * 1000
        answer = response.choices[0].message.content

        # Metricas IR
        ir_scores = compute_ir_metrics(relevances, max(len(ref_contexts), 1))

        raw_results.append({
            "question": question,
            "question_type": entry.get("question_type", "unknown"),
            "ground_truth": entry.get("ground_truth", ""),
            "answer": answer,
            "retrieved_chunks": retrieved,
            "reference_contexts": ref_contexts,
            "scores": ir_scores,
            "latency": {
                "retrieval_ms": round(t_retrieval, 1),
                "generation_ms": round(t_generation, 1),
                "total_ms": round(t_retrieval + t_generation, 1),
            },
        })

    # --- Metricas RAGAS (LLM-judge) ---
    print("Calculando metricas RAGAS (LLM-judge)...")

    ragas_dataset = Dataset.from_dict({
        "question": [r["question"] for r in raw_results],
        "answer": [r["answer"] for r in raw_results],
        "contexts": [
            [chunks_by_id[c["chunk_id"]]["text"] for c in r["retrieved_chunks"]]
            for r in raw_results
        ],
        "ground_truth": [r["ground_truth"] for r in raw_results],
    })

    llm = ChatOpenAI(model=LLM_MODEL)
    embeddings = HuggingFaceEmbeddings(
        model_name=BASE_MODEL,
        model_kwargs={"device": DEVICE},
    )

    ragas_result = ragas_evaluate(
        dataset=ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    # Agregar scores RAGAS a cada resultado raw
    ragas_df = ragas_result.to_pandas()
    for i, row in ragas_df.iterrows():
        raw_results[i]["scores"]["faithfulness"] = _safe_float(row.get("faithfulness"))
        raw_results[i]["scores"]["answer_relevancy"] = _safe_float(row.get("answer_relevancy"))
        raw_results[i]["scores"]["context_precision"] = _safe_float(row.get("context_precision"))
        raw_results[i]["scores"]["context_recall"] = _safe_float(row.get("context_recall"))

    # --- Calcular summary (promedios globales) ---
    all_metric_keys = [
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
        "ndcg_at_5", "recall_at_5", "precision_at_5",
    ]

    summary = _compute_averages(raw_results, all_metric_keys)

    # --- Calcular by_question_type ---
    question_types = sorted(set(r["question_type"] for r in raw_results))
    by_type = {}
    for qt in question_types:
        qt_results = [r for r in raw_results if r["question_type"] == qt]
        qt_summary = {"count": len(qt_results)}
        qt_summary.update(_compute_averages(qt_results, all_metric_keys))
        by_type[qt] = qt_summary

    # --- Compilar resultado final ---
    eval_result = {
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "testset_size": len(testset),
        "summary": summary,
        "by_question_type": by_type,
        "raw": raw_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)

    # --- Imprimir resumen ---
    _print_summary(summary, by_type, output_path)


# --- Helpers ---

def _safe_float(val):
    """Convierte a float redondeado, o None si no es numerico."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return round(float(val), 4)


def _compute_averages(results, metric_keys):
    """Calcula promedios de metricas sobre una lista de resultados."""
    avgs = {}
    for m in metric_keys:
        values = [r["scores"].get(m) for r in results if r["scores"].get(m) is not None]
        avgs[m] = round(sum(values) / len(values), 3) if values else None

    avgs["retrieval_latency_avg_ms"] = round(
        sum(r["latency"]["retrieval_ms"] for r in results) / len(results), 1
    )
    avgs["generation_latency_avg_ms"] = round(
        sum(r["latency"]["generation_ms"] for r in results) / len(results), 1
    )
    avgs["total_latency_avg_ms"] = round(
        sum(r["latency"]["total_ms"] for r in results) / len(results), 1
    )
    return avgs


def _print_summary(summary, by_type, output_path):
    """Imprime resumen en consola."""
    print("\nResumen general:")
    print("-" * 40)
    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
        "ndcg_at_5": "NDCG@5",
        "recall_at_5": "Recall@5",
        "precision_at_5": "Precision@5",
    }
    for key, label in metric_labels.items():
        val = summary.get(key)
        val_str = f"{val:.3f}" if val is not None else "N/A"
        print(f"  {label:<20s}  {val_str}")
    print(f"  {'Retrieval Latency':<20s}  {summary['retrieval_latency_avg_ms']:.0f}ms")
    print(f"  {'Generation Latency':<20s}  {summary['generation_latency_avg_ms']:.0f}ms")

    print("\nPor tipo de pregunta:")
    print("-" * 40)
    for qt, qs in by_type.items():
        ndcg = qs.get("ndcg_at_5", "N/A")
        recall = qs.get("recall_at_5", "N/A")
        faith = qs.get("faithfulness", "N/A")
        print(f"  {qt} (n={qs['count']}): NDCG@5={ndcg}, Recall@5={recall}, Faithfulness={faith}")

    print(f"\nGuardado en {output_path}")


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="RAGAS testset + evaluation")
    subparsers = parser.add_subparsers(dest="command")

    # Subcomando: testset
    ts_parser = subparsers.add_parser("testset", help="Generar testset sintetico")
    ts_parser.add_argument("--size", type=int, default=50, help="Cantidad de preguntas")
    ts_parser.add_argument("--max-docs", type=int, default=500, help="Max chunks para RAGAS (0=todos)")
    ts_parser.add_argument("--simple", type=float, default=0.4, help="Proporcion simple")
    ts_parser.add_argument("--reasoning", type=float, default=0.3, help="Proporcion reasoning")
    ts_parser.add_argument("--multi_context", type=float, default=0.3, help="Proporcion multi_context")

    # Subcomando: evaluate
    ev_parser = subparsers.add_parser("evaluate", help="Evaluar pipeline RAG")
    ev_parser.add_argument("mode", help="base o ft_L{n} (ej: ft_L1, ft_L2)")

    args = parser.parse_args()

    if args.command == "testset":
        cmd_testset(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
