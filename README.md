# RAG con Fine-Tuning de multilingual-e5-small

Ejercicio academico que demuestra como el fine-tuning de un modelo de embeddings de baja dimensionalidad (`multilingual-e5-small`, 384 dim) mejora la calidad de un pipeline RAG.

Se compara un RAG vanilla contra 4 variantes fine-tuned (1, 2, 3 y 4 capas descongeladas) usando el corpus CoWeSe (Corpus Web en Espanol).

## Requisitos

- Python 3.12+
- GPU con CUDA
- API key de OpenAI

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Agregar OPENAI_API_KEY
```

Colocar los datos en `data/`:
- `cowese_10.txt` — corpus para RAG (10% de [CoWeSe](https://zenodo.org/records/5513237))
- `cowese_ft_10.txt` — corpus separado para fine-tuning

## Ejecucion

```bash
./main.sh
```

Ejecuta todo el pipeline: chunking, indexacion, evaluacion base, fine-tuning (L1-L4), evaluacion fine-tuned y generacion del reporte final en `output/result.md`.

## Estructura

```
chunking.py      Chunking semantico (~400 tokens, overlap 200)
faiss_index.py   Indexacion FAISS (IndexFlatIP, cosine similarity)
ft_data.py       Genera pares de entrenamiento sinteticos via RAGAS
ft_process.py    Fine-tuning con sentence-transformers (MNR loss)
ragas_eval.py    Testset sintetico + evaluacion (RAGAS + IR metrics)
query.py         Consulta interactiva con generacion opcional
result.py        Genera reporte comparativo en markdown
```

## Metricas

- **LLM-Judge**: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- **IR**: NDCG@5, Recall@5, Precision@5
- **Latency**: retrieval + generacion
