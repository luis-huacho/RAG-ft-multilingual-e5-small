# Resultados - RAG con Fine-Tuning de multilingual-e5-small

Generado: 2026-05-22 20:48:55

## 1. Dataset

| Corpus | Archivo | Chunks | Tokens promedio/chunk |
|--------|---------|--------|-----------------------|
| RAG | cowese_10.txt | 479,355 | 385 |
| FT | cowese_ft_10.txt | 485,321 | 385 |

## 2. Configuracion de Fine-Tuning

| Variante | Capas descongeladas | Params entrenables | % del total | Epochs | LR | Duracion |
|----------|--------------------|--------------------|-------------|--------|----|----------|
| L1 | 1 (layers 11) | 1,774,464 | 1.5% | 3 | 2e-05 | 349.0s |
| L2 | 2 (layers 10, 11) | 3,548,928 | 3.0% | 3 | 2e-05 | 309.4s |
| L3 | 3 (layers 9, 10, 11) | 5,323,392 | 4.5% | 3 | 2e-05 | 302.4s |
| L4 | 4 (layers 8, 9, 10, 11) | 7,097,856 | 6.0% | 3 | 2e-05 | 304.9s |

## 3. Metricas de Evaluacion - Resumen General

| Metrica | base | ft_L1 | ft_L2 | ft_L3 | ft_L4 |
|---------|-------|-------|-------|-------|-------|
| Faithfulness | 0.606 | 0.578 | 0.546 | 0.554 | 0.545 |
| Answer Relevancy | 0.554 | 0.550 | 0.514 | 0.534 | 0.518 |
| Context Precision | 0.718 | 0.697 | 0.696 | 0.701 | 0.681 |
| Context Recall | 0.767 | 0.755 | 0.765 | 0.722 | 0.732 |
| NDCG@5 | 0.562 | 0.555 | 0.544 | 0.529 | 0.525 |
| Recall@5 | 1.320 | 1.330 | 1.230 | 1.250 | 1.190 |
| Precision@5 | 0.292 | 0.296 | 0.272 | 0.276 | 0.264 |
| Retrieval Latency (ms) | 27 | 27 | 27 | 27 | 27 |
| Generation Latency (ms) | 2022 | 1562 | 1334 | 1753 | 1608 |

## 4. Metricas por Tipo de Pregunta

### Multi_context

| Metrica | base | ft_L1 | ft_L2 | ft_L3 | ft_L4 |
|---------|-------|-------|-------|-------|-------|
| Faithfulness | 0.502 | 0.483 | 0.392 | 0.407 | 0.376 |
| Answer Relevancy | 0.444 | 0.433 | 0.321 | 0.378 | 0.318 |
| Context Precision | 0.634 | 0.639 | 0.632 | 0.597 | 0.647 |
| Context Recall | 0.771 | 0.731 | 0.731 | 0.731 | 0.731 |
| NDCG@5 | 0.630 | 0.623 | 0.610 | 0.609 | 0.604 |
| Recall@5 | 1.400 | 1.433 | 1.367 | 1.433 | 1.367 |
| Precision@5 | 0.373 | 0.387 | 0.360 | 0.373 | 0.360 |
| Retrieval Latency (ms) | 27 | 27 | 28 | 27 | 27 |
| Generation Latency (ms) | 2161 | 1372 | 1199 | 1680 | 1333 |

### Reasoning

| Metrica | base | ft_L1 | ft_L2 | ft_L3 | ft_L4 |
|---------|-------|-------|-------|-------|-------|
| Faithfulness | 0.667 | 0.578 | 0.578 | 0.528 | 0.517 |
| Answer Relevancy | 0.643 | 0.576 | 0.574 | 0.509 | 0.518 |
| Context Precision | 0.585 | 0.601 | 0.597 | 0.625 | 0.619 |
| Context Recall | 0.900 | 0.900 | 0.900 | 0.889 | 0.867 |
| NDCG@5 | 0.581 | 0.563 | 0.532 | 0.518 | 0.545 |
| Recall@5 | 1.267 | 1.267 | 1.200 | 1.200 | 1.267 |
| Precision@5 | 0.253 | 0.253 | 0.240 | 0.240 | 0.253 |
| Retrieval Latency (ms) | 26 | 26 | 26 | 27 | 27 |
| Generation Latency (ms) | 1656 | 1610 | 1277 | 1456 | 1660 |

### Simple

| Metrica | base | ft_L1 | ft_L2 | ft_L3 | ft_L4 |
|---------|-------|-------|-------|-------|-------|
| Faithfulness | 0.637 | 0.650 | 0.637 | 0.685 | 0.694 |
| Answer Relevancy | 0.570 | 0.618 | 0.613 | 0.669 | 0.668 |
| Context Precision | 0.880 | 0.813 | 0.818 | 0.837 | 0.754 |
| Context Recall | 0.665 | 0.665 | 0.690 | 0.590 | 0.632 |
| NDCG@5 | 0.496 | 0.497 | 0.503 | 0.476 | 0.451 |
| Recall@5 | 1.300 | 1.300 | 1.150 | 1.150 | 1.000 |
| Precision@5 | 0.260 | 0.260 | 0.230 | 0.230 | 0.200 |
| Retrieval Latency (ms) | 27 | 27 | 28 | 27 | 27 |
| Generation Latency (ms) | 2193 | 1669 | 1477 | 2031 | 1775 |

## 5. Informacion del Entorno

- **Device**: cpu (cpu)
- **Modelo base**: intfloat/multilingual-e5-small (384 dim, 12 layers)
- **LLM**: gpt-4o-mini-2024-07-18
- **Testset**: 50 preguntas
- **Pares de entrenamiento FT**: 537
