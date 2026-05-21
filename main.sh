#!/bin/bash
set -e

# Limitar threads para PyTorch/FAISS — 128 cores genera overhead excesivo
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

echo "════════════════════════════════════════════"
echo " RAG-FT-e5-small — Pipeline completo"
echo "════════════════════════════════════════════"
echo ""

# --- FASE 1: Ingest corpus RAG ---
echo "FASE 1: Ingest corpus RAG"
python chunking.py rag
python faiss_index.py base
echo ""

# --- FASE 2: Evaluacion base ---
echo "FASE 2: Evaluacion base"
python ragas_eval.py testset --size 50
python ragas_eval.py evaluate base
echo ""

# --- FASE 3: Preparacion fine-tuning ---
echo "FASE 3: Preparacion fine-tuning"
python chunking.py ft
python ft_data.py
echo ""

# --- FASE 4: Fine-tuning + evaluacion (4 variantes) ---
echo "FASE 4: Fine-tuning + evaluacion"
for L in 1 2 3 4; do
    echo ""
    echo "  -- Variante L${L} --"
    python ft_process.py --layers $L
    python faiss_index.py ft_L$L
    python ragas_eval.py evaluate ft_L$L
done
echo ""

# --- FASE 5: Reporte ---
echo "FASE 5: Generando reporte"
python result.py
echo ""

echo "════════════════════════════════════════════"
echo " Pipeline completado"
echo " Reporte: output/result.md"
echo "════════════════════════════════════════════"
