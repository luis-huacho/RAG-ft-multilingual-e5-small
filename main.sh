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
if [ -f output/chunks.pkl ]; then
    echo "  [skip] output/chunks.pkl ya existe"
else
    python chunking.py rag
fi
if [ -f output/faiss_base/index.faiss ]; then
    echo "  [skip] output/faiss_base/index.faiss ya existe"
else
    python faiss_index.py base
fi
echo ""

# --- FASE 2: Evaluacion base ---
echo "FASE 2: Evaluacion base"
if [ -f output/testset.json ]; then
    echo "  [skip] output/testset.json ya existe"
else
    python ragas_eval.py testset --size 50
fi
if [ -f output/eval_base.json ]; then
    echo "  [skip] output/eval_base.json ya existe"
else
    python ragas_eval.py evaluate base
fi
echo ""

# --- FASE 3: Preparacion fine-tuning ---
echo "FASE 3: Preparacion fine-tuning"
if [ -f output/chunks_ft.pkl ]; then
    echo "  [skip] output/chunks_ft.pkl ya existe"
else
    python chunking.py ft
fi
if [ -f output/ft_train_data.json ]; then
    echo "  [skip] output/ft_train_data.json ya existe"
else
    python ft_data.py
fi
echo ""

# --- FASE 4a: Fine-tuning (paralelo) ---
echo "FASE 4a: Fine-tuning (paralelo)"
PIDS_FT=()
for L in 1 2 3 4; do
    if [ -d "output/ft_model_L${L}" ]; then
        echo "  [skip] output/ft_model_L${L} ya existe"
    else
        echo "  Lanzando ft_process.py --layers $L (log: output/ft_L${L}.log)"
        python ft_process.py --layers $L > "output/ft_L${L}.log" 2>&1 &
        PIDS_FT+=("$!:L${L}")
    fi
done
FAIL=0
for entry in "${PIDS_FT[@]}"; do
    PID="${entry%%:*}"
    LABEL="${entry##*:}"
    if ! wait "$PID"; then
        echo "  [ERROR] ft_process $LABEL fallo — ver output/ft_${LABEL}.log"
        FAIL=1
    else
        echo "  [OK] ft_process $LABEL completado"
    fi
done
if [ $FAIL -ne 0 ]; then exit 1; fi
echo ""

# --- FASE 4b: Indexacion FAISS (paralelo) ---
echo "FASE 4b: Indexacion FAISS (paralelo)"
PIDS_IX=()
for L in 1 2 3 4; do
    if [ -f "output/faiss_ft_L${L}/index.faiss" ]; then
        echo "  [skip] output/faiss_ft_L${L}/index.faiss ya existe"
    else
        echo "  Lanzando faiss_index.py ft_L$L (log: output/faiss_ft_L${L}.log)"
        python faiss_index.py ft_L$L > "output/faiss_ft_L${L}.log" 2>&1 &
        PIDS_IX+=("$!:L${L}")
    fi
done
FAIL=0
for entry in "${PIDS_IX[@]}"; do
    PID="${entry%%:*}"
    LABEL="${entry##*:}"
    if ! wait "$PID"; then
        echo "  [ERROR] faiss_index $LABEL fallo — ver output/faiss_ft_${LABEL}.log"
        FAIL=1
    else
        echo "  [OK] faiss_index $LABEL completado"
    fi
done
if [ $FAIL -ne 0 ]; then exit 1; fi
echo ""

# --- FASE 4c: Evaluacion FT (secuencial — comparte cuota OpenAI) ---
echo "FASE 4c: Evaluacion FT"
for L in 1 2 3 4; do
    if [ -f "output/eval_ft_L${L}.json" ]; then
        echo "  [skip] output/eval_ft_L${L}.json ya existe"
    else
        echo "  Evaluando ft_L$L"
        python ragas_eval.py evaluate ft_L$L
    fi
done
echo ""

# --- FASE 5: Reporte (siempre regenerar) ---
echo "FASE 5: Generando reporte"
python result.py
echo ""

echo "════════════════════════════════════════════"
echo " Pipeline completado"
echo " Reporte: output/result.md"
echo "════════════════════════════════════════════"