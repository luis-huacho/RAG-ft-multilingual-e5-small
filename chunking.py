"""
chunking.py - Divide un corpus TXT en chunks con limites semanticos.

Uso:
    python chunking.py rag    # Procesa cowese_10.txt para RAG
    python chunking.py ft     # Procesa cowese_ft_10.txt para Fine-Tuning

Estrategia:
    - Maximo 400 tokens por chunk (margen bajo el limite de 512 de e5-small)
    - Overlap de 200 tokens entre chunks consecutivos
    - Corte en limites semanticos: parrafo > linea > punto
"""

import sys
import os
import pickle
import json
from transformers import AutoTokenizer
from tqdm import tqdm

TOKENIZER_NAME = "intfloat/multilingual-e5-small"
MAX_TOKENS = 400
OVERLAP = 200

MODES = {
    "rag": {
        "input": "data/cowese_10.txt",
        "output_pkl": "output/chunks.pkl",
        "output_json": "output/chunks_audit.json",
        "prefix": "rag",
    },
    "ft": {
        "input": "data/cowese_ft_10.txt",
        "output_pkl": "output/chunks_ft.pkl",
        "output_json": "output/chunks_ft_audit.json",
        "prefix": "ft",
    },
}


def find_boundary(text):
    """
    Busca el mejor punto de corte retrocediendo desde el final del texto.
    Prioridad: \\n\\n (parrafo) > \\n (linea) > '. ' (oracion)
    Busca en los ultimos ~200 caracteres (~50 tokens).
    Si no encuentra nada, retorna el largo completo del texto.
    """
    search_start = max(0, len(text) - 200)

    # Prioridad 1: doble salto de linea (fin de parrafo)
    pos = text.rfind("\n\n", search_start)
    if pos != -1:
        return pos + 2

    # Prioridad 2: salto de linea simple
    pos = text.rfind("\n", search_start)
    if pos != -1:
        return pos + 1

    # Prioridad 3: punto seguido de espacio
    pos = text.rfind(". ", search_start)
    if pos != -1:
        return pos + 2

    return len(text)


def generate_chunks(text, tokenizer):
    """Genera chunks del texto respetando limites semanticos."""
    print("Tokenizando texto completo...")
    all_tokens = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(all_tokens)
    print(f"  Total tokens: {total_tokens:,}")

    chunks = []
    start = 0

    pbar = tqdm(total=total_tokens, desc="Generando chunks", unit="tok")
    while start < total_tokens:
        end = min(start + MAX_TOKENS, total_tokens)
        window_tokens = all_tokens[start:end]
        window_text = tokenizer.decode(window_tokens, skip_special_tokens=True)

        if end >= total_tokens:
            # Ultimo chunk: tomar todo lo que queda
            chunk_text = window_text.strip()
            if chunk_text:
                chunks.append(chunk_text)
            pbar.update(end - start)
            break

        # Buscar el mejor punto de corte semantico
        cut_pos = find_boundary(window_text)
        chunk_text = window_text[:cut_pos].strip()

        if not chunk_text:
            chunk_text = window_text.strip()

        # Contar tokens reales del chunk para calcular avance
        chunk_token_count = len(tokenizer.encode(chunk_text, add_special_tokens=False))
        advance = max(chunk_token_count - OVERLAP, 1)

        chunks.append(chunk_text)
        pbar.update(advance)
        start += advance

    pbar.close()
    return chunks


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in MODES:
        print("Uso: python chunking.py <rag|ft>")
        sys.exit(1)

    mode = sys.argv[1]
    config = MODES[mode]
    prefix = config["prefix"]

    print(f"Modo: {mode}")
    os.makedirs("output", exist_ok=True)

    # Leer archivo
    input_path = config["input"]
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    print(f"Leyendo: {input_path} ({file_size_mb:.0f}MB)")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Cargar tokenizer
    print(f"Cargando tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Generar chunks
    raw_chunks = generate_chunks(text, tokenizer)

    # Construir metadata
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        token_count = len(tokenizer.encode(chunk_text, add_special_tokens=False))
        chunks.append({
            "chunk_id": f"{prefix}_chunk_{i:06d}",
            "source_file": os.path.basename(config["input"]),
            "text": chunk_text,
            "token_count": token_count,
            "char_count": len(chunk_text),
        })

    # Guardar pickle
    with open(config["output_pkl"], "wb") as f:
        pickle.dump(chunks, f)

    # Guardar JSON para auditoria
    with open(config["output_json"], "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    avg_tokens = sum(c["token_count"] for c in chunks) / len(chunks) if chunks else 0
    print(f"\nTotal: {len(chunks):,} chunks generados")
    print(f"Tokens promedio por chunk: {avg_tokens:.0f}")
    print(f"Guardado en {config['output_pkl']} y {config['output_json']}")


if __name__ == "__main__":
    main()
