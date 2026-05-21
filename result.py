"""
result.py - Genera output/result.md con el reporte comparativo final.

Lee todos los eval_*.json y logs de entrenamiento para construir
tablas markdown comparando base vs cada variante de fine-tuning.

Uso:
    python result.py
"""

import os
import json
import glob
from datetime import datetime


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_evals():
    """Carga todos los archivos eval_*.json disponibles."""
    evals = {}
    for path in sorted(glob.glob("output/eval_*.json")):
        data = load_json(path)
        evals[data["mode"]] = data
    return evals


def load_training_logs():
    """Carga el log mas reciente de cada variante L{n}."""
    logs = {}
    for path in sorted(glob.glob("output/logs/ft_L*.json")):
        data = load_json(path)
        # Extraer variante (ej: "L2" de "ft_L2_20260521_1430")
        run_id = data["run_id"]
        variant = run_id.split("_")[1]  # "L2"
        # Quedarse con el mas reciente (glob sorted por nombre = por timestamp)
        logs[variant] = data
    return logs


def load_chunks_stats():
    """Carga stats de los chunks desde los archivos de auditoria."""
    stats = {}
    for name, path in [("rag", "output/chunks_audit.json"), ("ft", "output/chunks_ft_audit.json")]:
        if os.path.exists(path):
            chunks = load_json(path)
            total_tokens = sum(c["token_count"] for c in chunks)
            stats[name] = {
                "count": len(chunks),
                "avg_tokens": round(total_tokens / len(chunks)) if chunks else 0,
            }
    return stats


def build_report(evals, logs, chunks_stats):
    """Construye el contenido del reporte markdown."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("# Resultados - RAG con Fine-Tuning de multilingual-e5-small")
    lines.append("")
    lines.append(f"Generado: {now}")
    lines.append("")

    # --- 1. Dataset ---
    lines.append("## 1. Dataset")
    lines.append("")
    lines.append("| Corpus | Archivo | Chunks | Tokens promedio/chunk |")
    lines.append("|--------|---------|--------|-----------------------|")
    if "rag" in chunks_stats:
        s = chunks_stats["rag"]
        lines.append(f"| RAG | cowese_10.txt | {s['count']:,} | {s['avg_tokens']} |")
    if "ft" in chunks_stats:
        s = chunks_stats["ft"]
        lines.append(f"| FT | cowese_ft_10.txt | {s['count']:,} | {s['avg_tokens']} |")
    lines.append("")

    # --- 2. Configuracion de Fine-Tuning ---
    if logs:
        lines.append("## 2. Configuracion de Fine-Tuning")
        lines.append("")
        lines.append("| Variante | Capas descongeladas | Params entrenables | % del total | Epochs | LR | Duracion |")
        lines.append("|----------|--------------------|--------------------|-------------|--------|----|----------|")
        for variant in sorted(logs.keys()):
            log = logs[variant]
            cfg = log["config"]
            layers_str = ", ".join(map(str, cfg["unfrozen_layers"]))
            lines.append(
                f"| {variant} "
                f"| {cfg['layers_unfrozen']} (layers {layers_str}) "
                f"| {log['trainable_params']:,} "
                f"| {log['trainable_pct']} "
                f"| {cfg['epochs']} "
                f"| {cfg['learning_rate']} "
                f"| {log['total_duration_s']}s |"
            )
        lines.append("")

    # --- 3. Metricas de Evaluacion - Resumen General ---
    if evals:
        modes = sorted(evals.keys(), key=_mode_sort_key)
        metric_keys = [
            ("faithfulness", "Faithfulness"),
            ("answer_relevancy", "Answer Relevancy"),
            ("context_precision", "Context Precision"),
            ("context_recall", "Context Recall"),
            ("ndcg_at_5", "NDCG@5"),
            ("recall_at_5", "Recall@5"),
            ("precision_at_5", "Precision@5"),
            ("retrieval_latency_avg_ms", "Retrieval Latency (ms)"),
            ("generation_latency_avg_ms", "Generation Latency (ms)"),
        ]

        lines.append("## 3. Metricas de Evaluacion - Resumen General")
        lines.append("")
        header = "| Metrica | " + " | ".join(modes) + " |"
        sep = "|---------|" + "|".join(["-------"] * len(modes)) + "|"
        lines.append(header)
        lines.append(sep)

        for key, label in metric_keys:
            row = f"| {label} |"
            for m in modes:
                val = evals[m]["summary"].get(key)
                if val is None:
                    row += " N/A |"
                elif "latency" in key:
                    row += f" {val:.0f} |"
                else:
                    row += f" {val:.3f} |"
            lines.append(row)
        lines.append("")

        # --- 4. Metricas por Tipo de Pregunta ---
        lines.append("## 4. Metricas por Tipo de Pregunta")
        lines.append("")

        # Recopilar todos los tipos de pregunta
        all_types = set()
        for m in modes:
            all_types.update(evals[m].get("by_question_type", {}).keys())

        for qt in sorted(all_types):
            lines.append(f"### {qt.capitalize()}")
            lines.append("")
            header = "| Metrica | " + " | ".join(modes) + " |"
            lines.append(header)
            lines.append(sep)

            for key, label in metric_keys:
                row = f"| {label} |"
                for m in modes:
                    qt_data = evals[m].get("by_question_type", {}).get(qt, {})
                    val = qt_data.get(key)
                    if val is None:
                        row += " N/A |"
                    elif "latency" in key:
                        row += f" {val:.0f} |"
                    else:
                        row += f" {val:.3f} |"
                lines.append(row)
            lines.append("")

    # --- 5. Informacion del Entorno ---
    lines.append("## 5. Informacion del Entorno")
    lines.append("")

    # Tomar device info del primer log disponible
    if logs:
        first_log = next(iter(logs.values()))
        dev = first_log.get("device", {})
        lines.append(f"- **Device**: {dev.get('type', 'N/A')} ({dev.get('name', 'N/A')})")
    lines.append(f"- **Modelo base**: intfloat/multilingual-e5-small (384 dim, 12 layers)")
    lines.append(f"- **LLM**: gpt-4o-mini-2024-07-18")

    if evals:
        first_eval = next(iter(evals.values()))
        lines.append(f"- **Testset**: {first_eval['testset_size']} preguntas")

    if logs:
        first_log = next(iter(logs.values()))
        lines.append(f"- **Pares de entrenamiento FT**: {first_log['config']['train_samples']:,}")

    lines.append("")
    return "\n".join(lines)


def _mode_sort_key(mode):
    """Ordena: base primero, luego L1, L2, L3, L4."""
    if mode == "base":
        return (0, 0)
    try:
        n = int(mode.split("L")[1])
        return (1, n)
    except (IndexError, ValueError):
        return (2, 0)


def main():
    print("Leyendo evaluaciones...")
    evals = load_evals()
    for mode in sorted(evals.keys(), key=_mode_sort_key):
        print(f"  eval_{mode}.json")

    print("Leyendo logs de entrenamiento...")
    logs = load_training_logs()
    for variant in sorted(logs.keys()):
        print(f"  {logs[variant]['run_id']}.json")

    print("Leyendo stats de chunks...")
    chunks_stats = load_chunks_stats()

    print("Generando reporte...")
    report = build_report(evals, logs, chunks_stats)

    output_path = "output/result.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Guardado en {output_path}")


if __name__ == "__main__":
    main()
