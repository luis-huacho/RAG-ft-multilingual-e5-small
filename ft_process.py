"""
ft_process.py - Fine-tunea multilingual-e5-small con sentence-transformers.

Congela todas las capas y descongelar las ultimas N capas del transformer.
Usa MultipleNegativesRankingLoss (in-batch negatives).

Uso:
    python ft_process.py --layers 2
    python ft_process.py --layers 4 --epochs 5 --batch-size 32 --lr 1e-5
"""

import argparse
import os
import json
import time
from datetime import datetime
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BASE_MODEL = "intfloat/multilingual-e5-small"
TOTAL_LAYERS = 12


def get_device_info():
    """Detecta y muestra informacion del device."""
    if DEVICE.startswith("cuda"):
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"Device: {DEVICE} ({name}, {mem:.0f}GB VRAM)")
        return {"type": DEVICE, "name": name, "vram_gb": round(mem, 1)}
    else:
        print(f"Device: {DEVICE}")
        return {"type": "cpu", "name": "cpu", "vram_gb": 0}


def freeze_layers(model, layers_to_unfreeze):
    """
    Congela todas las capas del transformer y descongelar las ultimas N.
    e5-small tiene 12 capas (0-11). --layers 2 descongelar layers 10 y 11.
    Siempre descongelar la capa de pooling.
    """
    # Congelar todo
    for param in model.parameters():
        param.requires_grad = False

    # Descongelar las ultimas N capas del encoder
    encoder = model[0].auto_model.encoder.layer
    unfrozen_indices = []
    for i in range(TOTAL_LAYERS - layers_to_unfreeze, TOTAL_LAYERS):
        for param in encoder[i].parameters():
            param.requires_grad = True
        unfrozen_indices.append(i)

    # Siempre descongelar pooling
    if len(model) > 1:
        for param in model[1].parameters():
            param.requires_grad = True

    return unfrozen_indices


def count_params(model):
    """Cuenta parametros totales y entrenables."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    parser = argparse.ArgumentParser(description="Fine-tune e5-small")
    parser.add_argument("--layers", type=int, required=True, help="Capas a descongelar (1-12)")
    parser.add_argument("--epochs", type=int, default=3, help="Epocas de entrenamiento")
    parser.add_argument("--batch-size", type=int, default=16, help="Tamano del batch")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    suffix = f"L{args.layers}"
    output_dir = f"output/ft_model_{suffix}/"
    logs_dir = "output/logs/"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    device_info = get_device_info()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Cargar datos de entrenamiento
    print("Cargando datos de entrenamiento desde output/ft_train_data.json")
    with open("output/ft_train_data.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    print(f"  {len(train_data):,} pares")

    # Preparar InputExamples con prefijos e5
    examples = [
        InputExample(texts=[f"query: {d['query']}", f"passage: {d['positive']}"])
        for d in train_data
    ]
    dataloader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)

    # Cargar modelo base
    print(f"Cargando modelo base: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL, device=DEVICE)

    # Congelar/descongelar capas
    unfrozen_layers = freeze_layers(model, args.layers)
    total_params, trainable_params = count_params(model)
    trainable_pct = trainable_params / total_params * 100

    print(f"Capas descongeladas: {args.layers}/{TOTAL_LAYERS} (layers {', '.join(map(str, unfrozen_layers))}) + pooling")
    print(f"  Parametros entrenables: {trainable_params:,} / {total_params:,} ({trainable_pct:.1f}%)")
    print(f"Configuracion:")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"  loss=MultipleNegativesRankingLoss")

    # Configurar loss
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    # Entrenar
    print("Entrenando...")
    t_start = time.perf_counter()

    model.fit(
        train_objectives=[(dataloader, loss_fn)],
        epochs=args.epochs,
        warmup_steps=int(len(dataloader) * 0.1),
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
        output_path=output_dir,
    )

    total_duration = time.perf_counter() - t_start
    print(f"Entrenamiento completado en {total_duration:.1f}s")
    print(f"Modelo guardado en {output_dir}")

    # Guardar log de entrenamiento
    log = {
        "run_id": f"ft_{suffix}_{run_timestamp}",
        "timestamp": datetime.now().isoformat(),
        "device": device_info,
        "config": {
            "base_model": BASE_MODEL,
            "layers_unfrozen": args.layers,
            "total_layers": TOTAL_LAYERS,
            "unfrozen_layers": unfrozen_layers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "loss": "MultipleNegativesRankingLoss",
            "train_samples": len(train_data),
        },
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": f"{trainable_pct:.1f}%",
        "total_duration_s": round(total_duration, 1),
        "output_dir": output_dir,
    }

    log_path = os.path.join(logs_dir, f"ft_{suffix}_{run_timestamp}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"Log guardado en {log_path}")


if __name__ == "__main__":
    main()
