"""
validate.py - Verifica que todo esta listo antes de ejecutar main.sh

Uso:
    python validate.py
"""

import os
import sys


def check(label, ok, detail=""):
    status = "OK" if ok else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def main():
    print("Validando entorno...\n")
    all_ok = True

    # 1. Archivos de datos
    print("1. Datos")
    for f in ["data/cowese_10.txt", "data/cowese_ft_10.txt"]:
        exists = os.path.exists(f)
        size = ""
        if exists:
            mb = os.path.getsize(f) / (1024 * 1024)
            size = f"{mb:.0f}MB"
        all_ok &= check(f, exists, size if exists else "no encontrado")

    # 2. .env
    print("\n2. Configuracion")
    env_exists = os.path.exists(".env")
    all_ok &= check(".env", env_exists, "" if env_exists else "copiar .env.example a .env")
    if env_exists:
        with open(".env") as f:
            content = f.read()
        has_key = "OPENAI_API_KEY" in content and "sk-your" not in content
        all_ok &= check("OPENAI_API_KEY", has_key, "configurada" if has_key else "falta o tiene valor placeholder")

    # 3. Dependencias
    print("\n3. Dependencias")
    deps = {
        "torch": "torch",
        "transformers": "transformers",
        "sentence_transformers": "sentence-transformers",
        "faiss": "faiss-cpu",
        "openai": "openai",
        "dotenv": "python-dotenv",
        "ragas": "ragas",
        "langchain": "langchain",
        "langchain_openai": "langchain-openai",
        "langchain_community": "langchain-community",
        "datasets": "datasets",
        "tqdm": "tqdm",
    }
    missing = []
    for module, pip_name in deps.items():
        try:
            __import__(module)
            check(pip_name, True)
        except ImportError:
            check(pip_name, False, f"pip install {pip_name}")
            missing.append(pip_name)
            all_ok = False

    # 4. CUDA
    print("\n4. GPU")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            check("CUDA", True, f"{name}, {mem:.0f}GB VRAM")
        else:
            check("CUDA", False, "no disponible, se usara CPU (sera lento)")
            all_ok = False
    except Exception:
        check("CUDA", False, "torch no instalado")
        all_ok = False

    # 5. Modelo e5-small (descarga si no esta en cache)
    print("\n5. Modelo e5-small")
    try:
        from sentence_transformers import SentenceTransformer
        print("  Cargando intfloat/multilingual-e5-small (puede descargar ~100MB)...")
        model = SentenceTransformer("intfloat/multilingual-e5-small")
        test_emb = model.encode(["query: test"], normalize_embeddings=True)
        check("Modelo cargado", True, f"dim={test_emb.shape[1]}")
        del model
    except Exception as e:
        check("Modelo", False, str(e))
        all_ok = False

    # 6. OpenAI API (test rapido)
    print("\n6. OpenAI API")
    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "user", "content": "Responde solo: ok"}],
            max_tokens=5,
        )
        check("gpt-4o-mini", True, f"respuesta: {resp.choices[0].message.content.strip()}")
    except Exception as e:
        check("gpt-4o-mini", False, str(e))
        all_ok = False

    # Resumen
    print("\n" + "=" * 40)
    if all_ok:
        print("Todo listo. Puedes ejecutar: ./main.sh")
    else:
        print("Hay problemas que resolver antes de ejecutar main.sh")
        if missing:
            print(f"\n  pip install {' '.join(missing)}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
