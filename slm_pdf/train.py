"""
train.py
--------
Bucle de entrenamiento desde cero para SmallLM.

Pasos:
  1. Extrae texto de data/pdfs/
  2. Entrena (o carga) el tokenizer BPE
  3. Construye el dataset de secuencias
  4. Crea el modelo SmallLM desde cero
  5. Entrena con bucle PyTorch puro
  6. Guarda modelo y tokenizer en output/model/final/

Uso:
    python3 train.py                          # small por defecto
    python3 train.py --size medium            # modelo más grande
    python3 train.py --size large             # grande (GPU recomendada)
    python3 train.py --n_layer 8 --n_head 8 --n_embd 512  # tamaño custom
    python3 train.py --size tiny --max_steps 20           # smoke-test
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from pdf_extractor import extract_text_from_pdfs
from tokenizer_trainer import train_and_save_tokenizer, load_tokenizer
from dataset import PDFTextDataset
from model import SmallLM, SLMConfig, build_model

# ---------------------------------------------------------------------------
BASE_DIR      = Path(__file__).parent
PDF_DIR       = BASE_DIR / "data" / "pdfs"
TOKENIZER_DIR = BASE_DIR / "output" / "tokenizer"
MODEL_DIR     = BASE_DIR / "output" / "model"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena SmallLM sobre PDFs")
    # -- Arquitectura -------------------------------------------------------
    p.add_argument(
        "--size",
        default="small",
        choices=["micro", "tiny", "small", "medium", "large", "xl"],
        metavar="SIZE",
        help="Tamaño: micro/tiny/small/medium/large/xl. Ignorado si se usan "
             "--n_layer/--n_head/--n_embd.",
    )
    p.add_argument("--n_layer",   type=int, default=None,
                   help="[custom] Número de capas Transformer.")
    p.add_argument("--n_head",    type=int, default=None,
                   help="[custom] Cabezas de atención (n_embd % n_head == 0).")
    p.add_argument("--n_embd",    type=int, default=None,
                   help="[custom] Dimensión del embedding.")
    p.add_argument("--n_kv_head", type=int, default=None,
                   help="[custom] Cabezas K/V para GQA (<= n_head).")
    p.add_argument("--architecture", type=str, default="transformer",
                   choices=["transformer", "hybrid"],
                   help="'transformer' (solo atención) o 'hybrid' (alterna GRU+Atención).")
    p.add_argument("--rnn_ratio",    type=float, default=0.5,
                   help="Fracción de capas que son GRU en modo hybrid (default: 0.5).")
    # -- Dataset ------------------------------------------------------------
    p.add_argument("--vocab_size", type=int, default=8_000)
    p.add_argument("--block_size", type=int, default=256,
                   help="Ventana de contexto en tokens.")
    # -- Entrenamiento -------------------------------------------------------
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch_size", type=int,   default=4,
                   help="Batch size. En M3 Pro puedes usar 8-16.")
    p.add_argument("--grad_accum", type=int,   default=2,
                   help="Gradient accumulation steps.")
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--max_steps",  type=int,   default=-1,
                   help="Límite de steps (-1 = sin límite). Útil para smoke-test.")
    p.add_argument("--eval_every", type=int,   default=50)
    p.add_argument("--save_every", type=int,   default=100)
    p.add_argument("--dtype",      type=str,   default="bfloat16",
                   choices=["float32", "bfloat16"],
                   help="Precision: bfloat16 es ~2x mas rapido en M3/M4 y CUDA.")
    p.add_argument("--retrain_tokenizer", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# LR Scheduler: warmup lineal + cosine decay
# ---------------------------------------------------------------------------
def get_lr(step: int, warmup_steps: int, total_steps: int,
           max_lr: float, min_lr: float) -> float:
    if total_steps <= 0:
        return max_lr
    if step < warmup_steps:
        return max_lr * max(step, 1) / max(warmup_steps, 1)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: SmallLM, loader: DataLoader, device: str,
             max_batches: int = 20) -> float:
    model.eval()
    total_loss, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ids    = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss, _ = model(ids, labels=labels)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # -- 1. PDFs -------------------------------------------------------------
    print("=" * 60)
    print("📂  Paso 1: Extracción de texto de PDFs")
    print("=" * 60)
    texts = extract_text_from_pdfs(PDF_DIR)

    # -- 2. Tokenizer --------------------------------------------------------
    print("\n" + "=" * 60)
    print("🔤  Paso 2: Tokenizer")
    print("=" * 60)
    tok_exists = (TOKENIZER_DIR / "tokenizer.json").exists()
    if tok_exists and not args.retrain_tokenizer:
        print(f"Cargando tokenizer desde: {TOKENIZER_DIR}")
        tokenizer = load_tokenizer(TOKENIZER_DIR)
    else:
        tokenizer = train_and_save_tokenizer(texts, TOKENIZER_DIR, args.vocab_size)

    # -- 3. Dataset ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("📚  Paso 3: Dataset")
    print("=" * 60)

    dataset = PDFTextDataset(texts, tokenizer, block_size=args.block_size)
    if len(dataset) < 4:
        args.block_size = args.block_size // 2
        print(f"⚠️  Dataset pequeño, reduciendo block_size a {args.block_size}")
        dataset = PDFTextDataset(texts, tokenizer, block_size=args.block_size)

    if len(dataset) < 2:
        raise RuntimeError(
            "Dataset demasiado pequeño. Añade más PDFs o reduce --block_size."
        )

    n_eval  = max(1, len(dataset) // 5)
    n_train = len(dataset) - n_eval
    train_ds, eval_ds = random_split(
        dataset, [n_train, n_eval],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"   Train: {n_train} | Eval: {n_eval} bloques")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
    )

    # -- 4. Modelo -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("🤖  Paso 4: Modelo (desde cero)")
    print("=" * 60)

    custom = any(v is not None for v in [args.n_layer, args.n_head, args.n_embd])
    if custom:
        cfg = getattr(SLMConfig, args.size)(vocab_size=len(tokenizer))
        if args.n_layer:    cfg.n_layer   = args.n_layer
        if args.n_head:     cfg.n_head    = args.n_head
        if args.n_embd:     cfg.n_embd    = args.n_embd
        if args.n_kv_head:  cfg.n_kv_head = args.n_kv_head
        else:               cfg.n_kv_head = cfg.n_head
        print(f"   Modo: arquitectura CUSTOM")
        print(f"   n_layer={cfg.n_layer} | n_head={cfg.n_head} | "
              f"n_kv_head={cfg.n_kv_head} | n_embd={cfg.n_embd}")
    else:
        cfg = getattr(SLMConfig, args.size)(vocab_size=len(tokenizer))
        print(f"   Modo: preset '{args.size}'")

    cfg.n_positions  = args.block_size
    cfg.architecture = args.architecture
    cfg.rnn_ratio    = args.rnn_ratio
    model = build_model(cfg)

    device = (
        "cuda" if torch.cuda.is_available() else
        ("mps"  if torch.backends.mps.is_available() else "cpu")
    )

    # Configurar dtype y autocast
    pt_dtype     = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    use_autocast = (args.dtype == "bfloat16")
    # MPS usa backend 'cpu' para autocast
    autocast_device = "cuda" if device == "cuda" else "cpu"

    print(f"   Dispositivo: {device} | Dtype: {args.dtype}")
    model.to(device)
    if args.dtype == "bfloat16" and device == "mps":
        model = model.to(torch.bfloat16)
        # nn.GRU no soporta bfloat16 en MPS — forzar a float32 explícitamente
        # (model._apply() bypasea nuestro override to() en GRUBlock)
        from model import GRUBlock
        for module in model.modules():
            if isinstance(module, nn.GRU):
                module.float()

    # -- 5. Optimizador ------------------------------------------------------
    decay_params   = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay_params,   "weight_decay": 0.1},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = max(steps_per_epoch * args.epochs, 1)
    warmup_steps = max(1, total_steps // 10)

    # -- 6. Bucle de entrenamiento -------------------------------------------
    print("\n" + "=" * 60)
    print("🚀  Paso 5: Entrenamiento")
    print("=" * 60)
    print(f"   Steps totales: {total_steps} | Warmup: {warmup_steps} | "
          f"Épocas: {args.epochs}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_eval_loss = float("inf")
    global_step    = 0
    accum_loss     = 0.0
    accum_count    = 0
    t0             = time.time()
    done           = False

    model.train()
    optimizer.zero_grad()

    for epoch in range(1, args.epochs + 1):
        if done:
            break
        for batch_idx, batch in enumerate(train_loader):
            if done:
                break

            ids    = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            if args.dtype == "bfloat16" and device == "mps":
                ids    = ids.to(torch.long)
                labels = labels.to(torch.long)

            with torch.autocast(autocast_device, dtype=pt_dtype, enabled=use_autocast):
                loss, _ = model(ids, labels=labels)
            (loss / args.grad_accum).backward()
            accum_loss  += loss.item()
            accum_count += 1

            last_batch = (batch_idx + 1 == len(train_loader))
            if (accum_count % args.grad_accum == 0) or last_batch:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                lr = get_lr(global_step, warmup_steps, total_steps,
                            args.lr, args.lr / 10)
                for gr in optimizer.param_groups:
                    gr["lr"] = lr

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = accum_loss / accum_count
                accum_loss, accum_count = 0.0, 0

                if global_step % 10 == 0 or global_step == 1:
                    elapsed = time.time() - t0
                    print(f"  Epoch {epoch:3d} | Step {global_step:5d}/{total_steps}"
                          f" | loss {avg_loss:.4f} | lr {lr:.2e} | {elapsed:.0f}s")

                if global_step % args.eval_every == 0:
                    eval_loss = evaluate(model, eval_loader, device)
                    print(f"  >>> Eval loss: {eval_loss:.4f}")
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        model.save(MODEL_DIR / "best")
                        tokenizer.save_pretrained(str(MODEL_DIR / "best"))
                        print(f"  >>> 💾 Mejor modelo guardado (loss={eval_loss:.4f})")

                if global_step % args.save_every == 0:
                    model.save(MODEL_DIR / f"ckpt_{global_step}")

                if args.max_steps > 0 and global_step >= args.max_steps:
                    done = True
                    break

    # -- 7. Guardado final ---------------------------------------------------
    print("\n" + "=" * 60)
    print("💾  Guardando modelo final")
    print("=" * 60)
    model.save(MODEL_DIR / "final")
    tokenizer.save_pretrained(str(MODEL_DIR / "final"))

    final_eval = evaluate(model, eval_loader, device)
    elapsed    = time.time() - t0
    print(f"\n✅ Completado en {elapsed / 60:.1f} min | "
          f"Steps: {global_step} | Eval loss final: {final_eval:.4f}")

    if final_eval > 4.0:
        print("\n💡 El eval loss todavía es alto. Sugerencias:")
        print("   - Añade más PDFs a data/pdfs/")
        print("   - Aumenta las épocas: --epochs 30")
        print("   - Prueba un modelo más pequeño: --size tiny")
    print("\nPara hacer preguntas:")
    print("  python3 qa_pipeline.py --question '¿De qué trata el documento?'")


if __name__ == "__main__":
    main()
