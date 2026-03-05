# SmallLM PDF — Small Language Model sobre tus PDFs

Sistema RAG (Retrieval-Augmented Generation) con un modelo Transformer entrenado desde cero sobre documentos PDF.

## Estructura

```
slm_pdf/
├── data/pdfs/              ← Coloca aquí tus PDFs
├── output/
│   ├── tokenizer/          ← Tokenizer BPE entrenado
│   └── model/              ← Modelos guardados
├── model.py                ← Arquitectura LLaMA-style
├── train.py                ← Script de entrenamiento
├── qa_pipeline.py          ← Pipeline RAG para preguntas
├── pdf_extractor.py        ← Extracción y limpieza de PDFs
├── dataset.py              ← Dataset PyTorch
├── tokenizer_trainer.py    ← Entrenador de tokenizer BPE
└── requirements.txt
```

## Instalación

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

### 1. Añadir PDFs
```bash
cp ruta/a/tu/documento.pdf data/pdfs/
```

### 2. Entrenar el modelo
```bash
# Modelo small (default, rápido en M3 Pro)
python3 train.py

# Modelo medium (más parámetros, mejor calidad)
python3 train.py --size medium --epochs 10 --batch_size 8 --dtype bfloat16

# Arquitectura personalizada
python3 train.py --n_layer 8 --n_head 8 --n_embd 512 --epochs 20
```

### 3. Hacer preguntas

```bash
# Ver PDFs indexados
python3 qa_pipeline.py --list-sources

# Pregunta sobre todos los documentos
python3 qa_pipeline.py --question "¿De qué trata el documento?"

# Filtrar por PDF específico
python3 qa_pipeline.py --question "¿Qué es la correlación?" --source "EDA"

# Modo generativo (requiere modelo entrenado)
python3 qa_pipeline.py --question "..." --mode generative
```

## Tamaños de modelo disponibles

| Size   | Params  | Dispositivo recomendado |
|--------|---------|------------------------|
| micro  | ~2M     | CPU, smoke-test         |
| tiny   | ~7M     | CPU o cualquier GPU     |
| small  | ~22M    | Mac M3 / GPU ≥4 GB     |
| medium | ~85M    | GPU ≥8 GB              |
| large  | ~250M   | GPU ≥16 GB             |
| xl     | ~500M   | GPU ≥24 GB             |

## Arquitectura

- **RMSNorm** — más estable y rápido que LayerNorm
- **SwiGLU** — activación gated (mejor que GELU)
- **RoPE** — Rotary Position Embeddings
- **GQA** — Grouped Query Attention
- **Flash Attention** — via `F.scaled_dot_product_attention`
- **Weight tying** — embedding ↔ LM head
