"""
qa_pipeline.py
--------------
Pipeline Preguntas & Respuestas (RAG) sobre PDFs usando SmallLM.

Modos:
  extractive (por defecto) — devuelve fragmentos relevantes del PDF formateados.
  generative               — usa SmallLM para generar texto (requiere modelo entrenado).

Uso:
    python3 qa_pipeline.py --question "De que trata el documento?"
    python3 qa_pipeline.py --question "..." --source "EDA"
    python3 qa_pipeline.py --list-sources
    python3 qa_pipeline.py --question "..." --mode generative
"""

import argparse
import logging
import os
import re
import textwrap
import warnings
from pathlib import Path
from typing import List, Tuple

# ---------- Silenciar logs ruidosos de HuggingFace / sentence-transformers ----------
os.environ.setdefault("TOKENIZERS_PARALLELISM",           "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS",     "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN",    "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING",  "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY",            "error")
warnings.filterwarnings("ignore")

# Redirigir stderr para eliminar el warning de autenticación de HF Hub
import sys as _sys
_devnull    = open(os.devnull, "w")
_real_stderr = _sys.stderr
_sys.stderr  = _devnull

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)


import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerFast

from pdf_extractor import extract_text_from_pdf

# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
PDF_DIR    = BASE_DIR / "data" / "pdfs"
MODEL_DIR  = BASE_DIR / "output" / "model" / "final"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 600   # caracteres


# ---------------------------------------------------------------------------
# Chunking por oraciones (nunca corta a mitad de palabra/frase)
# ---------------------------------------------------------------------------
def _split_sentences(text: str) -> List[str]:
    """Divide el texto en oraciones usando separadores de puntuación."""
    raw = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) >= 20:
            sentences.append(s)
    return sentences


def _chunk_sentences(text: str, max_chars: int = CHUNK_SIZE,
                     overlap_sents: int = 1) -> List[str]:
    """Agrupa oraciones en chunks de hasta max_chars caracteres."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current_sents: List[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) + 1 > max_chars and current_sents:
            chunks.append(" ".join(current_sents))
            current_sents = current_sents[-overlap_sents:]
            current_len   = sum(len(s) + 1 for s in current_sents)

        current_sents.append(sent)
        current_len += len(sent) + 1

    if current_sents:
        chunks.append(" ".join(current_sents))

    return chunks


def _clean_chunk(chunk: str) -> str:
    """Post-procesa un chunk eliminando artefactos."""
    # Referencias tipo [algo]
    chunk = re.sub(r"\[[\w\s,./:-]{1,40}\]", "", chunk)
    # Frases repetidas dentro del chunk
    phrases = re.findall(r"[A-Z][a-z\w\s]{10,60}(?= [A-Z]|$)", chunk)
    for phrase in set(phrases):
        if chunk.count(phrase) >= 2:
            first = chunk.index(phrase) + len(phrase)
            chunk = chunk[:first] + chunk[first:].replace(phrase, "")
    # Deduplicar oraciones consecutivas idénticas
    sentences = re.split(r"(?<=[.!?])\s+", chunk)
    seen: list = []
    for s in sentences:
        s = s.strip()
        if s and s not in seen:
            seen.append(s)
    chunk = " ".join(seen)
    chunk = re.sub(r"[ \t]{2,}", " ", chunk)
    lines = chunk.splitlines()
    lines = [l for l in lines if len(l.split()) > 3 or not l.strip()]
    return "\n".join(lines).strip()


def _format_answer(chunks_with_scores: List[Tuple[float, str]],
                   question: str, min_score: float = 0.15) -> str:
    """Combina chunks en una respuesta limpia, sin duplicados."""
    seen_content: List[str] = []
    parts: List[str] = []

    for score, chunk in chunks_with_scores:
        if score < min_score:
            continue
        clean = _clean_chunk(chunk)
        if not clean:
            continue

        words = set(clean.lower().split())
        is_dup = any(
            len(words & set(s.lower().split())) / max(len(words), 1) > 0.6
            for s in seen_content
        )
        if is_dup:
            continue

        seen_content.append(clean)
        parts.append(clean)

    if not parts:
        if chunks_with_scores:
            return _clean_chunk(chunks_with_scores[0][1]) or chunks_with_scores[0][1]
        return "No se encontró información relevante en los documentos."

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
class QAPipeline:
    """
    Pipeline RAG para responder preguntas sobre los PDFs.

    Parameters
    ----------
    pdf_dir         : directorio con los PDFs
    model_dir       : directorio con SmallLM entrenado
    embedding_model : nombre del modelo sentence-transformers
    """

    def __init__(
        self,
        pdf_dir: Path | str = PDF_DIR,
        model_dir: Path | str = MODEL_DIR,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.pdf_dir   = Path(pdf_dir)
        self.model_dir = Path(model_dir)

        print("🔍 Cargando modelo de embeddings...")
        self.embedder = SentenceTransformer(embedding_model)

        self.device = (
            "cuda" if torch.cuda.is_available() else
            ("mps"  if torch.backends.mps.is_available() else "cpu")
        )

        self.model: object | None = None
        self.tokenizer: object | None = None
        self._model_loaded = False

        self.chunks:  List[str] = []
        self.sources: List[str] = []
        self.index:   faiss.Index | None = None
        self._build_index()

    def _load_slm(self) -> None:
        if self._model_loaded:
            return
        from model import load_model, SmallLM, HybridSmallLM
        if not (self.model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"Modelo no encontrado en {self.model_dir}.\n"
                "Ejecuta primero: python3 train.py"
            )
        print("🤖 Cargando SLM entrenado...")
        self.model = load_model(self.model_dir, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(str(self.model_dir))
        self._model_loaded = True

    def _build_index(self) -> None:
        print("📂 Indexando PDFs...")
        pdf_files = sorted(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise RuntimeError(f"No se encontraron PDFs en: {self.pdf_dir}")

        self.chunks  = []
        self.sources = []

        for pdf_file in pdf_files:
            print(f"  🗂  {pdf_file.name}")
            text   = extract_text_from_pdf(pdf_file)
            chunks = _chunk_sentences(text, max_chars=CHUNK_SIZE)
            self.chunks  += chunks
            self.sources += [pdf_file.stem] * len(chunks)

        unique_sources = sorted(set(self.sources))
        print(f"   {len(self.chunks)} chunks de {len(unique_sources)} PDFs:")
        for src in unique_sources:
            n = self.sources.count(src)
            print(f"     [{n:4d}] {src}")

        if not self.chunks:
            raise RuntimeError("No se pudo generar ningún chunk. Revisa los PDFs.")

        embeddings = self.embedder.encode(
            self.chunks,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        dim        = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"✅ Índice FAISS listo ({self.index.ntotal} chunks, dim {dim}).\n")

    def list_sources(self) -> List[str]:
        """Devuelve la lista de PDFs indexados (sin extensión)."""
        return sorted(set(self.sources))

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
        source: str | None = None,
    ) -> List[Tuple[float, str, str]]:
        """
        Busca los chunks más relevantes.
        source: si se especifica, filtra solo chunks de ese PDF.
        Devuelve lista de (score, chunk_text, source_name).
        """
        q_emb = self.embedder.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        if source is None:
            scores, idxs = self.index.search(q_emb, top_k)
            return [
                (float(scores[0][i]), self.chunks[idxs[0][i]], self.sources[idxs[0][i]])
                for i in range(top_k)
                if idxs[0][i] >= 0
            ]
        else:
            source_lower = source.lower()
            search_k = min(len(self.chunks), top_k * 20)
            scores, idxs = self.index.search(q_emb, search_k)
            results = []
            for i in range(search_k):
                idx = idxs[0][i]
                if idx < 0:
                    break
                if source_lower in self.sources[idx].lower():
                    results.append(
                        (float(scores[0][i]), self.chunks[idx], self.sources[idx])
                    )
                if len(results) >= top_k:
                    break
            return results

    # ------------------------------------------------------------------
    def answer_extractive(
        self,
        question: str,
        top_k: int = 3,
        source: str | None = None,
        verbose: bool = True,
    ) -> str:
        results = self.retrieve(question, top_k=top_k, source=source)

        if verbose:
            src_label = f" [fuente: {source}]" if source else ""
            print(f"📎 Fragmentos recuperados{src_label}:")
            for rank, (score, chunk, src) in enumerate(results, 1):
                preview = textwrap.shorten(chunk, width=80, placeholder="...")
                print(f"  [{rank}] {src} | relevancia={score:.3f} | {preview}")
            print()

        return _format_answer([(s, c) for s, c, _ in results], question)

    def answer_generative(
        self,
        question: str,
        top_k: int = 3,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
        source: str | None = None,
        verbose: bool = True,
    ) -> str:
        self._load_slm()
        results = self.retrieve(question, top_k=top_k, source=source)

        if verbose:
            print("📎 Contextos recuperados:")
            for rank, (score, chunk, src) in enumerate(results, 1):
                preview = textwrap.shorten(chunk, width=80, placeholder="...")
                print(f"  [{rank}] {src} | relevancia={score:.3f} | {preview}")
            print()

        context = "\n---\n".join(_clean_chunk(c) for _, c, _ in results)
        prompt  = (
            f"Contexto:\n{context}\n\n"
            f"Pregunta: {question}\n"
            f"Respuesta:"
        )

        max_ctx = self.model.cfg.n_positions - max_new_tokens
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max(max_ctx, 1),
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(self.device)

        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.3,
        )
        new_tokens = output[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    def ask(
        self,
        question: str,
        top_k: int = 3,
        max_new_tokens: int = 150,
        mode: str = "extractive",
        source: str | None = None,
        verbose: bool = True,
    ) -> str:
        if mode == "generative":
            return self.answer_generative(question, top_k, max_new_tokens,
                                          source=source, verbose=verbose)
        return self.answer_extractive(question, top_k, source=source, verbose=verbose)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG QA sobre tus PDFs")
    p.add_argument("--question",  type=str, required=False, default=None)
    p.add_argument("--top_k",     type=int, default=3)
    p.add_argument("--mode",      type=str, default="extractive",
                   choices=["extractive", "generative"])
    p.add_argument("--max_new_tokens", type=int, default=150)
    p.add_argument("--source",    type=str, default=None,
                   help="Filtrar por PDF. Ej: --source EDA (nombre sin extension).")
    p.add_argument("--list-sources", action="store_true",
                   help="Lista los PDFs indexados y sale.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _print_answer(answer: str) -> None:
    """Formatea e imprime la respuesta con buen wrapping."""
    print(f"{'─'*60}")
    for block in answer.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith(("•", "-", "*", "–")):
                print(f"  {line}")
            else:
                for wrapped in textwrap.wrap(line, width=78,
                                              initial_indent="  ",
                                              subsequent_indent="    "):
                    print(wrapped)
        print()


if __name__ == "__main__":
    args = parse_args()
    pipeline = QAPipeline()

    if args.list_sources:
        print("📚 PDFs indexados:")
        for src in pipeline.list_sources():
            n = pipeline.sources.count(src)
            print(f"  [{n:4d} chunks] {src}")
        raise SystemExit(0)

    if not args.question:
        print("Usa --question '...' o --list-sources")
        raise SystemExit(1)

    print(f"❓ Pregunta: {args.question}")
    if args.source:
        print(f"   Fuente: {args.source}")
    print(f"   Modo: {args.mode}\n")

    answer = pipeline.ask(
        question=args.question,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        mode=args.mode,
        source=args.source,
        verbose=not args.quiet,
    )

    print(f"💬 Respuesta:")
    _print_answer(answer)
