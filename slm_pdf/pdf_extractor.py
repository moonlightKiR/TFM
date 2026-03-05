"""
pdf_extractor.py
----------------
Extrae y limpia texto de todos los PDFs en un directorio.

Uso:
    python3 pdf_extractor.py
    from pdf_extractor import extract_text_from_pdf, extract_text_from_pdfs, get_all_text
"""

import os
import re
from pathlib import Path
from typing import List

import pdfplumber


# Patrones de ruido habituales en PDFs de presentaciones/apuntes
_NOISE_PATTERNS = [
    # Cabeceras/pies de página con numeración: "Algo ... 48" o "Algo - 48"
    re.compile(r"^.{5,80}[\s\-–—]+\d{1,3}\s*$", re.MULTILINE),
    # Solo número de página
    re.compile(r"^\s*\d{1,3}\s*$", re.MULTILINE),
    # URLs sueltas en línea propia
    re.compile(r"^\s*https?://\S+\s*$", re.MULTILINE),
    # Líneas de guiones/puntos separadores
    re.compile(r"^[\s\-–—=_•·.]{5,}\s*$", re.MULTILINE),
    # Títulos de sección repetidos en slides (ej: "Visualizar distribuciones y relaciones")
    # — líneas que empiezan con mayúscula, no terminan en punto, y tienen 3-8 palabras
    re.compile(r"^[A-Z][a-záéíóúñü\w]+(?: [a-záéíóúñü\w]+){2,7}$", re.MULTILINE),
]


def _remove_unicode_noise(text: str) -> str:
    """
    Elimina caracteres unicode fuera del rango básico que producen basura
    (símbolos matemáticos, alfabetos matemáticos, etc.).
    Mantiene: ASCII, Latin Extended (ñ, á, é...) y puntuación habitual.
    """
    result = []
    for ch in text:
        cp = ord(ch)
        if (
            cp <= 0x024F        # ASCII + Latin Extended
            or 0x2010 <= cp <= 0x206F   # Puntuación general (guiones, etc.)
            or ch in "\n\t "
        ):
            result.append(ch)
        else:
            result.append(" ")
    return "".join(result)


# Sustituciones regex simples post-limpieza unicode
_CLEANUPS = [
    (re.compile(r"[ \t]{2,}"), " "),    # Espacios múltiples → uno
]


def _clean_text(text: str) -> str:
    """Limpieza profunda del texto extraído de un PDF."""

    # 1. Eliminar patrones de ruido (cabeceras, numeraciones, etc.)
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub("", text)

    # 2. Eliminar caracteres unicode matemáticos/basura
    text = _remove_unicode_noise(text)

    # 3. Sustituciones regex post-limpieza
    for pattern, replacement in _CLEANUPS:
        text = pattern.sub(replacement, text)

    # 4. Normalizar saltos de línea
    # Unir líneas que terminan abruptamente (no con punto/puntuación)
    text = re.sub(r"(?<![.!?:;\n])\n(?=[a-záéíóúñü])", " ", text, flags=re.IGNORECASE)
    # Reducir múltiples saltos de línea a máximo 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Eliminar líneas vacías que solo tienen espacios
    text = re.sub(r"\n[ \t]+\n", "\n\n", text)
    # Eliminar líneas demasiado cortas (< 15 chars) que suelen ser artefactos
    lines = [l for l in text.splitlines() if len(l.strip()) >= 15 or not l.strip()]
    text = "\n".join(lines)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extrae el texto de un único PDF."""
    pdf_path = Path(pdf_path)
    pages: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(layout=False)
            if page_text and len(page_text.strip()) > 20:
                pages.append(page_text)
    raw = "\n\n".join(pages)
    return _clean_text(raw)


def extract_text_from_pdfs(pdf_dir: str | Path) -> List[str]:
    """Extrae texto de todos los PDFs en un directorio."""
    pdf_dir   = Path(pdf_dir)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No se encontraron PDFs en: {pdf_dir}")

    texts: List[str] = []
    for pdf_file in pdf_files:
        print(f"  📄 Extrayendo: {pdf_file.name}")
        text = extract_text_from_pdf(pdf_file)
        if text:
            texts.append(text)

    print(f"✅ {len(texts)} PDFs procesados correctamente.")
    return texts


def get_all_text(pdf_dir: str | Path) -> str:
    """Devuelve todo el texto concatenado."""
    return "\n\n".join(extract_text_from_pdfs(pdf_dir))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    base_dir = Path(__file__).parent
    pdf_dir  = base_dir / "data" / "pdfs"
    print(f"Buscando PDFs en: {pdf_dir}\n")
    try:
        text = get_all_text(pdf_dir)
        print(f"\nTexto extraído y limpio (primeras 800 chars):\n{'─'*60}")
        print(text[:800])
        print(f"\nTotal: {len(text):,} caracteres")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
