"""
tokenizer_trainer.py
--------------------
Entrena un tokenizer BPE desde cero sobre el corpus de PDFs
y lo guarda como PreTrainedTokenizerFast (compatible con HuggingFace).

Uso:
    from tokenizer_trainer import train_and_save_tokenizer, load_tokenizer
"""
from pathlib import Path
from typing import List

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]


def train_and_save_tokenizer(
    texts: List[str],
    save_dir: Path | str,
    vocab_size: int = 8_000,
) -> PreTrainedTokenizerFast:
    """
    Entrena un tokenizer BPE Byte-Level y lo guarda en save_dir.

    Parameters
    ----------
    texts     : Lista de textos del corpus (uno por PDF)
    save_dir  : Directorio donde guardar el tokenizer
    vocab_size: Tamaño del vocabulario (default 8000)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔤 Entrenando tokenizer BPE (vocab_size={vocab_size})...")
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder       = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        min_frequency=2,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        pad_token="<|pad|>",
        unk_token="<|unk|>",
    )
    fast_tokenizer.save_pretrained(str(save_dir))
    print(f"✅ Tokenizer guardado en: {save_dir} | vocab_size={len(fast_tokenizer)}")
    return fast_tokenizer


def load_tokenizer(load_dir: Path | str) -> PreTrainedTokenizerFast:
    """Carga un tokenizer previamente entrenado."""
    return PreTrainedTokenizerFast.from_pretrained(str(load_dir))
