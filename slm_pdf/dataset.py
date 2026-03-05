"""
dataset.py
----------
Dataset PyTorch para entrenar SmallLM sobre texto concatenado de PDFs.
Divide el corpus tokenizado en bloques de longitud fija (block_size).
"""
from typing import List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast


class PDFTextDataset(Dataset):
    """
    Dataset de bloques de tokens de longitud fija (block_size).
    Cada item es un dict con 'input_ids' y 'labels' (desplazados en 1).
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizerFast,
        block_size: int = 256,
    ):
        print("📚 Tokenizando corpus...")
        full_text = " ".join(texts)
        tokens = tokenizer.encode(full_text)

        # Dividir en bloques de block_size tokens
        self.examples: List[List[int]] = []
        for i in range(0, len(tokens) - block_size, block_size):
            chunk = tokens[i: i + block_size + 1]
            if len(chunk) == block_size + 1:
                self.examples.append(chunk)

        print(
            f"✅ Dataset: {len(self.examples)} bloques de {block_size} tokens "
            f"(total {len(tokens):,} tokens en corpus)."
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        chunk = self.examples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return {"input_ids": x, "labels": y}
