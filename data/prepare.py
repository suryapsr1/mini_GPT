from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_bpe import load_tokenizer, encode


def prepare_dataset(
    corpus_path: str = "data/input.txt",
    rules_path: str = "tokenizer/bpe_rules.json",
    output_bin: str = "data/dataset.pt"
):
    """
    Encodes the corpus into an integer PyTorch tensor and saves it to disk.
    """
    print("Loading tokenizer rules...")
    merges, token_to_id, _ = load_tokenizer(rules_path)

    print(f"Reading {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Encoding text into subword token IDs...")
    token_ids = encode(text, merges, token_to_id)
    print(f"Total encoded tokens: {len(token_ids):,}")

    tensor_data = torch.tensor(token_ids, dtype=torch.long)
    output_path = Path(output_bin)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor_data, output_path)
    print(f"Saved pre-tokenized binary tensor to {output_bin} ({tensor_data.numel() * 8 / 1024:.2f} KB)")

if __name__ == "__main__":
    prepare_dataset()