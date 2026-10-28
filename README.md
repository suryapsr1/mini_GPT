# mini_GPT

A lightweight, from-scratch implementation of a decoder-only GPT model in PyTorch. 

This project covers the full pipeline: training a custom BPE tokenizer, gathering data, training the autoregressive transformer, and sampling text with optimized key-value (KV) caching.

---

## Project Structure

- `gather_data.py`: Downloads and prepares the text corpus.
- `train_bpe.py`: Trains a Byte-Pair Encoding (BPE) tokenizer on the raw text.
- `tokenizer/`: Stores tokenizer files, vocabularies, and merges.
- `model.py`: Core decoder-only transformer architecture (self-attention, MLP blocks, residual connections).
- `train.py`: Training loop, loss computation, and checkpoint saving.
- `sample.py`: Generates text completions from a trained model.
- `benchmark_kv.py`: Benchmarks inference speed comparing standard generation vs. KV cache generation.
- `checkpoints/`: Saved model weights and training states.

---

## Getting Started

### 1. Requirements

Ensure you have Python 3.8+ and PyTorch installed:

```bash
pip install torch
