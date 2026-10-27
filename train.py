import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import MiniGPT

# -----------------------------------------------------------------------------
# 1. Dataset for Continuous Stream
# -----------------------------------------------------------------------------
class TextChunkDataset(Dataset):
    def __init__(self, token_tensor: torch.Tensor, block_size: int):
        self.block_size = block_size
        self.data = token_tensor

    def __len__(self):
        # Non-overlapping windows
        return max(0, (len(self.data) - 1) // self.block_size)

    def __getitem__(self, idx: int):
        start = idx * self.block_size
        chunk = self.data[start : start + self.block_size + 1]
        return chunk[:-1], chunk[1:]

# -----------------------------------------------------------------------------
# 2. Learning Rate Scheduler (Linear Warmup + Cosine Annealing)
# -----------------------------------------------------------------------------
def get_lr(it: int, warmup_iters: int, lr_decay_iters: int, min_lr: float, max_lr: float) -> float:
    # 1. Linear warmup
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    # 2. If beyond decay horizon, return minimum lr
    if it > lr_decay_iters:
        return min_lr
    # 3. Cosine decay phase
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# -----------------------------------------------------------------------------
# 3. Training Loop
# -----------------------------------------------------------------------------
def train_model():
    # Hardware device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Hyperparameters
    dataset_path = "data/dataset.pt"
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")

    block_size = 128     # Context window
    batch_size = 32      # Sequences per batch
    d_model = 256        # Hidden feature dimension
    n_heads = 4          # Multi-head attention heads (64 dims per head)
    d_ff = 1024          # Feed-forward expansion (4x)
    n_layers = 4         # Transformer blocks
    dropout = 0.15

    epochs = 25
    max_lr = 6e-4
    min_lr = 6e-5
    warmup_iters = 100
    grad_clip = 1.0

    # Load dataset & create 90/10 Train/Validation Split (Slicing 1D tensor directly)
    full_data = torch.load(dataset_path, weights_only=True)
    n_train = int(len(full_data) * 0.9)
    train_data = full_data[:n_train]
    val_data = full_data[n_train:]

    train_dataset = TextChunkDataset(train_data, block_size)
    val_dataset = TextChunkDataset(val_data, block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    vocab_size = int(full_data.max().item()) + 1
    print(f"Vocab size: {vocab_size} | Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    # Instantiate model with Weight Tying & KV Cache support
    model = MiniGPT(
        vocab_size=vocab_size,
        max_seq_len=block_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        dropout=dropout
    ).to(device)

    # Optimizer: AdamW with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()

    # Mixed precision scaler (active when device is cuda)
    use_cuda = (device == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    total_iters = epochs * len(train_loader)
    lr_decay_iters = total_iters
    curr_iter = 0

    print(f"Starting training: {epochs} epochs | {total_iters:,} total optimization steps\n")
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Update dynamic learning rate
            lr = get_lr(curr_iter, warmup_iters, lr_decay_iters, min_lr, max_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_cuda):
                logits, _ = model(x)
                B, T, V = logits.size()
                loss = criterion(logits.view(B * T, V), y.view(B * T))

            # Scaled backward pass and step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            curr_iter += 1

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_cuda):
                    vlogits, _ = model(vx)
                    B, T, V = vlogits.size()
                    vloss = criterion(vlogits.view(B * T, V), vy.view(B * T))
                val_loss += vloss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {lr:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

        # Save best model checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state": model.state_dict(),
                "vocab_size": vocab_size,
                "block_size": block_size,
                "d_model": d_model,
                "n_heads": n_heads,
                "d_ff": d_ff,
                "n_layers": n_layers
            }, ckpt_path)

    print(f"\nTraining Complete. Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    train_model()