import time
import torch
from model import MiniGPT
from sample import sample_next_token

# -----------------------------------------------------------------------------
# 1. Generation WITHOUT KV-Cache (Quadratic O(T^2) recomputation)
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate_naive(model, prompt_tokens, max_new_tokens, device="cuda"):
    model.eval()
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    # Warmup CUDA
    _ = model(idx, kv_caches=None)
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    for _ in range(max_new_tokens):
        # Guard context window
        idx_cond = idx if idx.size(1) <= model.max_seq_len else idx[:, -model.max_seq_len:]
        logits, _ = model(idx_cond, kv_caches=None)
        last_logits = logits[:, -1, :]
        next_token = sample_next_token(last_logits, temperature=0.6, top_k=30, top_p=0.85)
        idx = torch.cat((idx, next_token), dim=1)
        
    torch.cuda.synchronize()
    return time.perf_counter() - start_time

# -----------------------------------------------------------------------------
# 2. Generation WITH KV-Cache (Linear O(T) latency / O(1) step)
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate_cached(model, prompt_tokens, max_new_tokens, device="cuda"):
    model.eval()
    curr_tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    # Warmup CUDA
    _, _ = model(curr_tokens, kv_caches=None)
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    logits, kv_caches = model(curr_tokens, kv_caches=None)
    last_logits = logits[:, -1, :]
    next_token = sample_next_token(last_logits, temperature=0.6, top_k=30, top_p=0.85)
    
    curr_token = next_token
    for _ in range(max_new_tokens - 1):
        logits, kv_caches = model(curr_token, kv_caches=kv_caches)
        last_logits = logits[:, -1, :]
        next_token = sample_next_token(last_logits, temperature=0.6, top_k=30, top_p=0.85)
        curr_token = next_token
        
    torch.cuda.synchronize()
    return time.perf_counter() - start_time

# -----------------------------------------------------------------------------
# 3. Benchmark Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "checkpoints/best_model.pt"
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Use the block_size stored directly in the checkpoint
    block_size = checkpoint.get("block_size", 256)
    
    model = MiniGPT(
        vocab_size=checkpoint.get("vocab_size", 977),
        max_seq_len=block_size,
        d_model=checkpoint.get("d_model", 512),
        n_heads=checkpoint.get("n_heads", 8),
        d_ff=checkpoint.get("d_ff", 2048),
        n_layers=checkpoint.get("n_layers", 6),
        dropout=0.0
    )
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Test up to safe sequence limits within block_size
    test_tokens = [25, 50, 75, 100]
    dummy_prompt = [1, 2, 3, 4, 5]
    
    print("\n" + "=" * 70)
    print(f"{'New Tokens':<12} | {'Naive (No Cache)':<18} | {'KV-Cache':<15} | {'Speedup':<10}")
    print("=" * 70)
    
    for n in test_tokens:
        time_naive = generate_naive(model, dummy_prompt, max_new_tokens=n, device=device)
        time_cached = generate_cached(model, dummy_prompt, max_new_tokens=n, device=device)
        speedup = time_naive / max(time_cached, 1e-6)
        print(f"{n:<12} | {time_naive * 1000:>8.2f} ms         | {time_cached * 1000:>8.2f} ms     | {speedup:>6.2f}x")
        
    print("=" * 70 + "\n")
