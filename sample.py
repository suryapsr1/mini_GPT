import argparse
import torch
import torch.nn.functional as F
from model import MiniGPT
from train_bpe import decode, encode, load_tokenizer

def sample_next_token(logits: torch.Tensor,
                      generated_tokens: list[int] = None,
                      repetition_penalty: float = 1.15,
                      temperature: float = 0.5,
                      top_k: int = 30,
                      top_p: float = 0.85) -> torch.Tensor:
    
    # 1. Apply repetition penalty to avoid repetitive loops (* radius * radius * radius)
    if generated_tokens and repetition_penalty > 1.0:
        for prev_token in set(generated_tokens[-30:]): # Penalize words from the last 30 tokens
            if logits[0, prev_token] > 0:
                logits[0, prev_token] /= repetition_penalty
            else:
                logits[0, prev_token] *= repetition_penalty

    # 2. Temperature scaling
    logits = logits / max(temperature, 1e-5)
    
    # 3. Top-k filtering
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        kth_val = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < kth_val, torch.full_like(logits, -float("Inf")), logits)
        
    # 4. Top-p (Nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))
        
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


@torch.no_grad()
def generate_clean(model: torch.nn.Module,
                   prompt_str: str,
                   merges: list,
                   token_to_id: dict,
                   id_to_token: dict,
                   max_new_tokens: int = 70,
                   temperature: float = 0.4,
                   top_k: int = 25,
                   top_p: float = 0.85,
                   device: str = "cpu") -> str:
    model.eval()
    model.to(device)
    
    prompt_tokens = encode(prompt_str, merges, token_to_id)
    curr_tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    # Initial prefill pass
    logits, kv_caches = model(curr_tokens, kv_caches=None)
    last_logits = logits[:, -1, :]
    
    next_token = sample_next_token(
        last_logits, 
        generated_tokens=prompt_tokens, 
        temperature=temperature, 
        top_k=top_k, 
        top_p=top_p
    )
    generated = prompt_tokens + [next_token.item()]
    curr_token = next_token
    
    # Autoregressive loop with KV-cache
    for _ in range(max_new_tokens - 1):
        logits, kv_caches = model(curr_token, kv_caches=kv_caches)
        last_logits = logits[:, -1, :]
        
        next_token = sample_next_token(
            last_logits, 
            generated_tokens=generated, 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p
        )
        token_id = next_token.item()
        generated.append(token_id)
        curr_token = next_token
        
        # Check text-level stop condition dynamically
        current_text = decode(generated, id_to_token)
        if "# <|" in current_text or "<|endoftext|>" in current_text:
            break

    # Clean text post-processing
    final_text = decode(generated, id_to_token)
    
    # Strip boundary artifacts
    for stop_seq in ["# <|", "<|endoftext|>", "<|"]:
        if stop_seq in final_text:
            final_text = final_text.split(stop_seq)[0]
            
    return final_text.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="def calculate_area(radius):")
    parser.add_argument("--temp", type=float, default=0.35)
    parser.add_argument("--tokens", type=int, default=60)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load("checkpoints/best_model.pt", map_location=device)
    merges, token_to_id, id_to_token = load_tokenizer("tokenizer/bpe_rules.json")
    
    model = MiniGPT(
        vocab_size=checkpoint.get("vocab_size", len(token_to_id)),
        max_seq_len=checkpoint.get("block_size", 256),
        d_model=checkpoint.get("d_model", 512),
        n_heads=checkpoint.get("n_heads", 8),
        d_ff=checkpoint.get("d_ff", 2048),
        n_layers=checkpoint.get("n_layers", 6),
        dropout=0.0
    )
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)

    print(f"\nPrompt: {args.prompt}")
    output = generate_clean(
        model=model,
        prompt_str=args.prompt,
        merges=merges,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        max_new_tokens=args.tokens,
        temperature=args.temp,
        device=device
    )
    print("Generated Output:\n")
    print(output)
