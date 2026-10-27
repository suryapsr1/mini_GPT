import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network: expands 512 -> 2048 -> 512 with GELU/ReLU.
    """
    def __init__(self, d_model: int =512, d_ff: int =2048, dropout: float =0.1):
        super().__init__()
        self.w1= nn.Linear(d_model, d_ff)
        self.w2= nn.Linear(d_ff, d_model)
        self.dropout= nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        #modern GPT often use Gelu
        return self.dropout(self.w2(F.gelu(self.w1(x))))


class GPTBlock(nn.Module):
    """
    A single Pre-LN Transformer Decoder Block (Self-Attention + FFN).
    """
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        #instatitation class
        # 1. Normalization layers placed BEFORE sublayers (Pre-LN)
        self.norm1= nn.LayerNorm(d_model)
        self.norm2= nn.LayerNorm(d_model)

        #2. Sublayers
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.ffn  = FeedForward(d_model= d_model, d_ff= d_ff, dropout= dropout)

    def forward(self, x: torch.Tensor, kv_cache=None):
        # Pass cache into attention sublayer
        attn_out, new_cache = self.attn(self.norm1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache






    
    
class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention supporting KV-caching during autoregressive generation.
    """
    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1, max_seq_len: int = 1024):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Combined projection for Q, K, V
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("bias", mask)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.size()

        # Step 1: Project Q, K, V
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)

        # Step 2: Reshape to [B, n_heads, T, d_k]
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Step 3: Handle KV Cache
        if kv_cache is not None:
            past_k, past_v = kv_cache
            # Concatenate along the sequence dimension (dim=2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Save current state for next generation step
        new_kv_cache = (k, v)

        # Total key sequence length (past cached + new)
        T_total = k.size(2)

        # Step 4: Scaled Dot-Product Attention: [B, n_heads, T, d_k] @ [B, n_heads, d_k, T_total] -> [B, n_heads, T, T_total]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Step 5: Masking (only needed if T > 1, e.g. during initial prompt prefill)
        if T > 1:
            scores = scores.masked_fill(self.bias[:, :, :T, :T_total] == 0, float('-inf'))

        # Step 6: Softmax and weighted sum
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # [B, n_heads, T, T_total] @ [B, n_heads, T_total, d_k] -> [B, n_heads, T, d_k]
        out = torch.matmul(weights, v)

        # Step 7: Recombine heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))

        return out, new_kv_cache



class MiniGPT(nn.Module):
    """
    Decoder-Only Autoregressive Transformer with Pre-LN and learned positional embeddings.
    """
    #instantiating 
    def __init__(
            self,
            vocab_size: int,
            max_seq_len: int =256,
            d_model: int=512,
            n_heads: int =8,
            d_ff: int =2048,
            n_layers: int =6,
            dropout: float =0.1

    ):
        super(). __init__()

        self.max_seq_len= max_seq_len
        self.d_model= d_model

        #token and learned position embeddings which will be trained
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)


        # Stack of pre-LN blocks
        self.blocks = nn.ModuleList([
            GPTBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
            for _ in range(n_layers)
        ])

        self.drop = nn.Dropout(dropout)

        # 3. Final Pre-LN LayerNorm
        self.ln_f = nn.LayerNorm(d_model)

        # 4. Language Model Head (Projects hidden states to vocabulary logits)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # -----------------------------------------------------------
        # WEIGHT TYING: Tie output projection weights to input embedding table
        # -----------------------------------------------------------
        self.lm_head.weight = self.tok_emb.weight





    def forward(self, idx: torch.Tensor, kv_caches=None):
        B, T = idx.size()

        # If caches exist, this is an incremental single-token step
        # Positional index must match the true current token position, not 0!
        if kv_caches is not None:
            # kv_caches[0][0] shape: [B, n_heads, past_len, d_k]
            past_len = kv_caches[0][0].size(2)
            pos = torch.tensor([past_len], dtype=torch.long, device=idx.device)
        else:
            past_len = 0
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_vecs = self.tok_emb(idx)
        pos_vecs = self.pos_emb(pos)
        x = self.drop(tok_vecs + pos_vecs)

        new_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, updated_cache = block(x, kv_cache=layer_cache)
            new_caches.append(updated_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits, new_caches


if __name__ == "__main__":
    vocab_size = 100
    model = MiniGPT(vocab_size=vocab_size, max_seq_len=32, d_model=64, n_heads=2, d_ff=128, n_layers=2)

    # 1. Prefill step: prompt with 3 tokens
    prompt = torch.tensor([[10, 20, 30]])
    logits, caches = model(prompt)
    print("Prefill output logits shape:", logits.shape)       # Expected: [1, 3, 100]
    print("Cached keys shape in Layer 0:", caches[0][0].shape) # Expected: [1, 2, 3, 32]

    # 2. Cached step: pass ONLY ONE new token
    new_token = torch.tensor([[40]])
    new_logits, updated_caches = model(new_token, kv_caches=caches)
    print("Single-token step logits shape:", new_logits.shape)          # Expected: [1, 1, 100]
    print("Updated cached keys shape:", updated_caches[0][0].shape)      # Expected: [1, 2, 4, 32]
    

