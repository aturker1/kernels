
import os
os.environ["XFORMERS_IGNORE_FLASH_VERSION_CHECK"] = "1"
import time
import torch
import xformers.ops as xops
from xformers.ops import MemoryEfficientAttentionCutlassOp
from triton.testing import do_bench

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters from attached
batch_size = 1
seq_len_q = 1800
seq_len_kv = 64
num_heads = 40
head_dim = 128
dtype = torch.float16
num_iterations = 100
warmup_iterations = 10

# Generate random tensors in B, S, H, D format
q = torch.randn(batch_size, num_heads, seq_len_q, head_dim, device=device, dtype=dtype)
k = torch.randn(batch_size, num_heads, seq_len_kv, head_dim, device=device, dtype=dtype)
v = torch.randn(batch_size, num_heads, seq_len_kv, head_dim, device=device, dtype=dtype)

def xformers_attn(q, k, v):
    q_ = q.permute(0, 2, 1, 3)
    k_ = k.permute(0, 2, 1, 3)
    v_ = v.permute(0, 2, 1, 3)
    out = xops.memory_efficient_attention(q_, k_, v_, attn_bias=None, op=MemoryEfficientAttentionCutlassOp)
    return out.permute(0, 2, 1, 3)

def torch_sdpa(q, k, v):
    B, H, Sq, D = q.shape
    Sk = k.shape[2]
    q_ = q.reshape(B * H, Sq, D)
    k_ = k.reshape(B * H, Sk, D)
    v_ = v.reshape(B * H, Sk, D)
    out = torch.nn.functional.scaled_dot_product_attention(q_, k_, v_, attn_mask=None, dropout_p=0.0, is_causal=False)
    return out.reshape(B, H, Sq, D)

if __name__ == "__main__":
    print(f"Device: {device}")
    print(f"Shapes: q={q.shape}, k={k.shape}, v={v.shape}")

    lambda_xformers = lambda : xformers_attn(q, k, v)
    lambda_sdpa = lambda : torch_sdpa(q, k, v)

    x_formers_time = do_bench(lambda_xformers)  
    print(f"Xformers average time: {x_formers_time:.4f} ms")

    sdpa_time = do_bench(lambda_sdpa)  
    print(f"SDPA average time: {sdpa_time:.4f} ms")

    print(f"Speedup (xformers / sdpa): {x_formers_time / sdpa_time:.2f}x") 

    # Xformers average time: 0.0748 ms
    # SDPA average time: 0.2156 ms
    # Speedup (xformers / sdpa): 0.35x
