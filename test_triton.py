import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, 'src')
from prime_moment_attention.triton_kernel import triton_prime_forward

def python_prime(q, k, v, decay, scaling):
    B, L, H, D = q.shape
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    q = q * scaling
    q_pos = F.elu(q) + 1.0
    k_pos = F.elu(k) + 1.0
    
    q_pos2 = q_pos**2
    k_pos2 = k_pos**2
    
    idx = torch.arange(L, device=q.device)
    diff = idx.unsqueeze(1) - idx.unsqueeze(0)
    causal_mask = diff >= 0
    decay_mat = torch.where(
        causal_mask,
        torch.pow(decay, diff.float()),
        torch.zeros(L, L, device=q.device)
    ).view(1, 1, L, L)
    
    A_1 = decay_mat * torch.matmul(q_pos, k_pos.transpose(-1, -2))
    A_2 = decay_mat * torch.matmul(q_pos2, k_pos2.transpose(-1, -2))
    
    decay_mat_sum = decay_mat.sum(dim=-1, keepdim=True)
    num = torch.matmul(decay_mat, v) + torch.matmul(A_1, v) + 0.5 * torch.matmul(A_2, v)
    den = (decay_mat_sum + torch.sum(A_1, dim=-1, keepdim=True) + 0.5 * torch.sum(A_2, dim=-1, keepdim=True)).clamp(min=1e-3)
    
    out = (num / den)
    return out.transpose(1, 2).contiguous()

q = torch.randn(2, 64, 4, 32, device='cuda', dtype=torch.float32)
k = torch.randn(2, 64, 4, 32, device='cuda', dtype=torch.float32)
v = torch.randn(2, 64, 4, 32, device='cuda', dtype=torch.float32)
decay = 0.99
scaling = 1.0 / (32**0.5)

out_py = python_prime(q, k, v, decay, scaling)
out_tr = triton_prime_forward(q, k, v, decay, scaling)

diff = torch.max(torch.abs(out_py - out_tr))
print(f"Max diff: {diff.item()}")

