import re

with open("src/prime_moment_attention/chunked_ssd.py", "r") as f:
    content = f.read()

old_intra = """    # 2. Intra-chunk attention via dense GEMM
    dot1 = torch.matmul(q_chunks, k_chunks.transpose(-1, -2))
    dot2 = 0.5 * torch.matmul(q_chunks**2, (k_chunks**2).transpose(-1, -2))
    A_intra = decay_mask * (1.0 + dot1 + dot2)
    num_intra = torch.matmul(A_intra, v_chunks)
    den_intra = torch.sum(A_intra, dim=-1, keepdim=True)"""

new_intra = """    # 2. Intra-chunk attention via positive feature map
    q_pos = F.elu(q_chunks) + 1.0
    k_pos = F.elu(k_chunks) + 1.0
    A_intra = decay_mask * torch.matmul(q_pos, k_pos.transpose(-1, -2))
    num_intra = torch.matmul(A_intra, v_chunks)
    den_intra = torch.sum(A_intra, dim=-1, keepdim=True)"""

content = content.replace(old_intra, new_intra)

old_deltas = """    w_chunk = torch.pow(decay, (C - 1 - idx).float()).view(1, 1, 1, C, 1).expand(B, H, num_chunks, C, 1).clone()
    if pad_len > 0:
        w_chunk[:, :, -1, -pad_len:, :] = 0.0

    k_w = k_chunks * w_chunk
    k2_w = (k_chunks**2) * w_chunk
    v_w = v_chunks * w_chunk

    chunk_S0 = torch.sum(v_w, dim=3, keepdim=True)
    chunk_S1 = torch.matmul(k_w.transpose(-1, -2), v_chunks)
    chunk_S2 = torch.matmul(k2_w.transpose(-1, -2), v_chunks)

    chunk_K0 = torch.sum(w_chunk, dim=3, keepdim=True)
    chunk_K1 = torch.sum(k_w, dim=3, keepdim=True)
    chunk_K2 = torch.sum(k2_w, dim=3, keepdim=True)"""

new_deltas = """    w_chunk = torch.pow(decay, (C - 1 - idx).float()).view(1, 1, 1, C, 1).expand(B, H, num_chunks, C, 1).clone()
    if pad_len > 0:
        w_chunk[:, :, -1, -pad_len:, :] = 0.0

    k_w = k_pos * w_chunk

    chunk_S1 = torch.matmul(k_w.transpose(-1, -2), v_chunks)
    chunk_K1 = torch.sum(k_w, dim=3, keepdim=True)
    # create zero states for S0, S2, K0, K2 API compat
    chunk_S0 = torch.zeros(B, H, num_chunks, 1, D, device=device)
    chunk_S2 = torch.zeros(B, H, num_chunks, D, D, device=device)
    chunk_K0 = torch.zeros(B, H, num_chunks, 1, 1, device=device)
    chunk_K2 = torch.zeros(B, H, num_chunks, 1, D, device=device)"""

content = content.replace(old_deltas, new_deltas)

old_readout = """    decay_into = torch.pow(decay, (idx + 1).float()).view(1, 1, 1, C, 1)
    q_in = q_chunks * decay_into
    q2_in = (q_chunks**2) * decay_into

    carry_num = (
        inter_S0 * decay_into
        + torch.matmul(q_in, inter_S1)
        + 0.5 * torch.matmul(q2_in, inter_S2)
    )
    carry_den = (
        inter_K0 * decay_into
        + torch.sum(q_in * inter_K1, dim=-1, keepdim=True)
        + 0.5 * torch.sum(q2_in * inter_K2, dim=-1, keepdim=True)
    )"""

new_readout = """    decay_into = torch.pow(decay, (idx + 1).float()).view(1, 1, 1, C, 1)
    q_in = q_pos * decay_into

    carry_num = torch.matmul(q_in, inter_S1)
    carry_den = torch.sum(q_in * inter_K1, dim=-1, keepdim=True)"""

content = content.replace(old_readout, new_readout)

with open("src/prime_moment_attention/chunked_ssd.py", "w") as f:
    f.write(content)
